"""Phase 1, Batch B: serve-model live path + local HTTP hardening.

W1.1 rollback sticks (high_water), W1.2 relative run_dir + visible
skips, W1.3 replace-not-stack, W1.5 provenance at the serve-model
boundary, W1.9 shared localhttp rules. Plain unittest, no torch, no
network (HF_HUB_OFFLINE=1); HTTP checks run against a real local
server on 127.0.0.1 with an ephemeral port.

Run from the kit directory:
  PYTHONPATH=. python3 -m unittest tests.test_phase1_serve -v
"""

import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from wsp import atomic, localhttp, loop, recipe, serve_model

MODEL_ID = "Qwen/Qwen3-0.6B"
LAYERS = [
    {"index": 27, "columns": [3, 4],
     "direction": [0.1, 0.2, 0.3], "alpha": 0.25},
    {"index": 26, "columns": [7],
     "direction": [0.4, 0.5, 0.6], "alpha": 0.25},
]
LAYER_INFOS = [
    {"index": 27, "hidden_size": 3, "intermediate_size": 16},
    {"index": 26, "hidden_size": 3, "intermediate_size": 16},
]
TOTAL_PARAMS = 1_000_000


class FakeModel:
    """A stand-in for RealModel: same methods, no weights."""

    def __init__(self, fail_layers=None):
        self.applied = []        # (layer_index, columns, alpha)
        self.restored = []       # saved handles
        self.calls = []          # ("apply"/"restore", layer_index)
        self.fail_layers = set(fail_layers or [])
        self.layer_infos = [dict(i) for i in LAYER_INFOS]
        self.total_params = TOTAL_PARAMS
        self.generate_calls = []
        self.generate_error = None

    def apply_edit(self, layer_index, columns, direction, alpha):
        if layer_index in self.fail_layers:
            raise ValueError(f"layer {layer_index} refused")
        self.calls.append(("apply", layer_index))
        self.applied.append((layer_index, list(columns), alpha))
        return len(columns) * 10, (layer_index, list(columns), "orig")

    def restore(self, saved):
        self.calls.append(("restore", saved[0]))
        self.restored.append(saved)

    def generate(self, prompt, max_new_tokens, temperature, seed):
        if self.generate_error is not None:
            raise self.generate_error
        self.generate_calls.append(
            (prompt, max_new_tokens, temperature, seed))
        return ("fake reply to " + prompt, True, 5)


def _write_version(versions_dir, version, run_name="run-1",
                   model_id=MODEL_ID, layers=None, verdict="SHIP",
                   report=True, latest_extra=None, run_dir_value=None):
    """Write one version: run dir (edit.json + report.json) + latest.

    ``verdict`` None means a report with no verdict (QUICK evidence).
    ``report`` False means no report.json at all. ``latest_extra``
    is merged into the latest.json entry; ``run_dir_value``
    overrides the stored run_dir (for traversal tests).
    """
    run_dir = os.path.join(versions_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    rec = {
        "format_version": 1,
        "model_id": model_id,
        "dtype": "bfloat16",
        "layers": layers if layers is not None
        else [dict(layer) for layer in LAYERS],
    }
    recipe.write_recipe(os.path.join(run_dir, "edit.json"), rec)
    if report:
        rep = {
            "gate": {"passed": True},
            "real_demo": {
                "edit_sha256": recipe.edit_sha256(rec),
                "model_id": model_id,
            },
        }
        if verdict is not None:
            rep["gate"]["verdict"] = verdict
        atomic.atomic_write_json(os.path.join(run_dir, "report.json"),
                                 rep)
    entry = {
        "version": version,
        "run_dir": (run_dir if run_dir_value is None
                    else run_dir_value),
        "model": model_id,
        "weights_changed": 30,
    }
    if latest_extra:
        entry.update(latest_extra)
    atomic.atomic_write_json(
        os.path.join(versions_dir, "latest.json"), entry)
    return run_dir, rec


def _live(model, **kwargs):
    return serve_model.LiveModel(model, MODEL_ID, log=lambda *a: None,
                                 **kwargs)


def _rep():
    return {
        "after": {"failing": 80},
        "before": {"failing": 40},
        "weights_changed": 100,
        "gate": {"passed": True, "verdict": "SHIP"},
        "real_demo": {"edit_sha256": "abc123", "model_id": "m"},
    }


# ---------------------------------------------------------------- W1.1

class TestHighWater(unittest.TestCase):
    def test_rollback_then_same_latest_not_reapplied(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            self.assertEqual(live.status()["current_version"], 1)
            self.assertTrue(live.rollback_last())
            status = live.status()
            self.assertEqual(status["current_version"], 0)
            self.assertEqual(status["high_water_version"], 1)
            self.assertEqual(status["rolled_back_versions"], [1])
            applied_before = list(model.applied)
            # Same latest.json again: NOT applied a second time.
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, applied_before)

    def test_new_version_after_rollback_applies(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            self.assertTrue(live.rollback_last())
            v2_layers = [{"index": 27, "columns": [5],
                          "direction": [0.1, 0.2, 0.3], "alpha": 0.5}]
            _write_version(versions_dir, 2, run_name="run-2",
                           layers=v2_layers)
            self.assertTrue(live.poll_once(versions_dir))
            status = live.status()
            self.assertEqual(status["current_version"], 2)
            self.assertEqual(status["high_water_version"], 2)
            self.assertEqual(model.applied[-1], (27, [5], 0.5))

    def test_rollback_over_http_then_poll_stays_rolled_back(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            handler = serve_model.make_handler(live)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=httpd.serve_forever,
                                      daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{httpd.server_address[1]}"

                def post(path, obj):
                    req = _json_request(base + path, obj)
                    with _urlopen(req) as resp:
                        return json.load(resp)

                def get(path):
                    with _urlopen(base + path) as resp:
                        return json.load(resp)

                out = post("/rollback", {})
                self.assertTrue(out["rolled_back"])
                # One watcher tick on the same latest.json: still 0.
                self.assertFalse(live.poll_once(versions_dir))
                self.assertEqual(get("/status")["current_version"], 0)
            finally:
                httpd.shutdown()
                httpd.server_close()


def _json_request(url, obj):
    import urllib.request

    return urllib.request.Request(
        url, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")


def _urlopen(req):
    import urllib.request

    return urllib.request.urlopen(req, timeout=5)


# ---------------------------------------------------------------- W1.2

class TestRunDirResolution(unittest.TestCase):
    def test_relative_out_dir_from_chdir(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = os.getcwd()
            try:
                os.chdir(td)
                os.makedirs("out/run-1")
                _write_version("out", 1)
                version = loop.record_version("out", "out/run-1",
                                              _rep(), "m")
                self.assertEqual(version, 1)
                with open("out/versions.json") as f:
                    entry = json.load(f)[0]
                self.assertEqual(entry["run_dir"],
                                 os.path.abspath("out/run-1"))
                self.assertEqual(entry["run_dir_rel"], "run-1")
                self.assertEqual(entry["edit_sha256"], "abc123")
                self.assertEqual(entry["verdict"], "SHIP")
                self.assertEqual(entry["evidence_level"], "ship")
                model = FakeModel()
                live = _live(model)
                self.assertTrue(live.poll_once("out"))
                self.assertEqual(
                    live.status()["edits"][0]["run_dir"],
                    os.path.abspath("out/run-1"))
            finally:
                os.chdir(cwd)

    def test_copied_out_dir_applies_via_rel(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out")
            os.makedirs(os.path.join(out, "run-1"))
            _write_version(out, 1)
            loop.record_version(out, os.path.join(out, "run-1"),
                                _rep(), "m")
            moved = os.path.join(td, "out-moved")
            shutil.move(out, moved)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(moved))
            self.assertTrue(
                live.status()["edits"][0]["run_dir"].startswith(moved))

    def test_traversal_run_dir_rel_refused(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            outside = os.path.join(td, "outside")
            os.makedirs(versions_dir)
            # A real edit sits outside; the traversal must still refuse.
            _write_version(outside, 1)
            latest = {"version": 1, "run_dir_rel": "../outside",
                      "model": MODEL_ID}
            atomic.atomic_write_json(
                os.path.join(versions_dir, "latest.json"), latest)
            model = FakeModel()
            live = _live(model)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, [])
            skipped = live.status()["skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]["version"], 1)
            self.assertIn("outside", skipped[0]["reason"])

    def test_absolute_run_dir_outside_refused(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            outside = os.path.join(td, "outside")
            os.makedirs(versions_dir)
            _write_version(outside, 1)
            _write_version(versions_dir, 1,
                           run_dir_value=os.path.join(outside, "run-1"))
            model = FakeModel()
            live = _live(model)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, [])
            self.assertEqual(len(live.status()["skipped"]), 1)

    def test_symlink_run_dir_refused(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            outside = os.path.join(td, "outside")
            os.makedirs(versions_dir)
            os.makedirs(os.path.join(outside, "run-1"))
            link = os.path.join(versions_dir, "sneaky")
            try:
                os.symlink(os.path.join(outside, "run-1"), link)
            except OSError:
                self.skipTest("OS refused the symlink")
            _write_version(versions_dir, 1,
                           latest_extra={"run_dir_rel": "sneaky"})
            model = FakeModel()
            live = _live(model)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, [])

    def test_corrupt_versions_json_renamed_and_numbering_kept(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out")
            os.makedirs(os.path.join(out, "run-1"))
            loop.record_version(out, os.path.join(out, "run-1"),
                                _rep(), "m")
            loop.record_version(out, os.path.join(out, "run-1"),
                                _rep(), "m")
            with open(os.path.join(out, "versions.json"), "w") as f:
                f.write("{not json")
            with self.assertRaises(RuntimeError):
                loop.record_version(out, os.path.join(out, "run-1"),
                                    _rep(), "m")
            corrupt = [n for n in os.listdir(out)
                       if n.startswith("versions.json.corrupt-")]
            self.assertEqual(len(corrupt), 1)
            # The founder restores a valid file with the two entries;
            # the next version is 3, numbering never restarts.
            entries = [
                {"version": 1, "run_dir": os.path.join(out, "run-1"),
                 "model": "m"},
                {"version": 2, "run_dir": os.path.join(out, "run-1"),
                 "model": "m"},
            ]
            atomic.atomic_write_json(
                os.path.join(out, "versions.json"), entries)
            self.assertEqual(
                loop.record_version(out, os.path.join(out, "run-1"),
                                    _rep(), "m"), 3)

    def test_loop_survives_corrupt_versions_json(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out = os.path.join(td, "out")
            os.makedirs(watch_dir)
            os.makedirs(out)
            with open(os.path.join(out, "versions.json"), "w") as f:
                f.write("{not json")
            eval_path = os.path.join(watch_dir, "eval.json")
            with open(eval_path, "w") as f:
                json.dump({"prompts": []}, f)

            def run_fn(model_id, path, out_dir, log=print, **kw):
                return _rep(), os.path.join(out_dir, "run-1"), 0

            state, summary = loop.loop_once(
                "m", watch_dir, out, {}, run_fn=run_fn,
                log=lambda *a: None)
            self.assertEqual(summary["versions"], 0)
            corrupt = [n for n in os.listdir(out)
                       if n.startswith("versions.json.corrupt-")]
            self.assertEqual(len(corrupt), 1)


# ---------------------------------------------------------------- W1.3

class TestReplaceNotStack(unittest.TestCase):
    def test_newer_version_replaces_active(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            self.assertEqual(live.status()["weights_changed"], 30)
            v2_layers = [{"index": 27, "columns": [5],
                          "direction": [0.1, 0.2, 0.3], "alpha": 0.5}]
            _write_version(versions_dir, 2, run_name="run-2",
                           layers=v2_layers)
            self.assertTrue(live.poll_once(versions_dir))
            # The old edit was restored (26 then 27) before v2 applied.
            self.assertEqual(model.calls, [
                ("apply", 27), ("apply", 26),
                ("restore", 26), ("restore", 27),
                ("apply", 27),
            ])
            status = live.status()
            self.assertEqual(status["applied_edits"], 1)
            self.assertEqual(status["weights_changed"], 10)
            self.assertEqual(status["active_edit"]["version"], 2)
            self.assertEqual(
                [e["version"] for e in status["history"]], [1, 2])

    def test_rollback_after_replace_restores_original_only(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1)
            model = FakeModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            v2_layers = [{"index": 27, "columns": [5],
                          "direction": [0.1, 0.2, 0.3], "alpha": 0.5}]
            _write_version(versions_dir, 2, run_name="run-2",
                           layers=v2_layers)
            self.assertTrue(live.poll_once(versions_dir))
            applies_before = list(model.applied)
            self.assertTrue(live.rollback_last())
            # Only v2's handles were restored; v1 was NOT re-applied.
            self.assertEqual(model.applied, applies_before)
            self.assertEqual(model.restored[-1], (27, [5], "orig"))
            self.assertEqual(live.status()["current_version"], 0)
            self.assertEqual(live.status()["active_edit"], None)

    def test_failed_v2_restores_v1(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            v1_layers = [{"index": 27, "columns": [3],
                          "direction": [0.1, 0.2, 0.3], "alpha": 0.25}]
            _write_version(versions_dir, 1, layers=v1_layers)
            model = FakeModel(fail_layers={26})
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            v2_layers = [{"index": 26, "columns": [7],
                          "direction": [0.4, 0.5, 0.6], "alpha": 0.25}]
            _write_version(versions_dir, 2, run_name="run-2",
                           layers=v2_layers)
            self.assertFalse(live.poll_once(versions_dir))
            # v1 was restored for the swap, v2 failed, v1 re-applied.
            self.assertEqual(model.calls, [
                ("apply", 27),
                ("restore", 27),
                ("apply", 27),
            ])
            status = live.status()
            self.assertEqual(status["active_edit"]["version"], 1)
            self.assertEqual(status["current_version"], 2)
            self.assertEqual(len(status["skipped"]), 1)
            self.assertEqual(status["skipped"][0]["version"], 2)

    def test_failed_v2_and_failed_v1_reapply_leaves_original(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            v1_layers = [{"index": 27, "columns": [3],
                          "direction": [0.1, 0.2, 0.3], "alpha": 0.25}]
            _write_version(versions_dir, 1, layers=v1_layers)

            class FlakyModel(FakeModel):
                def __init__(self):
                    super().__init__(fail_layers={26})
                    self._applies = 0

                def apply_edit(self, layer_index, columns, direction,
                               alpha):
                    self._applies += 1
                    if layer_index == 27 and self._applies > 1:
                        raise ValueError("layer 27 refused on re-apply")
                    return super().apply_edit(layer_index, columns,
                                              direction, alpha)

            model = FlakyModel()
            live = _live(model)
            self.assertTrue(live.poll_once(versions_dir))
            v2_layers = [{"index": 26, "columns": [7],
                          "direction": [0.4, 0.5, 0.6], "alpha": 0.25}]
            _write_version(versions_dir, 2, run_name="run-2",
                           layers=v2_layers)
            self.assertFalse(live.poll_once(versions_dir))
            status = live.status()
            self.assertEqual(status["active_edit"], None)
            self.assertEqual(status["weights_changed"], 0)
            self.assertEqual(status["applied_edits"], 0)
            self.assertEqual(len(status["skipped"]), 1)
            self.assertIn("re-apply", status["skipped"][0]["reason"])


# ---------------------------------------------------------------- W1.5

class TestProvenanceAtServe(unittest.TestCase):
    def test_quick_report_refused_without_flag(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1, verdict=None)
            model = FakeModel()
            live = _live(model)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, [])
            self.assertIn("--allow-quick",
                          live.status()["skipped"][0]["reason"])
            # Same edit, same server, with the flag: applies.
            live_quick = _live(FakeModel(), allow_quick=True)
            self.assertTrue(live_quick.poll_once(versions_dir))

    def test_missing_report_refused_even_with_flag(self):
        # Serve-model never applies without a report.json: the loop
        # always writes one, and --allow-quick only covers QUICK
        # reports (no verdict), not a missing report. The
        # --allow-unverified escape hatch exists only for the
        # manual apply-edit command.
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1, report=False)
            live = _live(FakeModel())
            self.assertFalse(live.poll_once(versions_dir))
            live_quick = _live(FakeModel(), allow_quick=True)
            self.assertFalse(live_quick.poll_once(versions_dir))
            self.assertIn("no report.json",
                          live.status()["skipped"][0]["reason"])

    def test_ship_report_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            _write_version(versions_dir, 1, verdict="SHIP")
            live = _live(FakeModel())
            self.assertTrue(live.poll_once(versions_dir))

    def test_wrong_hash_refused_even_with_flag(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            run_dir, rec = _write_version(versions_dir, 1)
            rep = {"gate": {"passed": True, "verdict": "SHIP"},
                   "real_demo": {"edit_sha256": "0" * 64,
                                 "model_id": MODEL_ID}}
            atomic.atomic_write_json(
                os.path.join(run_dir, "report.json"), rep)
            live = _live(FakeModel(), allow_quick=True)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(len(live.status()["skipped"]), 1)

    def test_report_for_other_model_refused(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            os.makedirs(versions_dir)
            run_dir, rec = _write_version(versions_dir, 1)
            rep = {"gate": {"passed": True, "verdict": "SHIP"},
                   "real_demo": {"edit_sha256":
                                 recipe.edit_sha256(rec),
                                 "model_id": "some/other-model"}}
            atomic.atomic_write_json(
                os.path.join(run_dir, "report.json"), rep)
            live = _live(FakeModel())
            self.assertFalse(live.poll_once(versions_dir))


# ---------------------------------------------------------------- W1.9

def _http(port, method, path, body=None, headers=None, skip_host=False):
    """One raw HTTP request; returns (status, parsed-or-raw body)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest(method, path, skip_host=skip_host)
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        if body is not None and "Content-Length" not in (headers or {}):
            conn.putheader("Content-Length", str(len(body)))
        conn.endheaders()
        if body is not None:
            conn.send(body)
        resp = conn.getresponse()
        data = resp.read()
        try:
            parsed = json.loads(data)
        except ValueError:
            parsed = data
        return resp.status, parsed
    finally:
        conn.close()


def _json_body(obj):
    return json.dumps(obj).encode()


class TestLocalHttp(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.live = _live(self.model)
        self.handler = serve_model.make_handler(self.live)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]
        self.own_origin = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _post_generate(self, obj, extra_headers=None):
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return _http(self.port, "POST", "/generate",
                     body=_json_body(obj), headers=headers)

    def test_wrong_host_403(self):
        status, _ = _http(self.port, "GET", "/status", skip_host=True,
                          headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_missing_host_403(self):
        status, _ = _http(self.port, "GET", "/status", skip_host=True)
        self.assertEqual(status, 403)

    def test_foreign_origin_403(self):
        status, _ = self._post_generate(
            {"prompt": "hi"},
            extra_headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_own_origin_ok(self):
        status, body = self._post_generate(
            {"prompt": "hi"},
            extra_headers={"Origin": self.own_origin})
        self.assertEqual(status, 200)
        self.assertEqual(body["text"], "fake reply to hi")

    def test_no_origin_ok(self):
        status, body = self._post_generate({"prompt": "hi"})
        self.assertEqual(status, 200)

    def test_post_without_json_content_type_415(self):
        status, body = _http(
            self.port, "POST", "/generate", body=b'{"prompt": "hi"}',
            headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_post_missing_content_length_400(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port,
                                          timeout=5)
        try:
            conn.putrequest("POST", "/generate")
            conn.putheader("Content-Type", "application/json")
            conn.endheaders()
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 400)
        finally:
            conn.close()

    def test_post_non_integer_content_length_400(self):
        status, _ = _http(
            self.port, "POST", "/generate", body=b"{}",
            headers={"Content-Type": "application/json",
                     "Content-Length": "abc"})
        self.assertEqual(status, 400)

    def test_post_negative_content_length_400(self):
        status, _ = _http(
            self.port, "POST", "/generate", body=b"{}",
            headers={"Content-Type": "application/json",
                     "Content-Length": "-5"})
        self.assertEqual(status, 400)

    def test_post_body_over_cap_413(self):
        status, _ = _http(
            self.port, "POST", "/generate", body=b"{}",
            headers={"Content-Type": "application/json",
                     "Content-Length": str(localhttp.MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)

    def test_post_body_not_object_400(self):
        status, _ = _http(
            self.port, "POST", "/generate", body=b"[1, 2]",
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_post_bad_json_400(self):
        status, _ = _http(
            self.port, "POST", "/generate", body=b"{nope",
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_prompt_empty_400(self):
        status, _ = self._post_generate({"prompt": ""})
        self.assertEqual(status, 400)

    def test_prompt_too_long_400(self):
        status, _ = self._post_generate(
            {"prompt": "x" * (localhttp.MAX_PROMPT_CHARS + 1)})
        self.assertEqual(status, 400)

    def test_max_new_tokens_clamped(self):
        status, _ = self._post_generate(
            {"prompt": "hi", "max_new_tokens": 0})
        self.assertEqual(status, 200)
        self.assertEqual(self.model.generate_calls[-1][1], 1)
        status, _ = self._post_generate(
            {"prompt": "hi", "max_new_tokens": 9999})
        self.assertEqual(status, 200)
        self.assertEqual(self.model.generate_calls[-1][1],
                         localhttp.MAX_NEW_TOKENS_CAP)

    def test_max_new_tokens_non_int_400(self):
        status, _ = self._post_generate(
            {"prompt": "hi", "max_new_tokens": "64"})
        self.assertEqual(status, 400)

    def test_temperature_out_of_range_400(self):
        for bad in (3, -0.5, "hot"):
            status, _ = self._post_generate(
                {"prompt": "hi", "temperature": bad})
            self.assertEqual(status, 400, bad)

    def test_seed_out_of_range_400(self):
        for bad in (-1, 2 ** 31, 1.5, "0"):
            status, _ = self._post_generate(
                {"prompt": "hi", "seed": bad})
            self.assertEqual(status, 400, bad)

    def test_token_gates_generate_and_rollback(self):
        handler = serve_model.make_handler(self.live, token="secret")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]

            def post(path, obj, auth=None):
                headers = {"Content-Type": "application/json"}
                if auth is not None:
                    headers["Authorization"] = auth
                return _http(port, "POST", path,
                             body=_json_body(obj), headers=headers)

            status, _ = post("/generate", {"prompt": "hi"})
            self.assertEqual(status, 403)
            status, _ = post("/generate", {"prompt": "hi"},
                             auth="Bearer wrong")
            self.assertEqual(status, 403)
            status, body = post("/generate", {"prompt": "hi"},
                                auth="Bearer secret")
            self.assertEqual(status, 200)
            status, _ = post("/rollback", {}, auth="Bearer secret")
            self.assertEqual(status, 200)
            # /status stays open without a token.
            status, _ = _http(port, "GET", "/status")
            self.assertEqual(status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_handler_timeout_is_30(self):
        self.assertEqual(self.handler.timeout,
                         localhttp.REQUEST_TIMEOUT_S)
        self.assertEqual(localhttp.REQUEST_TIMEOUT_S, 30)

    def test_internal_error_is_500_without_detail(self):
        self.model.generate_error = RuntimeError("boom secret detail")
        status, body = self._post_generate({"prompt": "hi"})
        self.assertEqual(status, 500)
        self.assertEqual(body, {"error": "internal error"})
        self.assertNotIn("boom", json.dumps(body))

    def test_status_answers_while_generate_blocked(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingModel(FakeModel):
            def generate(self, prompt, max_new_tokens, temperature,
                         seed):
                entered.set()
                release.wait(timeout=10)
                return ("done", True, 1)

        model = BlockingModel()
        live = _live(model)
        handler = serve_model.make_handler(live)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            port = httpd.server_address[1]
            result = {}

            def do_generate():
                result["generate"] = _http(
                    port, "POST", "/generate",
                    body=_json_body({"prompt": "hi"}),
                    headers={"Content-Type": "application/json"})

            gen_thread = threading.Thread(target=do_generate,
                                          daemon=True)
            gen_thread.start()
            self.assertTrue(entered.wait(timeout=5))
            status, body = _http(port, "GET", "/status")
            self.assertEqual(status, 200)
            self.assertEqual(body["current_version"], 0)
            release.set()
            gen_thread.join(timeout=5)
            self.assertEqual(result["generate"][0], 200)
        finally:
            release.set()
            httpd.shutdown()
            httpd.server_close()


class TestStatusPageHttp(unittest.TestCase):
    def test_host_origin_checks_and_escaping(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run-1")
            os.makedirs(run_dir)
            rep = {
                "gate": {"passed": True},
                "before": {"failing": 10},
                "after": {"failing": 90},
                "model": 'Evil <script>alert(1)</script> "quoted"',
                "real_demo": {
                    "eval_name": 'x" onmouseover="alert(1)',
                    "model_id": "m",
                },
                "weights_changed": 5,
                "weights_changed_pct": 0.001,
            }
            atomic.atomic_write_json(
                os.path.join(run_dir, "report.json"), rep)
            from wsp import serve

            handler = serve.make_handler(td)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=httpd.serve_forever,
                                      daemon=True)
            thread.start()
            try:
                port = httpd.server_address[1]
                status, page = _http(port, "GET", "/")
                self.assertEqual(status, 200)
                if isinstance(page, bytes):
                    page = page.decode("utf-8")
                self.assertNotIn("<script>alert(1)</script>", page)
                self.assertIn("&lt;script&gt;", page)
                self.assertNotIn('onmouseover="alert(1)', page)
                status, _ = _http(port, "GET", "/", skip_host=True,
                                  headers={"Host": "evil.example"})
                self.assertEqual(status, 403)
                status, _ = _http(port, "GET", "/",
                                  headers={"Origin":
                                           "https://evil.example"})
                self.assertEqual(status, 403)
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
