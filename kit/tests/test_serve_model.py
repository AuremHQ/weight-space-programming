"""Tests for `wsp serve-model` (live weight override). Plain unittest.

No torch needed: the model is a fake with the RealModel interface
(apply_edit / restore / generate). Run from the kit directory:
PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from wsp import recipe, serve_model


class FakeModel:
    """A stand-in for RealModel: same methods, no weights."""

    def __init__(self):
        self.applied = []   # (layer_index, columns, alpha)
        self.restored = []  # saved handles
        # Layer shape the fixtures use: hidden 3, intermediate 16.
        self.layer_infos = [
            {"index": 27, "hidden_size": 3, "intermediate_size": 16},
            {"index": 26, "hidden_size": 3, "intermediate_size": 16},
        ]
        self.total_params = 1_000_000

    def apply_edit(self, layer_index, columns, direction, alpha):
        self.applied.append((layer_index, list(columns), alpha))
        return len(columns) * 10, (layer_index, list(columns), "orig")

    def restore(self, saved):
        self.restored.append(saved)

    def generate(self, prompt, max_new_tokens, temperature, seed):
        return ("fake reply to " + prompt, True, 5)


def _write_edit(run_dir, model_id="Qwen/Qwen3-0.6B", layers=None):
    os.makedirs(run_dir, exist_ok=True)
    recipe_dict = {
        "format_version": 1,
        "model_id": model_id,
        "dtype": "bfloat16",
        "layers": layers or [
            {"index": 27, "columns": [3, 4],
             "direction": [0.1, 0.2, 0.3], "alpha": 0.25},
            {"index": 26, "columns": [7],
             "direction": [0.4, 0.5, 0.6], "alpha": 0.25},
        ],
    }
    recipe.write_recipe(os.path.join(run_dir, "edit.json"), recipe_dict)
    # A matching report.json: the gate pass this edit claims. The
    # fixture edit is a SHIP-level pass, so serve-model applies it
    # with default flags.
    with open(os.path.join(run_dir, "report.json"), "w") as f:
        json.dump({
            "gate": {"passed": True, "verdict": "SHIP"},
            "real_demo": {
                "edit_sha256": recipe.edit_sha256(recipe_dict),
                "model_id": model_id,
            },
        }, f)
    return recipe_dict


def _write_latest(versions_dir, version, run_dir):
    with open(os.path.join(versions_dir, "latest.json"), "w") as f:
        json.dump({"version": version, "run_dir": run_dir,
                   "model": "Qwen/Qwen3-0.6B",
                   "weights_changed": 128}, f)


class TestLoadRecipe(unittest.TestCase):
    def test_valid_and_bad(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run-1")
            recipe = _write_edit(run_dir)
            loaded = serve_model.load_recipe(
                os.path.join(run_dir, "edit.json"))
            self.assertEqual(loaded["model_id"], recipe["model_id"])
            bad = os.path.join(td, "bad.json")
            with open(bad, "w") as f:
                json.dump({"format_version": 99, "model_id": "m",
                           "layers": []}, f)
            with self.assertRaises(ValueError):
                serve_model.load_recipe(bad)


class TestApplyRecipe(unittest.TestCase):
    def test_applies_all_layers(self):
        model = FakeModel()
        recipe = {"format_version": 1, "model_id": "m", "layers": [
            {"index": 27, "columns": [3, 4],
             "direction": [0.1, 0.2, 0.3], "alpha": 0.25},
            {"index": 26, "columns": [7],
             "direction": [0.4, 0.5, 0.6], "alpha": 0.5},
        ]}
        changed, saved = serve_model.apply_recipe(model, recipe)
        self.assertEqual(changed, 30)  # 2 cols + 1 col, x10 each
        self.assertEqual(len(saved), 2)
        self.assertEqual(model.applied[0][0], 27)
        self.assertEqual(model.applied[1][2], 0.5)

    def test_failed_layer_restores_earlier_layers(self):
        class FailingModel(FakeModel):
            def apply_edit(self, layer_index, columns, direction,
                           alpha):
                if layer_index == 26:
                    raise ValueError("over the 1% cap")
                return super().apply_edit(layer_index, columns,
                                          direction, alpha)

        model = FailingModel()
        recipe = {"format_version": 1, "model_id": "m", "layers": [
            {"index": 27, "columns": [3],
             "direction": [0.1, 0.2, 0.3], "alpha": 0.25},
            {"index": 26, "columns": [7],
             "direction": [0.4, 0.5, 0.6], "alpha": 0.5},
        ]}
        with self.assertRaises(ValueError):
            serve_model.apply_recipe(model, recipe)
        # The first layer was applied, then restored exactly.
        self.assertEqual(len(model.restored), 1)
        self.assertEqual(model.restored[0][0], 27)


class TestLiveModel(unittest.TestCase):
    def test_poll_applies_new_version_once_then_rolls_back(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            run_dir = os.path.join(versions_dir, "run-1")
            os.makedirs(versions_dir)
            _write_edit(run_dir)
            _write_latest(versions_dir, 1, run_dir)
            model = FakeModel()
            live = serve_model.LiveModel(model, "Qwen/Qwen3-0.6B",
                                         log=lambda *a: None)
            self.assertTrue(live.poll_once(versions_dir))
            self.assertEqual(len(model.applied), 2)
            self.assertEqual(live.status()["current_version"], 1)
            self.assertEqual(live.status()["weights_changed"], 30)
            # Second poll: nothing new.
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(len(model.applied), 2)
            # Rollback restores both layers, version back to 0.
            self.assertTrue(live.rollback_last())
            self.assertEqual(len(model.restored), 2)
            self.assertEqual(live.status()["current_version"], 0)
            self.assertFalse(live.rollback_last())

    def test_wrong_model_edit_is_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            versions_dir = os.path.join(td, "out")
            run_dir = os.path.join(versions_dir, "run-1")
            os.makedirs(versions_dir)
            _write_edit(run_dir, model_id="some/other-model")
            _write_latest(versions_dir, 1, run_dir)
            model = FakeModel()
            live = serve_model.LiveModel(model, "Qwen/Qwen3-0.6B",
                                         log=lambda *a: None)
            self.assertFalse(live.poll_once(versions_dir))
            self.assertEqual(model.applied, [])
            # And it is not retried on the next poll.
            self.assertFalse(live.poll_once(versions_dir))

    def test_no_latest_json_is_quiet(self):
        with tempfile.TemporaryDirectory() as td:
            live = serve_model.LiveModel(FakeModel(), "m",
                                         log=lambda *a: None)
            self.assertFalse(live.poll_once(td))


class TestHttpEndpoints(unittest.TestCase):
    def test_generate_status_rollback(self):
        model = FakeModel()
        live = serve_model.LiveModel(model, "m", log=lambda *a: None)
        # Pretend an edit was applied, so rollback has something.
        live._saved_stack.append([(27, [3], "orig")])
        live.applied.append({"version": 1, "run_dir": "r",
                             "weights_changed": 10, "at": "t"})
        live.last_version = 1
        handler = serve_model.make_handler(live)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever,
                                  daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"

            def post(path, obj):
                req = urllib.request.Request(
                    base + path,
                    data=json.dumps(obj).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST")
                with urllib.request.urlopen(req) as resp:
                    return json.load(resp)

            def get(path):
                with urllib.request.urlopen(base + path) as resp:
                    return json.load(resp)

            out = post("/generate", {"prompt": "hello"})
            self.assertEqual(out["text"], "fake reply to hello")
            status = get("/status")
            self.assertEqual(status["current_version"], 1)
            out = post("/rollback", {})
            self.assertTrue(out["rolled_back"])
            self.assertEqual(get("/status")["current_version"], 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
