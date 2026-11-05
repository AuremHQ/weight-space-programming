"""Tests for wsp.status and wsp.serve: the green-only wins feed,
counters, chip states, the one-liner, and the running marker.

No network beyond localhost, no model loads, no hardcoded production
numbers — fixtures are small synthetic reports in the same format
(report v1) as the real ones.
"""

import http.server
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
import urllib.request

from wsp.status import (
    chip_state,
    parse_win,
    remove_running_marker,
    render_html,
    scan_reports,
    status_line,
    write_running_marker,
)
from wsp.serve import make_handler


def _report(path, passed, before, after, model="kit (2-layer MLP)",
            real=False, causal=None, weights=0, pct=0.0, eval_name=None):
    rep = {
        "report_version": 1,
        "kit_version": "0.2.0",
        "model": model,
        "weights_changed": weights,
        "weights_changed_pct": pct,
        "before": {"failing": before, "good": 100.0, "neighbours": 100.0},
        "after": {"failing": after, "good": 100.0, "neighbours": 100.0},
        "gate": {"passed": passed},
        "temperatures_checked": [0.0, 0.7],
        "editor": "test",
        "note": "test fixture",
    }
    if real:
        rep["real_demo"] = {
            "eval_name": eval_name or "loop-fix-test",
            "causal_proof": {"change_points": causal if causal is not None else 0.0},
        }
    with open(path, "w") as f:
        json.dump(rep, f)
    return path


class StatusTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="wsp-status-")
        self.t = 1_700_000_000  # fixed mtime base, so ordering is explicit

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _mk(self, name, mtime_offset, **kw):
        path = os.path.join(self.dir, name, "report.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _report(path, **kw)
        os.utime(path, (self.t + mtime_offset, self.t + mtime_offset))
        return path

    def test_gate_false_never_appears(self):
        """A refused run never reaches the feed or the page."""
        self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        self._mk("qwen-bad", 20, passed=False, before=31.11, after=31.11,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=32.22, eval_name="loop-fix-qwen3-0.6b")
        self._mk("gpt2-bad", 30, passed=False, before=28.33, after=28.33,
                 model="GPT2 (real open-weight model, CPU)",
                 real=True, causal=-1.11, eval_name="loop-fix-gpt2-124m")
        scan = scan_reports(self.dir)
        self.assertEqual(len(scan["wins"]), 1)
        self.assertEqual(scan["wins"][0]["short"], "pipeline test")
        page = render_html(scan)
        self.assertNotIn("loop-fix-qwen3-0.6b", page)
        self.assertNotIn("loop-fix-gpt2-124m", page)
        for banned in ("REFUSED", "FAILED", "failed", "refused",
                       "failure", "#dc2626", "#ef4444"):
            self.assertNotIn(banned, page, banned)
        self.assertIsNone(re.search(r"\bred\b", page, re.IGNORECASE),
                          "the word 'red' must not appear")
        # The honest record is linked, not shown.
        self.assertIn("REAL-RUNS.md", page)

    def test_counters_and_best_gain(self):
        self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        self._mk("real-win", 20, passed=True, before=30.0, after=36.5,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=10.0, weights=1000, pct=0.02,
                 eval_name="loop-fix-qwen3-0.6b")
        scan = scan_reports(self.dir)
        self.assertEqual(len(scan["wins"]), 2)
        self.assertEqual(scan["real_passes"], 1)
        line = status_line(self.dir)
        self.assertIn("2 approved improvements", line)
        self.assertIn("best +11.29 (pipeline test)", line)
        self.assertIn("1 real-model pass", line)

    def test_chip_idle_when_latest_run_did_not_pass(self):
        self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        self._mk("qwen-bad", 20, passed=False, before=31.11, after=31.11,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=32.22)
        scan = scan_reports(self.dir)
        self.assertEqual(chip_state(scan), "IDLE")
        self.assertIn("IDLE", status_line(self.dir))

    def test_chip_passed_when_latest_run_passed(self):
        self._mk("qwen-bad", 10, passed=False, before=31.11, after=31.11,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=32.22)
        self._mk("kit", 20, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        scan = scan_reports(self.dir)
        self.assertEqual(chip_state(scan), "PASSED")

    def test_status_line_exact(self):
        """The documented example, on a fixture matching the real
        reports: one kit win, four refused real runs, latest refused."""
        self._mk("kit", 10, passed=True, before=64.13885677119401,
                 after=75.42648011916296, weights=128, pct=0.7722007722007722)
        for i, name in enumerate(("qwen-1", "qwen-2", "qwen-clean", "gpt2")):
            self._mk(name, 20 + i, passed=False, before=31.11, after=31.11,
                     model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                     real=True, causal=32.22)
        self.assertEqual(
            status_line(self.dir),
            "wsp-kit by Aurem \u25cf IDLE \u2014 1 approved improvement "
            "\u00b7 best +11.29 (pipeline test) "
            "\u00b7 0 real-model passes yet")

    def test_no_reports(self):
        self.assertEqual(status_line(self.dir),
                         "wsp-kit by Aurem \u25cf IDLE \u2014 no runs found yet")

    def test_no_wins_but_runs(self):
        self._mk("qwen-bad", 10, passed=False, before=31.11, after=31.11,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=32.22)
        self.assertIn("no approved improvements yet",
                      status_line(self.dir))
        self.assertIn("1 run completed", status_line(self.dir))

    def test_non_report_json_is_ignored(self):
        junk = os.path.join(self.dir, "junk", "report.json")
        os.makedirs(os.path.dirname(junk), exist_ok=True)
        with open(junk, "w") as f:
            json.dump({"discovery": True, "notes": "not a report"}, f)
        self.assertEqual(len(scan_reports(self.dir)["reports"]), 0)

    def test_momentum_comes_from_real_reports(self):
        self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        self._mk("qwen-bad", 20, passed=False, before=31.11, after=31.11,
                 model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
                 real=True, causal=32.22)
        page = render_html(scan_reports(self.dir))
        self.assertIn("Pipeline proven end-to-end", page)
        self.assertIn("+32.22", page)
        self.assertIn("First passing real-model edit", page)

    def test_win_card_fields(self):
        path = self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                        weights=128, pct=0.77)
        with open(path) as f:
            rep = json.load(f)
        win = parse_win(rep, path)
        self.assertEqual(win["name"], "Pipeline test (smoke test)")
        self.assertFalse(win["is_real"])
        self.assertAlmostEqual(win["gain"], 11.29, places=2)
        self.assertEqual(win["weights_changed"], 128)
        self.assertAlmostEqual(win["weights_changed_pct"], 0.77, places=2)

    def test_gate_temps_from_string_keys(self):
        """Real reports store by_temperature keys as JSON strings
        ('0.7'); the card must still render them as numbers."""
        path = self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                        weights=128, pct=0.77)
        with open(path) as f:
            rep = json.load(f)
        rep["gate"]["by_temperature"] = {
            "0.7": {"passed": True}, "0.0": {"passed": True}}
        with open(path, "w") as f:
            json.dump(rep, f)
        win = parse_win(rep, path)
        self.assertEqual(win["gate_temps"], [0.0, 0.7])
        self.assertIn("Gate passed at 0, 0.7", render_html(scan_reports(self.dir)))

    def test_marker_alive_means_running(self):
        write_running_marker(self.dir, "Qwen/Qwen3-0.6B", "eval.json")
        scan = scan_reports(self.dir)
        self.assertIsNotNone(scan["running"])
        self.assertEqual(scan["running"]["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual(chip_state(scan), "RUNNING")
        self.assertIn("RUNNING", render_html(scan))
        remove_running_marker(self.dir)
        self.assertFalse(os.path.exists(
            os.path.join(self.dir, ".wsp-running")))
        self.assertEqual(chip_state(scan_reports(self.dir)), "IDLE")

    def test_marker_with_dead_pid_is_stale(self):
        marker = os.path.join(self.dir, ".wsp-running")
        with open(marker, "w") as f:
            json.dump({"pid": 999999, "model_id": "Qwen/Qwen3-0.6B",
                       "eval": "eval.json"}, f)
        scan = scan_reports(self.dir)
        self.assertIsNone(scan["running"])
        self.assertEqual(chip_state(scan), "IDLE")

    def test_serve_smoke(self):
        self._mk("kit", 10, passed=True, before=64.14, after=75.43,
                 weights=128, pct=0.77)
        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.dir))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_address[1]}/",
                    timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
            self.assertIn("Approved improvements", body)
            self.assertIn("Pipeline test", body)
            self.assertIn("+11.29", body)
            self.assertNotIn("REFUSED", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
