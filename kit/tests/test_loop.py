"""Tests for `wsp loop` (continuous background mode). Plain unittest.

Run from the kit directory: PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from wsp import loop


def _write_failures(path, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")


def _passing_run(calls):
    """A fake run_fn that always passes the gate, unique run dir."""
    def run_fn(model_id, eval_path, out_dir, log=print):
        n = len(calls) + 1
        calls.append(eval_path)
        run_dir = os.path.join(out_dir, f"run-{n}")
        os.makedirs(run_dir, exist_ok=True)
        rep = {
            "gate": {"passed": True},
            "before": {"failing": 50.0},
            "after": {"failing": 70.0},
            "weights_changed": 128,
        }
        return rep, run_dir, 0

    return run_fn


def _failing_run(calls):
    """A fake run_fn that always fails the gate."""
    def run_fn(model_id, eval_path, out_dir, log=print):
        n = len(calls) + 1
        calls.append(eval_path)
        run_dir = os.path.join(out_dir, f"run-{n}")
        os.makedirs(run_dir, exist_ok=True)
        rep = {
            "gate": {"passed": False, "reason": "no gain"},
            "before": {"failing": 50.0},
            "after": {"failing": 51.0},
            "weights_changed": 0,
        }
        return rep, run_dir, 1

    return run_fn


class TestParseFailures(unittest.TestCase):
    def test_kinds_and_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "failures.jsonl")
            _write_failures(path, [
                {"prompt": "f1", "expect": "e1", "kind": "failing"},
                {"prompt": "f2", "expect": "e2"},  # no kind -> failing
                {"prompt": "g1", "expect": "e3", "kind": "good"},
                {"prompt": "n1", "expect": "e4", "kind": "neighbour"},
                {"prompt": "bad"},  # no expect -> skipped
                {"kind": "failing"},  # no prompt -> skipped
            ])
            failing, good, neighbours, skipped = loop.parse_failures(path)
        self.assertEqual(failing, [("f1", "e1"), ("f2", "e2")])
        self.assertEqual(good, [("g1", "e3")])
        self.assertEqual(neighbours, [("n1", "e4")])
        self.assertEqual(skipped, 2)

    def test_missing_file_is_empty(self):
        failing, good, neighbours, skipped = loop.parse_failures(
            "/nonexistent/failures.jsonl")
        self.assertEqual((failing, good, neighbours, skipped),
                         ([], [], [], 0))


class TestNewFailureLines(unittest.TestCase):
    def test_first_call_reads_all_then_only_new(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "failures.jsonl")
            _write_failures(path, [
                {"prompt": "f1", "expect": "e1"},
                {"prompt": "g1", "expect": "e2", "kind": "good"},
            ])
            lines, entry = loop.new_failure_lines(path, None)
            self.assertEqual(len(lines), 2)
            lines, entry = loop.new_failure_lines(path, entry)
            self.assertEqual(lines, [])
            with open(path, "a") as f:
                f.write(json.dumps(
                    {"prompt": "f2", "expect": "e3"}) + "\n")
            lines, entry = loop.new_failure_lines(path, entry)
            self.assertEqual(len(lines), 1)
            self.assertIn("f2", lines[0])


class TestLoopOnce(unittest.TestCase):
    def test_failure_run_creates_version_then_stops(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            _write_failures(os.path.join(watch_dir, "failures.jsonl"), [
                {"prompt": "List 60 animals.", "expect": "dog"},
                {"prompt": "What is 2 + 3?", "expect": "5",
                 "kind": "good"},
            ])
            calls = []
            state = {}
            state, summary = loop.loop_once(
                "Qwen/Qwen3-0.6B", watch_dir, out_dir, state,
                run_fn=_passing_run(calls),
                log=lambda *a: None)
            self.assertEqual(summary["failure_runs"], 1)
            self.assertEqual(summary["versions"], 1)
            self.assertEqual(len(calls), 1)
            # Eval file was built from the failures.
            eval_file = os.path.join(
                out_dir, "evals", "loop-qwen-qwen3-0-6b.json")
            self.assertTrue(os.path.isfile(eval_file))
            # Version pointer exists and points at the run.
            with open(os.path.join(out_dir, "latest.json")) as f:
                latest = json.load(f)
            self.assertEqual(latest["version"], 1)
            self.assertEqual(latest["weights_changed"], 128)
            # Same state again: nothing new, no second run.
            state, summary = loop.loop_once(
                "Qwen/Qwen3-0.6B", watch_dir, out_dir, state,
                run_fn=_passing_run(calls),
                log=lambda *a: None)
            self.assertEqual(summary["failure_runs"], 0)
            self.assertEqual(len(calls), 1)

    def test_new_failure_line_retriggers(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            path = os.path.join(watch_dir, "failures.jsonl")
            _write_failures(path, [
                {"prompt": "f1", "expect": "e1"},
                {"prompt": "g1", "expect": "e2", "kind": "good"},
            ])
            calls = []
            state = {}
            loop.loop_once("m", watch_dir, out_dir, state,
                           run_fn=_passing_run(calls),
                           log=lambda *a: None)
            with open(path, "a") as f:
                f.write(json.dumps(
                    {"prompt": "f2", "expect": "e3"}) + "\n")
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, state,
                run_fn=_passing_run(calls), log=lambda *a: None)
            self.assertEqual(summary["failure_runs"], 1)
            self.assertEqual(len(calls), 2)
            with open(os.path.join(out_dir, "latest.json")) as f:
                latest = json.load(f)
            self.assertEqual(latest["version"], 2)

    def test_gate_fail_ships_no_version(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            _write_failures(os.path.join(watch_dir, "failures.jsonl"), [
                {"prompt": "f1", "expect": "e1"},
                {"prompt": "g1", "expect": "e2", "kind": "good"},
            ])
            calls = []
            state = {}
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, state,
                run_fn=_failing_run(calls), log=lambda *a: None)
            self.assertEqual(summary["failure_runs"], 1)
            self.assertEqual(summary["versions"], 0)
            self.assertFalse(
                os.path.exists(os.path.join(out_dir, "latest.json")))

    def test_gate_pass_logs_star_ask(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            _write_failures(os.path.join(watch_dir, "failures.jsonl"), [
                {"prompt": "f1", "expect": "e1"},
                {"prompt": "g1", "expect": "e2", "kind": "good"},
            ])
            lines = []
            state = {}
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, state,
                run_fn=_passing_run([]),
                log=lambda *a: lines.append(" ".join(str(x) for x in a)))
            self.assertEqual(summary["versions"], 1)
            self.assertTrue(
                any("star the repo" in ln and
                    "github.com/AuremHQ/weight-space-programming" in ln
                    for ln in lines),
                "gate pass must log the star ask")

    def test_gate_fail_logs_no_star_ask(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            _write_failures(os.path.join(watch_dir, "failures.jsonl"), [
                {"prompt": "f1", "expect": "e1"},
                {"prompt": "g1", "expect": "e2", "kind": "good"},
            ])
            lines = []
            state = {}
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, state,
                run_fn=_failing_run([]),
                log=lambda *a: lines.append(" ".join(str(x) for x in a)))
            self.assertEqual(summary["versions"], 0)
            self.assertFalse(
                any("star the repo" in ln for ln in lines),
                "gate fail must not log the star ask")

    def test_eval_file_in_watch_dir_runs(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            with open(os.path.join(watch_dir, "e.json"), "w") as f:
                json.dump({"format_version": 1}, f)
            calls = []
            state = {}
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, state,
                run_fn=_passing_run(calls), log=lambda *a: None)
            self.assertEqual(summary["eval_runs"], 1)
            self.assertEqual(summary["versions"], 1)


if __name__ == "__main__":
    unittest.main()
