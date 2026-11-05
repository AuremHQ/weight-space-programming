"""Tests for wsp-kit. Plain unittest, no extra dependencies.

Run from the kit directory: PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import os
import tempfile
import unittest

import numpy as np

from wsp import diagnose, edit, gate, pipeline, report, kit_test, weights


class TestEdit(unittest.TestCase):
    def test_edit_changes_only_one_row(self):
        w = kit_test.build_clean_weights()
        direction = np.zeros(kit_test.HIDDEN_DIM)
        direction[0] = 1.0
        new, changed, frac = edit.apply_edit(w, 0, direction, 1.0)
        self.assertEqual(changed, 1)
        self.assertTrue(np.array_equal(new["W1"], w["W1"]))
        self.assertTrue(np.array_equal(new["W2"][1:], w["W2"][1:]))

    def test_edit_rejects_over_cap(self):
        # A tiny model where one row is most of the model: must refuse.
        w = {"W2": np.zeros((2, 4))}
        direction = np.ones(4)
        with self.assertRaises(ValueError):
            edit.apply_edit(w, 0, direction, 1.0)


class TestGate(unittest.TestCase):
    def test_gate_passes_on_clear_gain(self):
        before = {"failing": 10.0, "good": 95.0, "neighbours": 94.0}
        after = {"failing": 80.0, "good": 95.0, "neighbours": 93.5}
        self.assertTrue(gate.check_gate(before, after)["passed"])

    def test_gate_fails_on_guard_regress(self):
        before = {"failing": 10.0, "good": 95.0, "neighbours": 94.0}
        after = {"failing": 80.0, "good": 90.0, "neighbours": 94.0}
        self.assertFalse(gate.check_gate(before, after)["passed"])

    def test_gate_fails_on_small_gain(self):
        before = {"failing": 10.0, "good": 95.0, "neighbours": 94.0}
        after = {"failing": 13.0, "good": 95.0, "neighbours": 94.0}
        self.assertFalse(gate.check_gate(before, after)["passed"])


class TestReport(unittest.TestCase):
    def test_report_carries_versions_and_note(self):
        from wsp import __version__

        rep = report.make_report("kit", 0, 128, 0.77, {}, {}, {"passed": True},
                                 [0.7, 0.0], "kit estimator", ["numpy"], 1, [])
        self.assertEqual(rep["report_version"], 1)
        self.assertEqual(rep["kit_version"], __version__)
        self.assertIn("verify independently", rep["note"])


class TestWeights(unittest.TestCase):
    def test_npz_roundtrip(self):
        w = kit_test.build_clean_weights()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            weights.save_weights(path, w)
            back = weights.load_weights(path)
        for k in w:
            self.assertTrue(np.array_equal(w[k], back[k]))

    def test_missing_file_errors_plainly(self):
        with self.assertRaises(FileNotFoundError):
            weights.load_weights("/nonexistent/weights.npz")


class TestPipeline(unittest.TestCase):
    def test_demo_pipeline_passes_gate(self):
        rep, edited, before, after, attempts, _sweep, cause = pipeline.run_pipeline()
        self.assertTrue(cause["causal"])
        self.assertIsNotNone(edited)
        self.assertTrue(rep["gate"]["passed"])
        self.assertGreaterEqual(after["failing"] - before["failing"], 5.0)
        self.assertLessEqual(before["good"] - after["good"], 2.0)
        self.assertLessEqual(before["neighbours"] - after["neighbours"], 2.0)
        self.assertLessEqual(rep["weights_changed_pct"], 1.0)
        self.assertLessEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
