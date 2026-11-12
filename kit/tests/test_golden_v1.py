"""Golden (characterization) tests for the v1 real-model pipeline.

Each test runs the CURRENT ``run_real_pipeline`` against a
``FakeRealModel`` scenario (see ``tests.fakes``) and compares the
written ``report.json`` with the committed golden file in
``tests/golden/v1/`` using the recursive subset check in
``tests.golden_utils``: every golden key/value must be present and
equal; extra keys are allowed, because reports are additive.

The goldens pin v1 MEASUREMENT behaviour (scores, gate, causal rule,
layer choice, alpha, sampling). Later phases may add report fields
but may never change or remove a v1 measurement value — if a golden
fails, the pipeline's measurements changed, and that needs a
deliberate decision, not a golden regeneration.

The ``pass`` scenario also pins the v1 ``edit.json`` the same way.
These tests are numpy-only: no torch, no model download, no network.
Run from the kit directory:
  PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from tests.fakes import FakeRealModel, SCENARIOS
from tests.golden_utils import assert_golden_subset, strip_volatile
from wsp.real_pipeline import run_real_pipeline

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden", "v1")
MODEL_ID = "fake-model"

# Exit codes the v1 pipeline returns per scenario (0 = gate passed and
# the edit shipped; 1 = stopped, nothing shipped).
EXPECTED_EXITS = {
    "cheap_fix": 1,
    "no_failure": 1,
    "not_causal": 1,
    "pass": 0,
    "no_attempt_passes": 1,
    "negative_causal": 0,
}


def _load_golden(name):
    with open(os.path.join(GOLDEN_DIR, name)) as f:
        return json.load(f)


class TestGoldenV1(unittest.TestCase):
    def _run_scenario(self, scenario):
        """Run the scenario; return (stripped report, out_dir files, code)."""
        with tempfile.TemporaryDirectory() as d:
            rep, out_dir, code = run_real_pipeline(
                model_id=MODEL_ID,
                eval_path=os.path.join(GOLDEN_DIR, "eval.json"),
                out_dir=d,
                model=FakeRealModel(scenario),
                log=lambda *a: None,
                protocol="v1",
            )
            with open(os.path.join(out_dir, "report.json")) as f:
                written = json.load(f)
            files = sorted(os.listdir(out_dir))
            if code == 0:
                with open(os.path.join(out_dir, "edit.json")) as f:
                    edit = json.load(f)
            else:
                edit = None
            self.assertEqual(rep, written,
                             "returned report differs from report.json")
            return strip_volatile(written), files, edit, code

    def test_golden_cheap_fix(self):
        self._check("cheap_fix")

    def test_golden_no_failure(self):
        self._check("no_failure")

    def test_golden_not_causal(self):
        self._check("not_causal")

    def test_golden_pass(self):
        self._check("pass")

    def test_golden_no_attempt_passes(self):
        self._check("no_attempt_passes")

    def test_golden_negative_causal(self):
        self._check("negative_causal")

    def _check(self, scenario):
        rep, files, edit, code = self._run_scenario(scenario)
        self.assertEqual(code, EXPECTED_EXITS[scenario],
                         f"{scenario}: exit {code}")
        golden = _load_golden(f"{scenario}.json")
        assert_golden_subset(self, golden, rep)
        if code == 0:
            # A passing run ships an edit. Both passing scenarios
            # produce the same recipe (same diagnosis, same chosen
            # variant), so both are pinned against pass.edit.json.
            self.assertIn("edit.json", files)
            self.assertIn("fake-model.json", files)
            golden_edit = _load_golden("pass.edit.json")
            assert_golden_subset(self, strip_volatile(golden_edit),
                                 strip_volatile(edit))
        else:
            self.assertNotIn("edit.json", files)


class TestGoldenFilesPresent(unittest.TestCase):
    def test_all_scenarios_have_goldens(self):
        # Only the v1 scenarios have goldens: "partial_pass" is a
        # v2-only scenario (it exercises the INCONCLUSIVE exit) and
        # has no v1 golden by design.
        for scenario in EXPECTED_EXITS:
            path = os.path.join(GOLDEN_DIR, f"{scenario}.json")
            self.assertTrue(os.path.exists(path), f"missing {path}")
        self.assertTrue(os.path.exists(
            os.path.join(GOLDEN_DIR, "pass.edit.json")))
        self.assertTrue(os.path.exists(
            os.path.join(GOLDEN_DIR, "eval.json")))


if __name__ == "__main__":
    unittest.main()
