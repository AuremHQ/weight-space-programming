"""Phase 2 (W2.4): check_gate_v2.

Synthetic per-prompt deltas only — no model, no files.
"""

import unittest


def _gate(before, after, guards=("good", "neighbours")):
    from wsp.gate import check_gate_v2

    return check_gate_v2(before, after, guards=guards)


class TestCheckGateV2(unittest.TestCase):
    def test_clear_effect_ships(self):
        out = _gate(
            {"failing": [30.0] * 10, "good": [100.0] * 10,
             "neighbours": [100.0] * 10},
            {"failing": [100.0] * 10, "good": [100.0] * 10,
             "neighbours": [100.0] * 10})
        self.assertEqual(out["verdict"], "SHIP")
        self.assertTrue(out["passed"])
        self.assertAlmostEqual(out["primary_gain_points"], 70.0)
        self.assertAlmostEqual(out["primary_ci"]["lower"], 70.0)
        self.assertEqual(out["n_prompts"]["failing"], 10)

    def test_legacy_keys_kept(self):
        out = _gate({"failing": [30.0] * 5, "good": [100.0] * 5,
                     "neighbours": [100.0] * 5},
                    {"failing": [100.0] * 5, "good": [100.0] * 5,
                     "neighbours": [100.0] * 5})
        for key in ("passed", "primary", "primary_gain_points",
                    "regressions_points", "min_gain_points",
                    "max_regress_points", "verdict", "n_prompts",
                    "primary_ci", "guard_ci"):
            self.assertIn(key, out)
        self.assertEqual(out["primary"], "failing")
        self.assertEqual(out["min_gain_points"], 2.0)
        self.assertEqual(out["max_regress_points"], 2.0)
        self.assertEqual(out["regressions_points"],
                         {"good": 0.0, "neighbours": 0.0})

    def test_noisy_plus_6_n3_is_inconclusive(self):
        # The +29.44 vs +10.56 noise case: point gain looks fine
        # (+6) but with n=3 and large variance the lower bound
        # does not clear zero.
        out = _gate({"failing": [30.0, 30.0, 30.0],
                     "good": [100.0] * 3, "neighbours": [100.0] * 3},
                    {"failing": [48.0, 30.0, 30.0],
                     "good": [100.0] * 3, "neighbours": [100.0] * 3})
        self.assertAlmostEqual(out["primary_gain_points"], 6.0)
        self.assertEqual(out["verdict"], "INCONCLUSIVE")
        self.assertFalse(out["passed"])
        self.assertLess(out["primary_ci"]["lower"], 0.0)

    def test_plus_1_9_rejects(self):
        out = _gate({"failing": [30.0] * 10, "good": [100.0] * 10,
                     "neighbours": [100.0] * 10},
                    {"failing": [31.9] * 10, "good": [100.0] * 10,
                     "neighbours": [100.0] * 10})
        self.assertEqual(out["verdict"], "REJECT")
        self.assertFalse(out["passed"])

    def test_guard_down_3_rejects(self):
        out = _gate({"failing": [30.0] * 10, "good": [100.0] * 10,
                     "neighbours": [100.0] * 10},
                    {"failing": [100.0] * 10, "good": [97.0] * 10,
                     "neighbours": [100.0] * 10})
        self.assertEqual(out["verdict"], "REJECT")
        self.assertAlmostEqual(out["regressions_points"]["good"], 3.0)

    def test_guard_point_ok_but_bound_below_minus_5_inconclusive(self):
        out = _gate(
            {"failing": [30.0, 30.0], "good": [100.0, 90.0],
             "neighbours": [100.0, 100.0]},
            {"failing": [40.0, 40.0], "good": [88.0, 100.0],
             "neighbours": [100.0, 100.0]})
        # Guard point regression is +1 (mean delta -1, within 2),
        # but with n=2 the lower bound is far below -5.
        self.assertAlmostEqual(out["regressions_points"]["good"], 1.0)
        self.assertLess(out["guard_ci"]["good"]["lower"], -5.0)
        self.assertEqual(out["verdict"], "INCONCLUSIVE")
        self.assertFalse(out["passed"])

    def test_n_one_is_inconclusive(self):
        out = _gate({"failing": [30.0], "good": [100.0],
                     "neighbours": [100.0]},
                    {"failing": [100.0], "good": [100.0],
                     "neighbours": [100.0]})
        self.assertEqual(out["verdict"], "INCONCLUSIVE")
        self.assertIsNone(out["primary_ci"]["lower"])

    def test_zero_variance_ships(self):
        out = _gate({"failing": [50.0] * 4, "good": [90.0] * 4,
                     "neighbours": [80.0] * 4},
                    {"failing": [60.0] * 4, "good": [90.0] * 4,
                     "neighbours": [80.0] * 4})
        self.assertEqual(out["verdict"], "SHIP")

    def test_nan_raises(self):
        with self.assertRaises(ValueError):
            _gate({"failing": [30.0, float("nan")],
                   "good": [100.0, 100.0],
                   "neighbours": [100.0, 100.0]},
                  {"failing": [100.0, 100.0],
                   "good": [100.0, 100.0],
                   "neighbours": [100.0, 100.0]})

    def test_deterministic(self):
        before = {"failing": [30.0, 40.0, 35.0],
                  "good": [100.0] * 3, "neighbours": [90.0] * 3}
        after = {"failing": [80.0, 90.0, 85.0],
                 "good": [100.0] * 3, "neighbours": [90.0] * 3}
        self.assertEqual(_gate(before, after), _gate(before, after))

    def test_legacy_check_gate_untouched(self):
        from wsp.gate import check_gate

        out = check_gate({"failing": 30.0, "good": 100.0,
                          "neighbours": 100.0},
                         {"failing": 100.0, "good": 100.0,
                          "neighbours": 100.0})
        self.assertTrue(out["passed"])
        self.assertNotIn("verdict", out)


class TestCombineVerdicts(unittest.TestCase):
    def test_ship_needs_every_temperature(self):
        from wsp.gate import combine_gate_verdicts

        self.assertEqual(
            combine_gate_verdicts(["SHIP", "SHIP"]), "SHIP")
        self.assertEqual(
            combine_gate_verdicts(["SHIP", "INCONCLUSIVE"]),
            "INCONCLUSIVE")
        self.assertEqual(
            combine_gate_verdicts(["SHIP", "REJECT"]), "REJECT")
        self.assertEqual(
            combine_gate_verdicts(["INCONCLUSIVE", "REJECT"]),
            "REJECT")


if __name__ == "__main__":
    unittest.main()
