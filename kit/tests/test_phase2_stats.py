"""Phase 2 (W2.2, W2.3): stats, splits, preflight, evalfile fields.

Red-first tests for the new modules ``wsp.stats`` and
``wsp.splits`` plus the ``family`` / ``valid_items`` additions to
``wsp.evalfile``. No network, no model loads.
"""

import copy
import math
import unittest

from wsp.evalfile import validate_eval


def _eval(failing=20, neighbours=16, good=12, family=None):
    """A minimal eval dict with N prompts per set."""
    def _prompts(prefix, n):
        out = []
        for i in range(n):
            p = {"prompt": f"{prefix} prompt {i}"}
            if family is not None:
                p["family"] = family[i % len(family)]
            out.append(p)
        return out

    return {
        "format_version": 1,
        "name": "phase2-test",
        "sets": {
            "failing": {"scorer": "loop",
                        "prompts": _prompts("failing", failing)},
            "good": {"scorer": "contains",
                     "prompts": [{"prompt": f"good prompt {i}",
                                  "expect": "x"}
                                 for i in range(good)]},
            "neighbours": {"scorer": "loop",
                           "prompts": _prompts("neighbour", neighbours)},
        },
    }


PREFLIGHT_TEXT = (
    "need at least 20 failing prompts for a decision (found 8). "
    "Add more of YOUR OWN failing prompts: the kit never invents "
    "them. For a quick personal check use --protocol v1 (result is "
    "labelled QUICK, not evidence)."
)


class TestTCrit(unittest.TestCase):
    def test_table_values_exact(self):
        from wsp.stats import t_crit_one_sided

        table = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
                 7: 1.895, 9: 1.833, 19: 1.729, 29: 1.699}
        for df, want in table.items():
            self.assertAlmostEqual(t_crit_one_sided(df), want, places=3)

    def test_interpolation_in_one_over_df(self):
        from wsp.stats import t_crit_one_sided

        # df=6 sits between 5 and 7 in 1/df space:
        # x=1/6, between x=0.2 (2.015) and x=1/7 (1.895).
        self.assertAlmostEqual(t_crit_one_sided(6), 1.945, places=3)
        # df=40 sits between 29 (1.699) and infinity (1.645).
        self.assertAlmostEqual(t_crit_one_sided(40), 1.684, places=3)
        # Between table points the value stays between them.
        self.assertTrue(1.895 < t_crit_one_sided(6) < 2.015)
        self.assertTrue(1.645 < t_crit_one_sided(40) < 1.699)

    def test_large_df_approaches_normal(self):
        from wsp.stats import t_crit_one_sided

        self.assertAlmostEqual(t_crit_one_sided(10000), 1.645, places=2)

    def test_bad_df_raises(self):
        from wsp.stats import t_crit_one_sided

        for bad in (0, -1, -5):
            with self.assertRaises(ValueError):
                t_crit_one_sided(bad)


class TestPairedLowerBound(unittest.TestCase):
    def test_zero_variance(self):
        from wsp.stats import paired_lower_bound

        out = paired_lower_bound([10.0, 10.0, 10.0])
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mean"], 10.0)
        self.assertAlmostEqual(out["sd"], 0.0)
        self.assertAlmostEqual(out["lower"], 10.0)

    def test_n_one_is_inconclusive(self):
        from wsp.stats import paired_lower_bound

        out = paired_lower_bound([10.0])
        self.assertEqual(out["n"], 1)
        self.assertIsNone(out["lower"])

    def test_empty_is_inconclusive(self):
        from wsp.stats import paired_lower_bound

        out = paired_lower_bound([])
        self.assertEqual(out["n"], 0)
        self.assertIsNone(out["lower"])

    def test_bound_shrinks_with_more_data(self):
        from wsp.stats import paired_lower_bound

        few = paired_lower_bound([18.0, 0.0, 0.0])
        many = paired_lower_bound([18.0, 0.0, 0.0] * 4)
        self.assertEqual(few["mean"], many["mean"])
        self.assertLess(few["lower"], many["lower"])

    def test_nan_and_inf_raise(self):
        from wsp.stats import paired_lower_bound

        for bad in ([float("nan"), 1.0], [float("inf"), 1.0],
                    [1.0, float("-inf")]):
            with self.assertRaises(ValueError):
                paired_lower_bound(bad)

    def test_deterministic(self):
        from wsp.stats import paired_lower_bound

        d = [3.0, -1.0, 7.0, 2.0, 5.0]
        self.assertEqual(paired_lower_bound(d), paired_lower_bound(d))


class TestAssignSplits(unittest.TestCase):
    def test_twenty_failing_gives_6_4_10(self):
        from wsp.splits import assign_splits
        from wsp.protocol import get_protocol

        splits = assign_splits(_eval(), get_protocol("v2"), seed=0)
        self.assertEqual(len(splits["failing"]["diagnose"]), 6)
        self.assertEqual(len(splits["failing"]["select"]), 4)
        self.assertEqual(len(splits["failing"]["holdout"]), 10)
        self.assertEqual(len(splits["neighbours"]["diagnose"]), 5)
        self.assertEqual(len(splits["neighbours"]["select"]), 3)
        self.assertEqual(len(splits["neighbours"]["holdout"]), 8)
        self.assertEqual(len(splits["good"]["select"]), 4)
        self.assertEqual(len(splits["good"]["holdout"]), 8)
        self.assertEqual(splits["good"]["diagnose"], [])

    def test_exact_whatever_the_file_order(self):
        from wsp.splits import assign_splits
        from wsp.protocol import get_protocol

        data = _eval()
        first = assign_splits(data, get_protocol("v2"), seed=0)
        shuffled = copy.deepcopy(data)
        for name in ("failing", "good", "neighbours"):
            shuffled["sets"][name]["prompts"].reverse()
        second = assign_splits(shuffled, get_protocol("v2"), seed=0)
        # The assignment is by prompt TEXT, so compare the texts
        # in each split (indices move when the file order moves).
        for name in ("failing", "good", "neighbours"):
            for split in ("diagnose", "select", "holdout"):
                first_texts = sorted(
                    data["sets"][name]["prompts"][i]["prompt"]
                    for i in first[name][split])
                second_texts = sorted(
                    shuffled["sets"][name]["prompts"][i]["prompt"]
                    for i in second[name][split])
                self.assertEqual(first_texts, second_texts,
                                 f"{name}/{split} changed with file order")

    def test_deterministic_and_disjoint(self):
        from wsp.splits import assign_splits
        from wsp.protocol import get_protocol

        data = _eval()
        a = assign_splits(data, get_protocol("v2"), seed=0)
        b = assign_splits(data, get_protocol("v2"), seed=0)
        self.assertEqual(a, b)
        for name in ("failing", "good", "neighbours"):
            seen = []
            for split in ("diagnose", "select", "holdout"):
                seen.extend(a[name][split])
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(
                len(seen), len(data["sets"][name]["prompts"]))

    def test_family_stratified(self):
        from wsp.splits import assign_splits
        from wsp.protocol import get_protocol

        # Two families of 10 failing prompts each: quotas apply
        # inside each family (3/2/5 + 3/2/5 = 6/4/10 overall).
        data = _eval(family=("a", "b"))
        splits = assign_splits(data, get_protocol("v2"), seed=0)
        self.assertEqual(len(splits["failing"]["diagnose"]), 6)
        self.assertEqual(len(splits["failing"]["select"]), 4)
        self.assertEqual(len(splits["failing"]["holdout"]), 10)
        prompts = data["sets"]["failing"]["prompts"]
        for split, want in (("diagnose", 3), ("select", 2),
                            ("holdout", 5)):
            for fam in ("a", "b"):
                got = sum(1 for i in splits["failing"][split]
                          if prompts[i].get("family") == fam)
                self.assertEqual(got, want, f"{split}/{fam}")

    def test_different_seed_different_split(self):
        from wsp.splits import assign_splits
        from wsp.protocol import get_protocol

        data = _eval()
        a = assign_splits(data, get_protocol("v2"), seed=0)
        b = assign_splits(data, get_protocol("v2"), seed=1)
        self.assertNotEqual(a["failing"], b["failing"])


class TestPreflight(unittest.TestCase):
    def test_ok_eval_has_no_problems(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        self.assertEqual(preflight(_eval(), get_protocol("v2")), [])

    def test_failing_short_exact_text(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        problems = preflight(_eval(failing=8), get_protocol("v2"))
        self.assertEqual(problems[0], PREFLIGHT_TEXT)

    def test_neighbours_and_good_minimums(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        problems = preflight(_eval(neighbours=3, good=2),
                             get_protocol("v2"))
        joined = " ".join(problems)
        self.assertIn("need at least 16 neighbour", joined)
        self.assertIn("need at least 12 good", joined)

    def test_duplicate_across_sets(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        data = _eval()
        data["sets"]["good"]["prompts"][0] = {
            "prompt": "failing prompt 0", "expect": "x"}
        problems = preflight(data, get_protocol("v2"))
        self.assertTrue(any("duplicate" in p.lower() for p in problems),
                        problems)

    def test_duplicate_within_set(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        data = _eval()
        data["sets"]["failing"]["prompts"][1] = dict(
            data["sets"]["failing"]["prompts"][0])
        problems = preflight(data, get_protocol("v2"))
        self.assertTrue(any("duplicate" in p.lower() for p in problems),
                        problems)

    def test_duplicate_ignores_case_and_spacing(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        data = _eval()
        data["sets"]["good"]["prompts"][0] = {
            "prompt": "  FAILING   prompt 0 ", "expect": "x"}
        problems = preflight(data, get_protocol("v2"))
        self.assertTrue(any("duplicate" in p.lower() for p in problems),
                        problems)

    def test_empty_split_flagged(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        problems = preflight(_eval(failing=1, neighbours=1, good=1),
                             get_protocol("v2"))
        self.assertTrue(any("empty" in p.lower() for p in problems),
                        problems)

    def test_v1_has_no_minimums(self):
        from wsp.splits import preflight
        from wsp.protocol import get_protocol

        self.assertEqual(
            preflight(_eval(failing=1, neighbours=1, good=1),
                      get_protocol("v1")), [])


class TestEvalfileV2Fields(unittest.TestCase):
    def test_family_accepted(self):
        data = _eval(family=("a", "b"))
        self.assertEqual(validate_eval(data)["sets"]["failing"]
                         ["prompts"][0]["family"], "a")

    def test_family_must_be_string(self):
        data = _eval()
        data["sets"]["failing"]["prompts"][0]["family"] = 7
        with self.assertRaises(ValueError):
            validate_eval(data)

    def test_valid_items_list_accepted_for_distinct(self):
        data = _eval()
        data["sets"]["failing"] = {
            "scorer": "distinct",
            "prompts": [{"prompt": "list animals",
                         "expect_count": 60,
                         "valid_items": ["dog", "cat"]}] * 20,
        }
        validate_eval(data)

    def test_valid_items_rejected_for_other_scorers(self):
        data = _eval()
        data["sets"]["failing"]["prompts"][0]["valid_items"] = ["x"]
        with self.assertRaises(ValueError):
            validate_eval(data)

    def test_valid_items_must_be_nonempty_strings(self):
        data = _eval()
        data["sets"]["failing"] = {
            "scorer": "distinct",
            "prompts": [{"prompt": "list animals",
                         "expect_count": 60,
                         "valid_items": ["dog", ""]}] * 20,
        }
        with self.assertRaises(ValueError):
            validate_eval(data)

    def test_valid_items_file_path_accepted(self):
        data = _eval()
        data["sets"]["failing"] = {
            "scorer": "distinct",
            "prompts": [{"prompt": "list animals",
                         "expect_count": 60,
                         "valid_items": "animals.txt"}] * 20,
        }
        validate_eval(data)


if __name__ == "__main__":
    unittest.main()
