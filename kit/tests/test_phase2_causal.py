"""Phase 2 (W2.8): causal v2 with placebo controls.

A stub model whose score depends only on which columns are
zeroed: the suspect columns {0, 1} improve the score, random
columns do not.
"""

import random
import unittest


class StubModel:
    """Zeroing columns changes the score by overlap with suspects."""

    SUSPECT = {0, 1}

    def __init__(self, base=30.0, ablated_full=70.0):
        self.base = base
        self.ablated_full = ablated_full
        self.zeroed = {}
        self.restore_calls = []

    def score(self):
        if not self.zeroed:
            return self.base
        cols = set()
        for layer_cols in self.zeroed.values():
            cols |= set(layer_cols)
        frac = len(cols & self.SUSPECT) / len(self.SUSPECT)
        return self.base + (self.ablated_full - self.base) * frac

    def zero_columns(self, layer_index, columns):
        self.zeroed[layer_index] = list(columns)
        return ("zero", layer_index, list(columns))

    def restore(self, saved):
        _kind, layer_index, _cols = saved
        self.zeroed.pop(layer_index, None)
        self.restore_calls.append(saved)


def _check(model, **kw):
    from wsp.causal import causal_check

    layers = kw.pop("layers", [0])
    columns = kw.pop("columns", {0: [0, 1]})
    intermediates = kw.pop("intermediate_sizes", {0: 8})
    return causal_check(model, model.score, layers, columns,
                        intermediates, **kw)


class TestCausalCheck(unittest.TestCase):
    def test_real_effect_passes(self):
        # Phase 4: a positive effect is a dampen proof (removal
        # improves the score), so this checks edit_type="dampen".
        model = StubModel()
        out = _check(model, seed=0, edit_type="dampen")
        self.assertEqual(out["sign"], "removal_improves")
        self.assertAlmostEqual(out["effect"], 40.0)
        self.assertLess(out["placebo_abs_max"], 40.0)
        self.assertTrue(out["passed"])
        self.assertEqual(model.zeroed, {})

    def test_effect_equal_to_placebo_fails(self):
        # Placebos that zero the suspects too: the largest placebo
        # equals the real effect, so the effect is not proven.
        model = StubModel()
        out = _check(model, seed=0, k_placebos=200)
        # With 200 draws over 8 columns, some placebo hits both
        # suspects: placebo_abs_max reaches 40 == effect.
        self.assertAlmostEqual(out["placebo_abs_max"], 40.0)
        self.assertFalse(out["passed"])

    def test_negative_sign_selects_subtract(self):
        # Phase 4: the sign selects the edit type. A worsening
        # ablation (-30) is a subtract proof, not a dampen proof
        # (replaces the old test_negative_sign_refuses).
        model = StubModel(base=70.0, ablated_full=40.0)
        out = _check(model, seed=0, edit_type="subtract")
        self.assertEqual(out["sign"], "removal_worsens")
        self.assertAlmostEqual(out["effect"], -30.0)
        self.assertTrue(out["passed"])
        self.assertEqual(model.zeroed, {})
        out = _check(model, seed=0, edit_type="dampen")
        self.assertFalse(out["passed"])
        self.assertEqual(model.zeroed, {})

    def test_small_effect_fails(self):
        model = StubModel(base=30.0, ablated_full=33.0)
        out = _check(model, seed=0)
        self.assertAlmostEqual(out["effect"], 3.0)
        self.assertFalse(out["passed"])

    def test_seeds_deterministic(self):
        a = _check(StubModel(), seed=7)
        b = _check(StubModel(), seed=7)
        self.assertEqual(a["placebo_effects"], b["placebo_effects"])
        self.assertEqual(a["placebo_abs_max"], b["placebo_abs_max"])

    def test_restore_called_when_scoring_raises(self):
        model = StubModel()
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom in scoring")
            return model.score()

        from wsp.causal import causal_check

        with self.assertRaises(RuntimeError):
            causal_check(model, flaky, [0], {0: [0, 1]}, {0: 8},
                         seed=0)
        self.assertEqual(model.zeroed, {})
        self.assertTrue(model.restore_calls)

    def test_placebo_same_size_same_layers(self):
        model = StubModel()
        seen = []

        orig_zero = model.zero_columns

        def spy(layer_index, columns):
            seen.append((layer_index, sorted(columns)))
            return orig_zero(layer_index, columns)

        model.zero_columns = spy
        _check(model, seed=0, k_placebos=3)
        # First call is the real ablation; the rest are placebos.
        self.assertEqual(seen[0], (0, [0, 1]))
        for _layer, cols in seen[1:]:
            self.assertEqual(len(cols), 2)
            self.assertTrue(all(0 <= c < 8 for c in cols))

    def test_expected_sign_for_other_edit_types(self):
        # Phase 4: the sign rule follows the edit type — subtract
        # expects removal to worsen the score, dampen expects it
        # to improve it. (The "hypothetical" case from Phase 2 is
        # now the real subtract rule.)
        model = StubModel(base=70.0, ablated_full=40.0)
        out = _check(model, seed=0, edit_type="subtract")
        self.assertTrue(out["passed"])
        self.assertEqual(out["sign"], "removal_worsens")
        out = _check(model, seed=0, edit_type="dampen")
        self.assertFalse(out["passed"])
        self.assertEqual(out["sign"], "removal_worsens")


if __name__ == "__main__":
    unittest.main()
