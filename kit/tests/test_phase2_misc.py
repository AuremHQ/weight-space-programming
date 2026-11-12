"""Phase 2 (W2.1, W2.6, W2.7b, W2.12): protocol, environment,
instrumentation, alpha bounds + characterisation.
"""

import os
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

from tests.fakes import FakeRealModel


class TestProtocol(unittest.TestCase):
    def test_v1_and_v2_known(self):
        from wsp.protocol import PROTOCOLS, get_protocol

        self.assertEqual(set(PROTOCOLS), {"v1", "v2"})
        v1 = get_protocol("v1")
        v2 = get_protocol("v2")
        self.assertEqual(v1.name, "v1")
        self.assertEqual(v2.name, "v2")
        self.assertEqual(v1.samples_per_prompt, 1)
        self.assertEqual(v2.samples_per_prompt, 3)
        self.assertEqual(v1.scorer_version, "v1")
        self.assertEqual(v2.scorer_version, "v2")
        self.assertEqual(v2.min_prompts,
                         {"failing": 20, "neighbours": 16, "good": 12})
        self.assertEqual(v1.min_prompts,
                         {"failing": 0, "neighbours": 0, "good": 0})

    def test_v2_split_quotas(self):
        from wsp.protocol import get_protocol

        v2 = get_protocol("v2")
        self.assertEqual(v2.split_quotas["failing"],
                         {"diagnose": 0.3, "select": 0.2,
                          "holdout": 0.5})
        self.assertEqual(v2.split_quotas["good"],
                         {"select": 0.3, "holdout": 0.7})

    def test_unknown_protocol_raises(self):
        from wsp.protocol import get_protocol

        with self.assertRaises(ValueError):
            get_protocol("v3")
        with self.assertRaises(ValueError):
            get_protocol("")

    def test_v2_gate_numbers(self):
        from wsp.protocol import get_protocol

        v2 = get_protocol("v2")
        self.assertEqual(v2.min_gain_points, 2.0)
        self.assertEqual(v2.max_regress_points, 2.0)
        self.assertEqual(v2.guard_margin_points, 5.0)
        self.assertEqual(v2.causal_min_change, 5.0)
        self.assertEqual(v2.causal_placebos, 10)


class TestEnvInfo(unittest.TestCase):
    def test_collect_environment_never_raises(self):
        from wsp.env_info import collect_environment

        env = collect_environment()
        for key in ("python", "platform", "numpy", "torch",
                    "transformers", "accelerate", "safetensors",
                    "device", "dtype", "kit_version", "kit_commit"):
            self.assertIn(key, env, key)
        self.assertTrue(env["python"].startswith("3."))
        self.assertIsInstance(env["numpy"], str)
        # torch is absent in the numpy-only env; present (a string)
        # or None in the real env — never an exception.
        self.assertTrue(env["torch"] is None
                        or isinstance(env["torch"], str))

    def test_kit_commit_is_none_or_hex(self):
        from wsp.env_info import collect_environment

        commit = collect_environment()["kit_commit"]
        self.assertTrue(commit is None
                        or (isinstance(commit, str)
                            and len(commit) == 40
                            and all(c in "0123456789abcdef"
                                    for c in commit)))

    def test_device_and_dtype_recorded(self):
        from wsp.env_info import collect_environment

        env = collect_environment(device="cpu", dtype="bfloat16")
        self.assertEqual(env["device"], "cpu")
        self.assertEqual(env["dtype"], "bfloat16")


class TestCountingModel(unittest.TestCase):
    def test_counts_generate_calls_and_tokens(self):
        from wsp.instrument import CountingModel

        fake = FakeRealModel("pass")
        proxy = CountingModel(fake)
        text, _fin, n = proxy.generate(
            "List 60 different animals, one per line.", 64, 0.7, 0)
        self.assertIn("Animal", text)
        self.assertEqual(proxy.eval_passes_spent, 1)
        self.assertEqual(proxy.generated_tokens, n)
        self.assertGreater(n, 0)

    def test_phase_context_and_prompt_log(self):
        from wsp.instrument import CountingModel

        fake = FakeRealModel("pass")
        proxy = CountingModel(fake)
        with proxy.phase("diagnose"):
            proxy.capture("List 60 different animals, one per line.",
                          64, [0, 1])
            proxy.generate("List 60 different animals, one per line.",
                           64, 0.0, 0)
        with proxy.phase("select"):
            proxy.generate("List 60 different animals, one per line.",
                           64, 0.0, 1)
        self.assertEqual(proxy.eval_passes_spent, 3)
        counts = proxy.phase_prompt_counts
        self.assertEqual(
            counts["diagnose"]["List 60 different animals, one per line."],
            2)
        self.assertEqual(
            counts["select"]["List 60 different animals, one per line."],
            1)
        self.assertEqual(
            [c[0] for c in proxy.calls],
            ["diagnose", "diagnose", "select"])

    def test_passthrough_attributes(self):
        from wsp.instrument import CountingModel

        fake = FakeRealModel("pass")
        proxy = CountingModel(fake)
        self.assertEqual(proxy.total_params, fake.total_params)
        self.assertEqual(proxy.layer_infos, fake.layer_infos)
        self.assertEqual(proxy.use_chat_template, True)
        self.assertIsNone(proxy.revision)
        self.assertEqual(proxy.nll("hello"), 2.0)
        self.assertEqual(proxy.top1("hello"), 5)


class TestAlphaV2(unittest.TestCase):
    def test_normal_alpha_unchanged(self):
        from wsp.real_edit import alpha_v2

        alpha, clamped = alpha_v2(3.0, 2.0)
        self.assertAlmostEqual(alpha, 1.5)
        self.assertFalse(clamped)

    def test_lower_bound_clamps(self):
        from wsp.real_edit import alpha_v2

        alpha, clamped = alpha_v2(0.001, 100.0)
        self.assertAlmostEqual(alpha, 1e-3)
        self.assertTrue(clamped)

    def test_hard_max_clamps(self):
        from wsp.real_edit import alpha_v2

        alpha, clamped = alpha_v2(1e6, 1.0)
        self.assertAlmostEqual(alpha, 100.0)
        self.assertTrue(clamped)

    def test_fallback_denom_uses_legacy_value(self):
        from wsp.real_edit import alpha_v2

        # Legacy fallback returns 1.0 for a ~zero denominator;
        # the v2 bounds keep it (inside [1e-3, 100]).
        alpha, clamped = alpha_v2(3.0, 0.0)
        self.assertAlmostEqual(alpha, 1.0)
        self.assertFalse(clamped)


@unittest.skipUnless(torch is not None, "torch not installed")
class TestRealModelGuards(unittest.TestCase):
    def test_nll_and_top1_on_tiny_model(self):
        import tempfile

        from tests.test_phase1_flows import _tiny_model_dir
        from wsp.real_runner import RealModel

        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "model")
            os.makedirs(model_dir)
            _tiny_model_dir(model_dir)
            model = RealModel(model_dir)
            # The tiny tokenizer's vocab is only a/b/c + endoftext,
            # so the sentence must use those bytes ("The river flows
            # quietly." tokenizes to [] and the forward pass fails
            # on an empty input — a fixture limit, not an nll bug).
            nll = model.nll("The cat sat.")
            self.assertIsInstance(nll, float)
            self.assertGreater(nll, 0.0)
            top = model.top1("The cat sat.")
            self.assertIsInstance(top, int)
            self.assertGreaterEqual(top, 0)
            # Deterministic: same input, same outputs.
            self.assertEqual(model.nll("The cat sat."),
                             nll)
            self.assertEqual(model.top1("The cat sat."),
                             top)


@unittest.skipUnless(torch is not None, "torch not installed")
class TestAlphaCharacterisation(unittest.TestCase):
    def test_legacy_alpha_overshoot_ratio(self):
        """CHARACTERISATION, not a fix (W2.7b / F05).

        Synthetic single-layer MLP with known activations: the
        legacy alpha divides the direction norm by the MEAN
        activation, but the edit shifts the output by the SUM over
        the selected columns — so the actual mean shift along the
        direction overshoots direction_norm by about the column
        count. Prints the ratio; the formula is NOT changed here.
        """
        import numpy as np

        from wsp.real_edit import (alpha_base_from_signals,
                                   apply_column_edit)

        hidden, n_cols = 4, 3
        direction = np.array([1.0, 0.0, 0.0, 0.0])
        direction_norm = 3.0
        # Known activations: mean per column 2.0, sum per prompt 6.0.
        activations = np.array([[2.0, 2.0, 2.0],
                                [1.0, 1.0, 1.0],
                                [3.0, 3.0, 3.0]])
        mean_act = float(activations.mean())
        alpha = alpha_base_from_signals(direction_norm, mean_act)
        weight = torch.zeros(hidden, n_cols)
        new_weight, _changed = apply_column_edit(
            weight, [0, 1, 2], direction, alpha)
        delta = (new_weight - weight).numpy()
        # Mean shift along the direction, over the known prompts.
        shifts = [float(np.dot(delta @ a, direction))
                  for a in activations]
        mean_shift = float(np.mean(shifts))
        ratio = abs(mean_shift) / direction_norm
        print(f"\n[alpha characterisation] legacy alpha={alpha:.4f} "
              f"mean_shift={mean_shift:.2f} "
              f"direction_norm={direction_norm:.2f} "
              f"ratio={ratio:.2f} (columns={n_cols})")
        self.assertAlmostEqual(ratio, float(n_cols), places=6)


if __name__ == "__main__":
    unittest.main()
