"""Tests for Merge mode (R6): same-base check, merge math, gate reuse.

Same-base check and merge math are pure numpy and need no model
download and no torch. Run from the kit directory:
  PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

import numpy as np

from wsp import gate, merge
from wsp.merge import (
    check_same_base,
    count_eval_passes,
    merge_linear,
    merge_slerp,
    merge_state_dicts,
)

try:
    import torch
except ImportError:
    torch = None


def _config(**over):
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "tie_word_embeddings": True,
    }
    cfg.update(over)
    return cfg


def _toks(**over):
    toks = {"tokenizer.json": "aaa", "tokenizer_config.json": "bbb"}
    toks.update(over)
    return toks


class TestSameBase(unittest.TestCase):
    def test_identical_configs_pass(self):
        self.assertEqual(
            check_same_base(_config(), _config(), _toks(), _toks()), [])

    def test_different_architecture_refused(self):
        self.assertTrue(check_same_base(
            _config(), _config(architectures=["Qwen2ForCausalLM"]),
            _toks(), _toks()))

    def test_different_vocab_refused(self):
        self.assertTrue(check_same_base(
            _config(), _config(vocab_size=32000), _toks(), _toks()))

    def test_different_hidden_size_refused(self):
        self.assertTrue(check_same_base(
            _config(), _config(hidden_size=2048), _toks(), _toks()))

    def test_different_layer_count_refused(self):
        self.assertTrue(check_same_base(
            _config(), _config(num_hidden_layers=24), _toks(), _toks()))

    def test_key_missing_in_one_refused(self):
        cfg_b = _config()
        del cfg_b["intermediate_size"]
        self.assertTrue(check_same_base(
            _config(), cfg_b, _toks(), _toks()))

    def test_tokenizer_hash_mismatch_refused(self):
        self.assertTrue(check_same_base(
            _config(), _config(),
            _toks(), _toks(**{"tokenizer.json": "zzz"})))

    def test_tokenizer_file_missing_in_one_refused(self):
        toks_b = _toks()
        del toks_b["tokenizer_config.json"]
        self.assertTrue(check_same_base(
            _config(), _config(), _toks(), toks_b))

    def test_slow_file_missing_ok_when_fast_tokenizer_matches(self):
        # vocab.json / merges.txt are redundant when tokenizer.json
        # matches — a model dir without them is the same tokenizer.
        toks_a = _toks(**{"vocab.json": "v", "merges.txt": "m"})
        self.assertEqual(check_same_base(
            _config(), _config(), toks_a, _toks()), [])

    def test_tokenizer_json_missing_in_one_refused(self):
        toks_b = _toks()
        del toks_b["tokenizer.json"]
        self.assertTrue(check_same_base(
            _config(), _config(), _toks(), toks_b))

    def test_slow_only_models_compare_slow_files(self):
        slow_a = {"vocab.json": "v", "merges.txt": "m"}
        slow_b = {"vocab.json": "v", "merges.txt": "m"}
        self.assertEqual(check_same_base(
            _config(), _config(), slow_a, slow_b), [])
        slow_c = {"vocab.json": "v", "merges.txt": "OTHER"}
        self.assertTrue(check_same_base(
            _config(), _config(), slow_a, slow_c))

    def test_all_mismatches_listed(self):
        # The refusal names every difference, not just the first.
        bad = check_same_base(
            _config(), _config(vocab_size=1, hidden_size=2),
            _toks(), _toks(**{"tokenizer.json": "zzz"}))
        self.assertEqual(len(bad), 3)


class TestMergeMath(unittest.TestCase):
    def test_linear_endpoints(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        np.testing.assert_array_equal(merge_linear(a, b, 0.0), a)
        np.testing.assert_array_equal(merge_linear(a, b, 1.0), b)

    def test_linear_midpoint_is_mean(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.testing.assert_allclose(
            merge_linear(a, b, 0.5), (a + b) / 2, rtol=1e-6)

    def test_linear_quarter_weight(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.ones(4, dtype=np.float32)
        np.testing.assert_allclose(
            merge_linear(a, b, 0.25), np.full(4, 0.25), rtol=1e-6)

    def test_linear_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge_linear(np.zeros(3), np.zeros(4), 0.5)

    def test_slerp_endpoints(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(merge_slerp(a, b, 0.0), a, atol=1e-6)
        np.testing.assert_allclose(merge_slerp(a, b, 1.0), b, atol=1e-6)

    def test_slerp_midpoint_is_45_degrees(self):
        # Orthogonal unit vectors: the SLERP midpoint is the
        # 45-degree unit vector, not the (shorter) linear mean.
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        mid = merge_slerp(a, b, 0.5)
        np.testing.assert_allclose(
            mid, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(mid)), 1.0, places=5)

    def test_slerp_parallel_falls_back_to_linear(self):
        a = np.array([2.0, 0.0], dtype=np.float32)
        b = np.array([4.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            merge_slerp(a, b, 0.5), merge_linear(a, b, 0.5), rtol=1e-6)

    def test_slerp_zero_vector_falls_back_to_linear(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        np.testing.assert_allclose(
            merge_slerp(a, b, 0.5), merge_linear(a, b, 0.5), rtol=1e-6)

    def test_slerp_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            merge_slerp(np.zeros(3), np.zeros(4), 0.5)

    def test_merge_preserves_dtype(self):
        a = np.array([1, 2], dtype=np.float16)
        b = np.array([3, 4], dtype=np.float16)
        self.assertEqual(merge_linear(a, b, 0.5).dtype, np.float16)
        self.assertEqual(merge_slerp(a, b, 0.5).dtype, np.float16)


class TestMergeStateDicts(unittest.TestCase):
    def _states(self):
        sa = {
            "w": np.array([1.0, 2.0], dtype=np.float32),
            "ids": np.array([7, 8], dtype=np.int64),
        }
        sb = {
            "w": np.array([3.0, 4.0], dtype=np.float32),
            "ids": np.array([7, 8], dtype=np.int64),
        }
        return sa, sb

    def test_floats_merged_ints_copied(self):
        sa, sb = self._states()
        merged, stats = merge_state_dicts(sa, sb, "linear", 0.5)
        np.testing.assert_allclose(merged["w"], [2.0, 3.0], rtol=1e-6)
        np.testing.assert_array_equal(merged["ids"], [7, 8])
        self.assertEqual(stats["tensors_merged"], 1)
        self.assertEqual(stats["tensors_copied"], 1)
        self.assertEqual(stats["params_merged"], 2)

    def test_slerp_method_selected(self):
        sa = {"w": np.array([1.0, 0.0], dtype=np.float32)}
        sb = {"w": np.array([0.0, 1.0], dtype=np.float32)}
        merged, _ = merge_state_dicts(sa, sb, "slerp", 0.5)
        np.testing.assert_allclose(
            merged["w"], [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)

    def test_key_mismatch_raises(self):
        sa, sb = self._states()
        sb["extra"] = np.zeros(1, dtype=np.float32)
        with self.assertRaises(ValueError):
            merge_state_dicts(sa, sb, "linear", 0.5)

    def test_shape_mismatch_raises(self):
        sa, sb = self._states()
        sb["w"] = np.zeros(3, dtype=np.float32)
        with self.assertRaises(ValueError):
            merge_state_dicts(sa, sb, "linear", 0.5)

    def test_int_tensor_mismatch_raises(self):
        sa, sb = self._states()
        sb["ids"] = np.array([7, 9], dtype=np.int64)
        with self.assertRaises(ValueError):
            merge_state_dicts(sa, sb, "linear", 0.5)

    def test_unknown_method_raises(self):
        sa, sb = self._states()
        with self.assertRaises(ValueError):
            merge_state_dicts(sa, sb, "ties", 0.5)

    def test_t_out_of_range_raises(self):
        sa, sb = self._states()
        with self.assertRaises(ValueError):
            merge_state_dicts(sa, sb, "linear", 1.5)


class TestEvalPasses(unittest.TestCase):
    def test_count_matches_evaluate(self):
        # 2 failing + 2 good + 2 neighbours; temps 0.7 (3 samples)
        # and 0.0 (1 sample); x2 models.
        data = {
            "sets": {
                "failing": {"prompts": [{}, {}]},
                "good": {"prompts": [{}, {}]},
                "neighbours": {"prompts": [{}, {}]},
            }
        }
        self.assertEqual(
            count_eval_passes(data, [0.7, 0.0], 3), (6 * 3 + 6 * 1) * 2)

    def test_fixture_eval_passes(self):
        path = os.path.join(os.path.dirname(__file__),
                            "merge-test-eval.json")
        with open(path) as f:
            data = json.load(f)
        # 6 prompts per temp at 0.7 x1 sample, 6 at 0.0 x1, x2 models.
        self.assertEqual(count_eval_passes(data, [0.7, 0.0], 1), 24)


class TestGateReuse(unittest.TestCase):
    def test_same_gate_thresholds(self):
        # The merge gate is the surgical gate, reused: +5 primary,
        # <=2 regression. A 5-point gain with no regression passes.
        before = {"failing": 30.0, "good": 100.0, "neighbours": 100.0}
        after = {"failing": 35.0, "good": 100.0, "neighbours": 100.0}
        self.assertTrue(gate.check_gate(before, after)["passed"])
        # 4.99 points is not a pass.
        after = {"failing": 34.99, "good": 100.0, "neighbours": 100.0}
        self.assertFalse(gate.check_gate(before, after)["passed"])
        # A 2.01-point guard regression fails even with a big gain.
        after = {"failing": 50.0, "good": 97.99, "neighbours": 100.0}
        self.assertFalse(gate.check_gate(before, after)["passed"])

    def test_merge_module_knows_its_methods(self):
        self.assertEqual(list(merge.METHODS), ["linear", "slerp"])
        self.assertEqual(merge.MAX_PARAMS, 3_000_000_000)


@unittest.skipIf(torch is None, "needs torch + safetensors")
class TestBf16RoundTrip(unittest.TestCase):
    def test_bf16_load_widens_and_save_restores(self):
        # Regression test: numpy has no bf16, so tensor.numpy()
        # raises TypeError on bf16 weights. load_model_dir must
        # widen to float32 for the math and save_merged_model must
        # restore the stored dtype.
        from safetensors.torch import load_file, save_file

        with tempfile.TemporaryDirectory() as d:
            w = torch.randn(2, 3).to(torch.bfloat16)
            save_file({"w": w, "ids": torch.tensor([1, 2])},
                      os.path.join(d, "model.safetensors"))
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"architectures": ["X"], "model_type": "x"}, f)
            config, state, _tok, total, dtypes = merge.load_model_dir(d)
            self.assertEqual(dtypes["w"], "torch.bfloat16")
            self.assertEqual(state["w"].dtype, np.float32)
            self.assertEqual(total, 8)
            np.testing.assert_allclose(
                state["w"], w.to(torch.float32).numpy(), rtol=1e-2)
            merged, _ = merge_state_dicts(state, state, "linear", 0.5)
            out = os.path.join(d, "out")
            merge.save_merged_model(out, config, d, merged, dtypes)
            back = load_file(os.path.join(out, "model.safetensors"))
            self.assertEqual(back["w"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(back["ids"], torch.tensor([1, 2])))


class TestCliMergeHelp(unittest.TestCase):
    def test_merge_subcommand_listed(self):
        # --help must work without torch (merge import is lazy).
        from wsp.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["merge", "--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
