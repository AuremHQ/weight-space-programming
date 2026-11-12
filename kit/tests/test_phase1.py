"""Phase 1 tests (W1.10, W1.5, W1.4, W1.6, W1.7): atomic writes,
recipes, the recipe-level cap, safe output dirs, pre-load size estimate.

Plain unittest, run from the kit directory:
  PYTHONPATH=. python3 -m unittest discover -s tests -v
No test touches the network (HF_HUB_OFFLINE=1).
"""

import hashlib
import importlib.util
import json
import math
import os
import tempfile
import unittest
from unittest import mock

from wsp import atomic, recipe
from wsp.real_edit import MAX_EDIT_FRACTION, check_recipe_cap
from wsp.safe_paths import assert_output_dir_safe, resolve_model_dirs
from wsp.size_check import estimate_param_count

try:
    import torch
except ImportError:
    torch = None


def _qwen_like_layer_infos(n_layers=28, hidden=1024, intermediate=3072):
    """layer_infos in the shape layers.describe_layers returns."""
    return [{
        "index": i,
        "block_path": "model.layers[%d]" % i,
        "mlp_name": "mlp",
        "down_proj_name": "down_proj",
        "weight_layout": "out_in",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "down_proj_numel": hidden * intermediate,
    } for i in range(n_layers)]


def _valid_recipe(model_id="Qwen/Qwen3-0.6B", layers=None):
    return {
        "format_version": 1,
        "model_id": model_id,
        "dtype": "bfloat16",
        "layers": layers if layers is not None else [
            {"index": 27, "columns": [3, 4],
             "direction": [0.1, 0.2, 0.3], "alpha": 0.25},
            {"index": 26, "columns": [7],
             "direction": [0.4, 0.5, 0.6], "alpha": 0.25},
        ],
    }


def _matching_report(rec, verdict=None):
    """A gate-passing report that matches ``rec`` (provenance-true)."""
    gate = {"passed": True}
    if verdict is not None:
        gate["verdict"] = verdict
    return {
        "report_version": 1,
        "gate": gate,
        "real_demo": {
            "model_id": rec["model_id"],
            "edit_sha256": recipe.edit_sha256(rec),
        },
    }


# ---------------------------------------------------------------- W1.10

class TestAtomicWriteJson(unittest.TestCase):
    def test_F10_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.json")
            atomic.atomic_write_json(path, {"a": 1, "b": [2, 3]})
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 1, "b": [2, 3]})

    def test_F10_crash_mid_write_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.json")
            atomic.atomic_write_json(path, {"a": 1})
            with mock.patch("wsp.atomic.json.dump",
                            side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    atomic.atomic_write_json(path, {"a": 2})
            # Previous content intact, no temp file left behind.
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 1})
            leftovers = [n for n in os.listdir(td) if n != "x.json"]
            self.assertEqual(leftovers, [])

    def test_F10_overwrite_replaces(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.json")
            atomic.atomic_write_json(path, {"a": 1})
            atomic.atomic_write_json(path, {"a": 2})
            with open(path) as f:
                self.assertEqual(json.load(f), {"a": 2})
            self.assertEqual(os.listdir(td), ["x.json"])


# ---------------------------------------------------------------- W1.5

class TestRecipeRoundTrip(unittest.TestCase):
    def test_F13_write_then_load(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edit.json")
            rec = _valid_recipe()
            recipe.write_recipe(path, rec)
            loaded = recipe.load_recipe(path)
            self.assertEqual(loaded["model_id"], rec["model_id"])
            self.assertEqual(loaded["layers"], rec["layers"])
            self.assertEqual(loaded["edit_type"], "subtract")
            self.assertEqual(loaded["kit_version"], recipe.KIT_VERSION)

    def test_F13_hash_stable_across_key_order_and_metadata(self):
        rec_a = _valid_recipe()
        rec_b = {"layers": list(reversed(rec_a["layers"])),
                 "dtype": "bfloat16",
                 "model_id": "Qwen/Qwen3-0.6B",
                 "format_version": 1}
        # Layer order is part of the edit, so reversed layers hash
        # differently; same edit with extra metadata hashes the same.
        self.assertNotEqual(recipe.edit_sha256(rec_a),
                            recipe.edit_sha256(rec_b))
        rec_c = dict(rec_a)
        rec_c["kit_version"] = "9.9.9"
        rec_c["created_at"] = "2999-01-01T00:00:00Z"
        self.assertEqual(recipe.edit_sha256(rec_a),
                         recipe.edit_sha256(rec_c))
        rec_d = _valid_recipe()
        rec_d["layers"][0]["columns"] = [3, 5]
        self.assertNotEqual(recipe.edit_sha256(rec_a),
                            recipe.edit_sha256(rec_d))

    def test_F13_hash_matches_sha256_of_canonical_bytes(self):
        rec = _valid_recipe()
        expect = hashlib.sha256(
            recipe.canonical_layers_bytes(rec)).hexdigest()
        self.assertEqual(recipe.edit_sha256(rec), expect)

    def test_F13_constants(self):
        self.assertEqual(recipe.MAX_RECIPE_BYTES, 5_000_000)
        self.assertEqual(recipe.MAX_RECIPE_LAYERS, 8)
        self.assertEqual(recipe.ALPHA_HARD_MAX, 100.0)


class TestLoadRecipeRejects(unittest.TestCase):
    def _write(self, td, obj, name="edit.json", raw=None):
        path = os.path.join(td, name)
        with open(path, "w") as f:
            f.write(raw if raw is not None else json.dumps(obj))
        return path

    def test_F13_bad_format_version(self):
        with tempfile.TemporaryDirectory() as td:
            bad = _valid_recipe()
            bad["format_version"] = 99
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, bad))

    def test_F13_missing_model_id_and_empty_layers(self):
        with tempfile.TemporaryDirectory() as td:
            bad = _valid_recipe()
            del bad["model_id"]
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, bad))
            bad = _valid_recipe(layers=[])
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, bad))

    def test_F13_too_many_layers(self):
        with tempfile.TemporaryDirectory() as td:
            layers = [{"index": i, "columns": [0],
                       "direction": [0.1], "alpha": 1.0}
                      for i in range(9)]
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, _valid_recipe(layers)))

    def test_F13_alpha_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            for bad_alpha in (0, -1.0, 100.01, 1e30):
                rec = _valid_recipe(layers=[
                    {"index": 0, "columns": [0],
                     "direction": [0.1], "alpha": bad_alpha}])
                with self.assertRaises(ValueError, msg=str(bad_alpha)):
                    recipe.load_recipe(self._write(td, rec))
            # Exactly 100 is the hard max: allowed.
            rec = _valid_recipe(layers=[
                {"index": 0, "columns": [0],
                 "direction": [0.1], "alpha": 100.0}])
            recipe.load_recipe(self._write(td, rec))

    def test_F13_nan_and_inf_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            # json.loads accepts NaN/Infinity by default; the recipe
            # loader must not.
            raw = ('{"format_version": 1, "model_id": "m", "layers": '
                   '[{"index": 0, "columns": [0], "direction": [NaN], '
                   '"alpha": 1.0}]}')
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, None, raw=raw))
            raw = ('{"format_version": 1, "model_id": "m", "layers": '
                   '[{"index": 0, "columns": [0], "direction": [0.1], '
                   '"alpha": Infinity}]}')
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, None, raw=raw))
            raw = ('{"format_version": 1, "model_id": "m", "layers": '
                   '[{"index": 0, "columns": [0], "direction": [0.1], '
                   '"alpha": 1e999}]}')
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, None, raw=raw))

    def test_F13_non_int_index_and_bool_columns(self):
        with tempfile.TemporaryDirectory() as td:
            rec = _valid_recipe(layers=[
                {"index": "27", "columns": [0],
                 "direction": [0.1], "alpha": 1.0}])
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, rec))
            rec = _valid_recipe(layers=[
                {"index": 27, "columns": [True],
                 "direction": [0.1], "alpha": 1.0}])
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, rec))
            rec = _valid_recipe(layers=[
                {"index": 27, "columns": [],
                 "direction": [0.1], "alpha": 1.0}])
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, rec))

    def test_F13_unknown_edit_type_refused(self):
        with tempfile.TemporaryDirectory() as td:
            rec = _valid_recipe()
            rec["edit_type"] = "dampen"  # Phase 4 only
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, rec))

    def test_F13_non_dict_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, [1, 2, 3]))
            with self.assertRaises(ValueError):
                recipe.load_recipe(self._write(td, None, raw="{oops"))

    def test_F13_oversize_file_refused(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edit.json")
            with open(path, "w") as f:
                f.write('{"format_version": 1, "model_id": "m", '
                        '"layers": [], "pad": "')
                f.write("x" * (recipe.MAX_RECIPE_BYTES + 1))
                f.write('"}')
            with self.assertRaises(ValueError):
                recipe.load_recipe(path)


class TestValidateAgainstModel(unittest.TestCase):
    def setUp(self):
        self.infos = _qwen_like_layer_infos()
        self.total = 596_049_920

    def test_F13_valid_passes(self):
        rec = _valid_recipe()
        # Directions in _valid_recipe are length 3; the fake model
        # here has hidden 1024, so use matching directions.
        rec["layers"] = [
            {"index": 27, "columns": [3, 4],
             "direction": [0.1] * 1024, "alpha": 0.25},
            {"index": 26, "columns": [7],
             "direction": [0.4] * 1024, "alpha": 0.25},
        ]
        recipe.validate_against_model(rec, self.infos, self.total)

    def test_F13_layer_index_out_of_range(self):
        rec = _valid_recipe(layers=[
            {"index": 28, "columns": [0],
             "direction": [0.1] * 1024, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)
        rec = _valid_recipe(layers=[
            {"index": -1, "columns": [0],
             "direction": [0.1] * 1024, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)

    def test_F13_column_out_of_range_and_duplicates(self):
        rec = _valid_recipe(layers=[
            {"index": 0, "columns": [3072],
             "direction": [0.1] * 1024, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)
        rec = _valid_recipe(layers=[
            {"index": 0, "columns": [3, 3],
             "direction": [0.1] * 1024, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)

    def test_F13_direction_length_must_equal_hidden_size(self):
        rec = _valid_recipe(layers=[
            {"index": 0, "columns": [0],
             "direction": [0.1] * 512, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)

    def test_F13_direction_must_be_finite(self):
        rec = _valid_recipe(layers=[
            {"index": 0, "columns": [0],
             "direction": [float("nan")] * 1024, "alpha": 1.0}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)

    def test_F13_over_cap_refused(self):
        # 2 layers x 3072 columns x 1024 hidden = 6,291,456 weights,
        # about 1.06% of 596,049,920 — over the 1% cap.
        layers = [{"index": i, "columns": list(range(3072)),
                   "direction": [0.1] * 1024, "alpha": 1.0}
                  for i in (27, 26)]
        rec = _valid_recipe(layers=layers)
        with self.assertRaises(ValueError):
            recipe.validate_against_model(rec, self.infos, self.total)


class TestVerifyProvenance(unittest.TestCase):
    def test_F13_matching_report_is_ship(self):
        rec = _valid_recipe()
        rep = _matching_report(rec, verdict="SHIP")
        self.assertEqual(recipe.verify_provenance(rec, rep), "ship")

    def test_F13_no_verdict_is_quick(self):
        rec = _valid_recipe()
        rep = _matching_report(rec)  # v1-style report, no verdict
        self.assertEqual(recipe.verify_provenance(rec, rep), "quick")

    def test_F13_missing_report_refused(self):
        with self.assertRaises(ValueError):
            recipe.verify_provenance(_valid_recipe(), None)

    def test_F13_gate_failed_refused(self):
        rec = _valid_recipe()
        rep = _matching_report(rec)
        rep["gate"]["passed"] = False
        with self.assertRaises(ValueError):
            recipe.verify_provenance(rec, rep)

    def test_F13_hash_mismatch_refused(self):
        rec = _valid_recipe()
        rep = _matching_report(rec)
        rep["real_demo"]["edit_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            recipe.verify_provenance(rec, rep)

    def test_F13_model_id_mismatch_refused(self):
        rec = _valid_recipe()
        rep = _matching_report(rec)
        rep["real_demo"]["model_id"] = "some/other-model"
        with self.assertRaises(ValueError):
            recipe.verify_provenance(rec, rep)


# ---------------------------------------------------------------- W1.4

class TestCheckRecipeCap(unittest.TestCase):
    def setUp(self):
        self.hidden = {i: 1024 for i in range(28)}
        self.total = 596_049_920

    def _layers(self, spec):
        return [{"index": i, "columns": list(cols)}
                for i, cols in spec]

    def test_F12b_two_full_layers_rejected(self):
        # Qwen3-0.6B with --top-columns 3072 on 2 layers: 6,291,456
        # of 596,049,920 weights, about 1.06% — over the cap.
        layers = self._layers([(27, range(3072)), (26, range(3072))])
        with self.assertRaises(ValueError):
            check_recipe_cap(layers, self.hidden, self.total)

    def test_F12b_one_full_layer_accepted(self):
        layers = self._layers([(27, range(3072))])
        frac = check_recipe_cap(layers, self.hidden, self.total)
        self.assertAlmostEqual(frac, 3072 * 1024 / self.total)

    def test_F12b_two_small_layers_accepted(self):
        layers = self._layers([(27, range(64)), (26, range(64))])
        frac = check_recipe_cap(layers, self.hidden, self.total)
        self.assertAlmostEqual(frac, 2 * 64 * 1024 / self.total)

    def test_F12b_exact_boundary_accepted(self):
        # 100 columns x 1024 hidden = 102,400 = exactly 1% of
        # 10,240,000. The cap is "above 1% raises", so exactly 1%
        # passes; one more column fails.
        layers = self._layers([(0, range(100))])
        frac = check_recipe_cap(layers, {0: 1024}, 10_240_000)
        self.assertAlmostEqual(frac, MAX_EDIT_FRACTION)
        layers = self._layers([(0, range(101))])
        with self.assertRaises(ValueError):
            check_recipe_cap(layers, {0: 1024}, 10_240_000)

    def test_F12b_duplicate_columns_count_once(self):
        layers = self._layers([(0, [1, 1, 2])])
        frac = check_recipe_cap(layers, {0: 8}, 10_000)
        self.assertAlmostEqual(frac, 2 * 8 / 10_000)

    def test_F12b_empty_layers_is_zero(self):
        self.assertEqual(check_recipe_cap([], self.hidden, self.total),
                         0.0)

    def test_F12b_unknown_layer_refused(self):
        layers = self._layers([(99, [0])])
        with self.assertRaises(ValueError):
            check_recipe_cap(layers, self.hidden, self.total)


# ---------------------------------------------------------------- W1.6

class TestAssertOutputDirSafe(unittest.TestCase):
    def _model_dir(self, td, name="model"):
        path = os.path.join(td, name)
        os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"model_type": "x"}, f)
        with open(os.path.join(path, "model.safetensors"), "wb") as f:
            f.write(b"weights")
        return path

    def test_F15_same_dir_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            with self.assertRaises(ValueError):
                assert_output_dir_safe(model, [model])

    def test_F15_parent_of_model_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            with self.assertRaises(ValueError):
                assert_output_dir_safe(td, [model])

    def test_F15_child_of_model_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            with self.assertRaises(ValueError):
                assert_output_dir_safe(os.path.join(model, "sub"),
                                       [model])

    def test_F15_symlink_to_model_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            link = os.path.join(td, "link")
            try:
                os.symlink(model, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not supported here")
            with self.assertRaises(ValueError):
                assert_output_dir_safe(link, [model])

    def test_F15_out_dir_with_weights_refused_unless_force(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td, "model-a")
            out = os.path.join(td, "out")
            os.makedirs(out)
            with open(os.path.join(out, "config.json"), "w") as f:
                json.dump({}, f)
            with self.assertRaises(ValueError):
                assert_output_dir_safe(out, [model])
            assert_output_dir_safe(out, [model], force=True)

    def test_F15_out_dir_with_only_reports_is_fine(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td, "model-a")
            out = os.path.join(td, "out")
            os.makedirs(out)
            with open(os.path.join(out, "report.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(out, "edit.json"), "w") as f:
                json.dump({}, f)
            assert_output_dir_safe(out, [model])  # no raise

    def test_F15_nonexistent_out_dir_is_fine(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            assert_output_dir_safe(os.path.join(td, "fresh-out"),
                                   [model])  # no raise

    def test_F15_non_directory_refs_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out")
            # A HuggingFace id is not a local directory: nothing to
            # protect on disk by itself. Its cached snapshot dir is
            # resolved separately (resolve_model_dirs) and checked
            # by the pipeline / merge / apply-edit call sites.
            assert_output_dir_safe(out, ["Qwen/Qwen3-0.6B"])  # no raise

    def test_F15_relative_paths_resolved(self):
        with tempfile.TemporaryDirectory() as td:
            model = self._model_dir(td)
            cwd = os.getcwd()
            try:
                os.chdir(td)
                with self.assertRaises(ValueError):
                    assert_output_dir_safe("model", ["model"])
                assert_output_dir_safe("fresh-out", ["model"])
            finally:
                os.chdir(cwd)


class TestResolveModelDirs(unittest.TestCase):
    """W1.6: HF ids resolve to their cached snapshot dirs."""

    def test_local_dir_resolves_to_itself(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(resolve_model_dirs([td]), [td])
            self.assertEqual(resolve_model_dirs([td, None, ""]), [td])

    def test_uncached_hub_id_resolves_to_nothing(self):
        # Nothing on disk yet: there are no local weights to clobber.
        self.assertEqual(resolve_model_dirs(["org/not-cached"]), [])

    @unittest.skipUnless(
        importlib.util.find_spec("huggingface_hub") is not None,
        "huggingface_hub not installed")
    def test_cached_hub_id_resolves_to_snapshot(self):
        import huggingface_hub.constants as hub_constants

        with tempfile.TemporaryDirectory() as td:
            repo = os.path.join(td, "hub", "models--org--tiny")
            snap = os.path.join(repo, "snapshots", "abc123def456")
            os.makedirs(os.path.join(repo, "refs"))
            os.makedirs(snap)
            with open(os.path.join(repo, "refs", "main"), "w") as f:
                f.write("abc123def456")
            with open(os.path.join(snap, "config.json"), "w") as f:
                json.dump({"model_type": "gpt2"}, f)
            with mock.patch.object(hub_constants, "HF_HUB_CACHE",
                                   os.path.join(td, "hub")):
                self.assertEqual(resolve_model_dirs(["org/tiny"]),
                                 [snap])


# ---------------------------------------------------------------- W1.7

@unittest.skipUnless(torch is not None, "torch not installed")
class TestEstimateParamCount(unittest.TestCase):
    def _tiny_model_dir(self, td):
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100, n_positions=64, n_embd=32, n_layer=2,
            n_head=2)
        model = GPT2LMHeadModel(config)
        model.save_pretrained(td)
        return model

    def test_F16_tiny_config_estimate_matches_count_params(self):
        from wsp.real_runner import count_params

        with tempfile.TemporaryDirectory() as td:
            model = self._tiny_model_dir(td)
            expect = count_params(model)
            self.assertEqual(estimate_param_count(td), expect)

    def test_F16_unknown_model_refuses_fail_closed(self):
        # No such model, no cache, no network (HF_HUB_OFFLINE=1):
        # the size cannot be determined, so it refuses.
        with self.assertRaises(ValueError):
            estimate_param_count("definitely-not-a-real-model-xyz-123")

    def test_F16_hub_metadata_fallback(self):
        from unittest import mock

        info = mock.Mock()
        info.safetensors = mock.Mock()
        info.safetensors.total = 7_000_000_000
        with mock.patch("huggingface_hub.HfApi") as api_cls:
            api_cls.return_value.model_info.return_value = info
            with mock.patch(
                    "huggingface_hub.snapshot_download",
                    side_effect=ValueError("not cached")):
                self.assertEqual(
                    estimate_param_count("some/7b-model"), 7_000_000_000)

    def test_F16_no_size_cap_on_load(self):
        # The 3B cap is removed: a model whose parameter count is 7B
        # loads fine. The count is still recorded — the 1% edit cap
        # needs it (apply_edit / dampen_columns).
        from unittest import mock

        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (GPT2Config, GPT2LMHeadModel,
                                  PreTrainedTokenizerFast)
        from wsp.real_runner import RealModel

        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "model")
            os.makedirs(model_dir)
            vocab = {"a": 0, "b": 1, "c": 2, "<|endoftext|>": 3}
            tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False)
            fast = PreTrainedTokenizerFast(
                tokenizer_object=tok, eos_token="<redacted>",
                pad_token="<redacted>")
            fast.save_pretrained(model_dir)
            config = GPT2Config(vocab_size=4, n_positions=64, n_embd=32,
                                n_layer=2, n_head=2, bos_token_id=3,
                                eos_token_id=3)
            GPT2LMHeadModel(config).save_pretrained(model_dir)
            with mock.patch("wsp.real_runner.count_params",
                            return_value=7_000_000_000):
                model = RealModel(model_dir)
            self.assertEqual(model.total_params, 7_000_000_000)


@unittest.skipUnless(torch is not None, "torch not installed")
class TestRealModelSourceDirSafety(unittest.TestCase):
    """W1.6 defence in depth: RealModel.save refuses the source dir."""

    def _tiny_model_dir(self, td):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (GPT2Config, GPT2LMHeadModel,
                                  PreTrainedTokenizerFast)

        vocab = {"a": 0, "b": 1, "c": 2, "<|endoftext|>": 3}
        tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False)
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tok, eos_token="<|endoftext|>",
            pad_token="<|endoftext|>")
        fast.save_pretrained(td)
        config = GPT2Config(vocab_size=4, n_positions=64, n_embd=32,
                            n_layer=2, n_head=2, bos_token_id=3,
                            eos_token_id=3)
        GPT2LMHeadModel(config).save_pretrained(td)

    def test_F15_save_refuses_source_dir(self):
        from wsp.real_runner import RealModel

        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "model")
            os.makedirs(model_dir)
            self._tiny_model_dir(model_dir)
            model = RealModel(model_dir)
            self.assertEqual(model.source_dir,
                             os.path.abspath(model_dir))
            with self.assertRaises(ValueError):
                model.save(model_dir)
            with self.assertRaises(ValueError):
                model.save(os.path.join(model_dir, "out"))
            out = os.path.join(td, "edited")
            self.assertEqual(model.save(out), out)
            self.assertTrue(os.path.exists(
                os.path.join(out, "config.json")))


if __name__ == "__main__":
    unittest.main()
