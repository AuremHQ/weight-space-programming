"""Phase 4 (W4.0-W4.5): evidence pack — dampen editor, causal-sign
rule, calibrated alpha, the missing evals + passing run reports,
`wsp apply-edit --verify`, `wsp reproduce`.

The founder's own tests live in tests/test_real.py (ported from the
combined-pipeline staging dir). This file covers the integration:
recipe/serve-model/apply-edit handling of "dampen", the sign-agnostic
causal check, the tensor-hash verify, the reproduce subcommand, the
evidence files (pinned by sha256 from the W4.0 inventory), and the
REAL-RUNS.md evidence column.

Run from the kit directory:
  PYTHONPATH=. python3 -m unittest tests.test_phase4 -v
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests.fakes import FakeRealModel
from wsp import cli, recipe

try:
    import torch
except ImportError:
    torch = None

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(KIT_DIR)

# W4.0 inventory (~/workspace/real-model-demo-build-backup-2026-11-06):
# sha256 of the FILE BYTES of each evidence file, as found.
SHA_NARROW_EVAL = ("66e74922b6848e649df0da95b5c3debda916a9bbc2aa37e23d"
                   "4273a0a1fed241")
SHA_SMOLLM2_EVAL = ("c25d05a010c0e6838f40e45a3f4055e52847705dedf05e97a"
                    "7a3b6e8f7b14c01")
SHA_DAMPEN_REPORT = ("0903367db18e7343223ee15a05bf357b2dbe6041b3000a1"
                     "121d93eef3129fed8")
SHA_DAMPEN_EDIT = ("0fde5a2600ead6ffa02c6b469e99d46d85bb12bb8f7681049"
                   "6d59558da8e9e21")
SHA_EDITOR_FIX_REPORT = ("7326f17508edfaa20eebbc14ec24618401c7fa00071f"
                         "98740c86f854ba5ba8bb")
SHA_DAMPEN_MODEL_FILE = ("b98602661bb2d0914f3c3e66a4867838861209433676"
                         "336a8734ff03ab30b125")
# sha256 of the CHANGED TENSORS of the edited Qwen model: raw BF16
# bytes of model.layers.26.mlp.down_proj.weight then
# model.layers.27.mlp.down_proj.weight (ascending layer order),
# read from wsp-real-out-qwen-narrow-dampen/model.safetensors.
SHA_DAMPEN_TENSORS = ("842972c87e1e99bbb6ee8cd40f8a1d5d96f554939a43673"
                      "b9f986f8a553a35fc")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_cli(fn, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(namespace)
    return code, out.getvalue()


def _dampen_recipe(layers=None):
    return {
        "format_version": 1,
        "model_id": "fake-model",
        "dtype": "bfloat16",
        "edit_type": "dampen",
        "layers": layers if layers is not None else [
            {"index": 0, "columns": [0, 1], "dampen_factor": 0.5},
        ],
    }


def _fake_layer_infos():
    return [{
        "index": i,
        "block_path": "model.layers[%d]" % i,
        "mlp_name": "mlp",
        "down_proj_name": "down_proj",
        "weight_layout": "out_in",
        "hidden_size": 8,
        "intermediate_size": 16,
        "down_proj_numel": 8 * 16,
    } for i in range(4)]


# ---------------------------------------------------------------- W4.2 recipe
class TestRecipeDampen(unittest.TestCase):
    def test_load_recipe_accepts_dampen(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edit.json")
            with open(path, "w") as f:
                json.dump(_dampen_recipe(), f)
            rec = recipe.load_recipe(path)
            self.assertEqual(rec["edit_type"], "dampen")
            self.assertEqual(rec["layers"][0]["dampen_factor"], 0.5)

    def test_load_recipe_rejects_dampen_factor_out_of_range(self):
        for bad in (1.5, -0.1, 2.0):
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "edit.json")
                with open(path, "w") as f:
                    json.dump(_dampen_recipe([
                        {"index": 0, "columns": [0],
                         "dampen_factor": bad}]), f)
                with self.assertRaises(ValueError):
                    recipe.load_recipe(path)

    def test_load_recipe_rejects_dampen_factor_missing_or_not_a_number(self):
        for layer in ({"index": 0, "columns": [0]},
                      {"index": 0, "columns": [0],
                       "dampen_factor": "half"}):
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "edit.json")
                with open(path, "w") as f:
                    json.dump(_dampen_recipe([layer]), f)
                with self.assertRaises(ValueError):
                    recipe.load_recipe(path)

    def test_load_recipe_rejects_dampen_layer_with_direction_or_alpha(self):
        # A dampen layer must not smuggle subtraction fields in.
        layer = {"index": 0, "columns": [0], "dampen_factor": 0.5,
                 "direction": [0.1] * 8, "alpha": 1.0}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edit.json")
            with open(path, "w") as f:
                json.dump(_dampen_recipe([layer]), f)
            with self.assertRaises(ValueError):
                recipe.load_recipe(path)

    def test_load_recipe_still_rejects_unknown_edit_type(self):
        rec = _dampen_recipe()
        rec["edit_type"] = "sideways"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "edit.json")
            with open(path, "w") as f:
                json.dump(rec, f)
            with self.assertRaises(ValueError):
                recipe.load_recipe(path)

    def test_validate_dampen_against_model(self):
        infos = _fake_layer_infos()
        recipe.validate_against_model(_dampen_recipe(), infos, 1_000_000)
        # Duplicate columns refused.
        bad = _dampen_recipe([{"index": 0, "columns": [0, 0],
                               "dampen_factor": 0.5}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(bad, infos, 1_000_000)
        # Out-of-range column refused.
        bad = _dampen_recipe([{"index": 0, "columns": [16],
                               "dampen_factor": 0.5}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(bad, infos, 1_000_000)
        # Unknown layer refused.
        bad = _dampen_recipe([{"index": 9, "columns": [0],
                               "dampen_factor": 0.5}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(bad, infos, 1_000_000)

    def test_validate_dampen_cap_counts_union(self):
        # Qwen-shaped stub: hidden 1024, intermediate 3072, 28
        # layers, total 596,049,920. Two layers x 3072 columns is
        # ~1.06% — over the cap; two layers x 64 is fine.
        infos = [{
            "index": i, "block_path": "m", "mlp_name": "mlp",
            "down_proj_name": "down_proj", "weight_layout": "out_in",
            "hidden_size": 1024, "intermediate_size": 3072,
            "down_proj_numel": 1024 * 3072,
        } for i in (26, 27)]
        over = _dampen_recipe([
            {"index": 26, "columns": list(range(3072)),
             "dampen_factor": 0.25},
            {"index": 27, "columns": list(range(3072)),
             "dampen_factor": 0.25}])
        with self.assertRaises(ValueError):
            recipe.validate_against_model(over, infos, 596_049_920)
        ok = _dampen_recipe([
            {"index": 26, "columns": list(range(64)),
             "dampen_factor": 0.25},
            {"index": 27, "columns": list(range(64)),
             "dampen_factor": 0.25}])
        recipe.validate_against_model(ok, infos, 596_049_920)

    def test_edit_sha256_changes_with_dampen_factor(self):
        a = _dampen_recipe()
        b = _dampen_recipe([{"index": 0, "columns": [0, 1],
                             "dampen_factor": 0.25}])
        self.assertNotEqual(recipe.edit_sha256(a),
                            recipe.edit_sha256(b))
        self.assertEqual(recipe.edit_sha256(a),
                         recipe.edit_sha256(_dampen_recipe()))

    def test_historical_dampen_edit_loads(self):
        path = os.path.join(REPO_ROOT, "runs", "qwen-narrow-dampen",
                            "edit.json")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_DAMPEN_EDIT)
        rec = recipe.load_recipe(path)
        self.assertEqual(rec["edit_type"], "dampen")
        self.assertEqual(rec["model_id"], "Qwen/Qwen3-0.6B")
        self.assertEqual([l["index"] for l in rec["layers"]], [27, 26])
        self.assertTrue(all(l["dampen_factor"] == 0.25
                            for l in rec["layers"]))
        self.assertTrue(all(len(l["columns"]) == 64
                            for l in rec["layers"]))


# ---------------------------------------------------------------- W4.2 causal
class TestCausalSignEither(unittest.TestCase):
    def _check(self, model, **kw):
        from wsp.causal import causal_check

        layers = kw.pop("layers", [0])
        columns = kw.pop("columns", {0: [0, 1]})
        intermediates = kw.pop("intermediate_sizes", {0: 8})
        return causal_check(model, model.score, layers, columns,
                            intermediates, **kw)

    def test_either_passes_on_negative_effect(self):
        from tests.test_phase2_causal import StubModel

        # SmolLM2's shape: zeroing hurts by 30 — the sign now
        # selects the edit type instead of refusing the run.
        model = StubModel(base=70.0, ablated_full=40.0)
        out = self._check(model, seed=0, edit_type="either")
        self.assertEqual(out["sign"], "removal_worsens")
        self.assertAlmostEqual(out["effect"], -30.0)
        self.assertTrue(out["passed"])

    def test_either_passes_on_positive_effect(self):
        from tests.test_phase2_causal import StubModel

        model = StubModel()
        out = self._check(model, seed=0, edit_type="either")
        self.assertEqual(out["sign"], "removal_improves")
        self.assertAlmostEqual(out["effect"], 40.0)
        self.assertTrue(out["passed"])

    def test_either_fails_on_small_effect(self):
        from tests.test_phase2_causal import StubModel

        model = StubModel(base=30.0, ablated_full=33.0)
        out = self._check(model, seed=0, edit_type="either")
        self.assertAlmostEqual(out["effect"], 3.0)
        self.assertFalse(out["passed"])

    def test_either_fails_when_effect_equals_placebo(self):
        from tests.test_phase2_causal import StubModel

        model = StubModel()
        out = self._check(model, seed=0, k_placebos=200,
                          edit_type="either")
        self.assertAlmostEqual(out["placebo_abs_max"], 40.0)
        self.assertFalse(out["passed"])

    def test_subtract_now_expects_removal_worsens(self):
        # Phase 4 rule (EDITOR-FIX-REPORT §4a): causal - means the
        # columns help, so subtraction (reshape) is the family.
        from tests.test_phase2_causal import StubModel

        worsening = StubModel(base=70.0, ablated_full=40.0)
        out = self._check(worsening, seed=0, edit_type="subtract")
        self.assertTrue(out["passed"])
        improving = StubModel()
        out = self._check(improving, seed=0, edit_type="subtract")
        self.assertFalse(out["passed"])

    def test_dampen_expects_removal_improves(self):
        from tests.test_phase2_causal import StubModel

        improving = StubModel()
        out = self._check(improving, seed=0, edit_type="dampen")
        self.assertTrue(out["passed"])
        worsening = StubModel(base=70.0, ablated_full=40.0)
        out = self._check(worsening, seed=0, edit_type="dampen")
        self.assertFalse(out["passed"])


# ------------------------------------------------- W4.2 serve-model / pipeline
class TestServeModelDampen(unittest.TestCase):
    def test_apply_recipe_dampen_changes_and_restores_bitwise(self):
        from wsp.serve_model import apply_recipe, rollback
        from wsp.verify import changed_tensors_sha256

        fake = FakeRealModel("pass")
        before = changed_tensors_sha256(
            fake, [{"index": 0, "columns": [0, 1]}])
        changed, saved = apply_recipe(fake, _dampen_recipe())
        # 2 columns x 8 hidden weights actually changed.
        self.assertEqual(changed, 16)
        after = changed_tensors_sha256(
            fake, [{"index": 0, "columns": [0, 1]}])
        self.assertNotEqual(after, before)
        rollback(fake, saved)
        self.assertEqual(changed_tensors_sha256(
            fake, [{"index": 0, "columns": [0, 1]}]), before)

    def test_apply_recipe_dampen_over_cap_refused(self):
        from wsp.serve_model import apply_recipe

        class TinyTotal:
            total_params = 100
            layer_infos = _fake_layer_infos()

            def dampen_columns(self, *a):
                raise AssertionError("must not be called")

        rec = _dampen_recipe([{"index": 0, "columns": [0, 1],
                               "dampen_factor": 0.5}])
        with self.assertRaises(ValueError):
            apply_recipe(TinyTotal(), rec)

    def test_apply_recipe_subtract_still_works(self):
        from wsp.serve_model import apply_recipe

        fake = FakeRealModel("pass")
        rec = {"format_version": 1, "model_id": "fake-model",
               "dtype": "bfloat16", "edit_type": "subtract",
               "layers": [{"index": 0, "columns": [3],
                           "direction": [0.1] * 8, "alpha": 0.25}]}
        changed, _saved = apply_recipe(fake, rec)
        self.assertEqual(changed, 8)


class TestProtocolV2AlphaBounds(unittest.TestCase):
    def test_v2_alpha_bounds_are_the_column_norm_cap(self):
        from wsp.protocol import get_protocol

        # Phase 4: v2 subtraction uses the column-norm-capped
        # alpha (max 1.0, the only proven subtraction alpha).
        self.assertEqual(get_protocol("v2").config_dict()["alpha_bounds"],
                         [1e-3, 1.0])
        # v1 freezes the legacy estimator's bounds.
        self.assertEqual(get_protocol("v1").config_dict()["alpha_bounds"],
                         [1e-3, 100.0])


# ---------------------------------------------------------------- W4.4 verify
class TestChangedTensorsHash(unittest.TestCase):
    def test_hash_deterministic_and_sensitive(self):
        from wsp.verify import changed_tensors_sha256

        fake = FakeRealModel("pass")
        layers = [{"index": 0, "columns": [0, 1]}]
        h0 = changed_tensors_sha256(fake, layers)
        self.assertEqual(h0, changed_tensors_sha256(fake, layers))
        _c, saved = fake.dampen_columns(0, [0, 1], 0.5)
        h1 = changed_tensors_sha256(fake, layers)
        self.assertNotEqual(h1, h0)
        fake.restore(saved)
        self.assertEqual(changed_tensors_sha256(fake, layers), h0)
        _c, saved = fake.dampen_columns(0, [0, 1], 0.25)
        self.assertNotEqual(changed_tensors_sha256(fake, layers), h1)
        fake.restore(saved)

    def test_hash_covers_the_full_tensor_not_just_edited_columns(self):
        # The hash is over the whole down-proj tensor per edited
        # layer (definition in wsp/verify.py): changing an
        # unedited column still changes the hash.
        from wsp.verify import changed_tensors_sha256

        fake = FakeRealModel("pass")
        layers = [{"index": 0, "columns": [0, 1]}]
        h0 = changed_tensors_sha256(fake, layers)
        fake._weights[0][:, 5] = 7.0
        self.assertNotEqual(changed_tensors_sha256(fake, layers), h0)

    def test_hash_layer_order_is_canonical(self):
        from wsp.verify import changed_tensors_sha256

        fake = FakeRealModel("pass")
        a = changed_tensors_sha256(
            fake, [{"index": 1}, {"index": 0}])
        b = changed_tensors_sha256(
            fake, [{"index": 0}, {"index": 1}])
        self.assertEqual(a, b)

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_tensor_bytes_handles_bf16(self):
        # RealModel weights are bf16: .numpy() raises on bf16, so
        # the helper must read raw bytes another way.
        from wsp.verify import _tensor_bytes

        w = torch.ones(2, 3, dtype=torch.bfloat16)
        raw = _tensor_bytes(w)
        self.assertEqual(len(raw), 12)
        self.assertEqual(raw, w.view(torch.uint8).numpy().tobytes())


class _FakeRealModelFactory:
    """Stand-in for RealModel in cmd_apply_edit: a fresh fake per call."""

    def __new__(cls, *args, **kwargs):
        return FakeRealModel("pass")


def _apply_edit_args(edit_path, out_dir, **over):
    kwargs = {"model": "fake-model", "edit": edit_path,
              "out_dir": out_dir, "force": False,
              "allow_quick": False, "allow_unverified": False,
              "verify": False, "expected_sha256": None}
    kwargs.update(over)
    return argparse.Namespace(**kwargs)


class TestApplyEditVerify(unittest.TestCase):
    def _setup(self, td, verdict="SHIP"):
        edit_path = os.path.join(td, "edit.json")
        rec = _dampen_recipe()
        recipe.write_recipe(edit_path, rec)
        rep = {"gate": {"passed": True, "verdict": verdict},
               "real_demo": {"edit_sha256": recipe.edit_sha256(rec),
                             "model_id": "fake-model"}}
        with open(os.path.join(td, "report.json"), "w") as f:
            json.dump(rep, f)
        return edit_path

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_verify_match(self):
        from wsp.verify import changed_tensors_sha256

        with tempfile.TemporaryDirectory() as td:
            edit_path = self._setup(td)
            # The expected hash: a fresh fake with the recipe applied.
            probe = FakeRealModel("pass")
            from wsp.serve_model import apply_recipe

            apply_recipe(probe, _dampen_recipe())
            expected = changed_tensors_sha256(
                probe, [{"index": 0, "columns": [0, 1]}])
            out_dir = os.path.join(td, "edited")
            os.makedirs(out_dir)
            with mock.patch("wsp.real_runner.RealModel",
                            _FakeRealModelFactory):
                code, text = _run_cli(
                    cli.cmd_apply_edit,
                    _apply_edit_args(edit_path, out_dir, verify=True,
                                     expected_sha256=expected))
            self.assertEqual(code, 0, text)
            self.assertIn("MATCH", text)
            self.assertNotIn("DIVERGED", text)
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                applied = json.load(f)
            self.assertEqual(applied["verify"], "MATCH")
            self.assertEqual(applied["expected_sha256"], expected)
            self.assertEqual(applied["actual_sha256"], expected)

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_verify_diverged(self):
        from wsp.verify import changed_tensors_sha256

        with tempfile.TemporaryDirectory() as td:
            edit_path = self._setup(td)
            # Expected hash for a DIFFERENT factor (0.25, not 0.5).
            probe = FakeRealModel("pass")
            probe.dampen_columns(0, [0, 1], 0.25)
            expected = changed_tensors_sha256(
                probe, [{"index": 0, "columns": [0, 1]}])
            out_dir = os.path.join(td, "edited")
            os.makedirs(out_dir)
            with mock.patch("wsp.real_runner.RealModel",
                            _FakeRealModelFactory):
                code, text = _run_cli(
                    cli.cmd_apply_edit,
                    _apply_edit_args(edit_path, out_dir, verify=True,
                                     expected_sha256=expected))
            self.assertEqual(code, 1, text)
            self.assertIn("DIVERGED", text)
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                applied = json.load(f)
            self.assertEqual(applied["verify"], "DIVERGED")

    def test_verify_needs_expected_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            edit_path = self._setup(td)
            out_dir = os.path.join(td, "edited")
            # No model patch: the check fires before any load.
            code, text = _run_cli(
                cli.cmd_apply_edit,
                _apply_edit_args(edit_path, out_dir, verify=True))
            self.assertEqual(code, 2, text)
            self.assertIn("--expected-sha256", text)

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_verify_without_flag_does_not_check(self):
        with tempfile.TemporaryDirectory() as td:
            edit_path = self._setup(td)
            out_dir = os.path.join(td, "edited")
            os.makedirs(out_dir)
            with mock.patch("wsp.real_runner.RealModel",
                            _FakeRealModelFactory):
                code, text = _run_cli(
                    cli.cmd_apply_edit,
                    _apply_edit_args(edit_path, out_dir))
            self.assertEqual(code, 0, text)
            self.assertNotIn("MATCH", text)
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                applied = json.load(f)
            self.assertIsNone(applied["verify"])

    def test_legacy_report_without_edit_hash_needs_allow_unverified(self):
        # The historical Qwen dampen report (v0.2.0) carries no
        # edit_sha256, so provenance cannot tie the recipe to it:
        # --allow-unverified is the honest path, not a relaxed check.
        run_dir = os.path.join(REPO_ROOT, "runs", "qwen-narrow-dampen")
        edit_path = os.path.join(run_dir, "edit.json")
        with open(os.path.join(run_dir, "report.json")) as f:
            report = json.load(f)
        rec = recipe.load_recipe(edit_path)
        with self.assertRaises(ValueError):
            recipe.verify_provenance(rec, report)

    def test_apply_edit_help_lists_verify(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                cli.main(["apply-edit", "--help"])
        text = out.getvalue()
        self.assertIn("--verify", text)
        self.assertIn("--expected-sha256", text)


# ---------------------------------------------------------------- W4.4 reproduce
class TestReproduce(unittest.TestCase):
    def _build_run(self, td, scenario="pass", seed=0, protocol="v2"):
        from tests.test_phase2_pipeline import build_v2_eval, write_eval
        from wsp.real_pipeline import run_real_pipeline

        eval_path = write_eval(td)
        run_dir = os.path.join(td, "run")
        rep, out_dir, code = run_real_pipeline(
            model_id="fake-model", eval_path=eval_path,
            out_dir=run_dir, model=FakeRealModel(scenario),
            protocol=protocol, top_columns=4, seed=seed,
            log=lambda *a: None)
        # The eval travels with the run, like kit/evals/ does.
        shutil.copy(eval_path, os.path.join(run_dir, "v2eval.json"))
        return run_dir, rep, code

    def test_reproduce_match(self):
        from wsp.reproduce import reproduce_run

        with tempfile.TemporaryDirectory() as td:
            run_dir, _rep, code = self._build_run(td)
            self.assertEqual(code, 0)
            lines = []
            result = reproduce_run(run_dir, model=FakeRealModel("pass"),
                                   log=lines.append)
            self.assertEqual(result["status"], "MATCH")
            self.assertEqual(result["code"], 0)
            self.assertIn("MATCH", "\n".join(lines))
            self.assertAlmostEqual(result["difference"], 0.0)

    def test_reproduce_diverged(self):
        from wsp.reproduce import reproduce_run

        with tempfile.TemporaryDirectory() as td:
            run_dir, _rep, _code = self._build_run(td)
            with open(os.path.join(run_dir, "report.json")) as f:
                rep = json.load(f)
            rep["after"]["failing"] = rep["after"]["failing"] + 20.0
            with open(os.path.join(run_dir, "report.json"), "w") as f:
                json.dump(rep, f)
            lines = []
            result = reproduce_run(run_dir, model=FakeRealModel("pass"),
                                   log=lines.append)
            self.assertEqual(result["status"], "DIVERGED")
            self.assertEqual(result["code"], 1)
            text = "\n".join(lines)
            self.assertIn("DIVERGED", text)
            self.assertNotIn("reproduced", text.lower())

    def test_reproduce_missing_fields(self):
        from wsp.reproduce import reproduce_run

        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run")
            os.makedirs(run_dir)
            # An early report: samples, seed and max tokens are None.
            rep = {"gate": {"passed": False},
                   "after": {"failing": 30.0},
                   "real_demo": {"model_id": "fake-model",
                                 "eval_file": "v2eval.json",
                                 "samples_per_prompt": None,
                                 "seed": None,
                                 "max_new_tokens": None}}
            with open(os.path.join(run_dir, "report.json"), "w") as f:
                json.dump(rep, f)
            lines = []
            with mock.patch("wsp.real_pipeline.run_real_pipeline") as rec:
                result = reproduce_run(run_dir, log=lines.append)
            rec.assert_not_called()
            self.assertEqual(result["status"], "CANNOT_REPRODUCE")
            self.assertEqual(result["code"], 2)
            self.assertIn(
                "cannot be reproduced exactly: missing "
                "samples_per_prompt, seed, max_new_tokens",
                "\n".join(lines))

    def test_reproduce_missing_eval_file(self):
        from wsp.reproduce import reproduce_run

        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "run")
            os.makedirs(run_dir)
            rep = {"gate": {"passed": True},
                   "after": {"failing": 30.0},
                   "real_demo": {"model_id": "fake-model",
                                 "eval_file": "no-such-eval.json",
                                 "samples_per_prompt": 3,
                                 "seed": 0,
                                 "max_new_tokens": 64}}
            with open(os.path.join(run_dir, "report.json"), "w") as f:
                json.dump(rep, f)
            lines = []
            result = reproduce_run(run_dir, log=lines.append)
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["code"], 2)
            self.assertIn("eval file", "\n".join(lines))

    def test_reproduce_tolerance_boundary(self):
        from wsp.reproduce import reproduce_run

        with tempfile.TemporaryDirectory() as td:
            run_dir, _rep, _code = self._build_run(td)
            with open(os.path.join(run_dir, "report.json")) as f:
                rep = json.load(f)
            rep["after"]["failing"] = rep["after"]["failing"] + 5.0
            with open(os.path.join(run_dir, "report.json"), "w") as f:
                json.dump(rep, f)
            # Exactly at the default tolerance: MATCH.
            result = reproduce_run(run_dir, model=FakeRealModel("pass"),
                                   log=lambda *a: None)
            self.assertEqual(result["status"], "MATCH")
            # Just past it: DIVERGED.
            rep["after"]["failing"] = rep["after"]["failing"] + 0.01
            with open(os.path.join(run_dir, "report.json"), "w") as f:
                json.dump(rep, f)
            result = reproduce_run(run_dir, model=FakeRealModel("pass"),
                                   log=lambda *a: None)
            self.assertEqual(result["status"], "DIVERGED")

    def test_reproduce_help_lists_tolerance(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                cli.main(["reproduce", "--help"])
        self.assertIn("--tolerance", out.getvalue())

    def test_cmd_reproduce_passes_tolerance(self):
        with mock.patch("wsp.reproduce.reproduce_run") as rec:
            rec.return_value = {"status": "MATCH", "code": 0}
            code = cli.cmd_reproduce(argparse.Namespace(
                run_dir="/tmp/run", tolerance=2.5, out_dir=None))
            self.assertEqual(code, 0)
            self.assertEqual(rec.call_args[1]["tolerance"], 2.5)


# ---------------------------------------------------------------- W4.3 evidence
class TestEvidencePack(unittest.TestCase):
    def test_narrow_animals_eval_byte_identical(self):
        path = os.path.join(KIT_DIR, "evals",
                            "loop-fix-qwen3-0.6b-narrow-animals.json")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_NARROW_EVAL)

    def test_smollm2_eval_byte_identical(self):
        path = os.path.join(KIT_DIR, "evals", "loop-fix-smollm2-1.7b.json")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_SMOLLM2_EVAL)

    def test_qwen_dampen_report_byte_identical(self):
        path = os.path.join(REPO_ROOT, "runs", "qwen-narrow-dampen",
                            "report.json")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_DAMPEN_REPORT)

    def test_qwen_dampen_edit_byte_identical(self):
        path = os.path.join(REPO_ROOT, "runs", "qwen-narrow-dampen",
                            "edit.json")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_DAMPEN_EDIT)

    def test_provenance_md_records_hashes(self):
        path = os.path.join(REPO_ROOT, "runs", "qwen-narrow-dampen",
                            "PROVENANCE.md")
        self.assertTrue(os.path.isfile(path), path)
        with open(path) as f:
            text = f.read()
        for want in (SHA_DAMPEN_REPORT, SHA_DAMPEN_EDIT,
                     SHA_NARROW_EVAL, SHA_DAMPEN_MODEL_FILE,
                     SHA_DAMPEN_TENSORS):
            self.assertIn(want, text)

    def test_editor_fix_report_added(self):
        path = os.path.join(REPO_ROOT, "experiments",
                            "EDITOR-FIX-REPORT.md")
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(_sha256_file(path), SHA_EDITOR_FIX_REPORT)

    def test_smollm2_run_report_absent_stop_rule(self):
        # STOP RULE (W4.0): no wsp-real-out-smollm2 report.json or
        # edit.json exists in either founder-approved search
        # location, so no runs/smollm2 report was invented.
        self.assertFalse(os.path.exists(os.path.join(
            REPO_ROOT, "runs", "smollm2", "report.json")))
        self.assertFalse(os.path.exists(os.path.join(
            REPO_ROOT, "runs", "smollm2", "edit.json")))


# ---------------------------------------------------------------- W4.5 docs
class TestRealRunsDoc(unittest.TestCase):
    def _text(self):
        with open(os.path.join(REPO_ROOT, "REAL-RUNS.md")) as f:
            return f.read()

    def test_evidence_level_column_and_quick_labels(self):
        text = self._text()
        self.assertIn("| Evidence |", text)
        self.assertIn("quick (3 failing prompts)", text)
        # Both passes carry the label, in the table and the sections.
        self.assertGreaterEqual(
            text.count("quick (3 failing prompts)"), 2)

    def test_reproduce_section_lists_verify_and_v1(self):
        text = self._text()
        section = text.split("## Reproduce", 1)[1]
        self.assertIn("wsp apply-edit", section)
        self.assertIn("--verify", section)
        self.assertIn("--protocol v1", section)
        self.assertIn("runs/qwen-narrow-dampen/edit.json", section)

    def test_smollm2_reproduce_listed_as_unavailable(self):
        text = self._text()
        section = text.split("## Reproduce", 1)[1]
        self.assertIn("SmolLM2", section)
        self.assertIn("NOT reproducible", section)
        self.assertIn("report.json", section)

    def test_failure_rows_untouched(self):
        text = self._text()
        for row in ("| Qwen run 1 |", "| Qwen run 2 |",
                    "| Qwen clean re-run |", "| GPT-2 run |",
                    "| Qwen narrow Try 1 |"):
            self.assertIn(row, text)
        self.assertIn("**FAILED — nothing shipped**", text)

    def test_qwen_dampen_section_points_at_runs_dir(self):
        text = self._text()
        self.assertIn("runs/qwen-narrow-dampen/report.json", text)


class TestReadmeFlags(unittest.TestCase):
    def test_readme_flags_table_has_verify_and_reproduce(self):
        with open(os.path.join(REPO_ROOT, "README.md")) as f:
            text = f.read()
        self.assertIn("--verify", text)
        self.assertIn("--expected-sha256", text)
        self.assertIn("reproduce", text)

    def test_kit_readme_lists_reproduce_command(self):
        with open(os.path.join(KIT_DIR, "README.md")) as f:
            text = f.read()
        self.assertIn("wsp reproduce", text)


if __name__ == "__main__":
    unittest.main()
