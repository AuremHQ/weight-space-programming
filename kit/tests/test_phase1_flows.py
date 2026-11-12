"""Phase 1, Batch C: flows — loop record_version (via Batch B),
merge honesty, apply-edit CLI, CLI flag plumbing, atomic wiring.

W1.8 merge writes to <out_dir>.partial-<pid> and moves into place
only on a gate pass; W1.11 cmd_apply_edit rewrite; W1.6 --force on
run / apply-edit / merge; W1.4 recipe cap before attempts in the
pipeline; W1.10 atomic report + .watch-seen writes. Plain unittest;
torch tests skipUnless. No network (HF_HUB_OFFLINE=1).

Run from the kit directory:
  PYTHONPATH=. python3 -m unittest tests.test_phase1_flows -v
"""

import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests.fakes import FakeRealModel
from wsp import atomic, cli, merge, recipe, watch

try:
    import torch
except ImportError:
    torch = None


def _tiny_model_dir(td):
    """A tiny GPT2 model + tokenizer, saved to a local directory."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)

    vocab = {"a": 0, "b": 1, "c": 2, "<|endoftext|>": 3}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|endoftext|>",
        pad_token="<|endoftext|>")
    fast.save_pretrained(td)
    config = GPT2Config(vocab_size=4, n_positions=64, n_embd=32,
                        n_layer=2, n_head=2, bos_token_id=3,
                        eos_token_id=3)
    GPT2LMHeadModel(config).save_pretrained(td)
    return config


def _run_cli(fn, namespace):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(namespace)
    return code, out.getvalue()


# ---------------------------------------------------------------- W1.6 + W1.4 + W1.10 (pipeline)

class TestPipelineFlows(unittest.TestCase):
    def _eval_path(self, td):
        import shutil as _shutil

        src = os.path.join(os.path.dirname(__file__),
                           "golden", "v1", "eval.json")
        dst = os.path.join(td, "eval.json")
        _shutil.copy(src, dst)
        return dst

    def test_out_dir_holding_model_files_refused(self):
        from wsp import real_pipeline

        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump({"model_type": "gpt2"}, f)
            fake = FakeRealModel("pass")
            with self.assertRaises(ValueError):
                real_pipeline.run_real_pipeline(
                    "fake-model", self._eval_path(td), out_dir,
                    model=fake, log=lambda *a: None, protocol="v1")
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "report.json")))

    def test_out_dir_holding_model_files_allowed_with_force(self):
        from wsp import real_pipeline

        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir)
            with open(os.path.join(out_dir, "config.json"), "w") as f:
                json.dump({"model_type": "gpt2"}, f)
            fake = FakeRealModel("pass")
            rep, _out, code = real_pipeline.run_real_pipeline(
                "fake-model", self._eval_path(td), out_dir,
                model=fake, force=True, log=lambda *a: None, protocol="v1")
            self.assertEqual(code, 0)
            self.assertTrue(rep["gate"]["passed"])
            # Phase 2 stamps the verdict: a passing v1 report is
            # QUICK evidence (SHIP is a v2 concept).
            self.assertEqual(rep["gate"]["verdict"], "QUICK_PASS")

    def test_recipe_cap_checked_before_attempts(self):
        from wsp import real_pipeline

        class TinyParamsFake(FakeRealModel):
            def __init__(self, scenario):
                super().__init__(scenario)
                self.total_params = 100

        with tempfile.TemporaryDirectory() as td:
            fake = TinyParamsFake("pass")
            with self.assertRaises(ValueError) as ctx:
                real_pipeline.run_real_pipeline(
                    "fake-model", self._eval_path(td),
                    os.path.join(td, "out"), model=fake,
                    log=lambda *a: None, protocol="v1")
            self.assertIn("1% cap", str(ctx.exception))
            self.assertEqual(fake._edited, {})

    def test_recipe_cap_fails_before_any_generation(self):
        from wsp import real_pipeline

        class TinyParamsFake(FakeRealModel):
            def __init__(self, scenario):
                super().__init__(scenario)
                self.total_params = 100

        with tempfile.TemporaryDirectory() as td:
            fake = TinyParamsFake("pass")
            calls = []
            orig_generate = fake.generate

            def counting_generate(*args, **kwargs):
                calls.append(1)
                return orig_generate(*args, **kwargs)

            fake.generate = counting_generate
            with self.assertRaises(ValueError):
                real_pipeline.run_real_pipeline(
                    "fake-model", self._eval_path(td),
                    os.path.join(td, "out"), model=fake,
                    log=lambda *a: None, protocol="v1")
            self.assertEqual(calls, [])
            self.assertEqual(fake._edited, {})

    def test_no_temp_files_left_after_run(self):
        from wsp import real_pipeline

        with tempfile.TemporaryDirectory() as td:
            out_dir = os.path.join(td, "out")
            fake = FakeRealModel("pass")
            rep, _out, code = real_pipeline.run_real_pipeline(
                "fake-model", self._eval_path(td), out_dir,
                model=fake, log=lambda *a: None, protocol="v1")
            self.assertEqual(code, 0)
            leftovers = glob.glob(os.path.join(out_dir, ".*.tmp-*"))
            self.assertEqual(leftovers, [])
            with open(os.path.join(out_dir, "report.json")) as f:
                written = json.load(f)
            with open(os.path.join(out_dir, "edit.json")) as f:
                edit = json.load(f)
            self.assertEqual(written["real_demo"]["edit_sha256"],
                             recipe.edit_sha256(edit))

    def test_save_seen_is_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            watch.save_seen(td, {"eval.json": {"size": 1}})
            leftovers = glob.glob(os.path.join(td, ".*.tmp-*"))
            self.assertEqual(leftovers, [])
            with open(os.path.join(td, ".watch-seen.json")) as f:
                self.assertEqual(json.load(f),
                                 {"eval.json": {"size": 1}})


# ---------------------------------------------------------------- W1.11

@unittest.skipUnless(torch is not None, "torch not installed")
class TestApplyEditCli(unittest.TestCase):
    def _setup(self, td, verdict="SHIP", report=True, layers=None):
        model_dir = os.path.join(td, "model")
        os.makedirs(model_dir)
        _tiny_model_dir(model_dir)
        edit_layers = layers if layers is not None else [
            {"index": 0, "columns": [3],
             "direction": [0.1] * 32, "alpha": 0.25},
        ]
        rec = {"format_version": 1, "model_id": model_dir,
               "dtype": "bfloat16", "layers": edit_layers}
        edit_path = os.path.join(td, "edit.json")
        recipe.write_recipe(edit_path, rec)
        if report:
            rep = {"gate": {"passed": True},
                   "real_demo": {"edit_sha256":
                                 recipe.edit_sha256(rec),
                                 "model_id": model_dir}}
            if verdict is not None:
                rep["gate"]["verdict"] = verdict
            atomic.atomic_write_json(os.path.join(td, "report.json"),
                                     rep)
        return model_dir, edit_path

    def _args(self, model_dir, edit_path, out_dir, **over):
        kwargs = {"model": model_dir, "edit": edit_path,
                  "out_dir": out_dir, "force": False,
                  "allow_quick": False, "allow_unverified": False}
        kwargs.update(over)
        return argparse.Namespace(**kwargs)

    def test_apply_edit_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td)
            out_dir = os.path.join(td, "edited")
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, out_dir))
            self.assertEqual(code, 0, text)
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "config.json")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "model.safetensors")))
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                applied = json.load(f)
            self.assertEqual(applied["evidence"], "ship")
            self.assertEqual(applied["weights_changed"], 32)
            self.assertIn("model_revision", applied)
            self.assertIn("revision", text)

    def test_bad_edit_exits_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = os.path.join(td, "model")
            os.makedirs(model_dir)
            _tiny_model_dir(model_dir)
            edit_path = os.path.join(td, "edit.json")
            with open(edit_path, "w") as f:
                f.write("{not json")
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path,
                           os.path.join(td, "edited")))
            self.assertEqual(code, 2)
            self.assertIn("error:", text)
            self.assertNotIn("Traceback", text)

    def test_missing_report_needs_allow_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td, report=False)
            out_dir = os.path.join(td, "edited")
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, out_dir))
            self.assertEqual(code, 2)
            self.assertIn("--allow-unverified", text)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, out_dir,
                           allow_unverified=True))
            self.assertEqual(code, 0, text)
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                self.assertEqual(json.load(f)["evidence"], "unverified")

    def test_quick_report_needs_allow_quick(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td, verdict=None)
            out_dir = os.path.join(td, "edited")
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, out_dir))
            self.assertEqual(code, 2)
            self.assertIn("--allow-quick", text)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, out_dir,
                           allow_quick=True))
            self.assertEqual(code, 0, text)
            with open(os.path.join(out_dir, "apply-edit.json")) as f:
                self.assertEqual(json.load(f)["evidence"], "quick")

    def test_wrong_hash_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td)
            rep = {"gate": {"passed": True, "verdict": "SHIP"},
                   "real_demo": {"edit_sha256": "0" * 64,
                                 "model_id": model_dir}}
            atomic.atomic_write_json(os.path.join(td, "report.json"),
                                     rep)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path,
                           os.path.join(td, "edited"),
                           allow_quick=True))
            self.assertEqual(code, 2)
            self.assertNotIn("Traceback", text)

    def test_out_dir_is_model_dir_refused(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path, model_dir))
            self.assertEqual(code, 2)
            self.assertIn("error:", text)
            self.assertNotIn("Traceback", text)

    def test_over_cap_recipe_refused(self):
        with tempfile.TemporaryDirectory() as td:
            layers = [{"index": 0, "columns": list(range(16)),
                       "direction": [0.1] * 32, "alpha": 0.25}]
            model_dir, edit_path = self._setup(td, layers=layers)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path,
                           os.path.join(td, "edited")))
            self.assertEqual(code, 2)
            self.assertIn("1% cap", text)

    def test_recipe_for_another_model_id_warns_but_applies(self):
        # The common case is a recipe written for an HF id applied
        # to a local copy of the same model, so this is a warning,
        # not a refusal — but it must never slip by silently.
        with tempfile.TemporaryDirectory() as td:
            model_dir, edit_path = self._setup(td)
            with open(edit_path) as f:
                rec = json.load(f)
            rec["model_id"] = "Qwen/Qwen3-0.6B"
            recipe.write_recipe(edit_path, rec)
            rep = {"gate": {"passed": True, "verdict": "SHIP"},
                   "real_demo": {"edit_sha256":
                                 recipe.edit_sha256(rec),
                                 "model_id": "Qwen/Qwen3-0.6B"}}
            atomic.atomic_write_json(os.path.join(td, "report.json"),
                                     rep)
            code, text = _run_cli(
                cli.cmd_apply_edit,
                self._args(model_dir, edit_path,
                           os.path.join(td, "edited")))
            self.assertEqual(code, 0, text)
            self.assertIn("warning: this edit is for Qwen/Qwen3-0.6B",
                          text)


# ---------------------------------------------------------------- W1.8

@unittest.skipUnless(torch is not None, "torch not installed")
class TestMergePartial(unittest.TestCase):
    def _pair(self, td):
        import torch as _torch
        from safetensors.torch import load_file, save_file

        dir_a = os.path.join(td, "model-a")
        dir_b = os.path.join(td, "model-b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        _tiny_model_dir(dir_a)
        # Model B: the same config + tokenizer, perturbed weights.
        state = load_file(os.path.join(dir_a, "model.safetensors"))
        state = {k: v.float() + 0.01 for k, v in state.items()}
        save_file(state, os.path.join(dir_b, "model.safetensors"),
                  metadata={"format": "pt"})
        # Byte-identical config + tokenizer files: the same base.
        for name in ("config.json", "generation_config.json",
                     "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(dir_a, name),
                        os.path.join(dir_b, name))
        return dir_a, dir_b

    def _eval_path(self):
        return os.path.join(os.path.dirname(__file__),
                            "merge-test-eval.json")

    def _run_merge(self, dir_a, dir_b, out_dir, merged_failing):
        created = []

        class FakeRM:
            def __init__(self, model_id, dtype="bfloat16"):
                self.model_id = model_id
                created.append(self)

        def fake_evaluate(model, data, temp, n, seed, log):
            if model is created[0]:
                return {"failing": 50.0, "good": 100.0,
                        "neighbours": 100.0}
            return {"failing": merged_failing, "good": 100.0,
                    "neighbours": 100.0}

        with mock.patch("wsp.real_runner.RealModel", FakeRM), \
                mock.patch("wsp.real_pipeline.evaluate", fake_evaluate):
            return merge.run_merge(
                dir_a, dir_b, "linear", 0.5, self._eval_path(), out_dir,
                log=lambda *a: None, protocol="v1")

    def test_gate_pass_moves_model_in(self):
        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            out_dir = os.path.join(td, "merged")
            rep, _out, code = self._run_merge(dir_a, dir_b, out_dir,
                                              merged_failing=80.0)
            self.assertEqual(code, 0)
            self.assertTrue(rep["gate"]["passed"])
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "model.safetensors")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "config.json")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "merge.json")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "report.json")))
            self.assertEqual(
                glob.glob(out_dir + ".partial-*"), [])

    def test_gate_fail_ships_no_model(self):
        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            out_dir = os.path.join(td, "merged")
            rep, _out, code = self._run_merge(dir_a, dir_b, out_dir,
                                              merged_failing=50.0)
            self.assertEqual(code, 1)
            self.assertFalse(rep["gate"]["passed"])
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "merge.json")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "report.json")))
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "model.safetensors")))
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "config.json")))
            self.assertEqual(
                glob.glob(out_dir + ".partial-*"), [])

    def test_out_dir_is_model_dir_refused(self):
        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            with self.assertRaises(ValueError):
                self._run_merge(dir_a, dir_b, dir_a,
                                merged_failing=80.0)

    def test_out_dir_is_model_dir_leaves_inputs_unchanged(self):
        import hashlib

        def _hashes(d):
            out = {}
            for root, _dirs, files in os.walk(d):
                for name in files:
                    path = os.path.join(root, name)
                    with open(path, "rb") as f:
                        out[os.path.relpath(path, d)] = (
                            hashlib.sha256(f.read()).hexdigest())
            return out

        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            before_a = _hashes(dir_a)
            before_b = _hashes(dir_b)
            with self.assertRaises(ValueError):
                self._run_merge(dir_a, dir_b, dir_a,
                                merged_failing=80.0)
            self.assertEqual(_hashes(dir_a), before_a)
            self.assertEqual(_hashes(dir_b), before_b)

    def test_cmd_merge_prints_only_on_pass(self):
        rep_pass = {"gate": {"passed": True, "by_temperature": {}},
                    "before": {"failing": 50.0, "good": 100.0,
                               "neighbours": 100.0},
                    "after": {"failing": 80.0, "good": 100.0,
                              "neighbours": 100.0}}
        rep_fail = {"gate": {"passed": False, "by_temperature": {}},
                    "before": {"failing": 50.0, "good": 100.0,
                               "neighbours": 100.0},
                    "after": {"failing": 50.0, "good": 100.0,
                              "neighbours": 100.0}}
        args = argparse.Namespace(
            model_a="a", model_b="b", method="linear", t=0.5,
            eval="e", out_dir="o", dtype="bfloat16",
            samples_per_prompt=1, seed=0, force=False)
        with mock.patch("wsp.merge.run_merge",
                        return_value=(rep_pass, "o", 0)):
            code, text = _run_cli(cli.cmd_merge, args)
            self.assertEqual(code, 0)
            self.assertIn("merged model in o", text)
        with mock.patch("wsp.merge.run_merge",
                        return_value=(rep_fail, "o", 1)):
            code, text = _run_cli(cli.cmd_merge, args)
            self.assertEqual(code, 1)
            self.assertIn("merged model NOT saved (gate failed)", text)
            self.assertNotIn("merged model in o", text)


# ---------------------------------------------------------------- CLI plumbing

class TestCliPlumbing(unittest.TestCase):
    def test_cmd_run_passes_force(self):
        rep = {"gate": {"passed": True},
               "before": {"failing": 10.0, "good": 100.0,
                          "neighbours": 100.0},
               "after": {"failing": 80.0, "good": 100.0,
                         "neighbours": 100.0},
               "weights_changed": 10, "weights_changed_pct": 0.01}
        args = argparse.Namespace(
            model="m", eval="e", out_dir="o", dtype="bfloat16",
            samples_per_prompt=1, seed=0, top_columns=64,
            max_attempts=3, force=True)
        with mock.patch("wsp.real_pipeline.run_real_pipeline",
                        return_value=(rep, "o", 0)) as run:
            code, _text = _run_cli(cli.cmd_run, args)
            self.assertEqual(code, 0)
            self.assertTrue(run.call_args.kwargs["force"])

    def test_cmd_serve_model_passes_token_and_allow_quick(self):
        args = argparse.Namespace(
            model="m", versions_dir="v", port=8900, interval=30,
            device="cpu", dtype="bfloat16", token="t",
            allow_quick=True)
        with mock.patch("wsp.serve_model.serve_model",
                        return_value=0) as serve:
            code, _text = _run_cli(cli.cmd_serve_model, args)
            self.assertEqual(code, 0)
            self.assertEqual(serve.call_args.kwargs["token"], "t")
            self.assertTrue(serve.call_args.kwargs["allow_quick"])

    def test_cmd_serve_model_token_from_env(self):
        args = argparse.Namespace(
            model="m", versions_dir="v", port=8900, interval=30,
            device="cpu", dtype="bfloat16", token=None,
            allow_quick=False)
        with mock.patch("wsp.serve_model.serve_model",
                        return_value=0) as serve, \
                mock.patch.dict(os.environ, {"WSP_TOKEN": "envtok"}):
            code, _text = _run_cli(cli.cmd_serve_model, args)
            self.assertEqual(code, 0)
            self.assertEqual(serve.call_args.kwargs["token"], "envtok")


class TestHelpFlags(unittest.TestCase):
    def _help(self, argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(argv)
        self.assertEqual(ctx.exception.code, 0)
        return out.getvalue()

    def test_run_help_lists_force(self):
        self.assertIn("--force", self._help(["run", "--help"]))

    def test_apply_edit_help_lists_flags(self):
        text = self._help(["apply-edit", "--help"])
        self.assertIn("--force", text)
        self.assertIn("--allow-quick", text)
        self.assertIn("--allow-unverified", text)

    def test_merge_help_lists_force(self):
        self.assertIn("--force", self._help(["merge", "--help"]))

    def test_serve_model_help_lists_token_and_allow_quick(self):
        text = self._help(["serve-model", "--help"])
        self.assertIn("--token", text)
        self.assertIn("--allow-quick", text)


if __name__ == "__main__":
    unittest.main()
