"""Phase 2 (W2.13, W2.14): CLI plumbing, loop preflight, status
verdicts, merge v2, --help coverage.

Every command is tested with a recorder patch on the pipeline
function, asserting the kwargs it received.
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from wsp import cli, loop, status, watch


def _run_cli(fn, args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = fn(args)
    return code, out.getvalue()


def _help(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            cli.main(argv)
        except SystemExit as exc:
            assert exc.code == 0, exc.code
    return out.getvalue()


class TestHelpFlagsPhase2(unittest.TestCase):
    def test_run_help_lists_protocol_dry_run_guards(self):
        text = _help(["run", "--help"])
        for flag in ("--protocol", "--dry-run", "--no-default-guards",
                     "--tokens-per-second", "--samples-per-prompt"):
            self.assertIn(flag, text, flag)

    def test_watch_help_lists_protocol(self):
        self.assertIn("--protocol", _help(["watch", "--help"]))

    def test_loop_help_lists_protocol(self):
        self.assertIn("--protocol", _help(["loop", "--help"]))

    def test_merge_help_lists_protocol(self):
        self.assertIn("--protocol", _help(["merge", "--help"]))

    def test_auto_help_lists_protocol(self):
        self.assertIn("--protocol", _help(["auto", "--help"]))


class TestCliPlumbingPhase2(unittest.TestCase):
    def _rep(self):
        return {"gate": {"passed": True},
                "before": {"failing": 10.0, "good": 100.0,
                           "neighbours": 100.0},
                "after": {"failing": 80.0, "good": 100.0,
                          "neighbours": 100.0},
                "weights_changed": 10, "weights_changed_pct": 0.01}

    def test_cmd_run_passes_protocol_and_dry_run(self):
        args = argparse.Namespace(
            model="m", eval="e", out_dir="o", dtype="bfloat16",
            samples_per_prompt=None, seed=0, top_columns=64,
            max_attempts=3, force=False, protocol="v2",
            dry_run=False, no_default_guards=True,
            tokens_per_second=None)
        with mock.patch("wsp.real_pipeline.run_real_pipeline",
                        return_value=(self._rep(), "o", 0)) as run:
            code, _text = _run_cli(cli.cmd_run, args)
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["protocol"], "v2")
            self.assertFalse(run.call_args.kwargs["dry_run"])
            self.assertTrue(
                run.call_args.kwargs["no_default_guards"])
            self.assertIsNone(run.call_args.kwargs["samples_per_prompt"])

    def test_cmd_run_dry_run_returns_without_gate_print(self):
        args = argparse.Namespace(
            model="m", eval="e", out_dir="o", dtype="bfloat16",
            samples_per_prompt=None, seed=0, top_columns=64,
            max_attempts=3, force=False, protocol="v2",
            dry_run=True, no_default_guards=False,
            tokens_per_second=None)
        with mock.patch("wsp.real_pipeline.run_real_pipeline",
                        return_value=(None, "o", 0)) as run:
            code, _text = _run_cli(cli.cmd_run, args)
            self.assertEqual(code, 0)
            self.assertTrue(run.call_args.kwargs["dry_run"])

    def test_cmd_watch_passes_protocol(self):
        args = argparse.Namespace(
            model="m", watch_dir="w", out_dir="o", interval=60,
            once=True, dtype="bfloat16", samples_per_prompt=None,
            seed=0, top_columns=64, max_attempts=3, protocol="v2")
        # cmd_watch preflights real_runner (torch); stand in a
        # fake module so the recorder runs in the numpy-only env.
        import sys

        with mock.patch("wsp.watch.watch_loop",
                        return_value=0) as wl, \
                mock.patch.dict(
                    sys.modules,
                    {"wsp.real_runner": mock.MagicMock()}):
            code, _text = _run_cli(cli.cmd_watch, args)
            self.assertEqual(code, 0)
            self.assertEqual(wl.call_args.kwargs["protocol"], "v2")
            self.assertIsNone(
                wl.call_args.kwargs["samples_per_prompt"])

    def test_cmd_loop_passes_protocol_in_pipeline_kwargs(self):
        args = argparse.Namespace(
            model="m", watch_dir="w", out_dir="o", interval=60,
            once=True, dtype="bfloat16", samples_per_prompt=None,
            seed=0, top_columns=64, max_attempts=3, protocol="v2")
        import sys

        with mock.patch("wsp.loop.run_loop",
                        return_value=0) as rl, \
                mock.patch.dict(
                    sys.modules,
                    {"wsp.real_runner": mock.MagicMock()}):
            code, _text = _run_cli(cli.cmd_loop, args)
            self.assertEqual(code, 0)
            kwargs = rl.call_args.kwargs["pipeline_kwargs"]
            self.assertEqual(kwargs["protocol"], "v2")
            self.assertIsNone(kwargs["samples_per_prompt"])

    def test_cmd_auto_passes_protocol_v1_by_default(self):
        args = argparse.Namespace(
            model="m", eval="e", out_dir="o", serve=False,
            protocol="v1")
        with mock.patch("wsp.auto.run_auto",
                        return_value=0) as ra:
            code, _text = _run_cli(cli.cmd_auto, args)
            self.assertEqual(code, 0)
            self.assertEqual(ra.call_args.kwargs["protocol"], "v1")

    def test_cmd_merge_passes_protocol(self):
        rep = {"gate": {"passed": True, "by_temperature": {}},
               "before": {"failing": 50.0, "good": 100.0,
                          "neighbours": 100.0},
               "after": {"failing": 80.0, "good": 100.0,
                         "neighbours": 100.0}}
        args = argparse.Namespace(
            model_a="a", model_b="b", method="linear", t=0.5,
            eval="e", out_dir="o", dtype="bfloat16",
            samples_per_prompt=None, seed=0, force=False,
            protocol="v2")
        with mock.patch("wsp.merge.run_merge",
                        return_value=(rep, "o", 0)) as rm:
            code, _text = _run_cli(cli.cmd_merge, args)
            self.assertEqual(code, 0)
            self.assertEqual(rm.call_args.kwargs["protocol"], "v2")

    def test_cli_defaults(self):
        # run/watch/loop/merge default v2; auto defaults v1 (D1).
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                cli.main(["run", "--help"])
        text = out.getvalue()
        self.assertIn("--protocol {v1,v2}", text)


class TestLoopPreflight(unittest.TestCase):
    def _write_failures(self, path, failing, good, neighbours=0):
        with open(path, "w") as f:
            for i in range(failing):
                f.write(json.dumps(
                    {"prompt": f"failing {i}", "expect": "x"}) + "\n")
            for i in range(good):
                f.write(json.dumps(
                    {"prompt": f"good {i}", "expect": "x",
                     "kind": "good"}) + "\n")
            for i in range(neighbours):
                f.write(json.dumps(
                    {"prompt": f"neighbour {i}", "expect": "x",
                     "kind": "neighbour"}) + "\n")

    def test_short_failures_skipped_without_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            self._write_failures(
                os.path.join(watch_dir, "failures.jsonl"),
                failing=3, good=2, neighbours=1)
            calls = []

            def run_fn(model_id, eval_path, out_dir, log=print,
                       **kw):
                calls.append(eval_path)
                return None, out_dir, 2

            logs = []
            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, {}, run_fn=run_fn,
                log=logs.append,
                pipeline_kwargs={"protocol": "v2"})
            self.assertEqual(calls, [])
            self.assertEqual(summary["failure_runs"], 0)
            self.assertTrue(
                any("need 17 more failing prompts" in ln
                    for ln in logs), logs)

    def test_enough_prompts_runs(self):
        with tempfile.TemporaryDirectory() as td:
            watch_dir = os.path.join(td, "watch")
            out_dir = os.path.join(td, "out")
            os.makedirs(watch_dir)
            self._write_failures(
                os.path.join(watch_dir, "failures.jsonl"),
                failing=20, good=12, neighbours=16)
            calls = []

            def run_fn(model_id, eval_path, out_dir, log=print,
                       **kw):
                calls.append(eval_path)
                rep = {"gate": {"passed": True},
                       "before": {"failing": 50.0},
                       "after": {"failing": 70.0},
                       "weights_changed": 128}
                return rep, os.path.join(out_dir, "run-1"), 0

            state, summary = loop.loop_once(
                "m", watch_dir, out_dir, {}, run_fn=run_fn,
                log=lambda *a: None,
                pipeline_kwargs={"protocol": "v2"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary["failure_runs"], 1)

    def test_default_run_fn_forwards_pipeline_kwargs(self):
        # Regression: _default_run_fn used to reject kwargs, so
        # `wsp loop` crashed as soon as pipeline_kwargs were
        # passed (cmd_loop always passes them).
        with mock.patch("wsp.watch.run_once",
                        return_value=(None, "d", 2)) as ro:
            loop._default_run_fn("m", "e.json", "o",
                                 log=lambda *a: None,
                                 protocol="v2", dtype="bfloat16")
            self.assertEqual(ro.call_args.kwargs["protocol"], "v2")
            self.assertEqual(ro.call_args.kwargs["dtype"], "bfloat16")


class TestStatusVerdicts(unittest.TestCase):
    def _report(self, path, passed, verdict=None, before=30.0,
                after=36.5):
        rep = {
            "report_version": 1,
            "kit_version": "0.2.0",
            "model": "Qwen/Qwen3-0.6B (real open-weight model, CPU)",
            "weights_changed": 1000,
            "weights_changed_pct": 0.02,
            "before": {"failing": before, "good": 100.0,
                       "neighbours": 100.0},
            "after": {"failing": after, "good": 100.0,
                      "neighbours": 100.0},
            "gate": {"passed": passed},
            "temperatures_checked": [0.0, 0.7],
            "editor": "test",
            "note": "test fixture",
        }
        if verdict is not None:
            rep["gate"]["verdict"] = verdict
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(rep, f)

    def test_status_line_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            self._report(os.path.join(td, "ship", "report.json"),
                         True, verdict="SHIP", before=30.0,
                         after=40.0)
            self._report(os.path.join(td, "quick", "report.json"),
                         True, verdict="QUICK_PASS", before=30.0,
                         after=36.5)
            self._report(os.path.join(td, "legacy", "report.json"),
                         True, verdict=None, before=30.0,
                         after=35.0)
            self._report(os.path.join(td, "bad", "report.json"),
                         False, verdict="REJECT", before=30.0,
                         after=31.0)
            line = status.status_line(td)
            self.assertIn("1 Approved (standard evidence)", line)
            self.assertIn("2 Quick check passed (not evidence)", line)
            self.assertIn("1 refused of 4 runs found here", line)

    def test_scan_reports_counts(self):
        with tempfile.TemporaryDirectory() as td:
            self._report(os.path.join(td, "a", "report.json"),
                         True, verdict="SHIP")
            self._report(os.path.join(td, "b", "report.json"),
                         False, verdict="REJECT")
            scan = status.scan_reports(td)
            self.assertEqual(scan["approved"], 1)
            self.assertEqual(scan["quick_passes"], 0)
            self.assertEqual(scan["refused"], 1)
            self.assertEqual(scan["total"], 2)

    def test_page_header_has_denominator_and_log_link(self):
        with tempfile.TemporaryDirectory() as td:
            self._report(os.path.join(td, "a", "report.json"),
                         True, verdict="SHIP")
            self._report(os.path.join(td, "b", "report.json"),
                         False, verdict="REJECT")
            page = status.render_html(status.scan_reports(td))
            self.assertIn("1 Approved (standard evidence)", page)
            self.assertIn("1 refused of 2 runs found here", page)
            self.assertIn("REAL-RUNS.md", page)


@unittest.skipUnless(
    __import__("importlib").util.find_spec("torch") is not None,
    "torch not installed")
class TestMergeV2(unittest.TestCase):
    def _pair(self, td):
        import torch as _torch
        from safetensors.torch import load_file, save_file

        from tests.test_phase1_flows import _tiny_model_dir

        dir_a = os.path.join(td, "model-a")
        dir_b = os.path.join(td, "model-b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        _tiny_model_dir(dir_a)
        state = load_file(os.path.join(dir_a, "model.safetensors"))
        state = {k: v.float() + 0.01 for k, v in state.items()}
        save_file(state, os.path.join(dir_b, "model.safetensors"),
                  metadata={"format": "pt"})
        import shutil

        for name in ("config.json", "generation_config.json",
                     "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(dir_a, name),
                        os.path.join(dir_b, name))
        return dir_a, dir_b

    def _run_merge_v2(self, dir_a, dir_b, out_dir, merged_failing):
        from wsp import merge

        created = []

        class FakeRM:
            def __init__(self, model_id, dtype="bfloat16"):
                self.model_id = model_id
                created.append(self)

        def fake_detailed(model, data, temp, n, seed, log,
                          split=None, scorer_version="v1"):
            from wsp.evalfile import SET_NAMES, set_prompts

            failing = (50.0 if model is created[0]
                       else merged_failing)
            out = {}
            for name in SET_NAMES:
                count = len(set_prompts(data, name))
                value = (failing if name == "failing" else 100.0)
                out[name] = {"per_prompt": [value] * count,
                             "mean": value}
            return out

        with mock.patch("wsp.real_runner.RealModel", FakeRM), \
                mock.patch("wsp.real_pipeline.evaluate_detailed",
                           fake_detailed):
            return merge.run_merge(
                dir_a, dir_b, "linear", 0.5,
                os.path.join(os.path.dirname(__file__),
                             "merge-test-eval.json"),
                out_dir, log=lambda *a: None, protocol="v2")

    def test_merge_v2_small_eval_is_inconclusive(self):
        # The merge test eval has 2 prompts per set: below the v2
        # minimums, so a v2 merge cannot SHIP — INCONCLUSIVE, and
        # no model ships.
        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            out_dir = os.path.join(td, "merged")
            rep, _out, code = self._run_merge_v2(
                dir_a, dir_b, out_dir, merged_failing=80.0)
            self.assertEqual(code, 1)
            self.assertEqual(rep["gate"]["verdict"], "INCONCLUSIVE")
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "model.safetensors")))

    def test_merge_v1_still_passes_on_small_eval(self):
        from wsp import merge

        with tempfile.TemporaryDirectory() as td:
            dir_a, dir_b = self._pair(td)
            out_dir = os.path.join(td, "merged")
            created = []

            class FakeRM:
                def __init__(self, model_id, dtype="bfloat16"):
                    self.model_id = model_id
                    created.append(self)

            def fake_evaluate(model, data, temp, n, seed, log):
                if model is created[0]:
                    return {"failing": 50.0, "good": 100.0,
                            "neighbours": 100.0}
                return {"failing": 80.0, "good": 100.0,
                        "neighbours": 100.0}

            with mock.patch("wsp.real_runner.RealModel", FakeRM), \
                    mock.patch("wsp.real_pipeline.evaluate",
                               fake_evaluate):
                rep, _out, code = merge.run_merge(
                    dir_a, dir_b, "linear", 0.5,
                    os.path.join(os.path.dirname(__file__),
                                 "merge-test-eval.json"),
                    out_dir, log=lambda *a: None, protocol="v1")
            self.assertEqual(code, 0)
            self.assertTrue(rep["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
