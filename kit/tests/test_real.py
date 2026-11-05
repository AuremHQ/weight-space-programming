"""Tests for the v0.2.0 universal real-model additions.

Scoring, eval-file and layer-finder tests need no model download and
no torch. Edit-math tests need torch and skip cleanly without it.
Run from the kit directory:
  PYTHONPATH=. python3 -m unittest discover -s tests -v
"""

import unittest
from types import SimpleNamespace

from wsp import evalfile, layers, scoring
from wsp.real_edit import MAX_EDIT_FRACTION, alpha_base_from_signals, check_cap
from wsp.real_pipeline import (failure_measured, gate_distance,
                               rank_attempts, select_best_passing,
                               suggest_next_steps)
from wsp import watch as watch_mod

try:
    import torch
except ImportError:
    torch = None


class FakeWeight:
    def __init__(self, out_features, in_features):
        self.shape = (out_features, in_features)
        self._numel = out_features * in_features

    def numel(self):
        return self._numel


class FakeLinear:
    def __init__(self, out_features, in_features):
        self.out_features = out_features
        self.in_features = in_features
        self.weight = FakeWeight(out_features, in_features)


class FakeConv1D:
    """GPT-2's Conv1D: nf/nx attrs, weight stored transposed (in, out)."""

    def __init__(self, nf, nx):
        self.nf = nf
        self.nx = nx
        self.weight = FakeWeight(nx, nf)


def qwen_style_model(n_layers=4):
    """Qwen/Llama-style: model.layers[i].mlp.down_proj."""
    blocks = [SimpleNamespace(mlp=SimpleNamespace(
        down_proj=FakeLinear(1024, 3072),
        gate_proj=FakeLinear(3072, 1024),
        up_proj=FakeLinear(3072, 1024))) for _ in range(n_layers)]
    return SimpleNamespace(model=SimpleNamespace(layers=blocks))


def gpt2_style_model(n_layers=3):
    """GPT-2-style: transformer.h[i].mlp.c_proj (real Conv1D)."""
    blocks = [SimpleNamespace(mlp=SimpleNamespace(
        c_fc=FakeConv1D(3072, 768),
        c_proj=FakeConv1D(768, 3072))) for _ in range(n_layers)]
    return SimpleNamespace(transformer=SimpleNamespace(h=blocks))


class TestScoring(unittest.TestCase):
    def test_loop_score_clean_text(self):
        text = "The cat sat on the mat. Dogs bark at night. Birds sing."
        self.assertEqual(scoring.loop_score(text), 100.0)

    def test_loop_score_repeated_block(self):
        block = "one two three four five six seven eight"
        text = " ".join([block] * 6)
        self.assertLess(scoring.loop_score(text), 10.0)

    def test_loop_score_short_text(self):
        self.assertEqual(scoring.loop_score("hello world"), 100.0)

    def test_contains_score(self):
        self.assertEqual(scoring.contains_score("Paris is lovely", "paris"), 100.0)
        self.assertEqual(scoring.contains_score("London calling", "paris"), 0.0)

    def test_contains_needs_expect(self):
        with self.assertRaises(ValueError):
            scoring.contains_score("anything", None)

    def test_finish_score(self):
        self.assertEqual(scoring.finish_score("done", True), 100.0)
        self.assertEqual(scoring.finish_score("cut off", False), 0.0)

    def test_unknown_scorer(self):
        with self.assertRaises(ValueError):
            scoring.score_text("nope", "text")

    def test_sequence_score_full(self):
        text = "1\n2\n3\n4\n5\n"
        self.assertEqual(
            scoring.score_text("sequence", text, expect_items=["1", "2", "3", "4", "5"]),
            100.0)

    def test_sequence_score_loop_cut(self):
        text = "\n".join(["1", "2", "3"] + ["3", "3", "3"] * 10) + "\n"
        self.assertEqual(
            scoring.score_text("sequence", text, expect_items=["1", "2", "3", "4", "5"]),
            60.0)

    def test_sequence_score_wrong_start(self):
        text = "9\n8\n7\n"
        self.assertEqual(
            scoring.score_text("sequence", text, expect_items=["1", "2", "3"]),
            0.0)

    def test_sequence_needs_items(self):
        with self.assertRaises(ValueError):
            scoring.score_text("sequence", "1\n2\n", expect_items=[])

    def test_distinct_all_distinct(self):
        text = "1. Lion\n2. Tiger\n3. Bear\n4. Wolf\n5. Fox\n"
        self.assertEqual(
            scoring.score_text("distinct", text, expect_count=5), 100.0)

    def test_distinct_cycling(self):
        # The observed small-model failure: numbered items cycle, so a
        # bare repetition check misses it but distinct catches it.
        items = ["Lion", "Tiger", "Bear"] * 10
        text = "".join(f"{i + 1}. {it}\n" for i, it in enumerate(items))
        self.assertAlmostEqual(
            scoring.score_text("distinct", text, expect_count=30), 10.0)

    def test_distinct_shortfall(self):
        text = "1. Lion\n2. Tiger\n"
        self.assertAlmostEqual(
            scoring.score_text("distinct", text, expect_count=10), 20.0)

    def test_distinct_case_insensitive(self):
        text = "1. Lion\n2. LION\n3. Tiger\n"
        self.assertAlmostEqual(
            scoring.score_text("distinct", text, expect_count=3),
            100.0 * 2 / 3)

    def test_distinct_preamble_ignored(self):
        text = "Here is the list:\n\n1. Lion\n2. Tiger\n3. Lion\n"
        self.assertAlmostEqual(
            scoring.score_text("distinct", text, expect_count=3),
            100.0 * 2 / 3)

    def test_distinct_empty_scores_zero(self):
        self.assertEqual(
            scoring.score_text("distinct", "", expect_count=5), 0.0)

    def test_distinct_needs_count(self):
        with self.assertRaises(ValueError):
            scoring.score_text("distinct", "1. Lion\n", expect_count=0)
        with self.assertRaises(ValueError):
            scoring.score_text("distinct", "1. Lion\n")


class TestEvalFile(unittest.TestCase):
    def _valid(self):
        return {
            "format_version": 1,
            "name": "test",
            "model": "Qwen/Qwen3-0.6B",
            "serving_temperature": 0.7,
            "gate_temperatures": [0.7, 0.0],
            "max_new_tokens": 64,
            "use_chat_template": True,
            "sets": {
                "failing": {"scorer": "loop",
                            "prompts": [{"prompt": "Count to 50."}]},
                "good": {"scorer": "contains",
                         "prompts": [{"prompt": "Capital of France?",
                                      "expect": "Paris"}]},
                "neighbours": {"scorer": "loop",
                               "prompts": [{"prompt": "Count to 5."}]},
            },
        }

    def test_valid_file_passes(self):
        data = evalfile.validate_eval(self._valid(), "test")
        self.assertEqual(evalfile.set_scorer(data, "good"), "contains")
        self.assertEqual(len(evalfile.set_prompts(data, "failing")), 1)

    def test_bad_version_rejected(self):
        data = self._valid()
        data["format_version"] = 2
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")

    def test_missing_set_rejected(self):
        data = self._valid()
        del data["sets"]["neighbours"]
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")

    def test_contains_without_expect_rejected(self):
        data = self._valid()
        del data["sets"]["good"]["prompts"][0]["expect"]
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")

    def test_unknown_scorer_rejected(self):
        data = self._valid()
        data["sets"]["failing"]["scorer"] = "vibes"
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")

    def test_distinct_without_count_rejected(self):
        data = self._valid()
        data["sets"]["failing"]["scorer"] = "distinct"
        data["sets"]["failing"]["prompts"] = [{"prompt": "List 60 animals."}]
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")
        data["sets"]["failing"]["prompts"] = [
            {"prompt": "List 60 animals.", "expect_count": 60}]
        evalfile.validate_eval(data, "test")  # passes with the count

    def test_sequence_without_items_rejected(self):
        data = self._valid()
        data["sets"]["failing"] = {
            "scorer": "sequence",
            "prompts": [{"prompt": "Count to 5."}],
        }
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")

    def test_bad_temperature_rejected(self):
        data = self._valid()
        data["serving_temperature"] = 3.0
        with self.assertRaises(ValueError):
            evalfile.validate_eval(data, "test")


class TestLayerFinder(unittest.TestCase):
    def test_qwen_style_found(self):
        infos = layers.describe_layers(qwen_style_model(4))
        self.assertEqual(len(infos), 4)
        self.assertEqual(infos[0]["down_proj_name"], "down_proj")
        self.assertEqual(infos[0]["hidden_size"], 1024)
        self.assertEqual(infos[0]["intermediate_size"], 3072)
        self.assertEqual(infos[0]["down_proj_numel"], 1024 * 3072)

    def test_gpt2_style_found(self):
        infos = layers.describe_layers(gpt2_style_model(3))
        self.assertEqual(len(infos), 3)
        self.assertEqual(infos[0]["down_proj_name"], "c_proj")
        self.assertEqual(infos[0]["hidden_size"], 768)
        self.assertEqual(infos[0]["intermediate_size"], 3072)
        self.assertEqual(infos[0]["weight_layout"], "in_out")

    def test_gpt2_linear_style_layout(self):
        # A Linear-shaped c_proj still works (layout "out_in").
        blocks = [SimpleNamespace(mlp=SimpleNamespace(
            c_proj=FakeLinear(768, 3072))) for _ in range(2)]
        model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
        infos = layers.describe_layers(model)
        self.assertEqual(infos[0]["weight_layout"], "out_in")
        self.assertEqual(infos[0]["hidden_size"], 768)
        self.assertEqual(infos[0]["intermediate_size"], 3072)

    def test_no_blocks_rejected(self):
        with self.assertRaises(ValueError):
            layers.describe_layers(SimpleNamespace())

    def test_no_mlp_output_rejected(self):
        model = SimpleNamespace(model=SimpleNamespace(
            layers=[SimpleNamespace(mlp=SimpleNamespace())]))
        with self.assertRaises(ValueError):
            layers.describe_layers(model)


class TestRealEdit(unittest.TestCase):
    def test_cap_enforced(self):
        with self.assertRaises(ValueError):
            check_cap(200, 10000)  # 2% > 1%
        self.assertAlmostEqual(check_cap(100, 10000), 0.01)

    def test_alpha_base_scales_with_signal(self):
        self.assertAlmostEqual(alpha_base_from_signals(4.0, 2.0), 2.0)
        self.assertEqual(alpha_base_from_signals(4.0, 0.0), 1.0)

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_column_edit_changes_only_columns(self):
        from wsp.real_edit import apply_column_edit

        w = torch.zeros(4, 6)
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        new, changed = apply_column_edit(w, [2, 5], direction, 1.5)
        self.assertTrue(torch.equal(w, torch.zeros(4, 6)))  # input untouched
        self.assertEqual(changed, 2)
        self.assertAlmostEqual(float(new[0, 2]), -1.5)
        self.assertAlmostEqual(float(new[0, 0]), 0.0)
        self.assertTrue(bool((new[:, [0, 1, 3, 4]] == 0).all()))

    @unittest.skipUnless(torch is not None, "torch not installed")
    def test_column_edit_conv1d_layout_edits_rows(self):
        from wsp.real_edit import apply_column_edit

        w = torch.zeros(6, 4)  # (intermediate, hidden) — Conv1D layout
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        new, changed = apply_column_edit(w, [1, 4], direction, 1.5,
                                         layout="in_out")
        self.assertTrue(torch.equal(w, torch.zeros(6, 4)))  # input untouched
        self.assertEqual(changed, 2)
        self.assertAlmostEqual(float(new[1, 0]), -1.5)
        self.assertAlmostEqual(float(new[4, 0]), -1.5)
        self.assertAlmostEqual(float(new[0, 0]), 0.0)
        self.assertTrue(bool((new[[0, 2, 3, 5], :] == 0).all()))

    def test_column_edit_bad_layout_rejected(self):
        from wsp.real_edit import apply_column_edit

        with self.assertRaises(ValueError):
            apply_column_edit(None, [0], None, 1.0, layout="sideways")


@unittest.skipUnless(torch is not None, "torch not installed")
class TestRunnerEditLayout(unittest.TestCase):
    """Runner edit/zero/restore on a Conv1D (transposed) down-proj."""

    def _runner_with_conv1d(self, total_params=10000):
        from wsp.real_runner import RealModel

        r = RealModel.__new__(RealModel)
        r.total_params = total_params
        r._down_projs = {0: SimpleNamespace(weight=torch.zeros(6, 4))}
        r._infos_by_index = {0: {"index": 0, "hidden_size": 4,
                                 "intermediate_size": 6,
                                 "weight_layout": "in_out"}}
        return r

    def test_apply_edit_conv1d_edits_rows_and_restores(self):
        r = self._runner_with_conv1d()
        direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
        changed, saved = r.apply_edit(0, [1, 4], direction, 1.5)
        self.assertEqual(changed, 2)
        w = r._down_projs[0].weight
        self.assertAlmostEqual(float(w[1, 0]), -1.5)
        self.assertAlmostEqual(float(w[4, 0]), -1.5)
        r.restore(saved)
        self.assertTrue(torch.equal(w, torch.zeros(6, 4)))

    def test_zero_columns_conv1d(self):
        r = self._runner_with_conv1d()
        w = r._down_projs[0].weight
        with torch.no_grad():
            w[2, :] = 3.0
        saved = r.zero_columns(0, [2])
        self.assertTrue(bool((w[2, :] == 0).all()))
        r.restore(saved)
        self.assertTrue(bool((w[2, :] == 3.0).all()))

    def test_apply_edit_cap_uses_hidden_size(self):
        r = self._runner_with_conv1d(total_params=100)
        # 2 cols x 4 hidden = 8 weights = 8% of 100 -> over the 1% cap.
        with self.assertRaises(ValueError):
            r.apply_edit(0, [1, 4], torch.zeros(4), 1.0)


def _gate(gain, regressions, passed):
    return {
        "passed": passed,
        "primary": "failing",
        "primary_gain_points": gain,
        "regressions_points": regressions,
        "min_gain_points": 5.0,
        "max_regress_points": 2.0,
    }


def _attempt(variant, gains, weights=100):
    """gains: {temperature: gate-dict}; passed iff all temps passed."""
    evaluated = sorted(gains)
    return {
        "variant": variant,
        "weights_changed": weights,
        "gate_by_temperature": {str(t): g for t, g in gains.items()},
        "evaluated_temperatures": evaluated,
        "passed": all(g["passed"] for g in gains.values())
        and len(evaluated) == 2,
    }


class TestBestPassing(unittest.TestCase):
    def test_failure_measured_gap(self):
        measured, gap = failure_measured({"good": 90.0, "failing": 30.0})
        self.assertTrue(measured)
        self.assertAlmostEqual(gap, 60.0)
        measured, _ = failure_measured({"good": 90.0, "failing": 87.0})
        self.assertFalse(measured)  # within 5 points: not a failure

    def test_gate_distance(self):
        g = _gate(12.0, {"good": -1.0, "neighbours": 0.5}, True)
        self.assertEqual(gate_distance({"0.7": g}), 0.0)
        g = _gate(2.0, {"good": 4.0, "neighbours": 0.0}, False)
        # 3 points short of min gain + 2 points excess regression
        self.assertAlmostEqual(gate_distance({"0.0": g}), 5.0)

    def test_best_passing_beats_first_passing(self):
        a1 = _attempt(1, {0.0: _gate(11.0, {"good": 0.0, "neighbours": 0.0}, True),
                          0.7: _gate(12.0, {"good": 0.0, "neighbours": 0.0}, True)})
        a2 = _attempt(2, {0.0: _gate(15.0, {"good": 0.0, "neighbours": 0.0}, True),
                          0.7: _gate(20.0, {"good": 0.0, "neighbours": 0.0}, True)})
        # Both pass; the best is variant 2 (higher serving gain),
        # even though variant 1 passed first.
        self.assertEqual(select_best_passing([a1, a2], 0.7), 1)

    def test_no_passing_returns_none(self):
        a = _attempt(1, {0.0: _gate(2.0, {"good": 0.0, "neighbours": 0.0}, False)})
        self.assertIsNone(select_best_passing([a], 0.7))

    def test_ranking_failing_closest_first(self):
        near = _attempt(1, {0.0: _gate(4.0, {"good": 0.0, "neighbours": 0.0}, False)})
        far = _attempt(2, {0.0: _gate(0.0, {"good": 5.0, "neighbours": 0.0}, False)})
        ranked = rank_attempts([far, near], 0.7)
        self.assertEqual(ranked[0]["attempt"], 1)  # 1 point short beats 8
        self.assertAlmostEqual(ranked[0]["distance_points"], 1.0)
        self.assertAlmostEqual(ranked[1]["distance_points"], 8.0)

    def test_suggestions_match_miss_type(self):
        primary_only = _gate(2.0, {"good": 0.0, "neighbours": 0.0}, False)
        s = suggest_next_steps({}, {"0.0": primary_only})
        self.assertTrue(any("did not move the failure" in x for x in s))
        self.assertTrue(any("No weights were shipped" in x for x in s))
        guard_only = _gate(12.0, {"good": 4.0, "neighbours": 0.0}, False)
        s = suggest_next_steps({}, {"0.0": guard_only})
        self.assertTrue(any("disturbed guards" in x for x in s))
        self.assertFalse(any("did not move the failure" in x for x in s))


class TestWatch(unittest.TestCase):
    def test_scan_finds_new_and_changed_only(self):
        import json
        import os
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "eval-a.json")
            with open(p, "w") as f:
                json.dump({"format_version": 1}, f)
            seen = {}
            self.assertEqual(watch_mod.scan_watch_dir(d, seen), [p])
            seen[os.path.basename(p)] = watch_mod._fingerprint(p)
            self.assertEqual(watch_mod.scan_watch_dir(d, seen), [])
            # Editing the file (new mtime/size) re-triggers it.
            time.sleep(0.01)
            with open(p, "w") as f:
                json.dump({"format_version": 1, "name": "b"}, f)
            self.assertEqual(watch_mod.scan_watch_dir(d, seen), [p])

    def test_seen_manifest_round_trip(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(watch_mod.load_seen(d), {})
            watch_mod.save_seen(d, {"a.json": {"size": 3, "mtime": 1.5}})
            self.assertEqual(watch_mod.load_seen(d),
                             {"a.json": {"size": 3, "mtime": 1.5}})

    def test_watch_once_runs_each_new_file_once(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as watch_dir, \
                tempfile.TemporaryDirectory() as out_dir:
            p = os.path.join(watch_dir, "eval-a.json")
            with open(p, "w") as f:
                json.dump({"format_version": 1}, f)
            calls = []

            def fake_run(model_id, eval_path, run_dir, log=print, **kw):
                calls.append(eval_path)
                return ({}, run_dir, 0)

            orig = watch_mod.run_once
            watch_mod.run_once = fake_run
            try:
                code = watch_mod.watch_loop("M", watch_dir, out_dir,
                                            once=True, log=lambda *a: None)
            finally:
                watch_mod.run_once = orig
            self.assertEqual(code, 0)
            self.assertEqual(calls, [p])
            # Second --once pass: nothing new, no rerun.
            watch_mod.run_once = fake_run
            try:
                watch_mod.watch_loop("M", watch_dir, out_dir,
                                     once=True, log=lambda *a: None)
            finally:
                watch_mod.run_once = orig
            self.assertEqual(calls, [p])

    def test_watch_rejects_missing_directory(self):
        with self.assertRaises(ValueError):
            watch_mod.watch_loop("M", "/nonexistent-watch-dir-xyz",
                                 "/tmp", once=True, log=lambda *a: None)


@unittest.skipUnless(torch is not None, "torch not installed")
class TestRunnerGenerate(unittest.TestCase):
    def _runner_with_stub(self):
        """A RealModel bypassing __init__, with a recording stub."""
        from wsp.real_runner import RealModel

        seen = {}

        class Ids(dict):
            def to(self, _device):
                return self

        class StubTokenizer:
            eos_token_id = 2
            pad_token_id = 0
            chat_template = None

            def __call__(self, text, return_tensors=None):
                ids = Ids({"input_ids": torch.tensor([[1, 2, 3]])})
                return ids

            def decode(self, ids, skip_special_tokens=True):
                return "stub output"

        class StubModel:
            generation_config = SimpleNamespace(eos_token_id=2)

            def generate(self, **kwargs):
                seen.update(kwargs)
                # 3 prompt tokens + 2 new tokens, last one EOS.
                return torch.tensor([[1, 2, 3, 4, 2]])

        r = RealModel.__new__(RealModel)
        r.model = StubModel()
        r.tokenizer = StubTokenizer()
        r.device = "cpu"
        r.use_chat_template = False
        r.thinking = False
        return r, seen

    def test_sampled_generate_has_no_generator_kwarg(self):
        # transformers 5.x rejects a `generator` kwarg; the runner must
        # seed globally instead (regression test for the demo-run
        # failure on the first sampled generation).
        r, seen = self._runner_with_stub()
        text, finished, n = r.generate("hi", 10, 0.7, 42)
        self.assertNotIn("generator", seen)
        self.assertTrue(seen["do_sample"])
        self.assertAlmostEqual(seen["temperature"], 0.7)
        self.assertTrue(finished)
        self.assertEqual(n, 2)

    def test_greedy_generate_is_deterministic(self):
        r, seen = self._runner_with_stub()
        r.generate("hi", 10, 0, 0)
        self.assertFalse(seen["do_sample"])
        self.assertNotIn("temperature", seen)
        self.assertNotIn("generator", seen)


if __name__ == "__main__":
    unittest.main()
