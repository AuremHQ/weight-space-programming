"""Phase 2 (W2.5, W2.7, W2.12, W2.14): the v2 pipeline end to end
with FakeRealModel — all six exits, the leak test, the schema
test, evaluate_detailed, and the dry-run cost model.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from tests.fakes import FakeRealModel
from wsp.protocol import get_protocol
from wsp.splits import assign_splits

CATS = ("animals", "countries", "fruits", "vegetables",
        "colors", "instruments", "sports", "cities")

GOOD_QA = (
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 3?", "5"),
    ("What color is the sky?", "blue"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is 7 + 8?", "15"),
    ("What color is grass?", "green"),
    ("Which is the largest planet?", "Jupiter"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who painted the Mona Lisa?", "da Vinci"),
    ("Who was the first president of the USA?", "Washington"),
)

# Seed 2 with the partial_pass scenario: holdout has 1 helped
# prompt (mean gain 7, lower bound -5.8) — point estimates pass,
# the bound fails: INCONCLUSIVE. Found by brute force over the
# deterministic hash split (see the phase report).
INCONCLUSIVE_SEED = 2

# Hand-computed generate-call total for a SHIP run on the v2
# fixture (top_columns=4, gate temps [0.7, 0.0], samples 3):
# cheap screen baseline + 8 candidates on select 9x11=99,
# select baselines 11x(3+1)=44, attempts 3x(11+33)=132,
# holdout baselines 26x(3+1)=104, base refusal probes 16,
# diagnosis captures 11, causal (2+10)x4=48, confirmation
# 26x(3+1)=104, edited refusal probes 16, sample before/after 2.
# Total 576. The dry run must print this exact total, and the
# run must spend exactly this many eval passes.
SHIP_TOTAL_PASSES = 576


def build_v2_eval():
    """A v2-size eval dict (20 failing, 16 neighbours, 12 good)."""
    failing = [
        {"prompt": f"List 60 different {CATS[i % 8]}, one per line. "
                   f"(case {i + 1})",
         "expect_count": 60}
        for i in range(20)
    ]
    good = [{"prompt": q, "expect": a} for q, a in GOOD_QA]
    neighbours = []
    for i in range(16):
        n = 5 if i % 2 == 0 else 10
        neighbours.append(
            {"prompt": f"List {n} different {CATS[i % 8]}, one per "
                       f"line. (near {i + 1})",
             "expect_count": n})
    return {
        "format_version": 1,
        "name": "v2-test",
        "model": "fake-model",
        "serving_temperature": 0.7,
        "gate_temperatures": [0.7, 0.0],
        "max_new_tokens": 64,
        "use_chat_template": True,
        "sets": {
            "failing": {"scorer": "distinct", "prompts": failing},
            "good": {"scorer": "contains", "prompts": good},
            "neighbours": {"scorer": "distinct",
                           "prompts": neighbours},
        },
    }


def write_eval(td, data=None):
    path = os.path.join(td, "v2eval.json")
    with open(path, "w") as f:
        json.dump(data or build_v2_eval(), f, indent=2)
    return path


def run_v2(scenario, td, seed=0, **kw):
    """Run the v2 pipeline with a fake model; return (rep, dir, code)."""
    from wsp.real_pipeline import run_real_pipeline

    out_dir = os.path.join(td, f"out-{scenario}-{seed}")
    rep, out_dir, code = run_real_pipeline(
        model_id="fake-model",
        eval_path=write_eval(td),
        out_dir=out_dir,
        model=FakeRealModel(scenario),
        protocol="v2",
        top_columns=4,
        seed=seed,
        log=lambda *a: None,
        **kw)
    return rep, out_dir, code


V2_REQUIRED_KEYS = (
    "protocol", "evidence_level", "eval_sha256",
    "eval_passes_spent", "generated_tokens", "environment",
    "model_revision", "generation_config_effective",
    "holdout_scores", "phase_prompt_counts", "known_limitations",
    "editor_info", "wall_seconds",
)


class TestV2Exits(unittest.TestCase):
    def test_preflight_exits_2_on_small_eval(self):
        from wsp.real_pipeline import run_real_pipeline

        golden = os.path.join(os.path.dirname(__file__), "golden",
                              "v1", "eval.json")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as ctx:
                run_real_pipeline(
                    model_id="fake-model", eval_path=golden,
                    out_dir=os.path.join(td, "out"),
                    model=FakeRealModel("pass"), protocol="v2",
                    log=lambda *a: None)
            self.assertEqual(
                str(ctx.exception),
                "need at least 20 failing prompts for a decision "
                "(found 8). Add more of YOUR OWN failing prompts: "
                "the kit never invents them. For a quick personal "
                "check use --protocol v1 (result is labelled QUICK, "
                "not evidence).")
            self.assertFalse(os.path.exists(
                os.path.join(td, "out", ".wsp-running")))

    def test_ship(self):
        with tempfile.TemporaryDirectory() as td:
            rep, out_dir, code = run_v2("pass", td)
            self.assertEqual(code, 0)
            self.assertEqual(rep["gate"]["verdict"], "SHIP")
            self.assertTrue(rep["gate"]["passed"])
            self.assertEqual(rep["protocol"], "v2")
            self.assertEqual(rep["evidence_level"], "standard")
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "edit.json")))
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "report.json")))
            # The recipe hash in the report matches edit.json.
            from wsp.recipe import (edit_sha256, load_recipe,
                                    validate_against_model)
            from tests.fakes import FakeRealModel

            with open(os.path.join(out_dir, "edit.json")) as f:
                edit = json.load(f)
            self.assertEqual(rep["real_demo"]["edit_sha256"],
                             edit_sha256(edit))
            # Phase 4: the "pass" fixture has a positive causal
            # effect, so the plan is dampen — the recipe carries
            # the factor, no alpha, and loads + validates.
            self.assertEqual(edit["edit_type"], "dampen")
            for layer in edit["layers"]:
                self.assertIn("dampen_factor", layer)
                self.assertNotIn("alpha", layer)
                self.assertNotIn("direction", layer)
            loaded = load_recipe(os.path.join(out_dir, "edit.json"))
            fake = FakeRealModel("pass")
            validate_against_model(loaded, fake.layer_infos,
                                   fake.total_params)
            for layer in rep["real_demo"]["layers"]:
                self.assertEqual(layer["edit_type"], "dampen")
                self.assertIn("dampen_factor", layer)
                self.assertNotIn("alpha", layer)
            # Phase 4: the calibrated alpha is ported, so the
            # legacy-alpha known limitation is gone.
            self.assertEqual(rep["known_limitations"], [])
            # Layer choice records both numbers + the full table.
            table = rep["real_demo"]["layer_table"]
            self.assertEqual(len(table), 4)
            for row in table:
                for key in ("index", "direction_norm", "failing_norm",
                            "contrast_norm", "relative_contrast"):
                    self.assertIn(key, row)
            self.assertEqual(rep["real_demo"]["layers"][0]["index"], 0)
            # Editor info: numerical, no LLM.
            self.assertEqual(rep["editor_info"],
                             {"type": "numerical_estimator",
                              "llm_used": False})
            # Holdout scores let anyone recompute the bound.
            self.assertEqual(
                len(rep["holdout_scores"]["before"]["0.7"]["failing"]),
                10)

    def test_cheap_fix_wins(self):
        with tempfile.TemporaryDirectory() as td:
            rep, out_dir, code = run_v2("cheap_fix", td)
            self.assertEqual(code, 1)
            self.assertIn("cheap fix wins: temperature 1.0",
                          rep["gate"]["reason"])
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "edit.json")))
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "fake-model.json")))

    def test_no_failure(self):
        with tempfile.TemporaryDirectory() as td:
            rep, _out, code = run_v2("no_failure", td)
            self.assertEqual(code, 1)
            self.assertIn("no failure measured",
                          rep["gate"]["reason"])
            self.assertEqual(rep["gate"]["verdict"], "REJECT")

    def test_not_causal(self):
        with tempfile.TemporaryDirectory() as td:
            rep, _out, code = run_v2("not_causal", td)
            self.assertEqual(code, 1)
            self.assertIn("not causal", rep["gate"]["reason"])
            self.assertEqual(rep["gate"]["verdict"], "REJECT")

    def test_negative_causal_ships_via_subtraction(self):
        # Phase 4 (replaces
        # test_negative_causal_refuses_with_exact_message): the
        # sign selects the edit type, so a -30.00 causal effect
        # now ships via subtraction with the column-norm-capped
        # alpha (mean column norm ~2.83 on the fake, capped at
        # 1.0) instead of refusing the run.
        with tempfile.TemporaryDirectory() as td:
            rep, out_dir, code = run_v2("negative_causal", td)
            self.assertEqual(code, 0)
            self.assertEqual(rep["gate"]["verdict"], "SHIP")
            self.assertTrue(rep["gate"]["passed"])
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "edit.json")))
            with open(os.path.join(out_dir, "edit.json")) as f:
                edit = json.load(f)
            self.assertEqual(edit["edit_type"], "subtract")
            self.assertEqual(edit["layers"][0]["alpha"], 1.0)
            self.assertEqual(
                rep["real_demo"]["layers"][0]["edit_type"],
                "subtract")
            self.assertEqual(
                rep["real_demo"]["layers"][0]["alpha"], 1.0)
            self.assertTrue(
                rep["real_demo"]["layers"][0]["alpha_clamped"])
            self.assertEqual(rep["known_limitations"], [])

    def test_no_attempt_passes_is_reject(self):
        with tempfile.TemporaryDirectory() as td:
            rep, out_dir, code = run_v2("no_attempt_passes", td)
            self.assertEqual(code, 1)
            self.assertEqual(rep["gate"]["verdict"], "REJECT")
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "edit.json")))

    def test_inconclusive_on_small_eval(self):
        with tempfile.TemporaryDirectory() as td:
            rep, out_dir, code = run_v2("partial_pass", td,
                                        seed=INCONCLUSIVE_SEED)
            self.assertEqual(code, 1)
            self.assertEqual(rep["gate"]["verdict"], "INCONCLUSIVE")
            self.assertFalse(rep["gate"]["passed"])
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "edit.json")))
            # The point estimates pass; the bound does not.
            by_temp = rep["gate"]["by_temperature"]
            self.assertGreaterEqual(
                by_temp["0.7"]["primary_gain_points"], 2.0)
            self.assertLess(by_temp["0.7"]["primary_ci"]["lower"], 0.0)

    def test_v1_still_quick(self):
        from wsp.real_pipeline import run_real_pipeline

        golden = os.path.join(os.path.dirname(__file__), "golden",
                              "v1", "eval.json")
        with tempfile.TemporaryDirectory() as td:
            rep, _out, code = run_real_pipeline(
                model_id="fake-model", eval_path=golden,
                out_dir=os.path.join(td, "out"),
                model=FakeRealModel("pass"), protocol="v1",
                log=lambda *a: None)
            self.assertEqual(code, 0)
            self.assertEqual(rep["gate"]["verdict"], "QUICK_PASS")
            self.assertEqual(rep["evidence_level"], "quick")
            self.assertEqual(rep["protocol"], "v1")


class TestV2Schema(unittest.TestCase):
    def test_required_keys_for_each_exit(self):
        cases = [
            ("cheap_fix", 0, 1), ("no_failure", 0, 1),
            ("not_causal", 0, 1), ("negative_causal", 0, 1),
            ("no_attempt_passes", 0, 1),
            ("partial_pass", INCONCLUSIVE_SEED, 1),
            ("pass", 0, 0),
        ]
        for scenario, seed, _code in cases:
            with tempfile.TemporaryDirectory() as td:
                rep, _out, _c = run_v2(scenario, td, seed=seed)
                for key in V2_REQUIRED_KEYS:
                    self.assertIn(key, rep, f"{scenario}: {key}")
                self.assertIn("verdict", rep["gate"])
                self.assertIn("protocol_config",
                              rep["real_demo"])
                self.assertEqual(
                    rep["real_demo"]["protocol_config"]["protocol"],
                    "v2")
                # eval_sha256 matches the eval file bytes.
                import hashlib

                with open(write_eval(td), "rb") as f:
                    want = hashlib.sha256(f.read()).hexdigest()
                self.assertEqual(rep["eval_sha256"], want)
                self.assertIsInstance(rep["wall_seconds"], float)
                self.assertGreaterEqual(rep["wall_seconds"], 0.0)


class TestLeakTest(unittest.TestCase):
    def test_no_holdout_prompt_leaks_into_selection(self):
        with tempfile.TemporaryDirectory() as td:
            rep, _out, code = run_v2("pass", td)
            self.assertEqual(code, 0)
            data = build_v2_eval()
            splits = assign_splits(data, get_protocol("v2"), seed=0)
            holdout = set()
            for name in ("failing", "good", "neighbours"):
                for i in splits[name]["holdout"]:
                    holdout.add(
                        data["sets"][name]["prompts"][i]["prompt"])
            counts = rep["real_demo"]["phase_prompt_counts"]
            for phase in ("diagnose", "causal", "select",
                          "cheap_screen"):
                touched = set(counts.get(phase, {}))
                leaked = touched & holdout
                self.assertEqual(leaked, set(),
                                 f"holdout leaked into {phase}: "
                                 f"{leaked}")
            for phase in ("baseline", "gate"):
                touched = set(counts.get(phase, {}))
                self.assertTrue(
                    touched <= holdout,
                    f"{phase} touched non-holdout prompts: "
                    f"{touched - holdout}")
                self.assertTrue(touched, f"{phase} touched nothing")


class TestEvaluateDetailed(unittest.TestCase):
    def test_per_prompt_means_and_split_seeds(self):
        from wsp.real_pipeline import evaluate, evaluate_detailed

        fake = FakeRealModel("pass")
        data = build_v2_eval()
        log = lambda *a: None
        full = evaluate_detailed(fake, data, 0.7, 3, 0, log)
        # One per-prompt mean per prompt in the set.
        self.assertEqual(len(full["failing"]["per_prompt"]), 20)
        self.assertAlmostEqual(
            full["failing"]["mean"],
            sum(full["failing"]["per_prompt"]) / 20)
        # A split evaluation uses the FULL-set prompt index for
        # its seed, so the same prompt scores identically.
        split = evaluate_detailed(fake, data, 0.7, 3, 0, log,
                                  split={"failing": [5, 7]})
        self.assertAlmostEqual(split["failing"]["per_prompt"][0],
                               full["failing"]["per_prompt"][5])
        self.assertAlmostEqual(split["failing"]["per_prompt"][1],
                               full["failing"]["per_prompt"][7])
        # evaluate() keeps its signature by wrapping it.
        means = evaluate(fake, data, 0.7, 3, 0, log)
        self.assertAlmostEqual(means["failing"],
                               full["failing"]["mean"])

    def test_scorer_v2_selected_by_protocol(self):
        from wsp.real_pipeline import evaluate_detailed

        fake = FakeRealModel("pass")
        data = build_v2_eval()
        log = lambda *a: None
        out = evaluate_detailed(fake, data, 0.0, 1, 0, log,
                                scorer_version="v2")
        # v2 distinct scoring of the fake's numbered lists gives
        # the same counts (items are distinct, normalised).
        self.assertAlmostEqual(out["failing"]["mean"],
                               evaluate_detailed(
                                   fake, data, 0.0, 1, 0, log,
                                   scorer_version="v1")
                               ["failing"]["mean"])


class TestDryRun(unittest.TestCase):
    def test_dry_run_prints_exact_total(self):
        with tempfile.TemporaryDirectory() as td:
            from wsp.real_pipeline import run_real_pipeline

            lines = []
            rep, out_dir, code = run_real_pipeline(
                model_id="fake-model",
                eval_path=write_eval(td),
                out_dir=os.path.join(td, "out"),
                model=FakeRealModel("pass"),
                protocol="v2",
                top_columns=4,
                dry_run=True,
                log=lines.append)
            self.assertEqual(code, 0)
            self.assertIsNone(rep)
            text = "\n".join(str(ln) for ln in lines)
            self.assertIn(f"total generate calls: {SHIP_TOTAL_PASSES}",
                          text)
            self.assertIn("failing: 20 prompts", text)
            self.assertIn("diagnose 6", text)
            self.assertIn("select 4", text)
            self.assertIn("holdout 10", text)
            # Nothing was written: no marker, no report, no dir.
            self.assertFalse(os.path.exists(out_dir))

    def test_dry_run_total_matches_actual_spend(self):
        with tempfile.TemporaryDirectory() as td:
            rep, _out, code = run_v2("pass", td)
            self.assertEqual(code, 0)
            self.assertEqual(rep["eval_passes_spent"],
                             SHIP_TOTAL_PASSES)

    def test_dry_run_imports_no_torch_in_subprocess(self):
        with tempfile.TemporaryDirectory() as td:
            eval_path = write_eval(td)
            kit_dir = os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))
            driver = (
                "import sys; "
                "from wsp.cli import main; "
                "code = main(['run', '--model', 'fake-model', "
                "'--eval', %r, '--out-dir', %r, '--protocol', 'v2', "
                "'--dry-run', '--top-columns', '4']); "
                "print('TORCH_IN_SYS_MODULES:', 'torch' in sys.modules); "
                "print('EXIT:', code)"
                % (eval_path, os.path.join(td, "out")))
            env = dict(os.environ)
            env["PYTHONPATH"] = "."
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                cwd=kit_dir,
                env=env, capture_output=True, text=True, timeout=120)
            self.assertIn("TORCH_IN_SYS_MODULES: False", proc.stdout,
                          proc.stdout + proc.stderr)
            self.assertIn("EXIT: 0", proc.stdout)
            self.assertFalse(os.path.exists(os.path.join(td, "out")))

    def test_tokens_per_second_prints_time_estimate(self):
        with tempfile.TemporaryDirectory() as td:
            from wsp.real_pipeline import run_real_pipeline

            lines = []
            run_real_pipeline(
                model_id="fake-model",
                eval_path=write_eval(td),
                out_dir=os.path.join(td, "out"),
                model=FakeRealModel("pass"),
                protocol="v2",
                top_columns=4,
                dry_run=True,
                tokens_per_second=100.0,
                log=lines.append)
            text = "\n".join(str(ln) for ln in lines)
            self.assertIn("estimated time", text)
            self.assertIn("100", text)


if __name__ == "__main__":
    unittest.main()
