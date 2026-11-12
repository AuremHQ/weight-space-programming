"""Phase 3 tests: claims equal code (F20-F27, items W3.1-W3.8).

Red-first: every test here failed before the Phase 3 implementation
(the registry, the doc rewrites, the nudge helper, the MCP tool,
the install.sh flags, the 0.3.0 release files, the verdict wording).
"""

import contextlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

KIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(KIT_DIR)


def _read(rel):
    with open(os.path.join(REPO_ROOT, rel)) as f:
        return f.read()


def _read_kit(rel):
    with open(os.path.join(KIT_DIR, rel)) as f:
        return f.read()


class _FakeTTY(io.StringIO):
    """A StringIO with a controllable isatty() answer."""

    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def _report(path, passed, before=30.0, after=40.0, verdict=None,
            model="Qwen/Qwen3-0.6B (real open-weight model, CPU)",
            eval_name="loop-fix-test"):
    rep = {
        "report_version": 1,
        "kit_version": "0.3.0",
        "model": model,
        "weights_changed": 1000,
        "weights_changed_pct": 0.02,
        "before": {"failing": before, "good": 100.0, "neighbours": 100.0},
        "after": {"failing": after, "good": 100.0, "neighbours": 100.0},
        "gate": {"passed": passed, "verdict": verdict},
        "temperatures_checked": [0.0, 0.7],
        "editor": "test",
        "note": "test fixture",
        "real_demo": {"eval_name": eval_name,
                      "causal_proof": {"change_points": 10.0}},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rep, f)
    return path


# ---------------------------------------------------------------- W3.1

class TestRefusals(unittest.TestCase):
    """W3.1: the refusal registry — five rows, truth as the code has it."""

    def test_registry_has_five_rows(self):
        from wsp.refusals import REFUSALS

        self.assertEqual(
            [r["id"] for r in REFUSALS],
            ["no_eval_file", "no_weights", "add_capability",
             "production_branch", "safety_removal"])
        for row in REFUSALS:
            for field in ("id", "text", "enforced_in_code", "where",
                          "test"):
                self.assertIn(field, row, row["id"])
            self.assertIsInstance(row["enforced_in_code"], bool)

    def test_registry_truth_values(self):
        from wsp.refusals import REFUSALS

        by_id = {r["id"]: r for r in REFUSALS}
        self.assertTrue(by_id["no_eval_file"]["enforced_in_code"])
        # Partial: no dedicated check, the loader fails instead.
        self.assertFalse(by_id["no_weights"]["enforced_in_code"])
        self.assertIn("PARTIAL", by_id["no_weights"]["where"])
        self.assertFalse(by_id["add_capability"]["enforced_in_code"])
        self.assertFalse(by_id["production_branch"]["enforced_in_code"])
        self.assertFalse(by_id["safety_removal"]["enforced_in_code"])
        self.assertIn("detected", by_id["safety_removal"]["where"])
        self.assertIn("not prevented",
                      by_id["safety_removal"]["where"])

    def test_enforced_rows_name_existing_tests(self):
        """Every enforced_in_code=True row names a real test method."""
        from wsp.refusals import REFUSALS

        if KIT_DIR not in sys.path:
            sys.path.insert(0, KIT_DIR)
        if os.path.join(KIT_DIR, "tests") not in sys.path:
            sys.path.insert(0, os.path.join(KIT_DIR, "tests"))
        for row in REFUSALS:
            if not row["enforced_in_code"]:
                continue
            self.assertIsNotNone(row["test"], row["id"])
            module_name, class_name, method_name = row["test"].split(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            self.assertTrue(callable(getattr(cls, method_name)),
                            row["test"])

    def test_readme_blocks_match_registry(self):
        """The refusal text in both READMEs is the rendered registry."""
        from wsp.refusals import render_refusal_block

        block = render_refusal_block()
        self.assertIn(block, _read("README.md"))
        self.assertIn(block, _read_kit("README.md"))

    def _cmd_run(self, model, eval_path, out_dir, protocol="v1"):
        import argparse

        from wsp.cli import cmd_run

        args = argparse.Namespace(
            model=model, eval=eval_path, out_dir=out_dir, dtype="bfloat16",
            samples_per_prompt=None, seed=0, top_columns=64,
            max_attempts=3, force=False, protocol=protocol,
            dry_run=False, no_default_guards=False,
            tokens_per_second=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cmd_run(args)
        return code, buf.getvalue()

    def test_no_eval_file_plain_message(self):
        with tempfile.TemporaryDirectory() as td:
            code, out = self._cmd_run(
                "fake-model", os.path.join(td, "no-such-eval.json"),
                os.path.join(td, "out"))
        self.assertEqual(code, 2)
        self.assertIn("error:", out)
        self.assertIn("no eval file", out)
        self.assertNotIn("Traceback", out)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "torch not installed")
    def test_no_weights_api_only_id_plain_message(self):
        """W3.1: an API-only id fails in the loader, plainly, offline."""
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
                code, out = self._cmd_run(
                    "gpt-4",
                    os.path.join(KIT_DIR, "tests", "golden", "v1",
                                 "eval.json"),
                    os.path.join(td, "out"), protocol="v1")
        self.assertEqual(code, 2)
        self.assertIn("error:", out)
        self.assertIn("gpt-4", out)
        self.assertNotIn("Traceback", out)


# ---------------------------------------------------------------- W3.2

class TestDocsConsistency(unittest.TestCase):
    """W3.2: the truthfulness sweep — stale phrases gone, true ones in."""

    STALE = [
        ("README.md", "the same pipeline is an MCP server"),
        ("README.md", "Real runs live: Qwen3-0.6B + GPT2-124M, honest numbers"),
        ("README.md", "\u221230.00 \u2014 FAILED"),
        ("README.md", "our real-model runs (Qwen + GPT-2)"),
        ("kit/README.md", "The same pipeline, callable from any MCP"),
        ("kit/README.md", "Refusals, enforced in code: no weights"),
        ("kit/README.md", "are never overwritten; the edited weights are a separate artifact"),
        ("kit/README.md", "no edit ever overwrites your original model."),
        ("kit/wsp/mcp_server.py", "MCP server: the same pipeline"),
        ("REAL-RUNS.md", "C:\\Users\\tejis"),
        ("REAL-RUNS.md", "need +5 points or stop"),
        ("REAL-RUNS.md", "\u221230.00 \u2014 FAILED"),
        ("docs/index.html", "the same way \u2014 subtracting a direction"),
        ("docs/index.html", "The original weights are never overwritten."),
        ("docs/index.html", "\u221230.00 (bar +5) \u2014 failed"),
        ("DEMO-PLAN.md", "No edit passed the gate, so **no edited weights ship**"),
        ("DEMO-PLAN.md", "None was.** The gate refused all four real runs"),
        ("DEMO-PLAN.md", "the gate works on two small models"),
        ("DEMO-PLAN.md", "Not yet: no real-model edit has passed the gate"),
        ("DEMO-PLAN.md", "refused all four real runs"),
    ]

    def test_stale_phrases_gone(self):
        for rel, phrase in self.STALE:
            text = _read(rel)
            self.assertNotIn(phrase, text, f"{rel}: {phrase!r}")

    def test_mcp_is_demo_only_toy_model(self):
        for rel in ("README.md", "kit/README.md",
                    "kit/wsp/mcp_server.py"):
            text = _read(rel)
            self.assertIn("toy model", text, rel)
            self.assertIn("demo", text.lower(), rel)
        self.assertIn("use `wsp run` for real models", _read("README.md"))

    def test_on_disk_wording_and_in_memory_sentence(self):
        for rel in ("kit/wsp/cli.py", "kit/README.md", "docs/index.html"):
            text = _read(rel)
            self.assertIn("on disk", text, rel)
            self.assertIn("in-memory copy", text, rel)

    def test_causal_rule_describes_v1_and_v2(self):
        text = _read("REAL-RUNS.md")
        self.assertIn("abs()", text)
        self.assertIn("placebo", text)
        self.assertIn("expected sign", text)

    def test_smollm2_causal_passed_the_v1_bar(self):
        """v1 is abs(): |-30| >= 5 passes — the old FAILED label lied."""
        text = _read("REAL-RUNS.md")
        self.assertIn("PASSED the v1 bar", text)
        self.assertIn("the v1 flaw", text)

    def test_plus_29_caveat_in_table_rows(self):
        row = [ln for ln in _read("REAL-RUNS.md").splitlines()
               if "+29.44" in ln and "|" in ln]
        self.assertTrue(row)
        self.assertIn("Penguin", row[0])
        site = [ln for ln in _read("docs/index.html").splitlines()
                if "+29.44" in ln]
        self.assertTrue(site)
        self.assertIn("caveat", site[0].lower())

    def test_both_passes_labelled_quick_evidence(self):
        # Phase 4 (W4.5): both passes are "quick (3 failing
        # prompts)". The old label also said "not reproducible
        # from this repo yet" — that is no longer true for the
        # Qwen pass, whose eval + report now ship. docs/index.html
        # and README.md still carry the old label (out of Phase 4
        # scope; follow-up).
        label = "quick (3 failing prompts)"
        self.assertGreaterEqual(_read("REAL-RUNS.md").count(label), 2)

    def test_no_absolute_home_paths_in_real_runs(self):
        text = _read("REAL-RUNS.md")
        self.assertNotIn("C:\\Users", text)
        self.assertNotIn("/home/", text)
        self.assertIn("<run dir>", text)

    def test_roadmap_names_three_models(self):
        self.assertIn("Qwen3-0.6B + GPT2-124M + SmolLM2-1.7B",
                      _read("README.md"))

    def test_task_notes_point_at_all_runs(self):
        note = json.loads(_read_kit("tasks/loop-fix-real/task.json"))["note"]
        self.assertIn("REAL-RUNS.md", note)

    def test_demo_plan_is_consistent(self):
        text = _read("DEMO-PLAN.md")
        self.assertIn("7 real runs, 2 gate passes", text)
        self.assertIn("260,969", text)
        self.assertIn("SmolLM2", text)


# ---------------------------------------------------------------- W3.3

class TestSkillFrontmatter(unittest.TestCase):
    """W3.3: SKILL.md frontmatter, provenance note, requirements line."""

    def _frontmatter(self):
        text = _read("SKILL.md")
        lines = text.split("\n")
        self.assertEqual(lines[0], "---")
        end = lines.index("---", 1)
        out = {}
        for line in lines[1:end]:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
        return out

    def test_frontmatter_parses_with_name_and_description(self):
        fm = self._frontmatter()
        self.assertEqual(fm["name"], "weight-space-programming")
        desc = fm["description"]
        self.assertTrue(desc)
        self.assertLess(len(desc), 400)
        self.assertIn("small models", desc)
        self.assertIn("Refuses", desc)
        # One or two sentences.
        sentences = [s for s in desc.split(". ") if s.strip()]
        self.assertGreaterEqual(len(sentences), 1)
        self.assertLessEqual(len(sentences), 2)

    def test_h1_kept_for_install_sh_grep(self):
        self.assertIn("# Weight-Space Programming", _read("SKILL.md"))
        self.assertIn("Weight-Space Programming", _read("install.sh"))

    def test_provenance_note_separates_mit_from_this_editor(self):
        skill = _read("SKILL.md")
        item5 = [ln for ln in skill.splitlines()
                 if "This note:" in ln]
        self.assertEqual(len(item5), 1)
        self.assertIn("blog post, not peer reviewed", item5[0])
        self.assertIn("numerical estimator", item5[0])
        self.assertIn("no LLM", item5[0])
        self.assertIn("MIT with GPT-6 Astra", item5[0])

    def test_report_note_same_wording(self):
        from wsp.report import _report_note

        for gate in ({"passed": True},
                     {"passed": False, "reason": "gate failed"}):
            note = _report_note(gate)
            self.assertIn("blog post, not peer reviewed", note)
            self.assertIn("numerical estimator", note)
            self.assertIn("no LLM", note)
            self.assertNotIn("demonstrated with GPT-6 Astra and is "
                             "unproven for other models. This edit",
                             note)

    def test_merge_note_same_wording(self):
        # The note is built from adjacent string literals, so the
        # wording is split across source lines; check the halves.
        src = _read_kit("wsp/merge.py")
        self.assertIn("blog post, ", src)
        self.assertIn("not peer reviewed", src)
        self.assertIn("arithmetic ", src)
        self.assertIn("on weights (no LLM, no editor model)", src)

    def test_requirements_line_offers_both_editors(self):
        skill = _read("SKILL.md")
        self.assertNotIn("A frontier-class model to do the reasoning "
                         "(the editor).", skill)
        self.assertIn("numerical estimator (`wsp run`", skill)
        self.assertIn("frontier-class model (agent-reasoned path", skill)


# ---------------------------------------------------------------- W3.4

class TestNudgePolicy(unittest.TestCase):
    """W3.4 (D2): one helper, maybe_nudge — every nudge path covered."""

    def _nudge(self, verdict, tty=True, env=None):
        from wsp.cli import maybe_nudge

        out = []
        fake = _FakeTTY(tty)
        patch_env = mock.patch.dict(os.environ, env or {}, clear=False)
        if env is None:
            patch_env = mock.patch.dict(os.environ, {}, clear=False)
            os.environ.pop("WSP_NO_NUDGE", None)
        with patch_env, mock.patch.object(sys, "stdout", fake):
            maybe_nudge({"gate": {"verdict": verdict}},
                        print_fn=out.append)
        return out

    def test_no_nudge_on_quick_pass(self):
        self.assertEqual(self._nudge("QUICK_PASS"), [])

    def test_no_nudge_on_quick_fail(self):
        self.assertEqual(self._nudge("QUICK_FAIL"), [])

    def test_no_nudge_when_stdout_is_not_a_tty(self):
        self.assertEqual(self._nudge("SHIP", tty=False), [])

    def test_nudge_silenced_by_env(self):
        self.assertEqual(
            self._nudge("SHIP", tty=True, env={"WSP_NO_NUDGE": "1"}), [])

    def test_nudge_shown_once_after_ship_on_tty(self):
        out = self._nudge("SHIP", tty=True)
        self.assertEqual(len(out), 1)
        self.assertIn("star", out[0])

    def test_install_sh_has_no_star_lines(self):
        self.assertNotIn("please star the repo", _read("install.sh"))

    def test_demo_keeps_its_star_line(self):
        """The toy demo is not a real-model run: its star line stays."""
        import argparse

        from wsp.cli import cmd_demo

        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(out_dir=os.path.join(td, "demo"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cmd_demo(args)
        self.assertIn("star the repo", buf.getvalue())

    def test_status_page_cta_stays(self):
        from wsp.status import render_html, scan_reports

        with tempfile.TemporaryDirectory() as td:
            _report(os.path.join(td, "a", "report.json"), True,
                    verdict="SHIP", before=30.0, after=40.0)
            page = render_html(scan_reports(td))
        self.assertIn("Star on GitHub", page)


# ---------------------------------------------------------------- W3.5

@unittest.skipUnless(
    importlib.util.find_spec("mcp") is not None, "mcp not installed")
class TestMcpPhase3(unittest.TestCase):
    """W3.5: the real import path, the pin, wsp_capabilities, smoke."""

    def test_import_path_is_mcpserver(self):
        import pkgutil

        import mcp.server
        import mcp.server.mcpserver as mod

        self.assertTrue(hasattr(mod, "MCPServer"))
        names = [m.name for m in pkgutil.iter_modules(mcp.server.__path__)]
        self.assertIn("mcpserver", names)

    def test_mcpserver_tried_first_fastmcp_is_fallback(self):
        src = _read_kit("wsp/mcp_server.py")
        self.assertLess(src.index("mcp.server.mcpserver"),
                        src.index("mcp.server.fastmcp"))

    def test_pyproject_pins_mcp_below_3(self):
        self.assertIn("mcp>=2,<3", _read_kit("pyproject.toml"))

    def test_wsp_capabilities_tool(self):
        from wsp.mcp_server import wsp_capabilities

        self.assertEqual(wsp_capabilities(), {
            "real_models": False,
            "note": "tools drive the built-in toy model; "
                    "use `wsp run` for real models",
        })

    def test_smoke_tools_in_order_match_run_pipeline(self):
        from wsp import pipeline
        from wsp.mcp_server import (kit_apply_edit, kit_baseline,
                                    kit_demo, kit_estimate_direction,
                                    kit_prove_cause, kit_run_gate,
                                    kit_temperature_sweep)

        base = kit_baseline()
        self.assertIn("baseline", base)
        sweep = kit_temperature_sweep()
        self.assertIn("0.7", sweep)
        cause = kit_prove_cause()
        self.assertTrue(cause["causal"])
        direction = kit_estimate_direction()
        self.assertIn("direction", direction)
        applied = kit_apply_edit(1.5)
        self.assertEqual(applied["weights_changed"], 128)
        gated = kit_run_gate()
        self.assertTrue(gated["passed"])
        self.assertEqual(kit_demo(), pipeline.run_pipeline()[0])


# ---------------------------------------------------------------- W3.6

class TestInstallRef(unittest.TestCase):
    """W3.6 (F26): WSP_REF / --ref, skill from the same checkout."""

    def _run(self, args, env=None):
        full_env = dict(os.environ)
        full_env["HOME"] = tempfile.mkdtemp(prefix="wsp-install-home-")
        if env:
            full_env.update(env)
        proc = subprocess.run(["bash", "install.sh"] + args,
                              cwd=REPO_ROOT, capture_output=True,
                              text=True, env=full_env, timeout=60)
        shutil.rmtree(full_env["HOME"], ignore_errors=True)
        return proc

    def test_bash_n(self):
        proc = subprocess.run(["bash", "-n", "install.sh"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_help_lists_ref_and_dry_run(self):
        proc = self._run(["--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--ref", proc.stdout)
        self.assertIn("WSP_REF", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)

    def test_dry_run_prints_ref(self):
        proc = self._run(["--dry-run", "--ref", "v0.3.0"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("v0.3.0", proc.stdout)

    def test_dry_run_defaults_to_main(self):
        proc = self._run(["--dry-run"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("main", proc.stdout)

    def test_dry_run_honours_wsp_ref_env(self):
        proc = self._run(["--dry-run"], env={"WSP_REF": "v0.3.0"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("v0.3.0", proc.stdout)

    def test_skill_comes_from_the_same_checkout(self):
        src = _read("install.sh")
        self.assertIn('--branch "$WSP_REF"', src)
        self.assertIn("refs/tags/$WSP_REF", src)
        self.assertIn("refs/heads/$WSP_REF", src)
        self.assertIn("rev-parse HEAD", src)
        self.assertIn('cp "$KIT_DIR/$SKILL_FILE"', src)
        self.assertNotIn("SKILL_URL", src)
        # Default stays main until a tag exists, with the comment
        # saying where the release tag goes.
        self.assertIn('WSP_REF="${WSP_REF:-main}"', src)
        self.assertIn("v0.3.0", src)


# ---------------------------------------------------------------- W3.7

class TestReleaseHygiene(unittest.TestCase):
    """W3.7 (F27): 0.3.0, CHANGELOG, SECURITY.md, package-data."""

    def test_versions_match_and_are_0_3_0(self):
        from wsp import __version__

        m = re.search(r'version = "([^"]+)"', _read_kit("pyproject.toml"))
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "0.3.0")
        self.assertEqual(__version__, "0.3.0")

    def test_changelog_sections(self):
        text = _read("CHANGELOG.md")
        for heading in ("### Added", "### Changed", "### Fixed",
                        "### Security"):
            self.assertIn(heading, text, heading)
        self.assertIn("0.3.0", text)
        self.assertIn("default protocol", text)
        self.assertIn("--protocol v1", text)

    def test_security_md_threat_model(self):
        text = _read("SECURITY.md")
        self.assertIn("hello@auremcto.com", text)
        self.assertIn("localhost", text)
        self.assertIn("What is checked", text)
        self.assertIn("What is not", text)
        section = text.split("## Threat model")[1].split("##")[0]
        lines = [ln for ln in section.splitlines() if ln.strip()]
        self.assertEqual(len(lines), 8)

    def test_package_data_entry_present(self):
        self.assertIn('wsp = ["data/guards/*.json"]',
                      _read_kit("pyproject.toml"))

    def test_readme_install_and_flags_updated(self):
        self.assertIn("--protocol", _read("README.md"))
        self.assertIn("mcp>=2,<3", _read_kit("README.md"))


# ---------------------------------------------------------------- W3.8

class TestVerdictWording(unittest.TestCase):
    """W3.8: status page + one-liner use the verdict wording."""

    def test_verdict_labels(self):
        from wsp.status import verdict_label

        self.assertEqual(verdict_label("SHIP"),
                         "Approved (standard evidence)")
        self.assertEqual(verdict_label("QUICK_PASS"),
                         "Quick check passed (not evidence)")
        self.assertEqual(verdict_label(None),
                         "Quick check passed (not evidence)")

    def test_status_line_uses_verdict_wording(self):
        from wsp.status import status_line

        with tempfile.TemporaryDirectory() as td:
            _report(os.path.join(td, "ship", "report.json"), True,
                    verdict="SHIP", before=30.0, after=40.0)
            _report(os.path.join(td, "quick", "report.json"), True,
                    verdict="QUICK_PASS", before=30.0, after=36.5)
            _report(os.path.join(td, "legacy", "report.json"), True,
                    verdict=None, before=30.0, after=35.0)
            _report(os.path.join(td, "bad", "report.json"), False,
                    verdict="REJECT", before=30.0, after=31.0)
            line = status_line(td)
        self.assertIn("1 Approved (standard evidence)", line)
        self.assertIn("2 Quick check passed (not evidence)", line)
        self.assertIn("1 refused of 4 runs found here", line)

    def test_win_cards_show_verdict_labels(self):
        from wsp.status import render_html, scan_reports

        with tempfile.TemporaryDirectory() as td:
            _report(os.path.join(td, "ship", "report.json"), True,
                    verdict="SHIP", before=30.0, after=40.0)
            _report(os.path.join(td, "quick", "report.json"), True,
                    verdict="QUICK_PASS", before=30.0, after=36.5)
            page = render_html(scan_reports(td))
        self.assertIn("Approved (standard evidence)", page)
        self.assertIn("Quick check passed (not evidence)", page)

    def test_hostile_strings_still_escaped(self):
        from wsp.status import render_html, scan_reports

        with tempfile.TemporaryDirectory() as td:
            _report(os.path.join(td, "x", "report.json"), True,
                    verdict="SHIP", before=30.0, after=40.0,
                    model="<script>alert(1)</script> (real model)",
                    eval_name='"><img src=x onerror=alert(1)>')
            page = render_html(scan_reports(td))
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("Approved (standard evidence)", page)


if __name__ == "__main__":
    unittest.main()
