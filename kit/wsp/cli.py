"""Command line: `wsp demo` runs the kit pipeline end to end, and
`wsp auto` is easy mode — 4 questions, then the kit runs by itself.
`wsp loop` keeps improving in the background (new eval files and
failures.jsonl lines trigger gated runs; gate passes become new
versions), and `wsp serve-model` serves one model and applies each
new passing edit to it live — the same model, its old weights
overridden in place, no reload, no swap, and `wsp merge` blends two
same-base models (linear/SLERP), then runs the eval gate on the
merged model. `wsp delta` lists, shows and applies saved weight
deltas: one verified edit each, stored as a sparse diff plus a JSON
proof next to the run that made it (`<out-dir>/deltas/`).

The demo prints every number it measures and writes two files into
the output directory: report.json and kit_edited.npz. The edited
weights are always a separate file — the kit never overwrites its
input on disk, and here the input is generated in memory, so there is
nothing to overwrite in the first place. serve-model changes only the
in-memory copy of the loaded model; the files on disk are never changed.
"""

import argparse
import json
import os
import sys

import numpy as np

from . import __version__, pipeline, weights


def _table(title, metrics):
    print(title)
    for name, value in metrics.items():
        print(f"  {name:<12} {value:6.2f}")
    print()


def cmd_demo(args):
    rep, edited, before, after, attempts, sweep, cause = pipeline.run_pipeline()
    print("baseline (serving temperature 0.7, defective model)")
    _table("  before", before)
    print("temperature sweep, failing set only (Step 0: the cheap knob first)")
    for t, m in sweep.items():
        print(f"  temperature {t:<4} failing {m['failing']:6.2f}  good {m['good']:6.2f}")
    print()
    print(
        f"causal proof: zeroing row {cause['suspect_row']} moves the failing "
        f"set from {cause['failing_before']:.2f} to "
        f"{cause['failing_after_ablation']:.2f} "
        f"({cause['change_points']:+.2f} points) — causal: {cause['causal']}"
    )
    print()
    for i, a in enumerate(attempts, 1):
        print(
            f"attempt {i}: alpha {a['alpha']} — "
            f"{a['weights_changed']} weights "
            f"({a['weights_changed_pct']:.2f}%) — passed: {a['passed']}"
        )
    print()
    if edited is None:
        print("gate failed on every attempt — discarding, shipping nothing")
        code = 1
    else:
        _table("  after (serving temperature 0.7)", after)
        print(
            f"gate passed: failing {after['failing'] - before['failing']:+.2f} points, "
            f"good {after['good'] - before['good']:+.2f} points, "
            f"neighbours {after['neighbours'] - before['neighbours']:+.2f} points"
        )
        print("If this helped, please star the repo: "
              "https://github.com/AuremHQ/weight-space-programming")
        code = 0

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if edited is not None:
        weights.save_weights(os.path.join(args.out_dir, "kit_edited.npz"), edited)
    print(f"wrote {args.out_dir}/report.json")
    if edited is not None:
        print(f"wrote {args.out_dir}/kit_edited.npz")
    return code


def maybe_nudge(rep, print_fn=print):
    """The star ask, gated (D2, W3.4): the ONE helper every
    real-model run path uses. Only after a standard-evidence SHIP,
    only on a TTY, silenced by WSP_NO_NUDGE=1. QUICK results never
    nudge. `wsp demo` is the toy pipeline, not a real-model run, so
    it keeps its own direct star line (see cmd_demo)."""
    verdict = (rep.get("gate") or {}).get("verdict")
    if verdict != "SHIP":
        return
    if os.environ.get("WSP_NO_NUDGE") == "1":
        return
    if not sys.stdout.isatty():
        return
    print_fn("If this helped, please star the repo: "
             "https://github.com/AuremHQ/weight-space-programming")


def cmd_run(args):
    """`wsp run`: the universal real-model pipeline.

    Example: wsp run --model Qwen/Qwen3-0.6B --eval eval-loop.json
    Needs the real extra (torch + transformers); without it, this
    prints the install hint instead of a traceback.
    """
    try:
        from . import real_pipeline
    except ImportError as exc:
        print(exc)
        return 2
    try:
        rep, out_dir, code = real_pipeline.run_real_pipeline(
            model_id=args.model,
            eval_path=args.eval,
            out_dir=args.out_dir,
            dtype=args.dtype,
            samples_per_prompt=args.samples_per_prompt,
            seed=args.seed,
            top_columns=args.top_columns,
            max_attempts=args.max_attempts,
            force=args.force,
            protocol=getattr(args, "protocol", "v2"),
            dry_run=getattr(args, "dry_run", False),
            no_default_guards=getattr(args, "no_default_guards", False),
            tokens_per_second=getattr(args, "tokens_per_second", None),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except ImportError as exc:
        # Missing torch/transformers: print the hint, not a traceback.
        print(exc)
        return 2
    if rep is None:
        # --dry-run: the plan was printed already; no report exists.
        return code
    print()
    print(f"gate passed: {rep['gate']['passed']}")
    if rep["gate"]["passed"]:
        print(f"  failing {rep['after']['failing'] - rep['before']['failing']:+.2f} points, "
              f"good {rep['after']['good'] - rep['before']['good']:+.2f} points, "
              f"neighbours {rep['after']['neighbours'] - rep['before']['neighbours']:+.2f} points")
        print(f"  {rep['weights_changed']} weights ({rep['weights_changed_pct']:.2f}%)")
        maybe_nudge(rep)
    else:
        print(f"  reason: {rep['gate'].get('reason')}")
        ranking = rep.get("real_demo", {}).get("attempt_ranking")
        if ranking:
            print("  attempt ranking (best first):")
            for r in ranking:
                print(f"    attempt {r['attempt']}: passed={r['passed']} "
                      f"distance={r['distance_points']:.2f} points "
                      f"serving_gain={r['primary_gain_serving']}")
            for s in rep["real_demo"].get("suggestions", []):
                print(f"  suggestion: {s}")
    print(f"wrote {out_dir}/report.json")
    return code


def cmd_auto(args):
    """`wsp auto`: easy mode — 4 questions, then the kit runs by itself.

    Answers in plain words; the kit builds the eval file, runs the
    gated pipeline, and shows the result. Nothing is invented: every
    prompt comes from the user's answers. With --eval, the questions
    are skipped and that eval file is used instead.
    """
    from . import auto

    return auto.run_auto(
        model_id=args.model,
        eval_path=args.eval,
        out_dir=args.out_dir,
        serve=args.serve,
        protocol=getattr(args, "protocol", "v1"),
    )


def cmd_watch(args):
    """`wsp watch`: monitor a directory for new eval files, run the
    gated pipeline on each. Never applies an edit to any model — it
    only produces artifacts (report + edited weights on a gate pass).
    Stop with Ctrl+C.
    """
    try:
        from . import watch
        from . import real_runner  # preflight: fail fast without torch
    except ImportError as exc:
        print(exc)
        return 2
    try:
        return watch.watch_loop(
            model_id=args.model,
            watch_dir=args.watch_dir,
            out_dir=args.out_dir,
            interval=args.interval,
            once=args.once,
            dtype=args.dtype,
            samples_per_prompt=args.samples_per_prompt,
            seed=args.seed,
            top_columns=args.top_columns,
            max_attempts=args.max_attempts,
            protocol=getattr(args, "protocol", "v2"),
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2


def cmd_loop(args):
    """`wsp loop`: keep improving in the background, forever.

    Watches a directory for new eval files AND new lines in
    failures.jsonl (one JSON object per line: prompt, expect, kind).
    Each trigger runs the gated pipeline; a gate pass becomes a new
    version in versions.json + latest.json. Nothing ships on a gate
    fail, and the same failure is never retried. Stop with Ctrl+C.
    """
    try:
        from . import loop
        from . import real_runner  # preflight: fail fast without torch
    except ImportError as exc:
        print(exc)
        return 2
    try:
        return loop.run_loop(
            model_id=args.model,
            watch_dir=args.watch_dir,
            out_dir=args.out_dir,
            interval=args.interval,
            once=args.once,
            pipeline_kwargs={
                "dtype": args.dtype,
                "samples_per_prompt": args.samples_per_prompt,
                "seed": args.seed,
                "top_columns": args.top_columns,
                "max_attempts": args.max_attempts,
                "protocol": getattr(args, "protocol", "v2"),
            },
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2


def cmd_serve_model(args):
    """`wsp serve-model`: serve one model, apply new passing edits to
    it live. The same model object, its old weights overridden in
    place — no reload, no restart, no swap. Rollback with
    POST /rollback. Stop with Ctrl+C.
    """
    from . import serve_model

    token = args.token or os.environ.get("WSP_TOKEN")
    return serve_model.serve_model(
        model_id=args.model,
        versions_dir=args.versions_dir,
        port=args.port,
        interval=args.interval,
        device=args.device,
        dtype=args.dtype,
        token=token,
        allow_quick=args.allow_quick,
    )


def cmd_apply_edit(args):
    """`wsp apply-edit`: reproduce a recorded edit from original weights.

    Loads --model fresh, validates the layers/columns/direction/alpha
    (or dampen_factor) in --edit (an edit.json written by `wsp run`
    on a gate pass) against the loaded model, checks the recipe-level
    1% cap and the provenance (a report.json next to the edit must
    show a gate pass for this exact edit), and saves the edited model
    to --out-dir. With --verify, the sha256 of the changed tensors
    is compared with --expected-sha256 and MATCH or DIVERGED is
    printed (exit 1 on DIVERGED). No eval, no gate — this is the
    reproducibility script the skill requires. Bad input exits 2
    with a plain message, never a traceback.
    """
    import json

    if getattr(args, "verify", False) and not getattr(
            args, "expected_sha256", None):
        print("error: --verify needs --expected-sha256 <hex> (the "
              "recorded sha256 of the changed tensors, e.g. from "
              "the run's PROVENANCE.md)")
        return 2
    try:
        from .atomic import atomic_write_json
        from .recipe import (edit_sha256, load_recipe, verify_provenance)
        from .real_runner import RealModel
        from .safe_paths import (assert_output_dir_safe,
                                 resolve_model_dirs)
        from .serve_model import apply_recipe
    except ImportError as exc:
        print(exc)
        return 2
    try:
        recipe = load_recipe(args.edit)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    edit_dir = os.path.dirname(os.path.abspath(args.edit))
    report_path = os.path.join(edit_dir, "report.json")
    report = None
    if os.path.isfile(report_path):
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (OSError, ValueError) as exc:
            print(f"error: report.json next to the edit is unreadable: "
                  f"{exc}")
            return 2
    if report is None:
        if not args.allow_unverified:
            print("error: no report.json next to this edit.json — the "
                  "edit cannot be tied to a gate pass; use "
                  "--allow-unverified to apply it anyway")
            return 2
        print("warning: no report.json next to this edit.json — "
              "applying UNVERIFIED (no gate pass tied to this edit)")
        evidence = "unverified"
    else:
        try:
            evidence = verify_provenance(recipe, report)
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        if evidence == "quick" and not args.allow_quick:
            print("error: QUICK evidence only (no SHIP verdict) — "
                  "refused without --allow-quick")
            return 2
    try:
        assert_output_dir_safe(
            args.out_dir, resolve_model_dirs([args.model]),
            force=args.force)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    try:
        model = RealModel(args.model,
                          dtype=recipe.get("dtype", "bfloat16"))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"loaded {args.model} (revision: "
          f"{model.revision if model.revision else 'none recorded'})")
    if recipe["model_id"] != args.model:
        # Not a refusal: the common case is a recipe written for an
        # HF id applied to a local copy of the same model. But an
        # edit for a DIFFERENT model must never slip by silently.
        print(f"warning: this edit is for {recipe['model_id']}, "
              f"you loaded {args.model} — check they are the same "
              "model before trusting the result")
    try:
        changed, _saved = apply_recipe(model, recipe)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 2
    verify = None
    actual_sha256 = None
    if getattr(args, "verify", False):
        from .verify import changed_tensors_sha256

        actual_sha256 = changed_tensors_sha256(model,
                                               recipe["layers"])
        expected = args.expected_sha256
        verify = ("MATCH" if actual_sha256 == expected else "DIVERGED")
        print(f"{verify}: changed tensors sha256 {actual_sha256} "
              f"(recorded {expected})")
    model.save(args.out_dir, force=args.force)
    applied = {
        "model": args.model,
        "model_revision": model.revision,
        "edit": os.path.abspath(args.edit),
        "edit_sha256": edit_sha256(recipe),
        "evidence": evidence,
        "weights_changed": changed,
        "verify": verify,
        "expected_sha256": (args.expected_sha256
                            if verify is not None else None),
        "actual_sha256": actual_sha256,
        "out_dir": os.path.abspath(args.out_dir),
    }
    atomic_write_json(os.path.join(args.out_dir, "apply-edit.json"),
                      applied)
    print(f"applied {changed} weights from {args.edit} "
          f"(evidence: {evidence})")
    print(f"saved edited model to {args.out_dir}")
    if verify == "DIVERGED":
        return 1
    return 0


def cmd_reproduce(args):
    """`wsp reproduce`: re-run a recorded run and compare the
    primary metric. Prints MATCH or DIVERGED with the numbers, or
    "cannot be reproduced exactly: missing <fields>" for early
    reports. Exit 0 MATCH, 1 DIVERGED, 2 anything else.
    """
    from .reproduce import reproduce_run

    try:
        result = reproduce_run(
            run_dir=args.run_dir,
            tolerance=getattr(args, "tolerance", 5.0),
            out_dir=getattr(args, "out_dir", None))
    except (ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    return result["code"]


def cmd_merge(args):
    """`wsp merge`: blend two same-base models, then gate the merge.

    Linear or SLERP average of --model-a and --model-b (same base
    required, enforced in code), saved to --out-dir, then the eval
    gate runs on the merged model against model A as the baseline
    (serving temperature AND greedy). Separate mode from surgical
    editing; NOT under the 1% cap. Needs the real extra.
    """
    try:
        from . import merge as merge_mod
    except ImportError as exc:
        print(exc)
        return 2
    try:
        rep, out_dir, code = merge_mod.run_merge(
            model_a=args.model_a,
            model_b=args.model_b,
            method=args.method,
            t=args.t,
            eval_path=args.eval,
            out_dir=args.out_dir,
            dtype=args.dtype,
            samples_per_prompt=args.samples_per_prompt,
            seed=args.seed,
            force=args.force,
            protocol=getattr(args, "protocol", "v2"),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    except ImportError as exc:
        # Missing torch/transformers: print the hint, not a traceback.
        print(exc)
        return 2
    print()
    print(f"gate passed: {rep['gate']['passed']}")
    before = rep["before"]
    after = rep["after"]
    if before is not None and after is not None:
        print(f"  failing {after['failing'] - before['failing']:+.2f} points, "
              f"good {after['good'] - before['good']:+.2f} points, "
              f"neighbours {after['neighbours'] - before['neighbours']:+.2f} points")
    if rep["gate"]["passed"]:
        maybe_nudge(rep)
    else:
        reason = rep["gate"].get("reason")
        if reason:
            print(f"  reason: {reason}")
        for temp, g in rep["gate"].get("by_temperature", {}).items():
            if not g["passed"]:
                print(f"  gate failed at temperature {temp}: "
                      f"gain {g['primary_gain_points']:+.2f} points "
                      f"(need {g['min_gain_points']}), "
                      f"regressions {g['regressions_points']}")
    print(f"wrote {out_dir}/report.json")
    if rep["gate"]["passed"]:
        print(f"merged model in {out_dir}")
    else:
        print("merged model NOT saved (gate failed)")
    return code


def cmd_version(_args):
    print(f"wsp-kit {__version__}")
    return 0


def cmd_status(args):
    """`wsp status`: one line — chip, approved improvements, best gain,
    real-model passes. Reads real report.json files; exits 0."""
    from .status import status_line
    print(status_line(args.runs_dir))
    return 0


def cmd_serve(args):
    """`wsp serve`: the status page on localhost until Ctrl+C."""
    from .serve import serve
    return serve(runs_dir=args.runs_dir, port=args.port)


def cmd_delta_list(args):
    """`wsp delta list`: one line per saved delta, oldest first."""
    from . import delta as delta_mod

    try:
        proofs = delta_mod.list_deltas(args.store)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    if not proofs:
        print(f"no deltas in {args.store}")
        return 0
    for p in proofs:
        g = p["gate"]
        e = p["edit"]
        print(f"{p['delta_id']}  {p['created_at']}  "
              f"{p['model']['model_id']}  "
              f"gain {g.get('primary_gain_points', 0):+.2f}  "
              f"{e['weights_changed']} weights "
              f"({e['weights_changed_pct']:.2%})  {g['verdict']}")
    return 0


def cmd_delta_show(args):
    """`wsp delta show`: the full proof of one delta, as JSON."""
    from . import delta as delta_mod

    try:
        _delta, proof = delta_mod.load_delta(args.store, args.delta_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(proof, indent=2))
    return 0


def cmd_delta_apply(args):
    """`wsp delta apply`: apply a delta to a NEW copy of a model.

    The local gate check runs first: the proof must record a gate
    pass, the base must match the delta's recorded base hash, the
    1% cap must hold, and the round-trip must be exact. Any failure
    refuses with the reason and writes nothing — the old model is
    never touched. ``--model`` is a .npz weights file, a local
    model directory, or a HuggingFace id; ``--out`` is where the
    new copy goes and must not exist yet.
    """
    from . import delta as delta_mod

    try:
        delta, proof = delta_mod.load_delta(args.store, args.delta_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    ok, reason = delta_mod.check_delta_gate(proof)
    if not ok:
        print(f"refused: {reason}")
        return 1
    model_path = os.path.abspath(args.model)
    out_path = os.path.abspath(args.out)
    if model_path == out_path:
        print("error: --out is the input model — refused "
              "(never overwrite)")
        return 2
    if os.path.exists(out_path):
        print(f"error: --out {args.out!r} already exists — refused "
              "(apply writes a new copy; pick a fresh path)")
        return 2
    if args.model.endswith(".npz"):
        try:
            base = weights.load_weights(args.model)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        try:
            new = delta_mod.apply_delta(base, delta, proof)
        except ValueError as exc:
            print(f"refused: {exc}")
            return 1
        weights.save_weights(args.out, new)
        print(f"applied delta {args.delta_id} "
              f"({proof['edit']['weights_changed']} weights) to a new "
              f"copy at {args.out}")
        print(f"gate: {proof['gate']['verdict']} "
              f"(gain "
              f"{proof['gate'].get('primary_gain_points', 0):+.2f} "
              f"points) — old model untouched")
        return 0
    try:
        from . import merge as merge_mod
        from .safe_paths import (assert_output_dir_safe,
                                 resolve_model_dirs)
    except ImportError as exc:
        print(exc)
        return 2
    try:
        assert_output_dir_safe(args.out,
                               resolve_model_dirs([args.model]))
        config, state, _tok, _total, dtypes = (
            merge_mod.load_model_dir(args.model))
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}")
        return 2
    try:
        new_state = delta_mod.apply_delta(state, delta, proof)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1
    try:
        model_dir_a = merge_mod._resolve_model_dir(args.model)
        merge_mod.save_merged_model(args.out, config, model_dir_a,
                                    new_state, dtypes)
    except (ValueError, ImportError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"applied delta {args.delta_id} "
          f"({proof['edit']['weights_changed']} weights) to a new "
          f"copy at {args.out}")
    print(f"gate: {proof['gate']['verdict']} "
          f"(gain {proof['gate'].get('primary_gain_points', 0):+.2f} "
          f"points) — old model untouched")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wsp",
        description="Weight-space programming, runnable. See the kit README for the full story.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_demo = sub.add_parser("demo", help="run the kit pipeline end to end")
    p_demo.add_argument("--out-dir", default="wsp-demo-out",
                        help="where report.json and kit_edited.npz go")
    p_demo.set_defaults(func=cmd_demo)
    p_run = sub.add_parser(
        "run", help="run the universal real-model pipeline on an eval file")
    p_run.add_argument("--model", required=True,
                       help="HuggingFace model id, e.g. Qwen/Qwen3-0.6B")
    p_run.add_argument("--eval", required=True,
                       help="eval file (JSON, format v1) with prompts + scorers")
    p_run.add_argument("--out-dir", default="wsp-real-out",
                       help="where report.json (and, on a gate pass, the "
                            "edited model + edit.json) go")
    p_run.add_argument("--dtype", default="bfloat16",
                       help="weight dtype to load (default bfloat16)")
    p_run.add_argument("--samples-per-prompt", type=int, default=None,
                       help="samples at non-greedy temperatures "
                            "(default: 1 for v1, 3 for v2)")
    p_run.add_argument("--seed", type=int, default=0,
                       help="base sampling seed (default 0)")
    p_run.add_argument("--top-columns", type=int, default=64,
                       help="columns edited per layer (default 64)")
    p_run.add_argument("--max-attempts", type=int, default=3,
                       help="edit attempts, max 3 per the skill (default 3)")
    p_run.add_argument("--force", action="store_true",
                       help="allow an out-dir that already holds model "
                            "files")
    p_run.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                       help="evaluation protocol: v2 is standard "
                            "evidence (default); v1 is the frozen "
                            "measurement, labelled QUICK")
    p_run.add_argument("--dry-run", action="store_true",
                       help="print the cost plan (prompt counts, exact "
                            "generate calls per phase, total) and exit "
                            "without loading or writing anything")
    p_run.add_argument("--tokens-per-second", type=float, default=None,
                       help="with --dry-run, also print a worst-case time "
                            "estimate at this generation speed")
    p_run.add_argument("--no-default-guards", action="store_true",
                       help="turn the default guard bundle off (v2 only; "
                            "the report records guards as disabled)")
    p_run.set_defaults(func=cmd_run)
    p_auto = sub.add_parser(
        "auto", help="easy mode: 4 questions, then the kit runs by itself")
    p_auto.add_argument("--model",
                        help="HuggingFace model id, e.g. Qwen/Qwen3-0.6B "
                             "(asked if omitted)")
    p_auto.add_argument("--eval",
                        help="use this eval file instead of asking "
                             "the questions")
    p_auto.add_argument("--out-dir", default="wsp-auto-out",
                        help="where the eval file and report.json go")
    p_auto.add_argument("--serve", action="store_true",
                        help="start the status page after the run")
    p_auto.add_argument("--protocol", choices=["v1", "v2"], default="v1",
                        help="evaluation protocol (default v1: easy mode "
                             "is a quick personal check, labelled QUICK)")
    p_auto.set_defaults(func=cmd_auto)
    p_apply = sub.add_parser(
        "apply-edit",
        help="reproduce a recorded edit from original weights")
    p_apply.add_argument("--model", required=True,
                         help="original HuggingFace model id")
    p_apply.add_argument("--edit", required=True,
                         help="edit.json written by `wsp run` on a gate pass")
    p_apply.add_argument("--out-dir", required=True,
                         help="where the edited model goes")
    p_apply.add_argument("--force", action="store_true",
                         help="allow an out-dir that already holds model "
                              "files")
    p_apply.add_argument("--allow-quick", action="store_true",
                         help="accept a report with no SHIP verdict "
                              "(QUICK evidence, incl. every v1 report)")
    p_apply.add_argument("--allow-unverified", action="store_true",
                         help="apply without a report.json next to the "
                              "edit (no gate pass tied to this edit)")
    p_apply.add_argument("--verify", action="store_true",
                         help="after applying, compare the sha256 of "
                              "the changed tensors with "
                              "--expected-sha256 and print MATCH or "
                              "DIVERGED (exit 1 on DIVERGED)")
    p_apply.add_argument("--expected-sha256", default=None,
                         help="recorded sha256 of the changed tensors "
                              "(from the run's PROVENANCE.md); "
                              "required with --verify")
    p_apply.set_defaults(func=cmd_apply_edit)
    p_repro = sub.add_parser(
        "reproduce",
        help="re-run a recorded run and compare the primary metric")
    p_repro.add_argument("run_dir",
                         help="directory holding report.json (and "
                              "edit.json) of the recorded run")
    p_repro.add_argument("--tolerance", type=float, default=5.0,
                         help="points of difference on the primary "
                              "metric still counted as MATCH "
                              "(default 5.0)")
    p_repro.add_argument("--out-dir", default=None,
                         help="where the fresh run goes (default: a "
                              "fresh temp dir)")
    p_repro.set_defaults(func=cmd_reproduce)
    p_merge = sub.add_parser(
        "merge",
        help="blend two same-base models (linear/SLERP), then run the "
             "eval gate on the merged model")
    p_merge.add_argument("--model-a", required=True,
                         help="first base model: HuggingFace id or local "
                              "directory (the gate baseline)")
    p_merge.add_argument("--model-b", required=True,
                         help="second base model: HuggingFace id or local "
                              "directory (must share A's base + tokenizer)")
    p_merge.add_argument("--method", default="linear",
                         choices=["linear", "slerp"],
                         help="blend method (default linear)")
    p_merge.add_argument("--t", type=float, default=0.5,
                         help="blend weight toward model B, in [0, 1] "
                              "(default 0.5)")
    p_merge.add_argument("--eval", required=True,
                         help="eval file (JSON, format v1) with prompts + "
                              "scorers")
    p_merge.add_argument("--out-dir", default="wsp-merge-out",
                         help="where the merged model, merge.json and "
                              "report.json go")
    p_merge.add_argument("--dtype", default="bfloat16",
                         help="weight dtype to load (default bfloat16)")
    p_merge.add_argument("--samples-per-prompt", type=int, default=None,
                         help="samples at non-greedy temperatures "
                              "(default: 1 for v1, 3 for v2)")
    p_merge.add_argument("--seed", type=int, default=0,
                         help="base sampling seed (default 0)")
    p_merge.add_argument("--force", action="store_true",
                         help="allow an out-dir that already holds model "
                              "files")
    p_merge.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                         help="evaluation protocol: v2 is standard "
                              "evidence (default); v1 is the frozen "
                              "measurement, labelled QUICK")
    p_merge.set_defaults(func=cmd_merge)
    p_watch = sub.add_parser(
        "watch",
        help="watch a directory for new eval files; run the gated "
             "pipeline on each (never applies edits to a live model)")
    p_watch.add_argument("--model", required=True,
                         help="HuggingFace model id to load per run")
    p_watch.add_argument("--watch-dir", required=True,
                         help="directory polled for new .json eval files")
    p_watch.add_argument("--out-dir", default="wsp-watch-out",
                         help="where each run's timestamped directory goes")
    p_watch.add_argument("--interval", type=int, default=60,
                         help="poll interval in seconds (default 60)")
    p_watch.add_argument("--once", action="store_true",
                         help="process pending files once, then exit")
    p_watch.add_argument("--dtype", default="bfloat16")
    p_watch.add_argument("--samples-per-prompt", type=int, default=None)
    p_watch.add_argument("--seed", type=int, default=0)
    p_watch.add_argument("--top-columns", type=int, default=64)
    p_watch.add_argument("--max-attempts", type=int, default=3)
    p_watch.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                         help="evaluation protocol (default v2)")
    p_watch.set_defaults(func=cmd_watch)
    p_loop = sub.add_parser(
        "loop",
        help="keep improving in the background: new eval files and "
             "new failures.jsonl lines trigger gated runs; gate passes "
             "become new versions")
    p_loop.add_argument("--model", required=True,
                        help="HuggingFace model id to load per run")
    p_loop.add_argument("--watch-dir", required=True,
                        help="directory polled for new .json eval files "
                             "and failures.jsonl")
    p_loop.add_argument("--out-dir", default="wsp-loop-out",
                        help="where each run's timestamped directory, "
                             "versions.json and latest.json go")
    p_loop.add_argument("--interval", type=int, default=60,
                        help="poll interval in seconds (default 60)")
    p_loop.add_argument("--once", action="store_true",
                        help="process pending items once, then exit")
    p_loop.add_argument("--dtype", default="bfloat16")
    p_loop.add_argument("--samples-per-prompt", type=int, default=None)
    p_loop.add_argument("--seed", type=int, default=0)
    p_loop.add_argument("--top-columns", type=int, default=64)
    p_loop.add_argument("--max-attempts", type=int, default=3)
    p_loop.add_argument("--protocol", choices=["v1", "v2"], default="v2",
                        help="evaluation protocol (default v2)")
    p_loop.set_defaults(func=cmd_loop)
    p_smodel = sub.add_parser(
        "serve-model",
        help="serve one model and apply new passing edits to it live "
             "(same model, weights overridden in place)")
    p_smodel.add_argument("--model", required=True,
                          help="HuggingFace model id to load once")
    p_smodel.add_argument("--versions-dir", required=True,
                          help="the loop output directory watched for "
                               "latest.json")
    p_smodel.add_argument("--port", type=int, default=8900,
                          help="localhost port (default 8900)")
    p_smodel.add_argument("--interval", type=int, default=30,
                          help="poll interval in seconds for new passing "
                               "edits (default 30)")
    p_smodel.add_argument("--device", default="cpu",
                          help="torch device (default cpu)")
    p_smodel.add_argument("--dtype", default="bfloat16")
    p_smodel.add_argument("--token", default=None,
                          help="bearer token required for POST /generate "
                               "and /rollback (or set WSP_TOKEN)")
    p_smodel.add_argument("--allow-quick", action="store_true",
                          help="accept reports with no SHIP verdict "
                               "(QUICK evidence, incl. every v1 report)")
    p_smodel.set_defaults(func=cmd_serve_model)
    p_ver = sub.add_parser("version", help="print the kit version")
    p_ver.set_defaults(func=cmd_version)
    p_status = sub.add_parser(
        "status",
        help="one line: chip, approved improvements, best gain")
    p_status.add_argument("--runs-dir", default=".",
                          help="directory scanned for report.json files "
                               "(default: current directory)")
    p_status.set_defaults(func=cmd_status)
    p_serve = sub.add_parser(
        "serve", help="the status page on localhost until Ctrl+C")
    p_serve.add_argument("--runs-dir", default=".",
                         help="directory scanned for report.json files "
                              "(default: current directory)")
    p_serve.add_argument("--port", type=int, default=8899,
                         help="localhost port (default 8899)")
    p_serve.set_defaults(func=cmd_serve)
    p_delta = sub.add_parser(
        "delta",
        help="list, show and apply saved weight deltas "
             "(one verified edit each)")
    p_delta_sub = p_delta.add_subparsers(dest="delta_command",
                                         required=True)
    p_dlist = p_delta_sub.add_parser(
        "list", help="list saved deltas with proof summaries")
    p_dlist.add_argument("--store", default="deltas",
                         help="deltas directory (default: deltas)")
    p_dlist.set_defaults(func=cmd_delta_list)
    p_dshow = p_delta_sub.add_parser(
        "show", help="print the full proof of one delta")
    p_dshow.add_argument("delta_id",
                         help="delta id, e.g. "
                              "dlt-20261111T093012Z-4f2a9c1e")
    p_dshow.add_argument("--store", default="deltas",
                         help="deltas directory (default: deltas)")
    p_dshow.set_defaults(func=cmd_delta_show)
    p_dapply = p_delta_sub.add_parser(
        "apply",
        help="apply a delta to a NEW copy of a model "
             "(never overwrites; local gate check first)")
    p_dapply.add_argument("delta_id",
                          help="delta id, e.g. "
                               "dlt-20261111T093012Z-4f2a9c1e")
    p_dapply.add_argument("--model", required=True,
                          help=".npz weights file, local model "
                               "directory, or HuggingFace id")
    p_dapply.add_argument("--out", required=True,
                          help="where the new copy goes "
                               "(must not exist yet)")
    p_dapply.add_argument("--store", default="deltas",
                          help="deltas directory (default: deltas)")
    p_dapply.set_defaults(func=cmd_delta_apply)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
