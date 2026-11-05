"""Command line: `wsp demo` runs the kit pipeline end to end.

The demo prints every number it measures and writes two files into
the output directory: report.json and kit_edited.npz. The edited
weights are always a separate file — the kit never overwrites its
input, and here the input is generated in memory, so there is nothing
to overwrite in the first place.
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
    if rep["gate"]["passed"]:
        print(f"  failing {rep['after']['failing'] - rep['before']['failing']:+.2f} points, "
              f"good {rep['after']['good'] - rep['before']['good']:+.2f} points, "
              f"neighbours {rep['after']['neighbours'] - rep['before']['neighbours']:+.2f} points")
        print(f"  {rep['weights_changed']} weights ({rep['weights_changed_pct']:.2f}%)")
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
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2


def cmd_apply_edit(args):
    """`wsp apply-edit`: reproduce a recorded edit from original weights.

    Loads --model fresh, applies the layers/columns/direction/alpha in
    --edit (an edit.json written by `wsp run` on a gate pass), and
    saves the edited model to --out-dir. No eval, no gate — this is
    the reproducibility script the skill requires.
    """
    import json

    try:
        from .real_edit import apply_column_edit, check_cap
        from .real_runner import RealModel
    except ImportError as exc:
        print(exc)
        return 2
    with open(args.edit) as f:
        recipe = json.load(f)
    if recipe.get("format_version") != 1:
        print(f"error: unsupported edit.json format_version "
              f"{recipe.get('format_version')!r}")
        return 2
    model = RealModel(recipe["model_id"], dtype=recipe.get("dtype", "bfloat16"))
    total_changed = 0
    for layer in recipe["layers"]:
        idx = layer["index"]
        cols = layer["columns"]
        direction = np.array(layer["direction"], dtype=np.float32)
        proj = model._down_projs[idx]
        info = model._infos_by_index[idx]
        intended = len(cols) * int(info["hidden_size"])
        check_cap(intended, model.total_params)
        new, changed = apply_column_edit(proj.weight.data, cols, direction,
                                         layer["alpha"],
                                         layout=info["weight_layout"])
        import torch

        with torch.no_grad():
            proj.weight.data.copy_(new)
        total_changed += changed
    model.save(args.out_dir)
    print(f"applied {total_changed} weights from {args.edit}")
    print(f"saved edited model to {args.out_dir}")
    return 0


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
    p_run.add_argument("--samples-per-prompt", type=int, default=1,
                       help="samples at non-greedy temperatures (default 1)")
    p_run.add_argument("--seed", type=int, default=0,
                       help="base sampling seed (default 0)")
    p_run.add_argument("--top-columns", type=int, default=64,
                       help="columns edited per layer (default 64)")
    p_run.add_argument("--max-attempts", type=int, default=3,
                       help="edit attempts, max 3 per the skill (default 3)")
    p_run.set_defaults(func=cmd_run)
    p_apply = sub.add_parser(
        "apply-edit",
        help="reproduce a recorded edit from original weights")
    p_apply.add_argument("--model", required=True,
                         help="original HuggingFace model id")
    p_apply.add_argument("--edit", required=True,
                         help="edit.json written by `wsp run` on a gate pass")
    p_apply.add_argument("--out-dir", required=True,
                         help="where the edited model goes")
    p_apply.set_defaults(func=cmd_apply_edit)
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
    p_watch.add_argument("--samples-per-prompt", type=int, default=1)
    p_watch.add_argument("--seed", type=int, default=0)
    p_watch.add_argument("--top-columns", type=int, default=64)
    p_watch.add_argument("--max-attempts", type=int, default=3)
    p_watch.set_defaults(func=cmd_watch)
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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
