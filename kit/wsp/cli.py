"""Command line: `wsp demo` runs the toy pipeline end to end.

The demo prints every number it measures and writes two files into
the output directory: report.json and toy_edited.npz. The edited
weights are always a separate file — the kit never overwrites its
input, and here the input is generated in memory, so there is nothing
to overwrite in the first place.
"""

import argparse
import json
import os
import sys

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
            f"({a['weights_changed_pct']:.2%}) — passed: {a['passed']}"
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
        weights.save_weights(os.path.join(args.out_dir, "toy_edited.npz"), edited)
    print(f"wrote {args.out_dir}/report.json")
    if edited is not None:
        print(f"wrote {args.out_dir}/toy_edited.npz")
    return code


def cmd_version(_args):
    print(f"wsp-kit {__version__}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wsp",
        description="Weight-space programming, runnable. See the kit README for the full story.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_demo = sub.add_parser("demo", help="run the toy pipeline end to end")
    p_demo.add_argument("--out-dir", default="wsp-demo-out",
                        help="where report.json and toy_edited.npz go")
    p_demo.set_defaults(func=cmd_demo)
    p_ver = sub.add_parser("version", help="print the kit version")
    p_ver.set_defaults(func=cmd_version)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
