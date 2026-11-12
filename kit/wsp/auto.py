"""`wsp auto`: easy mode — 4 questions, then the kit runs by itself.

The user answers in plain words; the kit builds the eval file, runs
the gated pipeline, and shows the result. Nothing is invented: every
prompt comes from the user's answers. If the user presses Enter on
the neighbours question, the good prompts are reused as neighbours
and the kit says so plainly.

The eval file uses the "contains" scorer for all three sets: each
prompt carries what the right answer should contain. A failing
prompt is one where the model does NOT produce it; a good prompt is
one where it does. Neighbours are nearby prompts that must not
change. The gate works unchanged: failing +2, guards/neighbours -2.

`wsp auto` never runs on a model the user did not point it at, and
it never overwrites the original model — on a gate pass the edited
model is a separate artifact in the output directory, as always.
"""

import json
import os
import re

from .evalfile import load_eval_file, validate_eval


def slugify_model(model_id):
    """A filesystem-safe name from a model id, e.g. Qwen/Qwen3-0.6B."""
    slug = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")
    return slug or "model"


def build_eval(model_id, name, failing, good, neighbours=None):
    """Build a format-v1 eval dict from (prompt, expect) pairs.

    ``failing``, ``good`` and ``neighbours`` are lists of
    (prompt, expect) tuples. ``neighbours=None`` reuses the good
    prompts. Raises ValueError on an empty prompt or a missing set —
    the eval is validated before anything runs.
    """
    if neighbours is None:
        neighbours = list(good)

    def _set(pairs):
        return {
            "scorer": "contains",
            "prompts": [
                {"prompt": p, "expect": e} for p, e in pairs
            ],
        }

    data = {
        "format_version": 1,
        "name": name,
        "model": model_id,
        "serving_temperature": 0.7,
        "gate_temperatures": [0.7, 0.0],
        "max_new_tokens": 192,
        "use_chat_template": True,
        "sets": {
            "failing": _set(failing),
            "good": _set(good),
            "neighbours": _set(neighbours),
        },
    }
    return validate_eval(data, "<wsp auto>")


def _ask_pairs(input_fn, print_fn, heading, max_n):
    """Collect up to max_n (prompt, expect) pairs. Empty prompt ends."""
    print_fn(heading)
    pairs = []
    while len(pairs) < max_n:
        prompt = input_fn(
            f"  prompt {len(pairs) + 1} (empty line to finish): ").strip()
        if not prompt:
            break
        expect = input_fn("    what should the answer contain? ").strip()
        if not expect:
            print_fn("    (needs an expected answer — skipped)")
            continue
        pairs.append((prompt, expect))
    return pairs


def ask_questions(input_fn=input, print_fn=print):
    """Ask the 4 questions. Returns (model, failing, good, neighbours).

    Raises EOFError/KeyboardInterrupt when the user cancels — the
    caller turns that into a clean stop, not a traceback.
    """
    print_fn("wsp auto — 4 questions, then the kit runs by itself.")
    print_fn("Answer in plain words. Nothing is invented: every test")
    print_fn("prompt comes from your answers.")
    model = ""
    while not model:
        model = input_fn(
            "1. Which model? (HuggingFace id, e.g. Qwen/Qwen3-0.6B): "
        ).strip()
        if not model:
            print_fn("   (a model id is required)")
    failing = _ask_pairs(
        input_fn, print_fn,
        "2. What does the model do WRONG? Give 1-3 example prompts.",
        3)
    while not failing:
        print_fn("   (at least one failing prompt is required)")
        failing = _ask_pairs(
            input_fn, print_fn,
            "2. What does the model do WRONG? Give 1-3 example prompts.",
            3)
    good = _ask_pairs(
        input_fn, print_fn,
        "3. Where is the model RIGHT? Give 1-2 prompts it gets right.",
        2)
    while not good:
        print_fn("   (at least one good prompt is required)")
        good = _ask_pairs(
            input_fn, print_fn,
            "3. Where is the model RIGHT? Give 1-2 prompts it gets right.",
            2)
    neighbours = _ask_pairs(
        input_fn, print_fn,
        "4. Similar prompts that should NOT change? "
        "(Enter to reuse the good ones)",
        2)
    if not neighbours:
        neighbours = list(good)
        print_fn("   (neighbours: reusing the good prompts)")
    return model, failing, good, neighbours


def run_auto(model_id=None, eval_path=None, out_dir="wsp-auto-out",
             serve=False, input_fn=input, print_fn=print, log=print,
             protocol="v1"):
    """Easy mode end to end. Returns an exit code (0 = gate passed).

    ``protocol`` defaults to v1 (D1): easy mode is a quick personal
    check, labelled QUICK — never standard evidence unless the
    caller asks for v2.
    """
    if eval_path:
        data = load_eval_file(eval_path)
        model_id = data.get("model")
        eval_file = eval_path
    else:
        try:
            model_id, failing, good, neighbours = ask_questions(
                input_fn, print_fn)
        except (EOFError, KeyboardInterrupt):
            print_fn("cancelled — nothing was run.")
            return 2
        name = "auto-" + slugify_model(model_id)
        eval_data = build_eval(model_id, name, failing, good, neighbours)
        os.makedirs(os.path.join(out_dir, "evals"), exist_ok=True)
        eval_file = os.path.join(out_dir, "evals", name + ".json")
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2)
        print_fn(f"wrote {eval_file}")
    try:
        from .real_pipeline import run_real_pipeline
    except ImportError as exc:
        # Missing torch/transformers: print the hint, not a traceback.
        print_fn(exc)
        return 2
    try:
        rep, _dir, code = run_real_pipeline(
            model_id, eval_file, out_dir, log=log, protocol=protocol)
    except (FileNotFoundError, ValueError) as exc:
        print_fn(f"error: {exc}")
        return 2
    except ImportError as exc:
        print_fn(exc)
        return 2
    print_fn()
    print_fn(f"gate passed: {rep['gate']['passed']}")
    if rep["gate"]["passed"]:
        print_fn(f"  failing {rep['after']['failing'] - rep['before']['failing']:+.2f} points, "
              f"good {rep['after']['good'] - rep['before']['good']:+.2f} points, "
              f"neighbours {rep['after']['neighbours'] - rep['before']['neighbours']:+.2f} points")
        print_fn(f"  {rep['weights_changed']} weights ({rep['weights_changed_pct']:.2f}%)")
        # Star ask, gated (D2, W3.4): only after a standard-evidence SHIP.
        from .cli import maybe_nudge

        maybe_nudge(rep, print_fn=print_fn)
    else:
        print_fn(f"  reason: {rep['gate'].get('reason')}")
    print_fn(f"wrote {out_dir}/report.json")
    if serve:
        from .serve import serve
        print_fn("Result page:")
        return serve(runs_dir=out_dir, port=8899)
    return code
