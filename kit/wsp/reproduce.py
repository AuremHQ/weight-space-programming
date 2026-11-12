"""`wsp reproduce`: re-run a recorded run and compare the primary metric.

Reads ``<run-dir>/report.json`` and re-runs the recorded eval with
the recorded model, protocol, seed and sampling, then compares the
fresh run's after.failing (serving temperature) with the recorded
value within ``--tolerance`` points:

- ``MATCH`` — the difference is within tolerance (exit 0).
- ``DIVERGED`` — outside tolerance, with both numbers printed
  (exit 1). The word "reproduced" is never printed for a DIVERGED
  run.
- ``cannot be reproduced exactly: missing <fields>`` — the report
  does not record samples_per_prompt, seed or max_new_tokens
  (every early report has None for all three), so no re-run is
  attempted (exit 2).

The eval file is resolved from the run dir first, then from
``kit/evals/`` (next to this package). The fresh run goes to a new
temp dir unless ``--out-dir`` is given. The recorded model
revision, when present, is printed; the load itself does not pin
a revision (RealModel has no revision pin — same as apply-edit,
which prints it). The sampling seed and samples-per-prompt come
from the report; top_columns is derived from the recorded recipe
(``edit.json`` next to the report, when present), else the
pipeline default.
"""

import json
import os
import tempfile

MISSING_FIELDS = ("samples_per_prompt", "seed", "max_new_tokens")
DEFAULT_TOLERANCE = 5.0


def _kit_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_eval_path(run_dir, eval_file):
    """Find the recorded eval: the run dir first, then kit/evals/."""
    candidates = [
        os.path.join(run_dir, eval_file),
        os.path.join(_kit_dir(), "evals", eval_file),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise ValueError(
        f"eval file {eval_file!r} not found in the run dir or in "
        f"kit/evals/ — cannot re-run this report")


def reproduce_run(run_dir, tolerance=DEFAULT_TOLERANCE, out_dir=None,
                  model=None, log=print):
    """Re-run a recorded run; return a result dict (and print it).

    The dict has ``status`` (MATCH / DIVERGED / CANNOT_REPRODUCE /
    ERROR), ``code`` (the CLI exit code), and, for MATCH/DIVERGED,
    ``recorded``, ``fresh`` and ``difference`` (the after.failing
    values at serving temperature). Nothing is raised for an
    expected failure (missing report, missing fields, missing eval,
    a fresh run that errors) — those return ERROR / CANNOT_REPRODUCE
    with the reason printed.
    """
    report_path = os.path.join(run_dir, "report.json")
    if not os.path.isfile(report_path):
        log(f"error: no report.json in {run_dir}")
        return {"status": "ERROR", "code": 2,
                "reason": "no report.json in the run dir"}
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, ValueError) as exc:
        log(f"error: report.json is unreadable: {exc}")
        return {"status": "ERROR", "code": 2,
                "reason": f"report.json is unreadable: {exc}"}
    demo = report.get("real_demo") or {}
    model_id = demo.get("model_id")
    eval_file = demo.get("eval_file")
    if not model_id or not eval_file:
        log("error: the report does not record a model_id and an "
            "eval_file — cannot re-run it")
        return {"status": "ERROR", "code": 2,
                "reason": "report is missing model_id or eval_file"}
    missing = [f for f in MISSING_FIELDS if demo.get(f) is None]
    if missing:
        log("cannot be reproduced exactly: missing "
            + ", ".join(missing))
        return {"status": "CANNOT_REPRODUCE", "code": 2,
                "missing": missing}
    try:
        eval_path = resolve_eval_path(run_dir, eval_file)
    except ValueError as exc:
        log(f"error: {exc}")
        return {"status": "ERROR", "code": 2, "reason": str(exc)}
    try:
        with open(eval_path) as f:
            eval_data = json.load(f)
    except (OSError, ValueError) as exc:
        log(f"error: the eval file is unreadable: {exc}")
        return {"status": "ERROR", "code": 2,
                "reason": f"eval file is unreadable: {exc}"}
    eval_max = eval_data.get("max_new_tokens", 192)
    if eval_max != demo["max_new_tokens"]:
        log("cannot be reproduced exactly: max_new_tokens differs "
            f"(report says {demo['max_new_tokens']}, the eval file "
            f"says {eval_max})")
        return {"status": "CANNOT_REPRODUCE", "code": 2,
                "missing": ["max_new_tokens"]}
    protocol = report.get("protocol") or "v1"
    revision = report.get("model_revision")
    log(f"reproducing {model_id} "
        f"(revision: {revision if revision else 'none recorded'}, "
        f"protocol {protocol}, seed {demo['seed']}, "
        f"{demo['samples_per_prompt']} samples per prompt)")
    # top_columns from the recorded recipe when there is one: the
    # recipe's widest layer is the columns-per-layer the run used.
    top_columns = 64
    edit_path = os.path.join(run_dir, "edit.json")
    if os.path.isfile(edit_path):
        try:
            with open(edit_path) as f:
                edit = json.load(f)
            top_columns = max(
                (len(layer.get("columns", []))
                 for layer in edit.get("layers", [])),
                default=64)
        except (OSError, ValueError):
            pass
    max_attempts = report.get("attempts_used") or 3
    fresh_dir = out_dir or tempfile.mkdtemp(prefix="wsp-reproduce-")
    from .real_pipeline import run_real_pipeline

    try:
        fresh, _out, _code = run_real_pipeline(
            model_id=model_id, eval_path=eval_path,
            out_dir=fresh_dir, model=model, protocol=protocol,
            samples_per_prompt=demo["samples_per_prompt"],
            seed=demo["seed"], top_columns=top_columns,
            max_attempts=max_attempts,
            dtype=demo.get("dtype", "bfloat16"),
            log=lambda *a: None)
    except (ValueError, FileNotFoundError) as exc:
        log(f"error: the fresh run failed: {exc}")
        return {"status": "ERROR", "code": 2,
                "reason": f"the fresh run failed: {exc}"}
    serving = demo.get("serving_temperature")
    if serving is None:
        serving = (eval_data.get("serving_temperature", 0.7))
    recorded = report.get("after", {}).get("failing")
    fresh_after = fresh.get("after", {}).get("failing")
    if recorded is None or fresh_after is None:
        log("error: no failing after-score to compare (the recorded "
            "or the fresh report has no after.failing)")
        return {"status": "ERROR", "code": 2,
                "reason": "no after.failing to compare"}
    difference = abs(float(fresh_after) - float(recorded))
    if difference <= tolerance:
        log(f"MATCH: primary metric failing after (serving "
            f"{serving}): recorded {recorded:.2f}, fresh "
            f"{fresh_after:.2f} (difference {difference:.2f} <= "
            f"tolerance {tolerance}) — fresh run in {fresh_dir}")
        return {"status": "MATCH", "code": 0, "recorded": recorded,
                "fresh": fresh_after, "difference": difference,
                "fresh_dir": fresh_dir}
    log(f"DIVERGED: primary metric failing after (serving "
        f"{serving}): recorded {recorded:.2f}, fresh "
        f"{fresh_after:.2f} (difference {difference:.2f} > "
        f"tolerance {tolerance}) — fresh run in {fresh_dir}")
    return {"status": "DIVERGED", "code": 1, "recorded": recorded,
            "fresh": fresh_after, "difference": difference,
            "fresh_dir": fresh_dir}
