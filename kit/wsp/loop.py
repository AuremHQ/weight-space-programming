"""`wsp loop`: continuous background mode — keep improving, forever.

The kit watches a folder. Two things trigger a run:

1. A new or changed eval file (.json) — same as `wsp watch`.
2. New lines in failures.jsonl — one JSON object per line::

       {"prompt": "...", "expect": "...", "kind": "failing"}
       {"prompt": "...", "expect": "...", "kind": "good"}

   ``kind`` is "failing", "good" or "neighbour" (default "failing").
   The kit builds an eval file from ALL the lines seen so far and
   runs the pipeline on it. So every new failure re-runs the whole
   set — the model is always tested on everything it has failed at.

On a gate pass, the edited model is saved as a NEW version — the run
directory itself, plus a pointer in versions.json / latest.json in
the output directory. The original model is never touched, and no
live model is ever swapped: the user (or their app) loads the latest
passing version when they want it. The status page (`wsp serve`)
shows every run, so the benefit is always visible.

On a gate fail, nothing ships. The failure is logged and marked
processed — the loop does NOT retry the same failure forever. A new
or changed failure (or eval file) re-triggers a run.

Safety: the gate (+5/-2), the 1% weight cap, and the 3B model cap
apply to every run, exactly as in `wsp run`. Max 3 edit attempts per
run (the pipeline's own cap). Stop with Ctrl+C.
"""

import json
import os
import time
from datetime import datetime, timezone

from . import auto, watch

FAILURES_FILE = "failures.jsonl"
VERSIONS_FILE = "versions.json"
LATEST_FILE = "latest.json"
FAILURES_STATE_KEY = "__failures__"


def parse_failures(path):
    """Parse a failures.jsonl file.

    Returns (failing, good, neighbours, skipped) — each of the first
    three a list of (prompt, expect) tuples, ``skipped`` the count of
    lines that were not valid failure objects (bad JSON, or a missing
    prompt/expect). A missing ``kind`` means "failing".
    """
    failing, good, neighbours = [], [], []
    skipped = 0
    if not os.path.isfile(path):
        return failing, good, neighbours, skipped
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if (not isinstance(obj, dict) or not obj.get("prompt")
                    or not obj.get("expect")):
                skipped += 1
                continue
            pair = (obj["prompt"], obj["expect"])
            kind = obj.get("kind", "failing")
            if kind == "good":
                good.append(pair)
            elif kind == "neighbour":
                neighbours.append(pair)
            elif kind == "failing":
                failing.append(pair)
            else:
                skipped += 1
    return failing, good, neighbours, skipped


def _failures_fingerprint(path):
    st = os.stat(path)
    with open(path) as f:
        lines = sum(1 for _ in f)
    return {"size": st.st_size, "mtime": st.st_mtime, "lines": lines}


def new_failure_lines(path, state):
    """New failure lines since ``state``, and the new state entry.

    ``state`` is the stored fingerprint dict (or None). Returns
    (lines, entry) where ``lines`` are the raw new lines and ``entry``
    the fingerprint to store. A missing file returns ([], None). A
    file that shrank (rewritten) is re-read from the start.
    """
    if not os.path.isfile(path):
        return [], None
    entry = _failures_fingerprint(path)
    if not state:
        with open(path) as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()], entry
    if entry["size"] < state.get("size", 0):
        with open(path) as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()], entry
    seen_lines = state.get("lines", 0)
    with open(path) as f:
        all_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return all_lines[seen_lines:], entry


def record_version(out_dir, run_dir, rep, model_id):
    """Append a gate-passing run to versions.json + latest.json.

    Returns the version number (1-based). The version IS the run
    directory — the edited model lives there, as the pipeline wrote
    it. Nothing is copied or moved.
    """
    os.makedirs(out_dir, exist_ok=True)
    versions_path = os.path.join(out_dir, VERSIONS_FILE)
    versions = []
    if os.path.isfile(versions_path):
        try:
            with open(versions_path) as f:
                versions = json.load(f)
            if not isinstance(versions, list):
                versions = []
        except (OSError, ValueError):
            versions = []
    entry = {
        "version": len(versions) + 1,
        "run_dir": run_dir,
        "model": model_id,
        "failing_gain": (rep["after"]["failing"]
                         - rep["before"]["failing"]),
        "weights_changed": rep["weights_changed"],
        "timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    versions.append(entry)
    with open(versions_path, "w") as f:
        json.dump(versions, f, indent=2)
    with open(os.path.join(out_dir, LATEST_FILE), "w") as f:
        json.dump(entry, f, indent=2)
    return entry["version"]


def _default_run_fn(model_id, eval_path, out_dir, log=print):
    return watch.run_once(model_id, eval_path, out_dir, log=log)


def loop_once(model_id, watch_dir, out_dir, state, run_fn=None,
              log=print, pipeline_kwargs=None):
    """One pass: run new eval files and new failures. Returns state.

    ``state`` is the seen-manifest dict (eval fingerprints plus the
    failures entry). ``run_fn`` defaults to ``watch.run_once`` and
    takes (model_id, eval_path, out_dir, log=log), returning
    (report, run_dir, exit_code). ``pipeline_kwargs`` are forwarded
    to the default run_fn only. Returns (state, summary) where
    ``summary`` counts eval runs, failure runs and new versions.
    """
    if run_fn is None:
        run_fn = _default_run_fn
    kwargs = pipeline_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    summary = {"eval_runs": 0, "failure_runs": 0, "versions": 0}

    def _maybe_version(rep, run_dir):
        if rep is not None and rep.get("gate", {}).get("passed"):
            version = record_version(out_dir, run_dir, rep, model_id)
            log(f"  new version v{version}: {run_dir}")
            log("If this helped, please star the repo: "
                "https://github.com/AuremHQ/weight-space-programming")
            summary["versions"] += 1

    for path in watch.scan_watch_dir(watch_dir, state):
        log(f"new eval file: {path}")
        rep, run_dir, _code = run_fn(model_id, path, out_dir, log=log,
                                     **kwargs)
        _maybe_version(rep, run_dir)
        summary["eval_runs"] += 1
        try:
            state[os.path.basename(path)] = watch._fingerprint(path)
        except OSError:
            state.pop(os.path.basename(path), None)
        watch.save_seen(out_dir, state)

    failures_path = os.path.join(watch_dir, FAILURES_FILE)
    new_lines, entry = new_failure_lines(
        failures_path, state.get(FAILURES_STATE_KEY))
    if new_lines:
        failing, good, neighbours, skipped = parse_failures(failures_path)
        if skipped:
            log(f"  failures.jsonl: {skipped} bad line(s) skipped")
        if not failing:
            log("  failures.jsonl: no failing prompts yet — need at "
                "least one failing prompt and one good prompt")
        elif not good:
            log("  failures.jsonl: no good prompts yet — need at least "
                "one failing prompt and one good prompt")
        else:
            name = "loop-" + auto.slugify_model(model_id)
            eval_data = auto.build_eval(model_id, name, failing, good,
                                        neighbours or None)
            os.makedirs(os.path.join(out_dir, "evals"), exist_ok=True)
            eval_file = os.path.join(out_dir, "evals", name + ".json")
            with open(eval_file, "w") as f:
                json.dump(eval_data, f, indent=2)
            log(f"new failures ({len(new_lines)} line(s)) — "
                f"running the pipeline on {len(failing)} failing, "
                f"{len(good)} good, {len(neighbours)} neighbour prompts")
            rep, run_dir, _code = run_fn(model_id, eval_file, out_dir,
                                         log=log, **kwargs)
            _maybe_version(rep, run_dir)
            summary["failure_runs"] += 1
        if entry is not None:
            state[FAILURES_STATE_KEY] = entry
            watch.save_seen(out_dir, state)
    return state, summary


def run_loop(model_id, watch_dir, out_dir, interval=60, once=False,
             run_fn=None, log=print, pipeline_kwargs=None):
    """Watch forever (or once) and keep improving. Returns 0.

    Raises ValueError if the watch directory does not exist.
    """
    if not os.path.isdir(watch_dir):
        raise ValueError(f"watch directory not found: {watch_dir}")
    os.makedirs(out_dir, exist_ok=True)
    state = watch.load_seen(out_dir)
    log(f"loop watching {watch_dir} every {interval}s "
        f"(model {model_id}; Ctrl+C to stop)")
    log("New eval files and new failures.jsonl lines trigger a run. "
        "Gate passes become new versions in versions.json.")
    try:
        while True:
            state, summary = loop_once(model_id, watch_dir, out_dir,
                                       state, run_fn=run_fn, log=log,
                                       pipeline_kwargs=pipeline_kwargs)
            if summary["eval_runs"] or summary["failure_runs"]:
                log(f"pass done: {summary['eval_runs']} eval run(s), "
                    f"{summary['failure_runs']} failure run(s), "
                    f"{summary['versions']} new version(s)")
            if once:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        log("loop stopped by user")
        return 0
