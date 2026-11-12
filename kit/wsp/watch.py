"""Opt-in watch mode: monitor a directory for new eval files and run
the gated pipeline on each one.

The user starts it (``wsp watch``) and stops it (Ctrl+C). Each new or
changed ``.json`` eval file in the watched directory triggers ONE run
of the same gated pipeline ``wsp run`` uses, writing to its own
timestamped output directory under the watch output directory, with
its own report. Each version of a file is processed once; editing a
file (which changes its size or mtime) re-triggers it.

Safety, stated plainly because it matters: watch mode NEVER applies
an edit to a live or production model. It only PRODUCES artifacts —
an edited-weights directory plus a report on a gate pass, a report
alone on a failure. Applying an edit is always the user's deliberate
step (``wsp apply-edit`` with an edit.json they chose). Watch mode
never pushes, never posts, and never touches the source model files.

A bad eval file does not kill the watch: the error is logged, the
file is marked processed for its current version, and watching
continues. Fix the file (edit it) to re-trigger.
"""

import json
import os
import time
from datetime import datetime, timezone

SEEN_FILE = ".watch-seen.json"


def _fingerprint(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def scan_watch_dir(watch_dir, seen):
    """Eval files that are new or changed since ``seen``.

    ``seen`` maps filename -> {"size", "mtime"}. Returns sorted paths
    for ``.json`` files whose fingerprint differs from ``seen``.
    """
    new = []
    for name in sorted(os.listdir(watch_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(watch_dir, name)
        if not os.path.isfile(path):
            continue
        fp = _fingerprint(path)
        if seen.get(name) != fp:
            new.append(path)
    return new


def load_seen(out_dir):
    """The persisted seen-manifest, or {} on first run."""
    path = os.path.join(out_dir, SEEN_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_seen(out_dir, seen):
    from .atomic import atomic_write_json

    os.makedirs(out_dir, exist_ok=True)
    atomic_write_json(os.path.join(out_dir, SEEN_FILE), seen)


def run_once(model_id, eval_path, out_dir, log=print, **pipeline_kwargs):
    """One gated run for one eval file. Returns (report, run_dir, code).

    ``run_dir`` is out_dir/<eval-stem>-<UTC timestamp>. Pipeline
    failures (gate fail, cheap fix, non-causal) are normal outcomes
    and return their report with code 1. Unexpected errors return
    (None, run_dir, 2) with the error written to run_dir/error.txt.
    """
    stem = os.path.splitext(os.path.basename(eval_path))[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(out_dir, f"{stem}-{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    from .real_pipeline import run_real_pipeline

    try:
        rep, _dir, code = run_real_pipeline(
            model_id, eval_path, run_dir, log=log, **pipeline_kwargs)
        log(f"  run finished: {run_dir} (exit {code})")
        return rep, run_dir, code
    except Exception as exc:  # per-file error must not kill the watch
        with open(os.path.join(run_dir, "error.txt"), "w") as f:
            f.write(f"{type(exc).__name__}: {exc}\n")
        log(f"  run failed: {exc} (see {run_dir}/error.txt)")
        return None, run_dir, 2


def watch_loop(model_id, watch_dir, out_dir, interval=60, log=print,
               once=False, **pipeline_kwargs):
    """Poll ``watch_dir`` forever (or once) and run new eval files.

    Returns 0 on a clean stop (Ctrl+C or --once completing). Raises
    ValueError if the watch directory does not exist.
    """
    if not os.path.isdir(watch_dir):
        raise ValueError(f"watch directory not found: {watch_dir}")
    os.makedirs(out_dir, exist_ok=True)
    seen = load_seen(out_dir)
    log(f"watching {watch_dir} every {interval}s "
        f"(model {model_id}; Ctrl+C to stop)")
    try:
        while True:
            for path in scan_watch_dir(watch_dir, seen):
                log(f"new eval file: {path}")
                run_once(model_id, path, out_dir, log=log,
                         **pipeline_kwargs)
                # Mark processed for its current version, whatever
                # the outcome — editing the file re-triggers it. A
                # file deleted mid-run simply drops out of `seen`.
                try:
                    seen[os.path.basename(path)] = _fingerprint(path)
                except OSError:
                    seen.pop(os.path.basename(path), None)
                save_seen(out_dir, seen)
            if once:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        log("watch stopped by user")
        return 0
