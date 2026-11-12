"""`wsp serve-model`: the same model, with its weights overridden live.

The kit's loop keeps improving the model in the background and writes
each gate-passing edit as an ``edit.json`` in a run directory, with
``latest.json`` pointing at the newest passing run. This server loads
the model ONCE and serves it — and whenever a new passing edit
appears, it applies the edit to the LIVE loaded model, in place:

- the old weights are overridden by the updated ones, in the same
  model object — no reload, no restart, no new model to swap in;
- the next prompt automatically uses the updated weights;
- only ONE edit is ever active: when a newer passing edit arrives,
  the active edit's saved original columns are restored first
  (bitwise), then only the new recipe is applied — edits replace,
  they never stack. Rollback restores the original weights; it does
  NOT reinstate the previous version.

A version is applied only when it proves itself (see poll_once):
the edit.json must validate against the loaded model, the recipe
must be inside the 1% cap, and a report.json must sit next to it
showing a gate pass for this exact edit (same edit hash, same model
id). A report with a SHIP verdict is full evidence; a report with
no verdict (every v1 report) is QUICK evidence and is accepted only
with --allow-quick. An edit that fails any of this is logged and
recorded in status()["skipped"] — the server keeps serving.

Rollback sticks: a rolled-back version is never re-applied. The
high-water mark only ever moves forward; poll_once applies only a
version above it. The run directory is resolved inside the versions
directory (run_dir_rel first), so a traversal, an absolute path
outside, or a symlink escaping the directory is refused.

Endpoints (127.0.0.1 only — this is a local server, not a public
service; see wsp.localhttp for the request checks):
  GET  /          short help
  GET  /status    model, current version, applied edits, weights changed
  POST /generate  {"prompt": "...", "max_new_tokens": 64,
                   "temperature": 0} -> {"text": "..."}
  POST /rollback  undo the last applied edit, exactly

Generation and edits never run at the same time: one lock guards
both, so an edit never lands mid-generation. /status never takes
the lock, so it answers even while a generation is running. Stop
with Ctrl+C.
"""

import json
import os
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

from . import localhttp
from .recipe import (load_recipe, validate_against_model,
                     verify_provenance)

LATEST_FILE = "latest.json"


def apply_recipe(model, recipe, log=print):
    """Apply one edit.json recipe to a loaded model, in place.

    Validates the recipe against the loaded model first (layer
    indices, column ranges, direction lengths / dampen factors,
    finite numbers, and the recipe-level 1% cap), then applies it
    layer by layer: subtract layers call ``model.apply_edit``,
    dampen layers call ``model.dampen_columns``. Returns (changed,
    saved) — total weights actually changed and the list of
    per-layer saved handles for rollback. Raises ValueError if any
    layer breaks the cap or validation (the model is left untouched
    for that layer; earlier layers of the same recipe are restored
    first).
    """
    import numpy as np

    validate_against_model(recipe, model.layer_infos,
                           model.total_params)
    changed = 0
    saved = []
    try:
        for layer in recipe["layers"]:
            if recipe.get("edit_type", "subtract") == "dampen":
                n, handle = model.dampen_columns(
                    layer["index"], layer["columns"],
                    layer["dampen_factor"])
            else:
                direction = np.array(layer["direction"],
                                     dtype=np.float32)
                n, handle = model.apply_edit(
                    layer["index"], layer["columns"], direction,
                    layer["alpha"])
            changed += n
            saved.append(handle)
    except Exception:
        # A failed layer must not leave a half-applied recipe behind.
        for handle in reversed(saved):
            model.restore(handle)
        raise
    return changed, saved


def rollback(model, saved, log=print):
    """Restore saved columns, last edit first. Returns nothing."""
    for handle in reversed(saved):
        model.restore(handle)


def resolve_run_dir(versions_dir, latest):
    """Resolve the run directory for a version entry.

    run_dir_rel (relative to versions_dir) wins when present;
    otherwise an absolute run_dir that exists; otherwise a legacy
    relative run_dir joined to versions_dir. The resolved realpath
    MUST be inside the versions_dir realpath — a traversal, an
    absolute path outside, or a symlink escaping the directory
    raises ValueError.
    """
    versions_real = os.path.realpath(versions_dir)
    rel = latest.get("run_dir_rel")
    if rel:
        candidate = os.path.realpath(os.path.join(versions_dir, rel))
    else:
        run_dir = latest.get("run_dir", "")
        if os.path.isabs(run_dir) and os.path.exists(run_dir):
            candidate = os.path.realpath(run_dir)
        else:
            candidate = os.path.realpath(
                os.path.join(versions_dir, run_dir))
    try:
        inside = (os.path.commonpath([candidate, versions_real])
                  == versions_real)
    except ValueError:
        inside = False  # different drive on Windows: outside
    if not inside:
        raise ValueError(
            f"run_dir {candidate!r} is outside the versions "
            f"directory {versions_real!r} — refused")
    return candidate


class LiveModel:
    """One loaded model, its applied edits, and the update watcher.

    ``model`` is a RealModel (or a test double with the same
    apply_edit / restore / generate methods, plus layer_infos and
    total_params). ``model_id`` must match the model_id in every
    edit.json applied. ``allow_quick`` accepts QUICK evidence
    (reports with no SHIP verdict, including every v1 report);
    without it only SHIP reports are applied.

    ``applied`` and ``_saved_stack`` hold the ACTIVE edit only
    (depth 0 or 1); ``history`` is the full record of applied
    edits. ``last_version`` is the newest version seen; the
    ``high_water`` mark only ever moves forward, so a rolled-back
    version is never re-applied.
    """

    def __init__(self, model, model_id, log=print, allow_quick=False):
        self.model = model
        self.model_id = model_id
        self.log = log
        self.allow_quick = allow_quick
        self.lock = threading.Lock()
        self.applied = []  # [{version, run_dir, weights_changed, ...}]
        self._saved_stack = []  # per-edit saved handles, same order
        self.history = []  # every applied edit, in order
        self.skipped = []  # [{version, reason, run_dir}]
        self.last_version = 0
        self.high_water = 0
        self.rolled_back = []  # versions rolled back, in order
        self._active_recipe = None

    def generate(self, prompt, max_new_tokens=64, temperature=0,
                 seed=0):
        with self.lock:
            text, _finished, _n = self.model.generate(
                prompt, max_new_tokens, temperature, seed)
        return text

    def status(self):
        # Deliberately lock-free: /status must answer even while a
        # generation is running (the lock is held by generate).
        return {
            "model": self.model_id,
            "current_version": self.last_version,
            "applied_edits": len(self.applied),
            "weights_changed": sum(e["weights_changed"]
                                   for e in self.applied),
            "edits": list(self.applied),
            "active_edit": self.applied[0] if self.applied else None,
            "history": list(self.history),
            "high_water_version": self.high_water,
            "rolled_back_versions": list(self.rolled_back),
            "skipped": list(self.skipped),
        }

    def _seen(self, version):
        self.last_version = version
        self.high_water = max(self.high_water, version)

    def _skip(self, version, reason, run_dir):
        self.skipped.append({"version": version, "reason": reason,
                             "run_dir": run_dir})
        self.log(f"  version {version}: skipped — {reason} "
                 f"({run_dir})")

    def poll_once(self, versions_dir):
        """Apply the newest passing edit, if there is one.

        Returns True if an edit was applied. A version that fails
        any check is logged, recorded in status()["skipped"], and
        marked seen — the server keeps serving the current weights
        and never retries it.
        """
        latest_path = os.path.join(versions_dir, LATEST_FILE)
        if not os.path.isfile(latest_path):
            return False
        try:
            with open(latest_path) as f:
                latest = json.load(f)
        except (OSError, ValueError) as exc:
            self.log(f"  latest.json unreadable: {exc}")
            return False
        version = latest.get("version", 0)
        if version <= self.high_water:
            return False
        try:
            run_dir = resolve_run_dir(versions_dir, latest)
        except ValueError as exc:
            self._skip(version, str(exc), None)
            self._seen(version)
            return False
        edit_path = os.path.join(run_dir, "edit.json")
        if not os.path.isfile(edit_path):
            self._skip(version, f"no edit.json in {run_dir}", run_dir)
            self._seen(version)
            return False
        try:
            recipe = load_recipe(edit_path)
        except (OSError, ValueError) as exc:
            self._skip(version, f"bad edit.json: {exc}", run_dir)
            self._seen(version)
            return False
        if recipe["model_id"] != self.model_id:
            self._skip(version,
                       f"edit is for {recipe['model_id']}, this "
                       f"server runs {self.model_id}", run_dir)
            self._seen(version)
            return False
        report = None
        report_path = os.path.join(run_dir, "report.json")
        if os.path.isfile(report_path):
            try:
                with open(report_path) as f:
                    report = json.load(f)
            except (OSError, ValueError) as exc:
                self._skip(version, f"report.json unreadable: {exc}",
                           run_dir)
                self._seen(version)
                return False
        try:
            evidence = verify_provenance(recipe, report)
        except ValueError as exc:
            self._skip(version, str(exc), run_dir)
            self._seen(version)
            return False
        if evidence == "quick" and not self.allow_quick:
            self._skip(version,
                       "QUICK evidence only (no SHIP verdict) — "
                       "refused without --allow-quick", run_dir)
            self._seen(version)
            return False
        with self.lock:
            old_saved = old_recipe = old_edit = None
            if self._active_recipe is not None:
                old_saved = (self._saved_stack[0]
                             if self._saved_stack else None)
                old_recipe = self._active_recipe
                old_edit = self.applied[0] if self.applied else None
                # Replace, not stack: restore the active edit's
                # saved original columns before the new recipe.
                rollback(self.model, old_saved, log=self.log)
            try:
                changed, saved = apply_recipe(self.model, recipe,
                                              log=self.log)
            except Exception as exc:
                if old_recipe is not None:
                    try:
                        _n, saved_old = apply_recipe(
                            self.model, old_recipe, log=self.log)
                        self._saved_stack = [saved_old]
                        self.applied = [old_edit]
                        self._active_recipe = old_recipe
                        reason = (f"apply failed: {exc} — the "
                                  f"previous edit (v{old_edit['version']}) "
                                  "was restored")
                    except Exception as exc2:
                        self._saved_stack = []
                        self.applied = []
                        self._active_recipe = None
                        reason = (f"apply failed: {exc}; re-apply of "
                                  "the previous edit also failed: "
                                  f"{exc2} — the model is back to its "
                                  "original weights")
                else:
                    reason = (f"apply failed: {exc} — kept the old "
                              "weights")
                self._skip(version, reason, run_dir)
                self._seen(version)
                return False
            self._saved_stack = [saved]
            self.applied = [{
                "version": version,
                "run_dir": run_dir,
                "weights_changed": changed,
                "evidence": evidence,
                "at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            }]
            self._active_recipe = recipe
            self.history.append(self.applied[0])
            self._seen(version)
        self.log(f"  live override: version {version} applied, "
                 f"{changed} weights changed — the same model now "
                 f"uses the updated weights")
        return True

    def rollback_last(self):
        """Undo the active edit, exactly. Returns True/False.

        The high-water mark is NOT lowered: the rolled-back version
        is never re-applied, and rollback does not reinstate any
        earlier version.
        """
        with self.lock:
            if not self._saved_stack:
                return False
            saved = self._saved_stack.pop()
            rollback(self.model, saved, log=self.log)
            edit = self.applied.pop()
            self._active_recipe = None
            self.last_version = 0
            self.rolled_back.append(edit["version"])
        self.log(f"  rolled back version {edit['version']}")
        return True


def make_handler(live, token=None):
    # The class body below cannot read the `token` parameter by the
    # same name (a name assigned in a class body is class-local), so
    # capture it under a different name first.
    _token = token

    class ModelHandler(localhttp.LocalHandler):
        token = _token

        def do_GET(self):
            try:
                self._do_get()
            except Exception as exc:
                self._internal_error(exc)

        def _do_get(self):
            if not self._check_host_origin():
                return
            if self.path == "/status":
                self._send_json(200, live.status())
            else:
                self._send_json(200, {
                    "wsp-kit live model": live.model_id,
                    "endpoints": {
                        "GET /status": "current version + applied edits",
                        "POST /generate": (
                            '{"prompt": "...", "max_new_tokens": 64, '
                            '"temperature": 0}'),
                        "POST /rollback": (
                            "undo the last applied edit, exactly"),
                    },
                })

        def do_POST(self):
            try:
                self._do_post()
            except Exception as exc:
                self._internal_error(exc)

        def _do_post(self):
            if not self._check_host_origin():
                return
            if self.path not in ("/generate", "/rollback"):
                self._send_json(404, {"error": "unknown endpoint"})
                return
            if not self._check_token():
                return
            body = self._read_json_body()
            if body is None:
                return
            if self.path == "/generate":
                params, err = self._validate_generate(body)
                if err is not None:
                    self._send_json(err[0], {"error": err[1]})
                    return
                text = live.generate(
                    params["prompt"],
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    seed=params["seed"])
                self._send_json(200, {"text": text})
            elif self.path == "/rollback":
                if live.rollback_last():
                    self._send_json(200, {"rolled_back": True})
                else:
                    self._send_json(200, {"rolled_back": False,
                                          "reason": "no edit applied yet"})

    return ModelHandler


def serve_model(model_id, versions_dir, port=8900, interval=30,
                device="cpu", dtype="bfloat16", model=None, log=print,
                token=None, allow_quick=False):
    """Load the model once, serve it, and apply new edits live.

    ``model`` is an injected loaded model (tests); otherwise a
    RealModel is loaded here (needs torch + transformers). The
    watcher thread polls ``versions_dir/latest.json`` every
    ``interval`` seconds and applies each new gate-passing edit to the
    live model. ``token`` (when set) is required as a bearer token
    for POST /generate and /rollback; ``allow_quick`` accepts QUICK
    evidence (reports with no SHIP verdict). Returns 0 on a clean
    stop (Ctrl+C).
    """
    if model is None:
        try:
            from .real_runner import RealModel
        except ImportError as exc:
            print(exc)
            print("hint: install the kit's model stack first "
                  "(torch + transformers), then run serve-model again")
            return 2
        log(f"loading {model_id} (this takes a while the first time)…")
        model = RealModel(model_id, dtype=dtype, device=device)
    live = LiveModel(model, model_id, log=log, allow_quick=allow_quick)
    stop = threading.Event()

    def _watch():
        while not stop.is_set():
            try:
                live.poll_once(versions_dir)
            except Exception as exc:  # never let the watcher die
                log(f"  watcher error (kept serving): {exc}")
            stop.wait(interval)

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    handler = make_handler(live, token=token)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        log(f"wsp-kit live model: http://127.0.0.1:"
            f"{httpd.server_address[1]} (model {model_id})", )
        if token is not None:
            log("  bearer token required for /generate and /rollback")
        log(f"watching {versions_dir} for new passing edits — "
            f"they are applied to this same model, live. "
            f"Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log("live model stopped.")
        finally:
            stop.set()
    return 0
