"""`wsp serve-model`: the same model, with its weights overridden live.

The kit's loop keeps improving the model in the background and writes
each gate-passing edit as an ``edit.json`` in a run directory, with
``latest.json`` pointing at the newest passing run. This server loads
the model ONCE and serves it — and whenever a new passing edit
appears, it applies the edit to the LIVE loaded model, in place:

- the old weights are overridden by the updated ones, in the same
  model object — no reload, no restart, no new model to swap in;
- the next prompt automatically uses the updated weights;
- the original columns are kept, so ``POST /rollback`` restores the
  last applied edit exactly (bitwise), instantly.

Only gate-passing edits are ever applied (the loop only writes an
edit.json on a gate pass), the edit must be for the model this server
loaded, and the 1% cap is enforced again at apply time. An edit that
fails validation is logged and skipped — the server keeps serving.

Endpoints (127.0.0.1 only — this is a local server, not a public
service):
  GET  /          short help
  GET  /status    model, current version, applied edits, weights changed
  POST /generate  {"prompt": "...", "max_new_tokens": 64,
                   "temperature": 0} -> {"text": "..."}
  POST /rollback  undo the last applied edit, exactly

Generation and edits never run at the same time: one lock guards
both, so an edit never lands mid-generation. Stop with Ctrl+C.
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LATEST_FILE = "latest.json"


def load_recipe(path):
    """Load and validate an edit.json recipe.

    Returns the recipe dict. Raises ValueError on a bad format
    version, a missing model id, or missing layers.
    """
    with open(path) as f:
        recipe = json.load(f)
    if recipe.get("format_version") != 1:
        raise ValueError(
            f"unsupported edit.json format_version "
            f"{recipe.get('format_version')!r}")
    if not recipe.get("model_id") or not recipe.get("layers"):
        raise ValueError("edit.json is missing model_id or layers")
    return recipe


def apply_recipe(model, recipe, log=print):
    """Apply one edit.json recipe to a loaded model, in place.

    Returns (changed, saved) — total weights changed and the list of
    per-layer saved handles for rollback. Raises ValueError if any
    layer breaks the 1% cap (the model is left untouched for that
    layer; earlier layers of the same recipe are restored first).
    """
    import numpy as np

    changed = 0
    saved = []
    try:
        for layer in recipe["layers"]:
            direction = np.array(layer["direction"], dtype=np.float32)
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


class LiveModel:
    """One loaded model, its applied edits, and the update watcher.

    ``model`` is a RealModel (or a test double with the same
    apply_edit / restore / generate methods). ``model_id`` must match
    the model_id in every edit.json applied.
    """

    def __init__(self, model, model_id, log=print):
        self.model = model
        self.model_id = model_id
        self.log = log
        self.lock = threading.Lock()
        self.applied = []  # [{version, run_dir, weights_changed, at}]
        self._saved_stack = []  # per-edit saved handles, same order
        self.last_version = 0

    def generate(self, prompt, max_new_tokens=64, temperature=0,
                 seed=0):
        with self.lock:
            text, _finished, _n = self.model.generate(
                prompt, max_new_tokens, temperature, seed)
        return text

    def status(self):
        return {
            "model": self.model_id,
            "current_version": self.last_version,
            "applied_edits": len(self.applied),
            "weights_changed": sum(e["weights_changed"]
                                   for e in self.applied),
            "edits": list(self.applied),
        }

    def poll_once(self, versions_dir):
        """Apply the newest passing edit, if there is one.

        Returns True if an edit was applied. A mismatched model, a
        bad recipe, or a cap failure is logged and skipped — the
        server keeps serving the current weights.
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
        if version <= self.last_version:
            return False
        run_dir = latest.get("run_dir", "")
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(versions_dir, run_dir)
        edit_path = os.path.join(run_dir, "edit.json")
        if not os.path.isfile(edit_path):
            self.log(f"  version {version}: no edit.json in {run_dir} "
                     f"— skipped")
            self.last_version = version
            return False
        try:
            recipe = load_recipe(edit_path)
        except (OSError, ValueError) as exc:
            self.log(f"  version {version}: bad edit.json: {exc} — "
                     f"skipped")
            self.last_version = version
            return False
        if recipe["model_id"] != self.model_id:
            self.log(f"  version {version}: edit is for "
                     f"{recipe['model_id']}, this server runs "
                     f"{self.model_id} — skipped")
            self.last_version = version
            return False
        with self.lock:
            try:
                changed, saved = apply_recipe(self.model, recipe,
                                              log=self.log)
            except Exception as exc:
                self.log(f"  version {version}: apply failed: {exc} — "
                         f"kept the old weights")
                self.last_version = version
                return False
            self._saved_stack.append(saved)
            self.applied.append({
                "version": version,
                "run_dir": run_dir,
                "weights_changed": changed,
                "at": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
            })
            self.last_version = version
        self.log(f"  live override: version {version} applied, "
                 f"{changed} weights changed — the same model now "
                 f"uses the updated weights")
        return True

    def rollback_last(self):
        """Undo the last applied edit, exactly. Returns True/False."""
        with self.lock:
            if not self._saved_stack:
                return False
            saved = self._saved_stack.pop()
            rollback(self.model, saved, log=self.log)
            edit = self.applied.pop()
            self.last_version = (self.applied[-1]["version"]
                                 if self.applied else 0)
        self.log(f"  rolled back version {edit['version']}")
        return True


def make_handler(live):
    class ModelHandler(BaseHTTPRequestHandler):
        def _send_json(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
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
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                self._send_json(400, {"error": "bad JSON"})
                return
            if self.path == "/generate":
                prompt = body.get("prompt", "")
                if not prompt:
                    self._send_json(400, {"error": "prompt is required"})
                    return
                text = live.generate(
                    prompt,
                    max_new_tokens=body.get("max_new_tokens", 64),
                    temperature=body.get("temperature", 0),
                    seed=body.get("seed", 0))
                self._send_json(200, {"text": text})
            elif self.path == "/rollback":
                if live.rollback_last():
                    self._send_json(200, {"rolled_back": True})
                else:
                    self._send_json(200, {"rolled_back": False,
                                          "reason": "no edit applied yet"})
            else:
                self._send_json(404, {"error": "unknown endpoint"})

        def log_message(self, *args):
            pass  # keep the terminal quiet; the log lines are the output

    return ModelHandler


def serve_model(model_id, versions_dir, port=8900, interval=30,
                device="cpu", dtype="bfloat16", model=None, log=print):
    """Load the model once, serve it, and apply new edits live.

    ``model`` is an injected loaded model (tests); otherwise a
    RealModel is loaded here (needs torch + transformers). The
    watcher thread polls ``versions_dir/latest.json`` every
    ``interval`` seconds and applies each new gate-passing edit to the
    live model. Returns 0 on a clean stop (Ctrl+C).
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
    live = LiveModel(model, model_id, log=log)
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
    handler = make_handler(live)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        log(f"wsp-kit live model: http://127.0.0.1:"
            f"{httpd.server_address[1]} (model {model_id})", )
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
