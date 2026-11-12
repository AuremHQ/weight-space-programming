"""Shared HTTP hardening for the two local servers.

``wsp serve-model`` and ``wsp serve`` both bind 127.0.0.1 only, but
a localhost server is still reachable by anything on the machine —
including a browser tab pointed at a malicious page. So every
request is checked before it is handled:

- Host must be ``127.0.0.1:<port>`` or ``localhost:<port>``, else 403.
  (A request with no Host at all is not one of those, so it is 403.)
- A non-own Origin gets 403. Same-origin requests and requests with
  no Origin at all (curl, scripts) are fine.
- POST needs ``Content-Type: application/json``, else 415.
- Content-Length missing / non-integer / negative -> 400; above
  MAX_BODY_BYTES -> 413 (checked before the body is read).
- The body must be a JSON object, else 400.
- For /generate: prompt is a non-empty str within MAX_PROMPT_CHARS;
  max_new_tokens is an int clamped to [1, MAX_NEW_TOKENS_CAP];
  temperature is a number in [0, 2]; seed is an int in
  [0, 2**31 - 1]. Anything else is 400.
- The handler timeout is REQUEST_TIMEOUT_S (a class attribute, so
  StreamRequestHandler.setup() applies it to the socket).
- An unexpected error returns ``{"error": "internal error"}`` (500);
  the detail goes only to the terminal log, never to the client.
- An optional bearer token (--token / WSP_TOKEN, compared with
  hmac.compare_digest) gates POST /generate and /rollback. /status
  stays open — it is read-only and lock-free.

Stdlib only. No new dependency.
"""

import hmac
import json
from http.server import BaseHTTPRequestHandler

MAX_BODY_BYTES = 65536
MAX_PROMPT_CHARS = 8000
MAX_NEW_TOKENS_CAP = 512
REQUEST_TIMEOUT_S = 30


class LocalHandler(BaseHTTPRequestHandler):
    """Base handler: the shared checks above, plus JSON helpers.

    Subclasses set the ``token`` class attribute (None = no token)
    and implement do_GET / do_POST on top of the helpers. Each
    handler method should wrap its work so an unexpected exception
    becomes a 500 with no detail (see _internal_error).
    """

    timeout = REQUEST_TIMEOUT_S
    token = None

    # -- responses ---------------------------------------------------

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, message):
        self._send_json(code, {"error": message})

    def _internal_error(self, exc):
        print(f"  internal error: {exc!r}", flush=True)
        try:
            self._send_json(500, {"error": "internal error"})
        except Exception:
            pass  # the client is gone; nothing more to do

    # -- request checks ----------------------------------------------

    def _check_host_origin(self):
        """Host + Origin allow-list. Sends 403 and returns False."""
        host = self.headers.get("Host")
        port = self.server.server_address[1]
        ok_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if host not in ok_hosts:
            self._send_error(403, "forbidden host")
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            ok_origins = {f"http://127.0.0.1:{port}",
                          f"http://localhost:{port}"}
            if origin not in ok_origins:
                self._send_error(403, "forbidden origin")
                return False
        return True

    def _check_token(self):
        """Bearer token gate (POST /generate and /rollback only).

        Returns True when no token is configured or the request
        carries the right one; sends 403 and returns False.
        """
        if self.token is None:
            return True
        auth = self.headers.get("Authorization", "")
        if (not auth.startswith("Bearer ")
                or not hmac.compare_digest(auth[7:], self.token)):
            self._send_error(403, "forbidden")
            return False
        return True

    def _read_json_body(self):
        """Read + parse a POST body. Returns the dict, or None after
        sending the matching error (415 / 400 / 413)."""
        if self.headers.get("Content-Type") != "application/json":
            self._send_error(415, "Content-Type must be application/json")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_error(400, "Content-Length is required")
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send_error(400, "Content-Length must be an integer")
            return None
        if length < 0:
            self._send_error(400, "Content-Length must not be negative")
            return None
        if length > MAX_BODY_BYTES:
            self._send_error(413, "request body too large")
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            self._send_error(400, "bad JSON")
            return None
        if not isinstance(body, dict):
            self._send_error(400, "request body must be a JSON object")
            return None
        return body

    # -- /generate params --------------------------------------------

    @staticmethod
    def _validate_generate(body):
        """Validate a /generate body.

        Returns (params, None) on success or (None, (code, message)).
        max_new_tokens is clamped, never rejected, when it is an int.
        """
        prompt = body.get("prompt")
        if (not isinstance(prompt, str) or not prompt
                or len(prompt) > MAX_PROMPT_CHARS):
            return None, (400, "prompt must be a non-empty string of "
                          f"at most {MAX_PROMPT_CHARS} characters")
        max_new_tokens = body.get("max_new_tokens", 64)
        if isinstance(max_new_tokens, bool) or not isinstance(
                max_new_tokens, int):
            return None, (400, "max_new_tokens must be an integer")
        max_new_tokens = max(1, min(MAX_NEW_TOKENS_CAP, max_new_tokens))
        temperature = body.get("temperature", 0)
        if (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not 0 <= float(temperature) <= 2):
            return None, (400, "temperature must be a number in [0, 2]")
        seed = body.get("seed", 0)
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed <= 2 ** 31 - 1):
            return None, (400, "seed must be an integer in "
                          f"[0, {2 ** 31 - 1}]")
        return {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
        }, None

    def log_message(self, *args):
        pass  # keep the terminal quiet; the log lines are the output
