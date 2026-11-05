"""`wsp serve`: the status page on localhost. Stdlib only.

Every request regenerates the page fresh from the report files, so
the page is always as current as the latest run. Serves on
127.0.0.1 only — this is a local status page, not a public service.
Stop with Ctrl+C.
"""

import http.server

from .status import render_html, scan_reports


def make_handler(runs_dir):
    class StatusHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            page = render_html(scan_reports(runs_dir))
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # keep the terminal quiet; the page is the output

    return StatusHandler


def serve(runs_dir=".", port=8899):
    """Serve the status page until Ctrl+C. Returns 0 on a clean stop."""
    handler = make_handler(runs_dir)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"wsp-kit status page: http://127.0.0.1:{httpd.server_address[1]}",
              flush=True)
        print("Serving until Ctrl+C.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("stopped.", flush=True)
    return 0
