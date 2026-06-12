#!/usr/bin/env python3
"""
Tiny webhook receiver for testing alert delivery (sprint3 STEP 8).

Usage:
  python scripts/webhook_listener.py --port 9000
  # then in .env:  ALERT_WEBHOOK_URL=http://127.0.0.1:9000/hook
"""
import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            pretty = json.dumps(json.loads(body), indent=2)
        except json.JSONDecodeError:
            pretty = body.decode(errors="replace")
        print(f"\n[{datetime.now():%H:%M:%S}] POST {self.path}\n{pretty}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args) -> None:  # silence default access log
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="third-eye webhook test listener")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    print(f"Listening for webhooks on http://127.0.0.1:{args.port} ... Ctrl+C to stop.")
    try:
        HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
