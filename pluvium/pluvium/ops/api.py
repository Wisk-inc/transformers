# Copyright 2026 Nathan. Apache-2.0.
"""
A small JSON API over the latest forecast (standard library only).

``GET /town?name=Port%20Antonio&country=JM``  one place's row and its hourly chances and words;
``GET /parish?country=JM&name=Portland``      a parish's row;
``GET /country?code=JM``                      every parish of a country, and the bulletin;
``GET /health``                               the issue time and age of the latest forecast.

It reads the files of the ``latest`` folder written by :mod:`pluvium.ops.loop`, re-reading them when they change.
"""

from __future__ import annotations

import csv
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


class Latest:
    def __init__(self, root) -> None:
        self.root = Path(root).expanduser() / "latest"
        self.loaded = None
        self.data = {}

    def get(self) -> dict:
        target = self.root.resolve() if self.root.exists() else None
        if target is not None and target != self.loaded:
            info = json.loads((target / "forecast.json").read_text())
            towns = list(csv.DictReader(open(target / "towns.csv", encoding="utf-8")))
            hourly = dict(np.load(target / "hourly.npz"))
            self.data = {"info": info, "towns": towns, "hourly": hourly, "folder": target}
            self.loaded = target
        return self.data


def _handler(latest: Latest):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload) -> None:
            body = json.dumps(payload, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:              # quiet
            pass

        def do_GET(self) -> None:                          # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            d = latest.get()
            if not d:
                return self._send(503, {"error": "no forecast yet"})
            if url.path == "/health":
                return self._send(200, {"t0": d["info"]["t0"], "issued": d["info"]["issued"],
                                        "age_minutes": (time.time() - d["folder"].stat().st_mtime) / 60})
            if url.path == "/town":
                name, country = q.get("name", "").lower(), q.get("country", "").upper()
                rows = [k for k, t in enumerate(d["towns"]) if t["place"].lower() == name and (not country or t["country"] == country)]
                if not rows:
                    return self._send(404, {"error": f"no place named {q.get('name')!r}"})
                k = rows[0]
                h = d["hourly"]
                return self._send(200, {"town": d["towns"][k], "p_wet": h["p_wet"][k].astype(float).tolist(),
                                        "thunder": h["thunder"][k].astype(float).tolist(),
                                        "expected_mm": h["expected_mm"][k].astype(float).tolist()})
            if url.path == "/parish":
                rows = [p for p in d["info"]["parishes"] if p["country"] == q.get("country", "").upper()
                        and p["parish"].lower() == q.get("name", "").lower()]
                return self._send(200 if rows else 404, rows[0] if rows else {"error": "no such parish"})
            if url.path == "/country":
                code = q.get("code", "").upper()
                bul = d["folder"] / f"bulletin_{code}.txt"
                return self._send(200, {"parishes": [p for p in d["info"]["parishes"] if p["country"] == code],
                                        "bulletin": bul.read_text() if bul.exists() else ""})
            return self._send(404, {"error": "unknown path"})

    return Handler


def serve(root, host: str = "127.0.0.1", port: int = 8750) -> ThreadingHTTPServer:
    """Start the server (blocking: ``serve(...).serve_forever()``)."""
    return ThreadingHTTPServer((host, port), _handler(Latest(root)))


__all__ = ["Latest", "serve"]
