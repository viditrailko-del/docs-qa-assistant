"""Replay every recorded scenario against a running demo and check each ends where the eval scored it.
Stdlib only: runs in CI against the freshly booted prod image, and from any machine against the public URL.

    python -m demo.check_replays http://localhost:8001
    python -m demo.check_replays https://<the-demo>.onrender.com

Also checks the guards a visitor could hit: uploads refused, unknown scenario 404, reset refused without a token.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parent.parent / "app" / "demo" / "scenarios"
UPLOAD = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="x.md"\r\nContent-Type: text/markdown\r\n\r\n'
          b"# Title\n\nSome text.\r\n--x--\r\n")


def call(method: str, url: str, data: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main(base: str) -> int:
    base = base.rstrip("/")
    failures = []
    for path in sorted(SCENARIOS.glob("*.json"), key=lambda p: json.loads(p.read_text(encoding="utf-8"))["order"]):
        exp = json.loads(path.read_text(encoding="utf-8"))["expected"]
        t0 = time.monotonic()
        code, body = call("POST", f"{base}/demo/replay/{path.stem}")
        secs = time.monotonic() - t0
        if code != 200:
            failures.append(path.stem)
            print(f"FAIL {path.stem}: HTTP {code} {body[:200]!r}")
            continue
        a = json.loads(body)
        got = (a["status"], sorted({f["code"] for f in a["flags"]}))
        ok = got == (exp["status"], exp["flags"])
        if not ok:
            failures.append(path.stem)
        print(f"{'ok  ' if ok else 'FAIL'} {path.stem:<30} {got[0]:<20} {got[1]} {secs:.1f}s")

    guards = {
        "page served at /": call("GET", f"{base}/")[0] == 200,
        "unknown scenario 404": call("POST", f"{base}/demo/replay/nope")[0] == 404,
        "upload refused 403": call("POST", f"{base}/documents", UPLOAD,
                                   {"Content-Type": "multipart/form-data; boundary=x"})[0] == 403,
        "reset without token refused": call("POST", f"{base}/demo/reset")[0] in (403, 404),
    }
    for name, ok in guards.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)
    print(f"\n{len(failures)} failure(s)" if failures else "\nall good")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"))
