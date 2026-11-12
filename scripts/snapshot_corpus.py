"""Download a frozen snapshot of selected GitLab Handbook pages into corpus/.

Stdlib only, so it runs with any Python 3.12 (no Docker needed).

    python -m scripts.snapshot_corpus            # download at PINNED_SHA
    python -m scripts.snapshot_corpus --verify   # check files against corpus/manifest.json
    python -m scripts.snapshot_corpus --sha <commit>   # re-snapshot at another commit
                                                       # (then update PINNED_SHA; the question set
                                                       #  must be re-checked against the new text)

Why pin a commit: the handbook changes every day. Test questions are labeled against
exact text, so the documents must never change underneath them.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = "gitlab-com/content-sites/handbook"
API = f"https://gitlab.com/api/v4/projects/{urllib.parse.quote(PROJECT, safe='')}"
PINNED_SHA = "7466a0c8323e9a5d31afc51bfa3c858055d05324"  # main @ 2026-11-12

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
DOCS = CORPUS / "handbook"
MANIFEST = CORPUS / "manifest.json"

# Sections are listed in full; INCLUDE patterns (relative to content/handbook/) pick pages.
# Theme: an employee helpdesk (policies, benefits, leave, communication, expenses, travel).
SECTIONS = ["people-policies", "total-rewards", "communication", "hiring", "finance"]
INCLUDE = [
    "people-policies/*",
    "total-rewards/*",
    "communication/*",
    "hiring/_index.md",
    "hiring/candidate-faq/_index.md",
    "hiring/referral-process.md",
    "hiring/referral-operations.md",
    "hiring/talent-acquisition-framework/internal-hiring-process.md",
    "finance/expenses.md",
    "finance/spending-company-money.md",
    "finance/travel/*",
    "finance/payroll.md",
    "finance/timekeeping/_index.md",
    "finance/authorization-matrix.md",
    "finance/accounts-payable/corp-credit-cards.md",
    "finance/procurement/office-equipment-supplies.md",
    "finance/procurement/individual-use-software.md",
    "finance/procurement/charitable-contributions.md",
]
# Pages centered on named individuals are never included.
EXCLUDE = ["*/readmes/*", "*readme*"]


def _get(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rag-corpus-snapshot"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def list_markdown(section: str, sha: str) -> list[str]:
    paths, page = [], 1
    while True:
        q = urllib.parse.urlencode(
            {"path": f"content/handbook/{section}", "ref": sha, "recursive": "true",
             "per_page": 100, "page": page}
        )
        items = json.loads(_get(f"{API}/repository/tree?{q}"))
        paths += [i["path"] for i in items if i["type"] == "blob" and i["path"].endswith(".md")]
        if len(items) < 100:
            return paths
        page += 1


def selected(rel: str) -> bool:
    low = rel.lower()
    if any(fnmatch.fnmatch(low, p) for p in EXCLUDE):
        return False
    return any(fnmatch.fnmatch(rel, p) for p in INCLUDE)


def raw_file(path: str, sha: str) -> bytes:
    enc = urllib.parse.quote(path, safe="")
    return _get(f"{API}/repository/files/{enc}/raw?ref={sha}")


def snapshot(sha: str) -> None:
    files = []
    for section in SECTIONS:
        for path in list_markdown(section, sha):
            rel = path.removeprefix("content/handbook/")
            if selected(rel):
                files.append(rel)
    files.sort()
    if not files:
        sys.exit("no files selected")

    if DOCS.exists():
        for old in sorted(DOCS.rglob("*"), reverse=True):
            old.unlink() if old.is_file() else old.rmdir()

    entries = []
    for i, rel in enumerate(files, 1):
        data = raw_file(f"content/handbook/{rel}", sha)
        out = DOCS / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        entries.append({
            "path": rel,
            "url": "https://handbook.gitlab.com/handbook/" + rel.removesuffix(".md").removesuffix("_index"),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        print(f"[{i}/{len(files)}] {rel} ({len(data)} B)")

    (CORPUS / "LICENSE").write_bytes(raw_file("LICENSE", sha))
    MANIFEST.write_text(json.dumps(
        {"source": f"https://gitlab.com/{PROJECT}", "commit": sha, "files": entries},
        indent=2) + "\n", encoding="utf-8")
    total = sum(e["bytes"] for e in entries)
    print(f"done: {len(entries)} files, {total / 1024:.0f} KiB, commit {sha}")


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    for e in manifest["files"]:
        p = DOCS / e["path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
            bad.append(e["path"])
    extra = {str(p.relative_to(DOCS).as_posix()) for p in DOCS.rglob("*.md")} - {
        e["path"] for e in manifest["files"]}
    if bad or extra:
        sys.exit(f"corpus mismatch: changed/missing={bad} extra={sorted(extra)}")
    print(f"ok: {len(manifest['files'])} files match commit {manifest['commit']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=PINNED_SHA)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    verify() if args.verify else snapshot(args.sha)
