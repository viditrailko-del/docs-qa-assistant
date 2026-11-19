"""Validate eval/questions/questions.json against the frozen corpus.

    python -m scripts.check_questions

Stdlib only. Checks that every quote really appears in the file it cites, that key facts
are backed by the cited sources, and that the question set matches the corpus commit.
This is the same idea as the assistant's citation check, applied to our own labels:
a label we can't verify in code is a label we shouldn't trust.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "corpus" / "handbook"
MANIFEST = ROOT / "corpus" / "manifest.json"
QUESTIONS = ROOT / "eval" / "questions" / "questions.json"

TYPES = {"simple", "multi", "tricky", "conflict", "injection", "unanswerable"}
EXPECT = {"simple": "answer", "multi": "answer", "tricky": "answer", "injection": "answer",
          "conflict": "answer_both", "unanswerable": "refuse"}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})


def normalize(text: str) -> str:
    """Same text, ignoring curly vs straight quotes and runs of whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return " ".join(text.split())


def check() -> list[str]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    known = {f["path"] for f in manifest["files"]}
    cache: dict[str, str] = {}
    errors: list[str] = []

    def text_of(path: str) -> str | None:
        if path not in known:
            return None
        if path not in cache:
            cache[path] = normalize((DOCS / path).read_text(encoding="utf-8"))
        return cache[path]

    if data.get("corpus_commit") != manifest["commit"]:
        errors.append(f"corpus_commit {data.get('corpus_commit')} != manifest {manifest['commit']}")

    ids = Counter(q["id"] for q in data["questions"])
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]

    for q in data["questions"]:
        qid, qtype = q["id"], q.get("type")
        if qtype not in TYPES:
            errors.append(f"{qid}: unknown type {qtype!r}")
            continue
        if q.get("expect") != EXPECT[qtype]:
            errors.append(f"{qid}: expect should be {EXPECT[qtype]!r} for type {qtype}")
        for field in ("question", "answer"):
            if not q.get(field, "").strip():
                errors.append(f"{qid}: empty {field}")

        sources, facts = q.get("sources", []), q.get("key_facts", [])
        if qtype == "unanswerable":
            if sources or facts:
                errors.append(f"{qid}: unanswerable questions have no sources/key_facts")
        else:
            if not sources:
                errors.append(f"{qid}: needs at least one source")
            if not facts:
                errors.append(f"{qid}: needs at least one key fact")
            if qtype == "conflict" and len({s["path"] + s["quote"] for s in sources}) < 2:
                errors.append(f"{qid}: a conflict needs two different sources")

        for kind in ("sources", "near_miss"):
            for s in q.get(kind, []):
                text = text_of(s["path"])
                if text is None:
                    errors.append(f"{qid}: {kind} path not in corpus: {s['path']}")
                elif normalize(s["quote"]) not in text:
                    errors.append(f"{qid}: quote not found in {s['path']}: {s['quote'][:70]!r}")

        cited = " ".join(text_of(s["path"]) or "" for s in sources).lower()
        for fact in facts:
            if normalize(fact).lower() not in cited:
                errors.append(f"{qid}: key fact {fact!r} not found in its cited files")

    return errors


if __name__ == "__main__":
    errs = check()
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    counts = Counter(q["type"] for q in data["questions"])
    print(f"{len(data['questions'])} questions: " + ", ".join(f"{t} {n}" for t, n in sorted(counts.items())))
    if errs:
        print(f"{len(errs)} problem(s):")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("ok: every quote and key fact is found in the corpus")
