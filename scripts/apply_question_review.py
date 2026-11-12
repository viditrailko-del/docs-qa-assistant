"""Record a human review (downloaded from review.html) into questions.json.

    python -m scripts.apply_question_review path/to/review.json --by <name>

Questions marked OK get verified_by / verified_at. Problems are listed (with notes) and
left unverified, to be fixed by hand and reviewed again. Re-running is safe (idempotent).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from scripts.check_questions import QUESTIONS, check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("review", type=Path)
    ap.add_argument("--by", required=True)
    args = ap.parse_args()

    if errs := check():
        sys.exit("questions.json fails its checks; fix those first:\n  " + "\n  ".join(errs))

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("commit") != data["corpus_commit"]:
        sys.exit("review was made for a different corpus commit")

    today = dt.date.today().isoformat()
    ok, problems, missing = 0, [], []
    for q in data["questions"]:
        r = review["reviews"].get(q["id"], {})
        if r.get("verdict") == "ok":
            q["verified_by"], q["verified_at"] = args.by, today
            ok += 1
        elif r.get("verdict") == "problem":
            q.pop("verified_by", None)
            q.pop("verified_at", None)
            problems.append(f"{q['id']}: {r.get('note') or '(no note)'}")
        else:
            missing.append(q["id"])

    QUESTIONS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"verified {ok}, problems {len(problems)}, not reviewed {len(missing)}")
    for p in problems:
        print("  problem", p)
    if missing:
        print("  not reviewed:", ", ".join(missing))


if __name__ == "__main__":
    main()
