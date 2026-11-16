"""Answer eval CLI: the whole pipeline (retrieval -> Gemini -> checks) on the labeled questions.

    python -m eval.run answer [--only q01,u02] [--limit 15] [--model M] [--fallback-model M]
                              [--prompt-version v2] [--k 8] [--pace 13]
        Calls REAL Gemini through the same answer_question() the API uses and freezes every
        result in eval/results/answers/<run_id>/raw/<qid>.json (gitignored), then scores.
    python -m eval.run answer --resume <run_id | latest> [--limit 15]
        Continues a run (same settings) with the questions not done yet. The free tier allows
        ~20 calls/day on the main model, so a full run (61 questions) is spread over days.
    python -m eval.run score [<run_id | latest>]
        Re-scores a frozen run. Offline and free: re-runs the checks, re-applies judgments.

Why two steps (same as the invoice project): the LLM call is the only slow, rate-limited,
non-deterministic part. Freezing it makes scoring reproducible, and lets a check or metric
change be measured on exactly the same answers.

Human verdicts: when the automatic key-fact verdict is wrong (a correct answer worded
differently), put {"q25": {"verdict": "correct", "by": "Vidit", "note": "..."}} in the run's
judgments.json and score again. Labels come from the documents, never from LLM output: a
judgment says whether THIS answer matches the label, it never changes the label.

No fallback model by default: one run = one model, so numbers are comparable. When the daily
quota runs out the run stops cleanly and tells you how to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.answering.client import DAILY_QUOTA, GeminiClient, LLMClient
from app.answering.core import Attempt, AnswerOutcome, answer_question, response_schema
from app.config import Settings, get_settings
from app.ingestion.embeddings import get_embedder
from app.retrieval.search import Method, search
from eval.scoring import QuestionScore, score_question, summarize
from scripts.check_questions import QUESTIONS

RESULTS = Path(__file__).parent / "results" / "answers"


class MemoryRecorder:
    """The eval's RunRecorder: keeps attempts in memory (the API's writes llm_calls rows)."""

    def __init__(self) -> None:
        self.attempts: list[Attempt] = []

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    def update(self, attempt: Attempt) -> None:
        pass  # same object, already in the list


# --- answer (calls Gemini) ----------------------------------------------------------


def answer_one(q: dict, db, embedder, llm: LLMClient, settings: Settings, k: int, method: Method) -> dict[str, Any]:
    recorder = MemoryRecorder()
    start = time.monotonic()
    hits = search(db, embedder, q["question"], k=k, method=method)
    crash = None
    try:
        outcome = answer_question(q["question"], hits, llm, settings, recorder)
    except Exception as e:  # same rule as the API: a bug becomes "failed", not a crash
        outcome, crash = None, f"{type(e).__name__}: {e}"
    return freeze(q["id"], hits, outcome, recorder.attempts, crash, int((time.monotonic() - start) * 1000))


def freeze(qid: str, hits, outcome: AnswerOutcome | None, attempts: list[Attempt], crash: str | None,
           wall_ms: int) -> dict[str, Any]:
    """Everything needed to score offline: the sources exactly as the model saw them (text
    included, so a re-ingest can't silently change what an old answer is compared with)."""
    return {
        "id": qid,
        "status": outcome.status.value if outcome else "failed",
        "answer": outcome.answer.model_dump() if outcome and outcome.answer else None,
        "flags": [f.to_dict() for f in outcome.flags] if outcome else [],
        "answered_by": _answered_by(attempts) if outcome and outcome.answer else None,
        "sources": [
            {"label": f"S{i}", "chunk_id": h.chunk_id, "path": h.path, "title": h.title,
             "heading_path": h.heading_path, "text": h.text, "similarity": round(h.similarity, 4)}
            for i, h in enumerate(hits, start=1)
        ],
        "attempts": [asdict(a) for a in attempts],
        "crash": crash,
        "wall_ms": wall_ms,
    }


def out_of_quota(raw: dict[str, Any]) -> bool:
    """Failed because every model hit its DAILY quota: retrying today is pointless."""
    errors = [a["error"] or "" for a in raw["attempts"]]
    return raw["status"] == "failed" and bool(errors) and DAILY_QUOTA.search(errors[-1]) is not None


def cmd_answer(args: argparse.Namespace) -> Path:
    from app.db.session import SessionLocal  # DB only needed here, not for scoring

    questions = load_questions()
    if args.resume:
        run_dir = resolve_run(args.resume)
        manifest = read_json(run_dir / "manifest.json")
        if manifest["questions_sha"] != questions_sha():
            print("warning: questions.json changed since this run started (labels re-checked?)")
    else:
        manifest = new_manifest(args, questions)
        run_dir = RESULTS / manifest["run_id"]
        write_json(run_dir / "manifest.json", manifest)

    settings = get_settings().model_copy(update={
        "gemini_model": manifest["model"], "gemini_fallback_model": manifest["fallback_model"],
        "prompt_version": manifest["prompt_version"], "gemini_thinking_level": manifest["thinking_level"],
        "min_similarity": manifest["min_similarity"],
    })
    if not settings.gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set: the answer step calls the real API")

    todo = [qid for qid in manifest["questions"] if not _done(run_dir, qid)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"run {run_dir.name}: {len(todo)} question(s) to answer with {settings.gemini_model}"
          f" (prompt {settings.prompt_version}, k={manifest['k']})", flush=True)

    llm = GeminiClient(settings.gemini_api_key, timeout_s=settings.gemini_timeout_s,
                       thinking_level=settings.gemini_thinking_level)
    embedder = get_embedder()
    with SessionLocal() as db:
        for i, qid in enumerate(todo, 1):
            if i > 1:
                time.sleep(args.pace)  # stay under the per-minute limit
            raw = answer_one(questions[qid], db, embedder, llm, settings, manifest["k"], Method(manifest["method"]))
            if out_of_quota(raw):
                print(f"daily quota reached at {qid}; nothing saved for it. Resume tomorrow:\n"
                      f"  python -m eval.run answer --resume {run_dir.name}")
                break
            write_json(run_dir / "raw" / f"{qid}.json", raw)
            cost = sum(float(a["cost_usd"] or 0) for a in raw["attempts"])
            print(f"[{i}/{len(todo)}] {qid}: {raw['status']}  {len(raw['attempts'])} call(s), "
                  f"{raw['wall_ms']} ms, ${cost:.4f}", flush=True)
    return run_dir


def _done(run_dir: Path, qid: str) -> bool:
    """A question is done once it has a frozen result that isn't `failed` (failures are retried)."""
    path = run_dir / "raw" / f"{qid}.json"
    return path.exists() and read_json(path)["status"] != "failed"


def new_manifest(args: argparse.Namespace, questions: dict[str, dict]) -> dict[str, Any]:
    s = get_settings()
    only = [x for x in (args.only or "").split(",") if x]
    unknown = sorted(set(only) - questions.keys())
    if unknown:
        sys.exit(f"unknown question id(s): {', '.join(unknown)}")
    model = args.model or s.gemini_model
    prompt_version = args.prompt_version or s.prompt_version
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{prompt_version}-{model}"
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": model,
        "fallback_model": args.fallback_model,
        "prompt_version": prompt_version,
        "thinking_level": s.gemini_thinking_level,
        "k": args.k or s.retrieval_k,
        "method": Method.vector.value,
        "min_similarity": s.min_similarity,
        "embed_model": s.embed_model,
        "chunk_max_chars": s.chunk_max_chars,
        # The response schema shapes answers as much as the prompt: fingerprint it.
        "schema_sha": _sha(json.dumps(response_schema(), sort_keys=True)),
        "questions_sha": questions_sha(),
        "git_commit": _git_commit(),
        "questions": only or list(questions),
    }


# --- score (offline) ---------------------------------------------------------------


def score_run(run_dir: Path) -> tuple[dict[str, Any], list[QuestionScore]]:
    manifest = read_json(run_dir / "manifest.json")
    questions = load_questions()
    judgments_path = run_dir / "judgments.json"
    judgments = read_json(judgments_path) if judgments_path.exists() else {}
    scores = []
    for qid in manifest["questions"]:
        path = run_dir / "raw" / f"{qid}.json"
        if path.exists():
            scores.append(score_question(questions[qid], read_json(path), judgments.get(qid),
                                         min_similarity=manifest["min_similarity"]))
    return manifest, scores


def cmd_score(run_dir: Path) -> None:
    manifest, scores = score_run(run_dir)
    if not scores:
        sys.exit(f"{run_dir.name}: no answers yet")
    summary = summarize(scores)
    write_json(run_dir / "report.json", {"manifest": manifest, "summary": summary,
                                         "questions": [asdict(s) for s in scores]})
    md = render_markdown(manifest, summary, scores)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print("\n" + md.split("\n## Questions")[0])
    print(f"full report: {run_dir / 'report.md'}")


def render_markdown(manifest: dict, s: dict, scores: list[QuestionScore]) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x:.0%}"

    cw = s["confident_wrong"]
    lines = [
        f"# Answer eval {manifest['run_id']}",
        "",
        f"model `{manifest['model']}` (fallback `{manifest['fallback_model'] or '-'}`), prompt `{manifest['prompt_version']}`, "
        f"k={manifest['k']}, {manifest['method']} retrieval, git `{manifest['git_commit']}`. "
        f"Answered {s['n']}/{len(manifest['questions'])} questions.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| **confident-wrong** (wrong, shown as verified) | **{cw['count']}** ({pct(cw['rate'])}) {', '.join(cw['ids'])} |",
        f"| answerable: correct (flagged or not) | {pct(s['answerable']['correct'])} of {s['answerable']['n']} |",
        f"| answerable: correct and unflagged | {pct(s['answerable']['correct_unflagged'])} |",
        f"| answerable: over-refusals | {pct(s['answerable']['over_refusal'])} {', '.join(s['answerable']['over_refusal_ids'])} |",
        f"| unanswerable: correct refusals | {pct(s['unanswerable']['correct_refusal'])} of {s['unanswerable']['n']} |",
        f"| flags: catch rate (wrong answers flagged) | {pct(s['flags']['catch_rate'])} |",
        f"| flags: precision (flagged answers that were wrong) | {pct(s['flags']['precision'])} |",
        f"| citations: quote found in cited source | {pct(s['citations']['valid_rate'])} of {s['citations']['total']} |",
        f"| citations: answer cites the labeled evidence | {pct(s['citations']['cites_labeled_evidence'])} |",
        f"| retrieval: evidence among sources shown (hit@k) | {pct(s['retrieval']['evidence_shown'])} |",
        f"| retrieval: all evidence shown (full@k) | {pct(s['retrieval']['all_evidence_shown'])} |",
        f"| conflicts: sources_disagree reported | {pct(s['conflict_disagree_reported'])} |",
        f"| failed | {s['failed']} |",
        f"| cost | total ${s['cost']['total_usd']}, ${s['cost']['per_question_usd']}/question ({s['cost']['unknown']} unknown) |",
        f"| LLM latency per question | mean {s['latency_ms']['mean']} ms, median {s['latency_ms']['median']} ms, max {s['latency_ms']['max']} ms |",
        f"| LLM calls / answered by | {s['llm_calls']} / {s['answered_by']} |",
        "",
        "Outcomes: " + ", ".join(f"{k} {v}" for k, v in s["outcomes"].items() if v),
        "",
        "By type: " + "; ".join(f"{t} " + ", ".join(f"{k} {v}" for k, v in c.items()) for t, c in s["by_type"].items()),
        "",
        f"Flags by check: {s['flags']['by_code'] or '-'}",
        "",
        f"**To review by hand** (auto verdict not correct, or labeled evidence not cited; "
        f"add judgments.json, then score again): "
        f"{', '.join(s['to_review']) or 'none'}",
        "",
        "## Questions",
        "",
    ]
    order = {o: i for i, o in enumerate(("confident_wrong", "wrong_flagged", "over_refusal", "failed",
                                          "correct_flagged", "correct", "correct_refusal"))}
    for q in sorted(scores, key=lambda q: (order[q.outcome], q.id)):
        verdict = f", verdict {q.verdict} ({q.verdict_by})" if q.verdict else ""
        lines.append(f"### {q.id} [{q.type}] {q.outcome}{verdict}")
        lines.append(f"Q: {q.question}  ")
        if q.answer_text:
            lines.append(f"A: {q.answer_text}  ")
        if q.refusal_reason:
            lines.append(f"Refused: {q.refusal_reason}  ")
        details = [f"flags {q.flags}" if q.flags else "", f"missing facts {q.missing_facts}" if q.missing_facts else "",
                   f"evidence shown {q.evidence_shown}, cited {q.evidence_cited}" if q.evidence_shown is not None else "",
                   f"status changed by re-check (was {q.stored_status})" if q.stored_status != q.status else ""]
        lines.append("; ".join(d for d in details if d))
        lines.append("")
    return "\n".join(lines)


# --- helpers -------------------------------------------------------------------------


def load_questions() -> dict[str, dict]:
    return {q["id"]: q for q in read_json(QUESTIONS)["questions"]}


def questions_sha() -> str:
    return _sha(QUESTIONS.read_text(encoding="utf-8"))


def resolve_run(name: str) -> Path:
    if name == "latest":
        runs = sorted(p for p in RESULTS.glob("*") if (p / "manifest.json").exists())
        if not runs:
            sys.exit("no answer eval runs yet: run `python -m eval.run answer` first")
        return runs[-1]
    path = RESULTS / name
    if not (path / "manifest.json").exists():
        sys.exit(f"no run {name} in {RESULTS}")
    return path


def _answered_by(attempts: list[Attempt]) -> str | None:
    """The last call that returned output without a validation error is the one used."""
    used = [a for a in attempts if a.raw_response is not None and not a.error]
    return used[-1].model if used else None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    # The container has no git binary, but the repo (with .git) is bind-mounted: read it directly.
    git = Path(__file__).resolve().parent.parent / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except OSError:
        pass
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run")
    sub = parser.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("answer", help="answer questions with real Gemini, freeze results, then score")
    an.add_argument("--resume", help="run id (or 'latest') to continue; its settings are reused")
    an.add_argument("--only", help="comma-separated question ids (new run)")
    an.add_argument("--limit", type=int, help="answer at most N questions now")
    an.add_argument("--model", help="new run: main model (default: GEMINI_MODEL)")
    an.add_argument("--fallback-model", default="", help="new run: fallback model (default: none)")
    an.add_argument("--prompt-version", help="new run: prompt version (default: PROMPT_VERSION)")
    an.add_argument("--k", type=int, help="new run: sources shown to the model (default: RETRIEVAL_K)")
    an.add_argument("--pace", type=float, default=13.0, help="seconds between questions (per-minute limit)")
    sc = sub.add_parser("score", help="re-score a frozen run (offline, free)")
    sc.add_argument("run", nargs="?", default="latest")
    args = parser.parse_args()

    if args.cmd == "answer":
        cmd_score(cmd_answer(args))
    else:
        cmd_score(resolve_run(args.run))


if __name__ == "__main__":
    main()
