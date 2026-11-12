"""Retrieval eval: how often does search put the right passage in front of the LLM?

    docker compose run --rm -T api uv run --frozen python -m eval.retrieval
    docker compose run --rm -T api uv run --frozen python -m eval.retrieval --methods hybrid --show q30,u02

No LLM involved: free, ~seconds. Uses the ingested dev DB and the real embedding model.
Writes eval/results/retrieval/<timestamp>.json (gitignored) and prints a comparison table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Chunk, Document
from app.db.session import SessionLocal
from app.ingestion.embeddings import get_embedder
from app.ingestion.markdown import clean_markdown
from app.retrieval.search import Method, search
from eval.metrics import KS, QuestionResult, first_relevant_rank, full_coverage_rank, is_trapped, summarize
from scripts.check_questions import QUESTIONS, normalize

RESULTS = Path(__file__).parent / "results" / "retrieval"
TOP = 10


def chunk_index(db) -> dict[str, list[tuple[int, str]]]:
    """path -> [(chunk id, normalized text)] for quote matching."""
    out: dict[str, list[tuple[int, str]]] = {}
    for cid, path, text in db.execute(select(Chunk.id, Document.path, Chunk.text).join(Document)):
        out.setdefault(path, []).append((cid, normalize(text)))
    return out


def chunks_with(index, path: str, quote: str) -> set[int]:
    needle = normalize(clean_markdown(quote))
    return {cid for cid, text in index.get(path, []) if needle in text}


def parse_variant(spec: str) -> tuple[str, Method, float]:
    """'vector' | 'keyword' | 'hybrid' | 'hybrid:0.3' (keyword weight in the fusion)."""
    name, _, weight = spec.partition(":")
    return spec, Method(name), float(weight) if weight else 1.0


def run(variants: list[tuple[str, Method, float]], show: set[str]) -> dict:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    embedder = get_embedder()
    report: dict = {
        "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "corpus_commit": questions["corpus_commit"],
        "embed_model": embedder.model_name,
        "chunk_max_chars": get_settings().chunk_max_chars,
        "methods": {},
    }
    with SessionLocal() as db:
        index = chunk_index(db)
        labels = {}
        for q in questions["questions"]:
            sources = [chunks_with(index, s["path"], s["quote"]) for s in q.get("sources", [])]
            near = set().union(*(chunks_with(index, s["path"], s["quote"]) for s in q.get("near_miss", [])))
            if any(not s for s in sources):
                raise SystemExit(f"{q['id']}: a source quote matches no chunk; re-ingest or fix the label")
            labels[q["id"]] = (sources, near)

        for label, method, weight in variants:
            results = []
            for q in questions["questions"]:
                sources, near = labels[q["id"]]
                hits = search(db, embedder, q["question"], k=TOP, method=method, keyword_weight=weight)
                ranked = [h.chunk_id for h in hits]
                r = QuestionResult(
                    id=q["id"], type=q["type"], answerable=q["expect"] != "refuse",
                    first_rank=first_relevant_rank(ranked, sources),
                    full_rank=full_coverage_rank(ranked, sources),
                    trapped=is_trapped(ranked, sources, near) if near else False,
                    top_similarity=max((h.similarity for h in hits), default=0.0),
                    ranked=ranked,
                )
                results.append(r)
                if q["id"] in show:
                    print(f"\n[{label}] {q['id']} {q['question']}  (first relevant: {r.first_rank})")
                    for h in hits[:5]:
                        tag = "REL " if any(h.chunk_id in s for s in sources) else ("TRAP" if h.chunk_id in near else "    ")
                        print(f"  {tag} {h.rank:>2} sim={h.similarity:.3f} {h.path} > {h.heading_path[-50:]}")
            report["methods"][label] = {
                "summary": summarize(results),
                "questions": [r.__dict__ for r in results],
            }
    return report


def print_table(report: dict) -> None:
    cols = [f"hit@{k}" for k in KS] + ["full@5", "full@10", "mrr"]
    print(f"\nRetrieval eval  embed={report['embed_model']}  chunk_max_chars={report['chunk_max_chars']}")
    print(f"{'method':<12}{'n':>4}" + "".join(f"{c:>9}" for c in cols) + "  trapped  missed@10")
    for method, data in report["methods"].items():
        s = data["summary"]
        a = s["answerable"]
        print(f"{method:<12}{a['n']:>4}" + "".join(f"{a[c]:>9.2f}" for c in cols)
              + f"  {len(s['trapped']):>7}  {','.join(a['missed@10']) or '-'}")
    for method, data in report["methods"].items():
        print(f"\n[{method}] by type (hit@5 / full@5 / mrr):")
        for t, r in data["summary"]["by_type"].items():
            print(f"  {t:<9} n={r['n']:<3} {r['hit@5']:.2f} / {r['full@5']:.2f} / {r['mrr']:.2f}")
        print(f"  trapped: {', '.join(data['summary']['trapped']) or '-'}")
        print(f"  refusal signal (top similarity): {data['summary']['refusal_signal']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="vector,keyword,hybrid",
                    help="comma-separated: vector, keyword, hybrid, hybrid:<keyword weight>")
    ap.add_argument("--show", default="", help="comma-separated question ids to print hits for")
    args = ap.parse_args()
    variants = [parse_variant(m) for m in args.methods.split(",")]
    report = run(variants, {s for s in args.show.split(",") if s})
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_table(report)
    print(f"\nsaved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
