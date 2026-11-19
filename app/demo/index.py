"""The demo's document index: the frozen corpus + the chunk vectors computed on the eval machine.

Why ship vectors instead of embedding on boot: on Render's free instance (0.1 CPU) the local model needs
~1.5 s per chunk, so the 1407 corpus chunks would take ~35 minutes. And reusing the exact float32 vectors the
eval runs searched means a recorded question retrieves exactly the chunks it retrieved then (replay mode
depends on it).

    docker compose run --rm -T api uv run --frozen python -m app.demo.index export   # dev DB -> app/demo/index/
    python -m app.demo.index seed                                                    # corpus + vectors -> DB

`seed` re-chunks the corpus files with the current code (pure, fast, no model) and takes each chunk's vector
from the export, matched by (path, seq) AND a hash of the chunk's heading + text. If the chunker or the corpus
changed since the export, nothing matches and it stops: re-run `export` on a freshly ingested dev DB.
Idempotent: documents already in the DB (same bytes) are skipped, so it runs on every boot.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import EMBED_DIM, Chunk, Document
from app.ingestion.markdown import chunk_markdown

ROOT = Path(__file__).resolve().parents[2]
CORPUS_MANIFEST = ROOT / "corpus" / "manifest.json"
CORPUS_DOCS = ROOT / "corpus" / "handbook"
INDEX_DIR = Path(__file__).parent / "index"
INDEX_FILE = INDEX_DIR / "chunks.json"
VECTORS_FILE = INDEX_DIR / "vectors.f32"  # float32, little-endian, EMBED_DIM per chunk, in INDEX_FILE order


class IndexMismatch(RuntimeError):
    """The corpus, the chunker or the vectors file disagree: the demo must not start on a wrong index."""


def chunk_key(heading_path: str, text: str) -> str:
    return hashlib.sha256(f"{heading_path}\n\n{text}".encode()).hexdigest()[:16]


def _floats(raw: bytes) -> array:
    vec = array("f")
    vec.frombytes(raw)
    if sys.byteorder != "little":  # pragma: no cover - every machine we run on is little-endian
        vec.byteswap()
    return vec


def load_index() -> tuple[dict, dict[tuple[str, int], tuple[str, list[float]]]]:
    """(header, {(path, seq): (key, vector)})."""
    meta = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    floats = _floats(VECTORS_FILE.read_bytes())
    rows = meta["chunks"]
    if meta["dim"] != EMBED_DIM or len(floats) != len(rows) * EMBED_DIM:
        raise IndexMismatch(f"{VECTORS_FILE.name} holds {len(floats)} floats, expected {len(rows)} x {EMBED_DIM}")
    vectors = {
        (path, seq): (key, floats[i * EMBED_DIM:(i + 1) * EMBED_DIM].tolist())
        for i, (path, seq, key) in enumerate(rows)
    }
    return meta, vectors


def corpus_files() -> list[dict]:
    """The manifest's files, each checked against its sha256 (never seed a corpus that was edited)."""
    files = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))["files"]
    for f in files:
        raw = (CORPUS_DOCS / f["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != f["sha256"]:
            raise IndexMismatch(f"corpus file changed since the snapshot: {f['path']}")
        f["raw"] = raw
    return files


def seed_index(db: Session, settings: Settings) -> dict[str, int]:
    """Insert every corpus document that isn't in the DB yet, with the exported vectors."""
    meta, vectors = load_index()
    present = set(db.scalars(select(Document.content_hash)))
    added = chunks = 0
    for f in corpus_files():
        if f["sha256"] in present:
            continue
        text = f["raw"].decode("utf-8")
        title, drafts = chunk_markdown(text, fallback_title=PurePosixPath(f["path"]).stem,
                                       max_chars=settings.chunk_max_chars)
        doc = Document(content_hash=f["sha256"], path=f["path"], title=title, source_url=f["url"], raw_text=text)
        for d in drafts:
            key, vec = vectors.get((f["path"], d.seq), (None, None))
            if key != chunk_key(d.heading_path, d.text):
                raise IndexMismatch(f"{f['path']} chunk {d.seq} differs from the exported index: re-run export")
            doc.chunks.append(Chunk(seq=d.seq, heading_path=d.heading_path, text=d.text, char_count=len(d.text),
                                    embedding=vec, embed_model=meta["embed_model"]))
        db.add(doc)
        added += 1
        chunks += len(drafts)
    db.commit()
    return {"documents_added": added, "chunks_added": chunks, "documents_present": len(present)}


def export_index(db: Session) -> dict[str, int]:
    """Write the dev DB's chunk vectors for the corpus files (run on the machine that ran the eval)."""
    files = corpus_files()
    by_hash = {f["sha256"]: f for f in files}
    docs = {d.content_hash: d for d in db.scalars(select(Document).where(Document.content_hash.in_(by_hash)))}
    missing = sorted(by_hash[h]["path"] for h in by_hash.keys() - docs.keys())
    if missing:
        raise IndexMismatch(f"not ingested in this DB: {missing[:5]}... run scripts.ingest_corpus first")
    rows, out, models = [], array("f"), set()
    for f in sorted(files, key=lambda f: f["path"]):
        for c in docs[f["sha256"]].chunks:  # ordered by seq
            rows.append([f["path"], c.seq, chunk_key(c.heading_path, c.text)])
            out.extend(float(x) for x in c.embedding)
            models.add(c.embed_model)
    if len(models) != 1:
        raise IndexMismatch(f"chunks embedded with several models: {models}")
    if sys.byteorder != "little":  # pragma: no cover
        out.byteswap()
    INDEX_DIR.mkdir(exist_ok=True)
    VECTORS_FILE.write_bytes(out.tobytes())
    INDEX_FILE.write_text(json.dumps({"embed_model": models.pop(), "dim": EMBED_DIM, "chunks": rows},
                                     separators=(",", ":")) + "\n", encoding="utf-8")
    return {"documents": len(files), "chunks": len(rows)}


def main(argv: list[str]) -> None:
    from app.db.session import SessionLocal

    cmd = argv[0] if argv else ""
    if cmd not in ("export", "seed"):
        sys.exit("usage: python -m app.demo.index export|seed")
    with SessionLocal() as db:
        result = export_index(db) if cmd == "export" else seed_index(db, get_settings())
    print(cmd, result)


if __name__ == "__main__":
    main(sys.argv[1:])
