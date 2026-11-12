# Docs Q&A Assistant (RAG)

Backend service: company documents → chunks + local embeddings in pgvector → hybrid retrieval → Gemini answer with cited quotes → deterministic citation checks → answer / flag / refuse → eval harness.

**The differentiator is the reliability layer around the LLM, not the chatbot.** The assistant must cite its sources, have those citations verified in code, and refuse instead of guessing. Every design choice should make answers measurable, auditable, and safe to be wrong.

Sister project: `../ai_backend_project` (invoice extraction). Reuse its patterns (Docker setup, versioned prompts, raw-response storage, pure checks, extract/score eval split) rather than reinventing them.

## Flow

```
Ingest (POST /documents)
  → store file + SHA-256 hash (idempotent: same file → same record)
  → split into chunks (with title / heading / position metadata)
  → embed locally → store in Postgres + pgvector

Ask (POST /ask)
  → hybrid retrieval (vector similarity + Postgres full-text) → top-k chunks
  → weak best match → refuse without calling the LLM
  → Gemini structured output: answer + claims[{chunk_id, quote}] or refused (raw response saved first)
  → Pydantic validation (1 retry with the validation error fed back)
  → deterministic checks → flags
  → result = answered | answered_with_flags | refused | failed
  → eval harness measures retrieval recall, answer accuracy, refusals, citation validity
```

## Stack

- Python 3.12, `uv` for deps/venv
- FastAPI, Pydantic v2
- SQLAlchemy 2.0 (typed `Mapped[]` style) + Alembic migrations
- Postgres 16 + **pgvector** via `docker-compose`
- Local embedding model (small, e.g. `bge-small` via sentence-transformers or fastembed — decide in Phase 1)
- `google-genai` SDK (Gemini) for answers only
- pytest

**Everything runs inside Docker.** This Windows machine has Smart App Control, which blocks compiled DLLs (pydantic-core, SQLAlchemy, psycopg, torch...), so a local `.venv` cannot run the app. Don't try to fix that locally; use the container. `uv` is installed via pip and isn't on PATH: use `python -m uv`.

Secrets live in `.env` (never committed); `.env.example` lists the keys.

## Commands

```bash
docker compose up -d --wait                                                # Postgres+pgvector (localhost:5433) + API (http://localhost:8001/docs)
docker compose run --rm -T api uv run --frozen pytest -q                   # all tests (rag_test DB, fake embedder)
docker compose run --rm -T api uv run --frozen alembic upgrade head        # apply migrations
docker compose run --rm -T api uv run --frozen alembic check               # models vs migrations drift
docker compose run --rm -T api uv run --frozen python -m scripts.ingest_corpus   # ingest corpus (~3.5 min first time, idempotent)
docker compose exec -T db psql -U rag -d rag -c "select ..."               # inspect DB
docker compose build api                                                   # after changing dependencies or the embedding model
python -m uv lock                                                          # after editing pyproject.toml (then rebuild)
python -m scripts.check_questions                                          # stdlib-only scripts also run on plain Windows Python
```

- Ports 5433/8001 so this runs next to the invoice project (5432/8000). Source is bind-mounted; code edits auto-reload.
- Tests use the separate `rag_test` DB (created + `vector` extension by `docker/init-test-db.sql`), built with `create_all`. The dev DB uses Alembic.
- Docker Desktop must be running (`"C:\Program Files\Docker\Docker\Docker Desktop.exe"`).

## Budget constraints (free only)

- No paid APIs. Gemini free tier: **~20 requests/day on the main model** (per-minute limit ~5). A full eval does not fit in one day on the main model: use the lite model, a subset, or spread runs over days.
- Embeddings are local, so ingestion and retrieval experiments are free and unlimited. Do most iteration in Phase 2, where no LLM is involved.
- Tests never call real Gemini. Only the eval harness does.

## Rules (important)

1. **Never trust an LLM citation.** Every quote is checked against its cited chunk in code.
2. **Refusing is a correct answer** when the docs don't cover the question. Never present an unverified answer as verified.
3. **Never overwrite history.** Raw LLM output is stored before parsing.
4. **Checks are pure functions**: no DB, no network. Each check gets unit tests (passing + failing + edge case).
5. **The Pydantic schema is the single source of truth** for the Gemini response schema. Don't hand-write a second one.
6. **Prompts are versioned files** (`prompts/v1.txt`...). Changing a prompt = new version; record `prompt_version` on every run.
7. LLM failures degrade to a `failed` result, never a 500 or crash.
8. Route handlers stay thin; logic lives in `retrieval/`, `answering/`, `checks/`.
9. Schema changes go through an Alembic migration.
10. **Label from the documents, never from LLM output.**

## Layout (target)

```
app/
  main.py            FastAPI app + routers
  api/               route handlers (thin)
  schemas/           Pydantic models (answer.py = LLM output schema, source of truth)
  ingestion/         loading, chunking, embedding
  retrieval/         vector + full-text search, fusion, refusal threshold
  answering/         Gemini client, prompts/ (versioned)
  checks/            pure citation/number checks
  db/                SQLAlchemy models, session
alembic/
corpus/              frozen snapshot of the source documents + license/attribution
eval/
  questions/         labeled question set
  run.py             eval CLI (answer → freeze raw; score → offline, free)
tests/
```

## Phase plan

### Phase 0: corpus + question set (before any code)
- Corpus: **GitLab Handbook** (checked 2026-11-12). Source repo `gitlab.com/gitlab-com/content-sites/handbook`; content is Markdown under `content/handbook/<section>/` (e.g. `people-policies`, `total-rewards`, `security`, `support`, `communication`, `hiring`).
  - License: the repo's `LICENSE` file is **MIT** ("Copyright (c) 2011-present GitLab B.V."); a 2016 GitLab blog post says the handbook is **CC BY-SA 4.0**. Both allow reuse; to satisfy either, keep the MIT notice in `corpus/LICENSE`, credit GitLab with a link, and keep any modified handbook text under CC BY-SA. Our own code stays under our license.
  - "GitLab" is their trademark: say "built on the public GitLab Handbook", never imply endorsement or brand the assistant as GitLab's.
  - Skip pages centered on named individuals (team member bios, personal READMEs).
- **Snapshot done:** 60 pages (~1 MB), commit `7466a0c8` (2026-11-12), theme "employee helpdesk": `people-policies`, `total-rewards` (benefits per country, leave, comp, stock), `communication`, part of `hiring` (referrals, internal hiring) and `finance` (expenses, travel, spending, cards). Files in `corpus/handbook/`, hashes + source URLs in `corpus/manifest.json`, attribution in `corpus/SOURCE.md`. The corpus is meant to be committed to git (license allows it).
  - `python -m scripts.snapshot_corpus --verify` checks the files (stdlib only: runs on plain Windows Python, no Docker). Never hand-edit corpus files; change `INCLUDE` in the script and re-snapshot. A new commit means re-checking every question label.
  - Pointer pages (`finance/payroll.md`, `timekeeping`, `authorization-matrix`, `procurement/office-equipment-supplies.md`) only link to GitLab's *internal* handbook. Kept on purpose: good unanswerable questions (refuse, mention the link).
  - Pages contain Hugo shortcodes (`{{% details %}}`, `{{% panel %}}`, `{{< youtube >}}`...) and YAML front matter: ingestion must strip/handle them. `communication/top-misused-terms.md` is just a shortcode (no real content).
  - Benefits pages list vendor contacts, some named individuals' work emails: never write questions whose answer is a person's name or email.
- Write 40–60 questions, each with expected answer + source passage(s). Mix:
  - simple (one passage)
  - multi-passage (combine two sections)
  - **unanswerable** (plausible, but the docs don't say; tests refusals)
  - tricky (similar wording in the wrong section, numbers, dates)
- Human-review every label.
- **Question set drafted** (2026-11-12): `eval/questions/questions.json`, 61 questions (37 simple, 8 multi, 5 tricky, 3 conflict, 8 unanswerable). Each has `expect` (answer / answer_both / refuse), `key_facts`, `sources` (path + exact quote) and optional `near_miss` (passages that look relevant but don't answer: retrieval/LLM traps).
  - `python -m scripts.check_questions` verifies every quote exists in its file and every key fact appears in the cited files (whitespace/curly-quote normalized). Run it after any edit; it will become a pytest.
  - Review loop: `python -m scripts.make_question_review` → open `eval/questions/review.html` (gitignored) → mark OK/Problem → Download (or "Copy review to clipboard" when the download is blocked, which happened on file://) → `python -m scripts.apply_question_review <review.json> --by <name>` (sets `verified_by/at`).
  - **All 61 human-verified** by Vidit on 2026-11-12. Any later edit to a question (or a new corpus commit) means re-reviewing it.
  - Real corpus inconsistencies found while labeling (kept as `conflict` questions): business travel insurance trip length (Travel page: < 180 days; Benefits page: up to 365 days); duplicate-referral credit (first submitter vs ask the hire and split); team meal limit ($100/person on purchasing card vs $80/person/meal reimbursed).
  - "Unanswerable" must be checked against the whole corpus, not one page: the gym question was moved to `tricky` because HealthKick offers fitness discounts.
  - The Browser pane's `preview_start` was still reading the invoice project's launch.json in the session that created this repo; open `review.html` directly if `question-review` doesn't start.

### Phase 1: skeleton + ingestion
FastAPI + Postgres/pgvector + Alembic. `POST /documents` with idempotent hashing, chunking with metadata, local embeddings, `GET` to inspect chunks.

**Done (2026-11-12).** What exists and why:
- Tables `documents` (content_hash unique, path, title, source_url, raw_text never modified) and `chunks` (seq, heading_path, text, char_count, `vector(384)` embedding, embed_model). No vector index: exact scan over ~1.4k chunks is fast and exact (matters for eval). Migration `0001` creates the `vector` extension.
- `app/ingestion/markdown.py` (pure): front matter → title; strips Hugo shortcodes, HTML, comments, mermaid, link URLs; keeps panel headers / `<summary>` labels as bold lines (context like "GitLab LTD (UK)"); keeps Markdown emphasis + tables. Splits by headings (`heading_path` = "Title > H2 > H3"), packs paragraphs/lists/tables up to `chunk_max_chars`=1200, splits only blocks that are too long.
- Embedding input = `heading_path + "\n\n" + text` (a bare "Limit is set to $60/year" gets its subject). Stored `text` is the passage alone.
- Embeddings: fastembed 0.9 + `BAAI/bge-small-en-v1.5` (384 dims, ONNX, CPU), model baked into the image at `/opt/models`. **Questions use `embed_query`, chunks `embed_passages`** (bge uses a query prefix). `get_embedder()` is a cached FastAPI dependency; tests override it with `FakeEmbedder` (hash-based vectors).
- `POST /documents` (multipart `file`, optional `path`, `source_url`): .md only (415), ≤ 5 MB (413), UTF-8 + some readable text (422), same bytes → 200 + existing document (no re-embedding). `GET /documents`, `/documents/{id}`, `/documents/{id}/chunks`, `/health`.
- `scripts/ingest_corpus.py` verifies the corpus hashes, then ingests through the same service as the API, with the manifest's handbook URL as `source_url`.
- Corpus result: 60 docs → **1407 chunks** (median ~600 chars). Tests assert every eval quote (sources + near misses, 86) lands inside one chunk, so a cited passage is never split.
- First look at plain vector search (preview of Phase 2): hotel/home-office questions hit the right sections (similarity ~0.69-0.72); "how many weeks of parental leave" ranks **country pages** (Canada/NZ/Korea, ~0.91-0.93) above the company-wide 16-weeks passage; the unanswerable "payday" question tops out at ~0.68, but so do some answerable ones, so a refusal threshold on raw similarity alone won't separate them.

### Phase 2: retrieval + retrieval eval (no LLM, free)
Vector search, then hybrid (vectors + full-text, e.g. reciprocal rank fusion). Measure recall@k / MRR against the question set. Tune chunk size/overlap here.

**Done (2026-11-12).**
- `app/retrieval/search.py`: `vector` (cosine via pgvector), `keyword` (Postgres full-text on generated column `chunks.search_tsv`, migration `0002`, GIN index; heading weight B, text A; the question's terms are OR-ed because `plainto_tsquery` ANDs them), `hybrid` (weighted RRF, k=60, pool 50 per method). Every `Hit` carries `similarity` (cosine) whatever the method: it's the refusal signal.
- `GET /search?q=&k=&method=&keyword_weight=`: retrieval alone, for inspection.
- Eval: `docker compose run --rm -T api uv run --frozen python -m eval.retrieval [--methods vector,hybrid:0.3] [--show q30,u02]`. Free, ~1 min. Relevant chunks = chunks containing a label quote (same normalization as the tests). Metrics in `eval/metrics.py` (pure, tested): hit@k (any evidence in top k), full@k (all evidence: matters for multi/conflict), MRR, traps (a near-miss ranked above the evidence), refusal signal. Results in gitignored `eval/results/retrieval/`.
- **Result (bge-small, 1200-char chunks, 53 answerable questions):**

  | method | hit@1 | hit@5 | hit@10 | full@5 | MRR | missed@10 |
  |---|---|---|---|---|---|---|
  | vector | 0.62 | **0.92** | 0.96 | 0.83 | **0.75** | q30, q53 |
  | keyword | 0.40 | 0.68 | 0.77 | 0.58 | 0.52 | 12 |
  | hybrid (kw weight 1) | 0.57 | 0.81 | 0.87 | 0.72 | 0.67 | 7 |
  | hybrid (kw weight 0.3) | 0.64 | 0.89 | 0.91 | 0.81 | 0.74 | 5 |

  **Default = vector.** Hybrid never beat it at any keyword weight. Likely cause: `ts_rank` has no IDF, so common words ("team member", "GitLab") weigh as much as rare ones ("USERRA", "walking pads"). Also `pads/treadmills` is tokenized as a file path, not two words. Real BM25 (with IDF) is the open experiment before giving up on keywords.
- Weak spots (vector): **q30** (parental leave length): the company-wide "16 weeks" chunk is not in the top 10; country pages (Canada, NZ, Korea... similarity ~0.90-0.93) win. **q53** (gym/HealthKick) missed. Tricky questions hit@5 only 0.60; full@5 for conflict 0.33 (both sides of a contradiction rarely both retrieved) and multi 0.62.
- Refusal signal (vector top similarity): answerable min 0.693 / median 0.776; unanswerable median 0.705 / max 0.836. Best threshold 0.693 refuses 4/8 unanswerable, 0/53 answerable wrongly, but it's fit on the same 61 questions and sits exactly at the answerable minimum: too fragile to use alone. Phase 4 needs more than a similarity cutoff.
- Unanswerable questions retrieve their decoys (u01 RSU→option vesting, u07 timekeeping, u08 software list): the LLM step must resist them.

### Phase 3: answer generation
Gemini structured output: `answer` + `claims[{chunk_id, quote}]` or `refused` + reason. Raw response saved, validation retry, versioned prompts, tokens/cost/latency recorded per call.

### Phase 4: verification checks (the differentiator)
- `quote_not_in_source`: quote not found in the cited chunk (after normalizing case/spacing)
- `cited_unretrieved`: cites a chunk that wasn't retrieved
- `number_unsupported`: number/date in the answer appears in no cited chunk
- `low_retrieval_confidence`: best match too weak → refuse before calling the LLM

### Phase 5: eval harness
`answer` (real Gemini, freezes raw answers) / `score` (pure, re-runnable, free). Metrics:
- **Headline: confident-wrong answers** = wrong answer with no flag and no refusal
- correct refusals on unanswerable questions; over-refusals on answerable ones
- citation validity rate, recall@k, cost and latency per question
- correctness judged deterministically where possible (key facts/numbers); any LLM judge is spot-checked by hand

### Phase 6: hardening
Structured JSON logging, prod Docker image, CI, README with real eval numbers and a findings → fixes table. Optional minimal chat page, only after the eval works.

## Scope discipline

Build in phase order. No auth, queues, frontend, or agent features until the eval harness (Phase 5) works.

| Phase | Goal | Status |
|---|---|---|
| 0 | Corpus snapshot + labeled question set | done (60 pages, 61 verified questions) |
| 1 | Skeleton + ingestion (chunks, embeddings, pgvector) | done (60 docs, 1407 chunks, 24 tests) |
| 2 | Hybrid retrieval + retrieval eval | done (vector default: hit@5 0.92, MRR 0.75; hybrid measured, lost) |
| 3 | Gemini answers with cited quotes | not started |
| 4 | Verification checks + refusal policy | not started |
| 5 | Eval harness (confident-wrong answers headline) | not started |
| 6 | Hardening: logging, Docker prod, README numbers, CI | not started |

Update the Status column when a phase is done.

## Working style

- Small, reviewable changes; run the relevant tests after each change.
- When unsure about a design choice that affects the data model or API contract, ask before implementing.
- The user is a beginner building a freelance portfolio: briefly explain *why* when introducing a new pattern or tool (embeddings, pgvector, hybrid search, recall@k...).
