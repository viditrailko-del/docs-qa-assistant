# Deploying the public demo (free: Render + Neon + Gemini free tier)

Everything in the repo is ready (`render.yaml`, `docker/demo-start.sh`, `.github/workflows/demo-reset.yml`).
What's left needs your own accounts. About 20 minutes, most of it waiting for the first build.

Secrets (the database URL, the Gemini key, the reset token) go only into Render and GitHub, never into a file or
a chat.

## 0. Push `main` to GitHub
Render deploys from the GitHub repo, so the demo code has to be there first. CI boots the production image on a
fresh database and replays every recorded question before you deploy anything.

## 1. The Gemini key
Reuse the demos' existing free-tier key (its own Google project, billing OFF). The demos share that project's
daily quota: this one allows 30 live questions a day (usually 1 call each, at most 3), next to the support
demo's 30 tickets. With billing off, the worst a visitor can do is use up the day's free quota; it can never
cost money.

## 2. Neon (the database)
1. Open the existing Neon project (Postgres 16, AWS US East) → **Databases** → **New database** → name it
   `rag_demo`, owner the project's default role. One project, one more database: no new project.
2. **Connect** → pick the database `rag_demo` → turn **Connection pooling OFF** → copy the connection string
   (`postgresql://...neon.tech/rag_demo?sslmode=require...`). Use the direct one, not the pooled one: our driver
   (psycopg 3) uses prepared statements, which the pooler can break. The app converts `postgresql://` to its
   own driver by itself, so paste it as it is.
3. Nothing else to set up: the first boot runs the migrations (which enable the `pgvector` extension, supported
   on Neon) and loads the documents. The demo's data is about 7 MB (60 documents, 1,407 passages with their
   384-number vectors), far below the free plan's storage.

## 3. Render (the app)
1. **New → Blueprint** → pick the `docs-qa-assistant` repo, branch `main`, blueprint path `render.yaml` (the
   default). Render shows one free web service, `docs-qa-assistant-demo`, in Virginia.
2. It asks for two secret values: `DATABASE_URL` = the Neon string, `GEMINI_API_KEY` = the demos' key.
   `DEMO_MODE` and `DEMO_RESET_TOKEN` fill themselves in.
3. **Apply**. The first build takes ~5-10 minutes (the image carries the embedding model). When the service
   shows **Live**, open its URL. The first start loads the documents into Neon by itself (rehearsed locally
   at the free instance's 0.1 CPU / 512 MB: healthy in ~45 s, ~320 MB used).

## 4. The nightly reset (GitHub Actions)
1. Render → the service → **Environment** → copy the value of `DEMO_RESET_TOKEN`.
2. GitHub repo → **Settings → Secrets and variables → Actions**:
   - **Secrets** tab → New secret `DEMO_RESET_TOKEN` = that value.
   - **Variables** tab → New variable `DEMO_URL` = the Render URL, without a trailing slash.
3. **Actions** tab → **demo-reset** → **Run workflow** once, to check it goes green.

## 5. Send me the URL
I'll check it from here with `python -m demo.check_replays <url>`: every recorded question, the page, the
guards, then one live question and the reset.

## Good to know
- **Free instances sleep** after about 15 minutes with no visitors. The next visit wakes it in about a minute
  (it loads the embedding model). Open the link yourself a minute before you share it on a call. Nothing keeps
  it awake on purpose.
- **Every push to `main` redeploys** (`autoDeploy: true`).
- **Live mode caps**: 30 live questions a day for everyone, 5 per visitor per hour, 300 characters each.
  Change them with `DEMO_LIVE_DAILY_LIMIT` / `DEMO_LIVE_PER_IP_PER_HOUR` / `DEMO_QUESTION_MAX_CHARS` on Render.
  The per-visitor cap trusts the `X-Forwarded-For` header, which a determined visitor can fake; the daily cap
  is the one that protects the quota. `POST /ask` spends the same quota in demo mode, and uploads are off.
- **If the documents, the chunker or a prompt file change**, rebuild the index and the recordings on the eval
  machine (`python -m app.demo.index export`, `python -m demo.build_replay`): the boot refuses vectors that
  don't match the corpus, and a replay whose prompt changed ends "stopped safely" instead of showing an answer
  to a question the model never saw. CI catches both before a deploy.
- **Scheduled workflows pause** after 60 days without commits to the repo. GitHub emails you; re-enable it in
  the Actions tab.
