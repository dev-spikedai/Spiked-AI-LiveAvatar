# feat-fastembeddings deployment plan: test on personal GCP first, prod later

**Status: test-deploy target identified, code fixes landed, `gcloud run deploy`
not yet run.** This supersedes the original GCS-backed full-corpus warm-index
plan below the line — that design was replaced mid-session by a simpler
lazy per-user hot-load refactor that needs no GCS bucket at all. Kept the
old plan further down for history/context, not as active guidance.

## Hard boundary — read this before touching any deploy command

**Under no circumstances does this work publish to, redeploy, or repoint
`spikedai-production-application` (backend-one) in production.** That
service keeps running exactly as it is, untouched, indefinitely as far as
this plan is concerned. `feat-fastembeddings` is not a migration of
backend-one's traffic — it is a new, standalone service: **LiveAvatar's (and
the Console's, for testing) answer backend**, serving the `/ask/regular` +
`/cognitive` + `/documents` + `/websites` slice on its own Cloud Run URL.
Nothing in this plan ever calls `gcloud run deploy` against backend-one's
service name/project, and no step here modifies `BASE_URL`/
`SPIKED_BACKEND_URL`/`VITE_FAST_ASK_API_URL` in any production config —
repointing those (in test or in prod) is a separate, explicit decision the
user makes later, not an automatic consequence of this deploy.

## Decision: two-stage rollout

Explicit direction this session: deploy to a **personal/testing GCP project
first**, validate there, and only decide on production (the actual
spikedai-production project) afterward. The two are not the same project —
confirmed via `gcloud` this session, see below. Do not point real customer
traffic at the test-stage deployment.

## Stage 1 target: personal GCP project (current)

- **Project**: `einsteini-485316` (authenticated as `einsteini@spiked.ai`)
- **Region**: `us-central1`
- **Artifact Registry repo**: `spikedai-backend` (Docker, already exists,
  currently empty — no images pushed yet)
- **Cloud Run**: no service deployed in this project/region yet (confirmed via
  `gcloud run services list --region us-central1` → 0 items)
- This project also hosts unrelated services (einsteini-backend, LinkedIn
  bot backend/frontend, stripe-backend, email-orchestration-server) — mostly
  in `asia-south1`/`asia-east1`. `spikedai-backend`'s existence in
  `us-central1` specifically (matching `BASE_URL`'s region) suggests it was
  pre-provisioned for exactly this deploy, but nothing has been pushed to it.

**Not the same project as production.** `core/config.py`'s `BASE_URL`
default and the LiveAvatar architecture doc reference Cloud Run URLs under
project numbers `822359826336` / `409019309412`, neither of which is
`einsteini-485316`. Confirm explicitly before ever repointing real traffic
here — this project is additive/isolated from whatever's already live.

## What shipped this session (code, not yet deployed)

1. **Embedding model benchmark** (`scripts/embedding_model_bench.py`,
   `scripts/embedding_model_bench_real.py`) — `e5-small-v2` vs `e5-large-v2`
   on real chunks from `sai@spiked.ai`'s `SPIKEDAI_PRODUCT_MASTER.pdf`:
   ~8x faster query-embed latency (110ms → 13ms avg), no wrong-document
   top-1 misses. `e5-large-v2` stays load-bearing everywhere else (every
   other account's retrieval, the hybrid RPC's expected 1024-dim param, the
   cognitive pipeline's grounding check, and the stored `chunks.embedding`
   column itself is `public.vector(1024)`, confirmed read-only via the
   Supabase PostgREST OpenAPI schema) — this is additive, not a replacement.
2. **Locked-down small-model demo path** (`services/warm_index.py`): a
   second, self-contained `e5-small-v2` pipeline scoped to
   `SMALL_MODEL_DEMO_USER_IDS = {"ac73135a-7d06-4182-b67f-59a8db613265"}`
   (`sai@spiked.ai`). Re-embeds chunk `content` entirely in memory — never
   touches Supabase's `embedding` column, no writes anywhere. Wired into
   `build_rag_context` (`routers/search.py`) ahead of the large-model path,
   so this one account never pays the large-model embed cost at all.
   Verified end-to-end through the real console (real auth, real
   `client_id`/`source_ids`), not just standalone scripts.
3. **Lazy per-user warm loading** (`services/warm_index.py`): removed the
   hardcoded `TARGET_EMAILS`/`_resolve_user_ids_by_email`/`_load_targets`
   demo-account scoping entirely. `kick_off_warm_load(user_id)` fires a
   background load on any user's first warm-index miss; every request after
   that from the same user hits the warm path. No fixed account list, no
   GCS snapshot needed — this is what replaced the full-corpus-eager-load
   design below. Trade-off: first request per user per process is still a
   cold RPC fallback, and each Cloud Run instance builds its own warm set
   independently (see session-affinity note below).
4. **Auth-bypass fix** (`Dockerfile`): removed hardcoded
   `ENV APP_ENV=development`. `core/dependencies.py` and
   `core/kyc_database.py` fall back to `DEFAULT_TEST_USER_ID` — a real,
   unauthenticated pass — for any request whenever `APP_ENV=="development"`
   (missing header, expired/invalid token, or any exception during
   validation). Shipping the old Dockerfile as-is would have deployed a full
   auth bypass. `core/config.py` already defaults `APP_ENV` to
   `"production"` when unset, so removing the override is fail-secure.
5. **`PORT` default** (`Dockerfile`): added `ENV PORT=8080` to match
   `EXPOSE 8080`. Cloud Run injects its own `$PORT` regardless; this only
   fixes bare `docker run` without `-e PORT`.

## Remaining before `gcloud run deploy` (stage 1, test project)

- [ ] Confirm this is really where you want to test (project
      `einsteini-485316`, region `us-central1`, repo `spikedai-backend`) —
      asked and confirmed this session, listed here for the record.
- [ ] Supply `SUPABASE_KEY` and `SUPABASE_SERVICE_ROLE_KEY` via
      `--set-env-vars`/`--set-secrets` at deploy time — not baked into the
      image (`.dockerignore` already excludes `.env`).
- [ ] Supply `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` the same way.
      Consider a **separate Groq key from production** for this test
      deploy — production backend-one shares the current key, and this
      session already hit a live 429 from local testing colliding with it.
- [ ] `--memory` ≥2Gi (not the 512Mi default) — `e5-large-v2` +
      `e5-small-v2` (once the demo account warms) + growing per-user chunk
      content in memory adds up under torch.
- [ ] `--min-instances=1` — without it, Cloud Run scales to zero on idle and
      the next request pays full model-load cold-start time; also, every
      scale-up event spins a fresh instance with an empty warm index.
- [ ] `--session-affinity` — new consideration specific to lazy per-user
      loading (didn't matter under the old full-corpus-eager plan, where
      every instance had every account warm from boot): without it, a
      user's second request can land on a different instance than the one
      that warmed them, and they're cold there too.
- [ ] Confirm `/documents` and `/websites` are live on the deployed
      revision, not just `/ask/regular` — the orchestrator calls both
      first to resolve `source_ids`; their absence was a past demo blocker.
- [ ] Decide who repoints traffic at this deployment once it's up — nothing
      currently points at it, this is a fresh URL.

## Before stage 2 (production), separately

Stage 2 means promoting `feat-fastembeddings` to a real, durable Cloud Run
service that LiveAvatar and the Console point at for real traffic — it does
**not** mean touching backend-one (see hard boundary above; that stays out
of scope permanently, not just until stage 2).

- Decide whether `SMALL_MODEL_DEMO_USER_IDS` stays a single hardcoded
  account (fine for a demo) or needs a real per-account opt-in mechanism
  before more users get the small-model path.
- Explicit, separate decision: repointing `SPIKED_BACKEND_URL`
  (LiveAvatar) and `VITE_FAST_ASK_API_URL` (Console) at the new service's
  URL, in the real production project — not automatic, not part of this
  plan's deploy steps.

---

## [Superseded] Original plan: GCS-backed full-corpus eager load

The section below is the original plan from earlier in this project, before
the lazy per-user refactor replaced it. Kept for history only — none of this
is the current direction.

`feat-fastembeddings`'s `warm_index.py` (`_WarmIndex`) was originally scoped
as a demo hack: chunk embeddings for 4 hardcoded email addresses
(`TARGET_EMAILS`) were loaded into memory once at process start, with a local
pickle file as a dev-restart convenience. The original fix considered here
was a **full-corpus eager load**: a standalone script would embed every
account's chunks, upload `embeddings.npz` + `chunks.parquet` to GCS, and the
service would download the whole corpus at startup before serving traffic,
gated by `APP_ENV`.

That design is no longer the plan. It was replaced by the lazy
`kick_off_warm_load` per-user mechanism above, which needs no GCS bucket, no
snapshot build script, and no `APP_ENV`-gated loader branch — every account
warms itself on first contact instead of the service needing to know the
full account list up front. The `APP_ENV`-gated branch this plan originally
proposed for `warm_index.py` was never built; `warm_index.py` does not
branch on `APP_ENV` in the current code at all.
