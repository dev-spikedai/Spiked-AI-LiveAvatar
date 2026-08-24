# feat/fastembeddings

Stripped fork of `backend-one` (`prod` @ `ce21921`) carrying **only** the
`/ask/regular` answer-generation path, run as a standalone service so its
`WarmIndex` can hot-load every active bot's chunk embeddings into memory at
startup and keep retrieval off the request path entirely.

## Why this exists

`build_rag_context` in backend-one blocked every `/ask/regular` request on
a `supabase.rpc("hybrid_match_chunks_by_sources", ...)` call — a synchronous
network+DB round trip **not** wrapped in `asyncio.to_thread`, so it also
stalled the event loop for every other in-flight request on that worker.
Benchmarked RAG time-to-first-byte: 1.4–3.8s (see the "Tom Latency
Benchmark" artifact). This service replaces that hop with an in-process
cosine search over embeddings already sitting in RAM.

## What's here vs. what's gone

Kept: `main.py`, `core/{auth,config,dependencies,kyc_database,persona_styles,prompts}.py`,
`models/schemas.py`, `routers/search.py` (ask_regular + /cognitive poll only —
`/ask/handsfree`, `/followup`, `/askBeyond` removed), `services/{ai_helpers,rag_classifier}.py`,
and the new `services/warm_index.py`.

Deleted: every other feature module (`ai_assistant`, `drive_sync`,
`learning_module`, `mindmap_hub`, `notetaker`, `proposal_hub`), every other
router (`clients`, `documents`, `gap_analysis`, `help_bot`, `kyc_router`,
`meeting_goals`, `meeting_logs`, `settings`, `stripe_webhooks`, `training`,
`tutorials`), document-processing/OCR/Drive/Stripe services, and the legacy
`app.py` monolith (unused — `main.py` never imported it).

## Known gaps — read before treating this as prod-ready

- **`WarmIndex` is dense-only.** backend-one's retrieval is hybrid (dense +
  BM25, fused with RRF). This fork's warm path does cosine top-k only — no
  sparse arm, no RRF. Good enough to validate the latency win; needs a local
  BM25 pass (e.g. `rank_bm25`) before it's a quality-neutral replacement for
  the RPC on real traffic.
- **Refresh is a 5-minute freshness check, not event-driven.** A newly
  ingested or deleted doc can be stale in the warm index for up to
  `WARM_INDEX_TTL_SECONDS` (default `300`). The first request after expiry
  falls back to the RPC while a single background refresh reloads the user.
  No webhook off `sources.ingestion_status` yet.
- **Cold users fall through to the Supabase RPC**, same as backend-one today
  — `build_rag_context` only skips the DB when `warm_index.search()` finds
  the user already loaded.
- **Two-generation pipeline (Groq draft + Gemini cognitive refine via
  `/cognitive` poll) is unchanged.** Collapsing that into a single formatted
  generation was the other half of the latency discussion — not done here;
  this fork only attacks the retrieval-side cost.

## Running it

Alongside the rest of the local stack (`npm run dev` from the `twin-avatar`
root spawns it as the `fast-embed` pane on `:8090`), or standalone:

```
cd feat-fastembeddings
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
# needs: SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY, GEMINI/GOOGLE API key, OPENAI_API_KEY
uvicorn main:app --port 8090 --reload
```

`GET /health` reports `warm_bots`, the count of users currently hot in memory.
