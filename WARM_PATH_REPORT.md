# The Warm Path

What a separate, stripped-down `/ask/regular` backend with an in-memory retrieval index actually bought Tom — and what it didn't.

*feat/fastembeddings, forked from backend-one @ ce21921 — 2026-08-20*

## Headline numbers

| | |
|---|---|
| **1.5–2x** | faster time-to-first-byte, local vs. prod, across 3 clean benchmark runs |
| **~4x** | drop on the RAG/Groq leg specifically, measured live in a real Tom meeting |
| **9** | real bugs found and fixed along the way, one of them a total demo blocker |
| **1** | provider swap tested and rejected on real data, not assumed |

## 1. Why a separate backend at all

Backend-one's `/ask/regular` blocks every request on `supabase.rpc("hybrid_match_chunks_by_sources", ...)` — a synchronous network round trip *not* wrapped in a thread offload, so it stalls the event loop for every other in-flight request too. Measured across the Tom Latency Benchmark's V2 and V3 sessions, that single call averaged **2.89s**, ranging 1.39–3.78s, for functionally identical queries.

Rather than trying to speed up that call in place inside the full production codebase, we forked backend-one down to just the answer-generation path and rebuilt retrieval as an in-memory operation: a warm-loaded dense index instead of a database round trip on every question.

## 2. The mechanism

**Cold path (prod backend-one)** — 2.89s avg
1. Embed the question locally (CPU)
2. Network round trip to Postgres
3. Hybrid dense+BM25 RRF query executes
4. Rows stream back over the wire

**Warm path (this fork)** — ~150–250ms
1. Embed the question locally (CPU)
2. Cosine similarity against an in-process matrix
3. Top-k selected, no network hop

Four accounts' chunk embeddings (`spiked@test`, `test@gmail.com`, `test@spiked.ai`, `sai@spiked.ai`, resolved by email via Supabase's admin API) are held in memory as plain `float32` arrays, loaded once at startup and cached to disk so a restart doesn't mean a cold crawl again. A cold user still falls through to the original RPC — nothing breaks, it just isn't fast.

**Scope note carried since the first line of code**: this replicates only the *dense* arm of backend-one's hybrid retrieval. No BM25, no RRF fusion. Good enough to prove the latency case — not yet a quality-neutral replacement, and it showed up as real `HALLUCINATED` classifier verdicts during testing that the RPC path didn't produce for the same questions.

## 3. What actually moved, end to end

Retrieval got dramatically faster in isolation. The full voice-in→voice-out pipeline moved less, because retrieval stopped being the bottleneck — something else always is.

| Stage | V3 baseline (n=7) | Tonight, live meeting (n=2) |
|---|---|---|
| **RAG / Groq ttfb** — the leg this fork touches | 3.06s | **0.74s** |
| Classification — Gemini turn-intent call, untouched | 1.32s | 1.48s |
| Dispatch → speaking — LiveKit + HeyGen TTS start, untouched | 1.10s | 1.09s |
| **Total, speech-end → speaking** | ~6.1s | ~5.8s |

RAG went from the single largest cost in the pipeline to the smallest of the four measured legs. Classification (Gemini 3.5 Flash-Lite) is now the biggest number on the board, followed by the TTS dispatch floor — both entirely outside this fork's reach. That's the honest shape of the win: real, but bounded by everything it doesn't touch.

## 4. What building it in isolation actually surfaced

The bigger, less predictable value: standing up a separate instance and stress-testing it end to end found real bugs a latency number alone wouldn't have caught.

| Found | Severity | What it was |
|---|---|---|
| Missing `/documents`, `/websites` | **demo blocker** | Stripped out as "not needed for ask_regular" — except the orchestrator calls them *first* to resolve source_ids. Every question fail-closed to "I don't have verified information," silently, for the whole session. |
| Blocking startup | perf | The warm-load awaited a full crawl before uvicorn would accept any connection — every dev restart was a ~60s dead window that looked exactly like a hang. |
| OFFSET pagination | perf | Got slower every page against a filtered join; reliably hit a Postgres statement timeout past ~3,000 rows. Switched to keyset pagination. |
| pgvector as string | correctness | PostgREST returns embedding columns as literal text, not a JSON array. Crashed the warm-load until parsed explicitly. |
| Blocking cognitive write | perf | The live answer's stream couldn't start until a Supabase "pending" row was written and confirmed — on every single request, for a race that didn't need a synchronous write to close. |
| Case-sensitive filter | correctness | `ingestion_status` is stored as `COMPLETED`, not `completed` — the warm-load silently matched zero rows until caught. |
| Port mismatch | **demo blocker** | The dev-pane launcher ran this service on :8090; both the orchestrator's and frontend's env files pointed at :8000. Nothing was reachable and it looked like a crash. |
| Uncached KYC overlay | perf | Base user settings were cached; the per-client overlay read wasn't — a full Supabase round trip on every question regardless of warm/cold retrieval state. |

## 5. A swap we tested and didn't make

With RAG no longer the bottleneck, classification (Gemini 3.5 Flash-Lite, ~1.3s) became the largest labeled stage — and Groq's raw inference speed made it a tempting target. Rather than assume, we benchmarked the orchestrator's actual classify-and-draft-reply shape (structured JSON output, ~300 tokens) against every general-purpose model currently on Groq.

| Model | Avg | Result |
|---|---|---|
| **Gemini 3.5 Flash-Lite** (current) | **1.05s** | Fastest and fully reliable |
| Groq `gpt-oss-120b` | 1.75s | Reasoning overhead outweighs raw hardware speed |
| Groq `gpt-oss-20b` | 2.18s | One run spiked to 7.16s — unreliable |
| Groq `qwen3.6-27b` | — | Failed JSON validation outright |
| Groq `compound-mini` | 1.57s | Slower, and dropped required schema fields |

Groq wins for long free-text generation, which is exactly why it's still what generates Tom's live answers. It loses here because every available model either reasons before answering or isn't tuned for small structured output — the opposite shape of task. Left as Gemini.

## 6. What it isn't, yet

- **Dense-only retrieval** — no BM25 arm, no RRF fusion. Real quality gap, not just a theoretical one — produced `HALLUCINATED` classifier verdicts in testing that the full hybrid RPC didn't.
- **Four hardcoded accounts** — scoped deliberately for this demo. Scaling to real traffic means resolving which accounts to warm dynamically, not a fixed email list.
- **Single-instance only** — in-memory index plus a local disk cache. Fine for one demo process; means nothing on a real multi-instance deployment without a shared warm-index service.
- **Two-generation chain intact** — the Groq-draft-then-Gemini-refine cognitive pipeline was disabled for the demo (polling turned off), not collapsed into one call. Still there, still slow, if re-enabled.

## So, was it useful?

Yes, on both counts it set out to test. The retrieval leg is a real, repeatable, measured win — not marketing, not a guess. And building it in isolation, then wiring it into the real live-meeting path end to end, found eight production-adjacent bugs a synthetic benchmark alone would never have surfaced, including the one thing that would have made the demo fail outright. The honest ceiling is that total end-to-end latency barely moved, because retrieval was never the whole pipeline — it was just the loudest part of it until now.

---
*Compiled from this session's benchmark runs, live-meeting logs, and the Tom Latency Benchmark (V2/V3) artifact. Numbers not independently re-verified after this report was written.*
