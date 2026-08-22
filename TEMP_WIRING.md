# TEMPORARY WIRING — remove when the backend's Groq is healthy

**Added 2026-08-17.** The prod backend's live answer stream (`/ask/regular` →
`stream_groq_response`) is failing platform-wide (Groq HTTP errors, surfaced as
`"Error: Service unavailable."`). The background **cognitive** pipeline on the
same backend still works, so Tom temporarily falls back to it.

## What is wired

In `src/live_avatar.py`:

- `query_spiked_rag` captures the `X-Cognitive-Key` response header. When the
  live answer comes back as one of the backend's short `Error: ...` strings, it
  polls `GET /cognitive?cognitive_key=...` (same user JWT) until the background
  answer lands, then returns that instead.
- `_poll_cognitive_answer` — the poll loop itself.
- Config (env-overridable):
  - `AGENT_COGNITIVE_FALLBACK` (default `false`) — the kill switch.
  - `AGENT_COGNITIVE_FALLBACK_TIMEOUT_S` (default 25)
  - `AGENT_COGNITIVE_POLL_INTERVAL_S` (default 1.5)
- The one-shot retry in `_generate_grounded_reply` is suppressed while the
  fallback is on, so a failed turn costs one fallback budget, not two.

## Cost while active

When the live stream fails, a spoken answer takes roughly 5–30s extra (the
cognitive pipeline is a wide-retrieval + rerank + generation pass bounded at
35s server-side). Tom will feel slow but not dumb. When the live stream is
healthy, the fallback never triggers and adds zero latency.

## How to remove

1. Set `AGENT_COGNITIVE_FALLBACK=false` in `.env` (instant, no code change), or
2. Delete `_poll_cognitive_answer`, the fallback block at the end of
   `query_spiked_rag` (marked `TEMPORARY (see TEMP_WIRING.md)`), the three
   `AGENT_COGNITIVE_*` config lines, and restore the plain retry condition in
   `_generate_grounded_reply`. Then delete this file.

Root cause to actually fix: the backend's Groq key/quota — check Cloud Run logs
for `Groq HTTP error: <status>` from `services/ai_helpers.py`.

## Interaction with the single-shot company_knowledge path (2026-08-17, separate change)

`company_knowledge` turns no longer run a second Gemini compose call — the
reply shape (word budget, closing question) is folded directly into the
question sent to `/ask/handsfree`, and that text is spoken close to verbatim
(see `_generate_grounded_reply` in `live_avatar.py`). If the cognitive fallback
above triggers, the text it returns comes from the backend's *background*
pipeline (a different, longer system prompt with no word cap and no
tailored closing question) — so a fallback answer will not have Tom's usual
shape. The word-budget backstop still applies (prevents an unbounded read),
but the closing question may be missing on fallback turns. Acceptable given
the fallback is itself temporary and currently disabled.
