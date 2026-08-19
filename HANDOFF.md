# Handoff — RAG latency, streaming, and turn-taking hardening (2026-08-18/19)

Session scope: Tom (the LiveAvatar meeting bot) was giving RAG answers that
diverged wildly from the console's, taking 9–15s to speak, and losing its
session after ~3 minutes idle. All three are fixed, plus a round of turn-taking
and voice-command features. This doc is the state of things at the end of the
session — what changed, why, what's confirmed live vs. still open.

## The one bug that mattered most

`_generate_grounded_reply` (`src/live_avatar.py`) was sending a ~90-word
persona/format instruction paragraph as the `question` field to backend-one's
`/ask/regular`. That field is used **verbatim** for both the retrieval
embedding and the final LLM prompt (`routers/search.py` in `forks/backend-one`)
— there's no separate field for instructions. The paragraph dominated the
embedding, pulled different chunks than the console's clean question, and
inflated generation time. Replaced with a 24-word persona one-liner
(`persona_hint` in `_generate_grounded_reply`) that includes an explicit word
count — an earlier version of the one-liner dropped the word count too and
answers ballooned to 135–249 words as a result; that's restored now.

**Confirmed live**: RAG backend_total dropped from 6–9s to 1.4–4s for
comparable questions once the paragraph was gone.

## Latency fixes, in order of impact

1. **The persona-paragraph fix above.**
2. **Sentence-by-sentence TTS dispatch.** `query_spiked_rag` now takes an
   optional `on_sentence` callback and fires it on every sentence boundary as
   the backend streams, instead of buffering the whole answer. `_speak_chunk`
   sends one sentence and awaits its `avatar_speak_ended` (matched by a
   `chunk_id`, not just `turn_id`) before sending the next, so audio never
   overlaps. First words now start playing well before the full answer has
   generated. Wired through `_generate_grounded_reply` (`run`/`turn_id`
   params, opt-in — omit both to get the untouched non-streaming behavior,
   which is what Level 1 insight prefetch still does) and both callers
   (`_finalize_turn`'s `respond()`, `/invoke`'s `deliver()`).
   **Confirmed live, dispatching multiple chunks correctly** — not yet
   confirmed whether HeyGen's `speak_text` has any audible gap between
   sequential chunks (couldn't tell from logs alone).
3. **Single-shot classify+compose merge** for `coaching`/`meeting_context`/
   `social`/`command` intents — one Gemini call drafts the reply inline
   instead of a second compose call after classification. `company_knowledge`
   is unaffected (still needs retrieval before it can answer).
4. **Source-id cache prewarm** at bot deploy time (`_deploy_live_avatar_bot`)
   — the first real question no longer pays `resolve_source_ids`'s ~0.7–1.9s
   inline.
5. Endpoint is `/ask/regular`, not `/ask/handsfree` — measured faster for this
   workload in a real side-by-side, not assumed.

### A real, accepted timing-log caveat

`[RAG][TIMING]` now reports `backend_total` / `playback_wait` / `wall_total`
separately. `backend_total` is pure backend response time (what the old
`total` meant). `playback_wait` is time spent inside `on_sentence` waiting for
TTS chunks to actually finish playing — real time, but it's Tom talking, not
the backend being slow. Before this split, a wall-clock `total=21s` on a
streamed turn looked like a huge regression; it was mostly playback.

## Reliability fixes

- **LiveAvatar session keep-alive.** LiveAvatar auto-closes idle sessions;
  Tom's silent-unless-addressed design meant every meeting eventually hit
  that. `_keep_avatar_session_alive` pings `POST /v1/sessions/keep-alive`
  every `LIVEAVATAR_KEEPALIVE_INTERVAL_S` (default 60s). **Confirmed live.**
- **`avatar.js` reconnect** on `RoomEvent.Disconnected` — re-fetches fresh
  credentials and retries with backoff, since the old session may be dead
  server-side. Not yet exercised live (no disconnect happened during testing).
- **Barge-in hangover tolerance.** `SustainedSpeechDetector` required an
  unbroken streak of voiced VAD frames for the full 700ms — a single dropped
  frame (plosive, breath, resampling artifact) reset the whole counter to
  zero, making it nearly impossible to trigger on real speech. Now tolerates
  a short run of unvoiced frames (`UNVOICED_TOLERANCE_FRAMES`) before
  resetting. **Confirmed live, firing repeatedly and correctly this session.**
- **Follow-up phrase-shape check now decays with distance from the anchor
  turn.** The very first nameless follow-up after Tom finishes speaking
  accepts *any* reply (question or declarative) — "I want to see the
  integrations" in reply to "what would you like to see next?" no longer
  silently fails. Later follow-ups in the same exchange still require
  question-shape or a known direct-reply starter, so an unrelated aside
  doesn't keep riding the window. **Confirmed live.**
- **Streaming-aware barge-in fix.** An in-place interrupt during `SPEAKING`
  releases the floor (state → `LISTENING`) without changing `active_turn_id`
  at all — the streaming dispatch loop's checks originally only looked at
  `active_turn_id`, so a barge-in landing between two chunks (not mid-chunk)
  wouldn't have stopped the next one from being spoken. Both `_speak_chunk`
  and the `on_sentence` handler now also check `state`. Not yet isolated in a
  live test (no long multi-chunk answer + well-timed interrupt combo tried).

## New feature: "Tom, stay quiet for N seconds"

- `detect_mute_command` (`agent_policy.py`) — deterministic regex, no Gemini
  call. Requires the wake name. Recognizes a wide phrase set (stay/be/go
  quiet, stop talking/listening, don't talk/speak/respond/answer, shut up,
  pause/wait/hold on, give me a sec/minute, go silent, stand by/down) and a
  duration in seconds/minutes/**hours** (all three units, both digit and
  **spelled-out** durations — "thirty seconds" parses the same as "30
  seconds"; `_words_to_digits` handles compounds like "forty five"). Clamped
  5s–1hr.
- `_set_mute`/`_clear_mute` (`live_avatar.py`) manage a self-expiring
  `asyncio.Task` per run, push `agent_muted`/`agent_unmuted` over both the rep
  console socket and the avatar's own control socket.
- **The mute gate lives in the actual meeting video feed, not just the
  console** — `avatar.html`/`avatar.js` render a bottom-center countdown ring
  everyone in the call sees, driven by `_push_control`.
- "Ask Tom" (`/invoke`) always clears an active mute unconditionally, before
  the busy-check — an explicit request should never be silently eaten by a
  stale mute.
- **Confirmed live, multiple durations, correct expiry, correct clearing.**

## Classification prompt fixes

Two live misfires, same root cause each time — the prompt's own category
definitions didn't mention the case, so the model guessed:

- **"Who are you?"** was going to `company_knowledge` → RAG → "I don't have
  information about a Tom in the provided documents." Added identity
  questions to the `social` definition.
- **"What is the current sentiment?"** (no possessive) went to
  `company_knowledge` and hallucinated from an unrelated document, while
  **"What is *my* current sentiment?"** correctly hit `meeting_context` and
  used real `CallIntelligence` data. Added sentiment/mood/engagement to the
  `meeting_context` definition explicitly. **The live-data linkage already
  existed** — this was purely a routing gap, not missing plumbing.

Both **confirmed live** on retest.

## Cutoff bug (backstop trimming)

Removing the word-budget instruction from the retrieval question (the fix
above) meant backend-one started returning full document-length answers,
which the local word-count backstop trimmed with a raw `words[:N]` slice —
producing answers that visibly trailed off mid-sentence ("...three CI
architecture orchestrates."). Fixed in both the non-streaming path and the
streaming `on_sentence` handler: the backstop now stops at the last complete
sentence that fits the budget, never mid-clause. **Confirmed live** (and unit
tested with a real multi-sentence case, not just a synthetic word blob).

## backend-one: a real perf finding, not yet merged

Branch `perf/unblock-live-answer-from-cognitive-write` in `forks/backend-one`
(2 commits, not pushed): `launch_cognitive_background` was `await`ed directly
in front of every live Groq stream and blocked on a full Supabase upsert (the
`/cognitive` "pending" row) that has nothing to do with the answer being
waited on. Moved the write into the background task instead of the awaited
path — `/cognitive` already treats a missing row as "keep polling," not a
failure, so this is safe. `PERF_NOTES.md` on that branch also documents: the
query-embedding step is CPU-bound (`intfloat/e5-large-v2`, no GPU by default)
and a plausible source of RAG time variance — GPU-move is the safe fix (same
vector space, no re-embed); switching embedding providers entirely is a
migration, not a quick win. Also flagged, unrelated to latency: `get_embeddings`
unconditionally prefixes `"passage: "` even when the caller already sent
`"query: "`, producing a double-prefixed embedding — possible retrieval-quality
bug, not fixed.

**Not pushed or opened as a PR — needs your go-ahead.**

## Known open issues (not fixed this session)

1. **"Ask Tom" can be silently cancelled while thinking.** Barge-in during
   `THINKING` cancels any pending generation, including an explicit `/invoke`
   — observed live: clicked Ask Tom, someone talked ~3s later, the answer was
   dropped with no feedback. This is existing, deliberate design for
   voice-triggered turns ("the room moved past the question"), but applying
   it identically to an explicit button click is a real UX question, not
   obviously correct. **Needs a decision**: exempt invoke-triggered turns from
   THINKING-phase barge-in, require a longer sustained threshold, or leave as
   is.
2. **Possible echo bleeding into the barge-in detector.** One live turn had
   `[Barge-In] sustained_ms=700` fire 0.58s after Tom started speaking — too
   fast to be a genuine reaction, and `SustainedSpeechDetector` operates on
   raw audio energy from Recall's separated human channel, with zero
   awareness of `EchoSuppressor` (which is text-based and doesn't protect the
   VAD path at all). Unconfirmed without audio inspection — could be genuine
   rapid interruption during active testing instead.
3. **Dismissal-phrase misclassification, never fixed.** "ok tom, that's
   useful, we'll circle back later" still costs a full classification round
   trip and risks being misjudged as `respond` instead of `acknowledge`/
   `silent` — flagged early in the session, a deterministic detector (same
   shape as `detect_mute_command`) was proposed but never built.
4. **Streaming duplicate-reply suppression gap.** `_finish_streamed_reply`
   can't call `governor.is_duplicate()` before speaking, since there's no
   full text until streaming is already underway — a literally-repeated
   question within the duplicate window won't be pre-emptively suppressed on
   the streaming path the way it is on the non-streaming one. Accepted, not
   fixed.
5. **HeyGen inter-chunk audio quality unconfirmed** (see streaming section
   above) — need a native English listen-through of a long, multi-sentence
   streamed answer to know if it's seamless.

## Reference: the benchmark artifact

Published earlier this session, three versions (V1 baseline from a separate
debugging session / V2 measured mid-session / V3 roadmap) — worth republishing
with tonight's final numbers as a V4 if useful going forward:
https://claude.ai/code/artifact/97d7830a-be27-4d51-b097-af82fbe9b1be

## Suggested next steps, roughly in priority order

1. Decide + fix the invoke-vs-barge-in-during-THINKING question (#1 above).
2. Push and open the backend-one PR (or explicitly decide not to).
3. Build the dismissal-phrase detector (#3 above) — same pattern as the mute
   command, well understood, not started.
4. Get a real listen-through of a long streamed answer to confirm HeyGen
   handles sequential `speak_text` calls cleanly.
5. Investigate the echo/barge-in-timing question (#2) if it recurs.
