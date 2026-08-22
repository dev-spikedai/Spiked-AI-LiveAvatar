# Handoff — pluggable providers + shared prompt document (2026-08-22)

Branch `refactor/providers`, 5 commits off `main` (`4ea5537`..`66a8421`), pushed.
45 files, +4101/-930. **129 Python tests + 14 browser tests, all green.**
Not merged, not deployed.

Scope: turn the service from "the LiveAvatar backend" into a shared agent core
that any avatar vendor plugs into, then collapse the avatar persona into one
document. Benchmarked against the `simli` branch, which had solved the same
problem a second way.

## What the two branches revealed

`main` (LiveAvatar/HeyGen) and `origin/simli` (Simli + Cartesia) are **parallel**,
not sequential — merge base `8b25f6f`, main +6, simli +2. simli was missing ~1400
lines of main's agent work (`call_intelligence`, autospeak, insights, `/ws/rep`,
`/invoke`, teardown). So simli was used as a *reference implementation*, not a
base: `refactor/providers` branched off `main` and simli's ~250 lines of provider
code were ported in as adapters.

Having two independent implementations is what identified the real seam. Both had
converged on the same control-socket message set without coordinating — that, not
the Python class layout, is the interface a video vendor actually implements.

## The design, in one idea

Vendors disagree about where the avatar ends and the voice begins. HeyGen FULL is
a face *and* a voice. Simli is a face only. Anam is a face, optionally a voice,
optionally a brain. Modelling all three as "an avatar" forces an interface that
fits none of them.

So the core **always emits a sentence stream**, and each adapter declares
`accepts: "text" | "audio"`. Text providers get the sentence; audio providers get
PCM synthesized by a `TtsProvider`. The agent loop never knows which is live.
`src/core/speech.py` is the fork. Anam and Simli both want PCM16 LE / 16 kHz /
mono, which is exactly what Cartesia already produced, so nothing resamples.

Three swappable roles, selected per run:

| role | adapters |
|---|---|
| video/face | `liveavatar` (default), `anam` (driven + native), `simli` |
| TTS/voice | `cartesia` — only for providers that just lip-sync |
| answer engine | `spiked` (/ask), or the vendor's own LLM |

`registry.resolve()` rejects invalid combinations at `/start` rather than at the
first turn. That exists for a specific failure: an audio-only provider with no
TTS raises nothing anywhere — it just mouths silence for an entire meeting.

Every provider is **two halves**: a Python adapter and a browser module under
`public/providers/`. `public/avatar.js` is now a vendor-agnostic shell that
imports whatever `/credentials` names. Vendor SDKs load on demand, so a Simli
page never downloads LiveKit.

## Anam, specifically

One adapter, two modes, switched by a single field:

- **driven** — `llmId: "CUSTOMER_CLIENT_V1"` (the documented "no brain" sentinel).
  Our engine answers; we stream it in with talk commands. Drop-in for HeyGen FULL.
  **This is the launch config.**
- **native** — a real model UUID + `systemPrompt`. Anam composes and speaks; we
  send the gated transcript with `sendUserMessage`.

Both set `disableInputAudio`. The page never listens in any configuration —
Recall → Deepgram owns that — so **the turn gate survives native mode**.
`agent_policy` still decides *whether* Tom speaks; native mode only gives up
control of *what he says*.

## Verified vs not

**Verified.** Default resolution is LiveAvatar/text/no-TTS/spiked with the 60s
keepalive. Text-mode wire output is byte-identical to pre-refactor. `/credentials`
still carries the legacy `livekit_*` keys. The browser suite executes the real
`avatar.js` and `liveavatar.js` and confirms `stop_listening` on connect,
`speak_text` on the right topic, and echoes round-tripping both `turn_id` and
`chunk_id`. Persona blocks render byte-identical to the literals they replaced.

**Not verified.** No live meeting has run on this branch. Real LiveKit/WebRTC
handshakes, HeyGen echo timing, and autoplay in Recall's headless Chrome are all
untested. **Nothing Anam has ever touched a real session** — it is written from
the published API, which already proved wrong three times (`streamToVideoElement`
not `streamToVideoAndAudioElements`; message role `persona` not `assistant`; no
documented interrupt API). Expect a live run to find more.

## Prompt library (Priority 5)

The persona lived in three inline copies that could drift, and a fourth consumer —
the vendor system prompt for Anam native — was never set at all, so that path
would have run with an empty persona while the others carried the real one.

`prompts/avatar_prompt.md` is now the single source. `### name` blocks, `>` lines
are editor notes that never reach a model, placeholders validated at load so a
typo fails on startup instead of shipping a literal brace to Gemini. Editing the
persona is a one-file change with no code edit.

`tests/test_persona.py` greps `src/` for persona phrases and fails if a fifth copy
appears. That test, not the refactor, is what keeps this true.

## Things that would have broken production

Caught before deploy, listed because they show where this area is sharp:

1. **`prompts/` was not in the Dockerfile.** `persona.py` reads it at import, so
   the container would not have started at all. Fixed, with a test on the COPY.
2. **`avatar_speak_end` carried no `chunk_id`.** An audio provider's ack would
   read as single-shot and release the floor mid-answer.
3. **Anam message role.** Checking `"assistant"` meant native-mode echo
   suppression would never fire — Tom would hear himself through Deepgram and
   answer his own reply.

## Known open issues

- **`live_avatar.py` is still 2803 lines** and `src/app.py` does not exist. This
  is blocked on a real trade, not effort: ~40 tests monkeypatch
  `live_avatar.query_spiked_rag`, `_generate_grounded_reply`,
  `_take_floor_and_speak` and `_judge_interjection` **by module attribute**.
  Moving that code silently disconnects the patches — the tests keep passing
  while testing nothing. Doing it properly means moving code and re-pointing
  every patch target in one change. Do not attempt it piecemeal.
- `src/core/floor.py` is 487 lines; `src/agent_policy.py` is 711 and untouched.
- Anam driven mode acks a sentence when Anam *accepts* it, not when it has been
  heard — Anam has no per-chunk playback event. Pacing is open-loop there and
  barge-in cuts less precisely than on LiveAvatar.
- `anam.js` / `simli.js` have no browser tests. Deliberate: neither has run
  against its vendor, and a green stub test would imply confidence that does not
  exist.
- Per-client provider config reads a `client_provider_configs` table that does
  not exist yet. Absent table is handled as "no overrides", so this is inert
  until someone creates it.
- Repo, Cloud Run service and `PUBLIC_BASE_URL` still say `LiveAvatar`. Only the
  internal name changed (`SpikedMeetingAgent`). Renaming the rest touches the
  Recall webhook URL and needs coordination.

## Next steps, in priority order

1. **Run one live LiveAvatar meeting on this branch.** Everything up to the
   vendor boundary is confirmed; the boundary itself is not.
2. **Then one live Anam driven-mode meeting.** Needs `ANAM_API_KEY` and
   `ANAM_AVATAR_ID`. Verify the SDK export shape against the installed package
   before trusting `anam.js`.
3. Only after 2: native mode, and measure whether `MESSAGE_HISTORY_UPDATED`
   arrives before the bot's own audio returns through Deepgram. If it is late,
   native mode is not safe to ship.
4. The `src/app.py` split, as one deliberate change (see above).
5. Decide whether Simli ships or stays a reference adapter.

## Where to read next

- `CLAUDE.md` — conventions, traps, and the monkeypatch warning.
- `docs/PROVIDER_REFACTOR_PLAN.md` §9 — current status of every step.
- `docs/CONTROL_PROTOCOL.md` — the wire contract, and the two details in it that
  are load-bearing.
- `tests/browser/README.md` — what the browser suite does and does not cover.
