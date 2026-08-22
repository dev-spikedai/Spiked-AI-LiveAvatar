# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this service is

A meeting agent ("Tom") that joins Zoom/Meet/Teams via Recall.ai, listens through
Deepgram, decides whether to speak, answers from the Spiked document RAG, and
renders a talking avatar. The avatar vendor is pluggable — LiveAvatar (HeyGen) is
the default, Anam and Simli are also implemented. Internally the service is named
**SpikedMeetingAgent**; the repo, Cloud Run service and `PUBLIC_BASE_URL` still
say `LiveAvatar` and renaming those is a deliberate, deploy-affecting change.

## Commands

```bash
pip install -r requirements-dev.txt

# Python suite (the regression gate; must be green before and after any change)
.venv/Scripts/python.exe -m pytest -q          # Windows
pytest -q                                       # elsewhere
pytest tests/test_grounded_reply.py -q          # one file
pytest tests/test_persona.py::test_missing_block_fails_loudly -q   # one test

# Browser suite — the other half of the provider seam
node --test tests/browser/*.test.mjs
# NB: `node --test tests/browser` fails; pass files or a glob.

uvicorn src.live_avatar:app --host 0.0.0.0 --port 8080 --reload
gcloud builds submit --config cloudbuild.yaml   # deploy
```

There is no linter or formatter configured.

## Architecture

### The three provider seams

The core is provider-agnostic; adapters plug into it. Read
`docs/PROVIDER_REFACTOR_PLAN.md` before changing anything under `src/providers/`.

- **video/face** — `src/providers/video/{liveavatar,anam,simli}.py`
- **TTS/voice** — `src/providers/tts/cartesia.py`, needed only when the video
  provider cannot speak for itself
- **answer engine** — `src/providers/answer/spiked.py`, or the vendor's own LLM

The load-bearing idea: **the core always emits a sentence stream**, and each
video adapter declares `accepts: "text" | "audio"`. Text providers get the
sentence; audio providers get PCM synthesized by a `TtsProvider`. The agent loop
never knows which is live. `src/core/speech.py` is where that fork happens.

A provider is **two halves**: a Python adapter plus a browser module under
`public/providers/<name>.js`. `/api/runs/{id}/credentials` tells the page which
one to import. Adding a vendor means adding both files and registering it — not
editing `public/avatar.js`.

`src/providers/registry.py` validates combinations at `/start`, not at first
turn. That exists because an audio-only provider with no TTS raises nothing — it
just mouths silence for a whole meeting.

Provider selection is per-run: explicit request field → per-client Supabase
config → env default (`pick_provider` in `src/live_avatar.py`).

### Layout

- `src/live_avatar.py` (~2800 lines) — still the service module: FastAPI routes,
  retrieval (`query_spiked_rag`), Gemini prompting, turn-taking, websockets.
- `src/core/` — provider-agnostic. **Nothing here may import `src/providers/`.**
  `protocol.py` (wire contract), `speech.py` (dispatch), `floor.py` (floor state,
  speech dispatch, notification fan-out), `asr.py` (Deepgram), `runs.py`
  (registry + teardown), `persona.py`.
- `src/agent_policy.py` — the turn-taking brain: `FloorState`, `EchoSuppressor`,
  `SpeechGovernor`, `SustainedSpeechDetector`, `evaluate_turn`, `compose_reply`.
  Pure logic, no I/O. This is the actual product; treat it carefully.
- `prompts/avatar_prompt.md` — the single source for the avatar's persona.

### Control socket

`docs/CONTROL_PROTOCOL.md` is the contract between backend and page. Two things
in it are load-bearing and easy to break:

- `chunk_id` on `avatar_speak_ended` distinguishes "this sentence finished" from
  "the whole answer finished". Drop it and the floor releases mid-answer.
- `avatar_vendor_reply` is the *only* source of spoken text in delegated mode. It
  feeds `EchoSuppressor`; without it the agent hears itself through Deepgram and
  answers its own reply.

### Run state

`_ACTIVE_RUNS` (in `src/core/runs.py`) is **in-memory and process-local**. Cloud
Run is pinned to one instance for this reason. It also means a rolling deploy
never hands an old run to a new process, so "runs created before X" fallbacks are
dead code.

## Conventions in this codebase

- **Comments are one line.** Rationale belongs in `docs/*.md`, not inline. The
  pre-refactor `live_avatar.py` grew to 3300 lines of multi-paragraph comments;
  the codebase is deliberately moving away from that. Do not match the older
  heavy-comment style you will still find in places.
- **No wrapper function called from exactly one place** — inline it.
- **Keep files well under 1000 lines.**
- **Never write persona text into a prompt.** Add a block to
  `prompts/avatar_prompt.md` and call `persona.block(...)`. `tests/test_persona.py`
  greps `src/` and fails if a second copy appears.
- Tests assert on *behaviour with a stated failure mode*, not on implementation.
  A test that reimplements the logic it is checking is worse than no test.

## Traps

- **`tests/` monkeypatch `live_avatar.query_spiked_rag`, `_generate_grounded_reply`,
  `_take_floor_and_speak` and `_judge_interjection` by module attribute** (~40
  sites). Moving those functions to another module silently disconnects the
  patches — the tests keep passing while testing nothing. Any such move must
  re-point every patch target in the same change. This is what currently blocks
  finishing the `src/app.py` split.
- `prompts/` must be `COPY`d in the `Dockerfile`. `persona.py` reads it at
  import, so a missing copy is a service that will not start.
- Deepgram rejects `utterance_end_ms < 1000` with a 400, which looks exactly like
  an agent that hears nobody. `src/core/asr.py` clamps it.
- LiveAvatar allows one concurrent session per key. A leaked session blocks every
  later start with `4032 Session concurrency limit reached`, so
  `VideoProvider.close()` in teardown matters more than it looks.
- `TEMP_WIRING.md` documents a temporary cognitive-answer fallback
  (`AGENT_COGNITIVE_FALLBACK`, default off). Remove it when the backend's Groq
  path is healthy.
- Anam has no per-chunk "finished playing" event, so driven mode acks a sentence
  on acceptance rather than on playback. Pacing there is open-loop.

## State of the provider work

`docs/PROVIDER_REFACTOR_PLAN.md` §9 is the current status. Two things are worth
knowing before trusting anything:

- **Nothing Anam has run against a live session.** It is written from the
  published API, which already proved wrong three times (`streamToVideoElement`
  not `streamToVideoAndAudioElements`; message role `persona` not `assistant`; no
  documented interrupt API).
- Simli/Cartesia are ports of code that ran on the `simli` branch, but not in the
  current shell.
