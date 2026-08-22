# Provider Refactor Plan — pluggable video / TTS / answer providers

Status: in progress on `refactor/providers`. Written 2026-08-22.
Benchmark inputs: `main` (LiveAvatar/HeyGen) and `origin/simli` (Simli + Cartesia).

**Progress against §7:** steps 1, 3, 4, 5 done; step 2 partial; steps 6–8 open.
See §9 at the bottom for exactly what is and is not wired.

---

## 1. Goal

Turn `Spiked-AI-LiveAvatar` into one repo with a provider-agnostic agent core and
swappable adapters, so a run can be launched against any combination of:

- **video/face** provider — LiveAvatar (HeyGen), Anam, (Simli as reference)
- **TTS** provider — only needed when the video provider takes audio rather than text
- **answer engine** — Spiked `/ask`, or the video vendor's own brain

selected **per run**, from the request payload or per-client config, not per deploy.

### Non-goals (this pass)

- Wiring the fast-embeddings answer engine. The `AnswerEngine` interface must
  leave a slot for it; no implementation is built now.
- Shipping Simli to production. It is ported only as the proof that the
  audio-input path is real, and as a second implementation that keeps the
  abstraction honest.
- Touching `feat-fastembeddings` / `backend-one`. Scope is this repo only.

---

## 2. What the two branches taught us

`main` and `origin/simli` are **parallel**, not sequential — merge base `8b25f6f`,
main +6 commits, simli +2, both current as of 2026-08-21. simli lacks ~1400 lines
of main's agent work (`call_intelligence.py`, autospeak, insights, `/ws/rep`,
`/invoke`, `/api/active-runs`, teardown, streaming sentence handler, keepalive).

**Do not merge simli forward.** Branch `refactor/providers` off `main` and port
simli's ~250 lines of provider code in as an adapter.

### The one decision everything hangs on: text vs audio

| provider | takes | mechanism |
|---|---|---|
| LiveAvatar FULL | **text** | `avatar.speak_text` over LiveKit data channel |
| LiveAvatar LITE | **audio** | `agent.speak` with PCM |
| Anam (brainless) | **text** | `createTalkMessageStream()` / `streamMessageChunk()` / `endMessage()` |
| Anam (passthrough) | **audio** | `createAgentAudioInputStream()` / `sendAudioChunk()` / `endSequence()` |
| Simli | **audio** only | raw PCM frames over its p2p WebRTC WS |

Anam and Simli both want **PCM16 signed LE, 16 kHz, mono** — byte-identical to
what simli's Cartesia path already emits (`CARTESIA_SAMPLE_RATE=16000`,
`output_format: {container: raw, encoding: pcm_s16le}`). One TTS shim serves both.

So the core does **not** pick text or audio. It always emits a **sentence stream**,
and the adapter declares what it accepts:

```python
class VideoProvider:
    accepts: Literal["text", "audio"]
```

`accepts == "audio"` → the executor routes sentences through a `TtsProvider`
before handing frames to the adapter. `accepts == "text"` → forward directly.
The agent loop never knows which is live.

### Every video provider is two halves

`public/avatar.js` differs almost totally between the branches — LiveKit `Room`
subscription vs a hand-rolled `RTCPeerConnection` against Simli's p2p WS. A
provider ships as a **pair**: `src/providers/video/<name>.py` +
`public/providers/<name>.js`.

### The control WS protocol is already the adapter interface

Both branches independently converged on the same message set
(`avatar_speak`, `avatar_audio`, `avatar_speak_end`, `avatar_interrupt`,
`avatar_speak_started`, `avatar_speak_ended`, `heard`, plus main's `mute`/`rep`).
Version it, freeze it, and the browser half becomes a pure protocol implementation.

---

## 3. Target layout

```
src/
  core/                     # provider-agnostic; nothing here imports an adapter
    agent_policy.py         # unchanged — floor, echo, governor, endpointing
    call_intelligence.py    # unchanged
    asr.py                  # Deepgram ParticipantTranscriber (extracted)
    meeting.py              # Recall.ai deploy/leave/teardown (extracted)
    runs.py                 # _ACTIVE_RUNS registry, credentials, run lifecycle
    turn.py                 # TurnExecutor — the only thing that touches providers
    protocol.py             # control-WS message constructors + version const
  providers/
    base.py                 # VideoProvider / TtsProvider / AnswerEngine ABCs
    registry.py             # name -> class, per-run resolution
    video/
      liveavatar.py         # from main
      anam.py               # new
      simli.py              # from origin/simli (reference tier)
    tts/
      cartesia.py           # from origin/simli
    answer/
      spiked.py             # from main: query_spiked_rag + source_ids + fallback
      anam_native.py        # delegated — Anam's own LLM
      # fastembed.py        # SLOT, deliberately unimplemented
  app.py                    # FastAPI routes only
public/
  avatar.html               # dynamic-imports the module named by /credentials
  providers/
    liveavatar.js
    anam.js
    simli.js
```

`src/live_avatar.py` (3306 lines) is decomposed; it stays as a thin
back-compat shim re-exporting from `src.app` until the deploy is switched.

---

## 4. Interfaces

```python
# src/providers/base.py

@dataclass
class VideoSession:
    provider: str                 # goes to the browser, selects the JS module
    credentials: dict             # provider-specific, returned by /credentials
    session_id: Optional[str]

class VideoProvider(ABC):
    name: str
    accepts: Literal["text", "audio"]
    browser_module: str                       # "/providers/anam.js"
    audio_format: Optional[AudioFormat]       # required when accepts == "audio"

    async def create_session(self, ctx: RunContext) -> VideoSession: ...
    async def keepalive(self, session: VideoSession) -> None: ...   # no-op default
    async def close(self, session: VideoSession) -> None: ...

class TtsProvider(ABC):
    name: str
    sample_rate: int
    async def stream(self, text: str) -> AsyncIterator[bytes]: ...  # PCM16 mono LE

class AnswerEngine(ABC):
    name: str
    mode: Literal["stream", "delegated"]

    # mode == "stream": yields sentences; core speaks them
    def stream_answer(self, ctx: TurnContext) -> AsyncIterator[str]: ...

    # mode == "delegated": vendor composes AND speaks; returns when done
    async def delegate_turn(self, ctx: TurnContext) -> DelegatedResult: ...
```

`TurnExecutor` is the single seam the agent loop calls:

```python
async def run_turn(run, turn_id, ctx) -> Optional[str]
```

Two paths inside it, both wrapped by the **existing** duplicate guard, echo
registration, `_push_rep`, watchdog and floor release from `_dispatch_reply` /
`_speak_chunk` — those move into `core/turn.py` unchanged:

- **composed**: `AnswerEngine.stream_answer` → sentences → (`TtsProvider` if
  `accepts=="audio"`) → `VideoProvider`
- **delegated**: `AnswerEngine.delegate_turn` → vendor speaks → core awaits the
  provider's speech-ended event, and takes the spoken text from the vendor's
  message-history event to feed `EchoSuppressor` / `SpeechGovernor`

---

## 5. Anam adapter — verified details

Session token: `POST https://api.anam.ai/v1/auth/session-token`,
`Authorization: Bearer <API key>`, body `{ personaConfig: {...}, sessionOptions: {...} }`
→ `{ "sessionToken": "<JWT>" }`. `personaConfig` is either ephemeral (`name`,
`avatarId`, `voiceId`, `llmId`, `systemPrompt`) or stateful (`personaId`).
Relevant extras: `avatarModel` (`cara-3` | `cara-4` | `cara-4-latest`),
`skipGreeting`, `maxSessionLengthSeconds`, `voiceDetectionOptions`, `tools`;
`sessionOptions.region` (`eu` | `us`), `videoQuality`, `egress`.

**One adapter, two modes, switched by `llmId`:**

| mode | `llmId` | who answers | mechanism |
|---|---|---|---|
| `anam` (driven) | `CUSTOMER_CLIENT_V1` | Spiked `/ask` | `createTalkMessageStream()` → `streamMessageChunk(text, false)` → `endMessage()` |
| `anam_native` | a real LLM UUID + `systemPrompt` | Anam's brain | `sendUserMessage(transcript)`; Anam composes and speaks |

Both set `disableInputAudio: true` — the avatar page never hears the meeting;
Recall → backend → Deepgram owns listening in every configuration. **Your turn
gate survives native mode**: `agent_policy` still decides *whether* the turn
happens, `sendUserMessage` just hands off *what to say* about it.

Mappings that fall out cleanly:

- `chunk_id` ≡ Anam's `correlationId`; `AnamEvent.TALK_STREAM_INTERRUPTED` ≡
  barge-in. `_speak_chunk`'s await-then-continue loop maps 1:1 onto
  `TalkMessageStream.getState()` / `isActive()`.
- `MESSAGE_HISTORY_UPDATED` is the only source of spoken text in native mode —
  it must feed `EchoSuppressor.note_bot_speech`, or Tom hears himself.

**To verify before building:** the custom-TTS guide says `enableAudioPassthrough: true`
is set at session-token creation, but the session-token API reference does not
list the field. Confirm placement (top level vs `sessionOptions`) against a live
call. Passthrough also imposes an **800 ms audio buffer before render** — a
latency floor that text mode does not pay, which is a reason to prefer
`accepts="text"` for Anam.

---

## 6. Per-run provider selection

Resolution order, first hit wins:

1. explicit fields on `POST /start` / `POST /create-live-avatar-bot`:
   `video_provider`, `tts_provider`, `answer_engine`
2. per-client config row in Supabase, keyed on `client_id`
3. env defaults (`DEFAULT_VIDEO_PROVIDER=liveavatar`, …)

Resolved once at deploy time into `run["providers"]: ProviderSet` and never
re-read. `GET /api/runs/{run_id}/credentials` grows a `provider` and
`browser_module` field; `avatar.html` does
`const mod = await import(creds.browser_module)` and hands it the credentials
blob plus the control socket. Credential keys stay provider-namespaced
(`livekit_url` / `livekit_token` vs `anam_session_token` vs `simli_session_token`)
rather than being flattened into a lowest-common-denominator shape.

Combination validity is checked at resolve time, not at first speak:
`accepts == "audio"` with no `tts_provider` → 400 at `/start`, not dead air
mid-meeting. `answer_engine == "anam_native"` with `video_provider != "anam"` → 400.

---

## 7. Migration sequence

Each phase leaves `main` deployable and behaviourally identical.

1. **Freeze the protocol.** Extract `core/protocol.py` from the message literals
   already in `live_avatar.py` + `avatar.js`. Pure rename, no behaviour change.
2. **Extract the core.** Move ASR, Recall, run registry, floor/watchdog/echo out
   of `live_avatar.py` into `src/core/`. Still hard-wired to LiveAvatar. Existing
   tests (`test_agent_policy`, `test_agent_routing`, `test_autospeak`,
   `test_grounded_reply`) must pass untouched — that is the regression gate.
3. **Introduce the ABCs + `TurnExecutor`,** with `liveavatar` + `spiked` as the
   only registered adapters. Deploy. Nothing observable changes.
4. **Port `cartesia` + `simli`** from `origin/simli`. First real exercise of the
   `accepts == "audio"` path, and of a second `browser_module`.
5. **Build the `anam` adapter** in driven mode (`CUSTOMER_CLIENT_V1`). This is
   the launch target with the Spiked answer engine.
6. **Add `anam_native`** as a delegated `AnswerEngine`, incl. the
   `MESSAGE_HISTORY_UPDATED` → echo-suppressor wiring.
7. **Turn on per-run selection** (§6) and per-client config.
8. Retire the `src/live_avatar.py` shim.

---

## 8. Risks / open items

- **Anam passthrough field placement and the 800 ms render buffer** — see §5.
- **Native-mode echo suppression** depends on `MESSAGE_HISTORY_UPDATED` arriving
  before the meeting audio echoes back through Deepgram. If it lands late,
  Tom can hear himself. Needs a live latency measurement before shipping step 6.
- **Native mode loses word-limit control.** `AGENT_MAX_REPLY_WORDS` and
  `normalize_reply` are enforced in `compose_reply`; in delegated mode the only
  lever is `systemPrompt` + `directorNotes`. Accept, or don't ship native.
- **Keepalive is provider-specific.** LiveAvatar needs the 60 s ping
  (`LIVEAVATAR_KEEPALIVE_INTERVAL_S`); Simli uses `maxIdleTime` on the token;
  Anam uses `maxSessionLengthSeconds`. Hence `keepalive()` on the interface
  rather than a shared task.
- **`origin/simli`'s `src/rag_client.py` is dropped** (direct-Supabase +
  bot-service-user modes). Confirm nothing depends on the
  `liveavatar-bot@spiked.ai` service user before deleting.
- **Both branches drift daily.** Steps 1–3 should land fast; the longer
  `refactor/providers` lives beside `main`, the more of main's agent work has to
  be re-reconciled.

---

## 9. Status — what is wired on `refactor/providers`

Test suite: **98 passed** (78 pre-existing, 4 updated for the filler feature,
16 new). Every step below left it green.

### Done

- **Step 1 — protocol frozen.** `src/core/protocol.py`, `PROTOCOL_VERSION = 1`.
  Constructors verified to emit dicts byte-identical to the literals they
  replaced. Two messages added for delegated mode: `avatar_user_message`
  (out) and `avatar_vendor_reply` (in).
- **Step 3 — provider layer.** `src/providers/base.py` (`VideoProvider`,
  `TtsProvider`, `AnswerEngine`, `RunContext`, `TurnContext`, `VideoSession`,
  `DelegatedResult`), `src/providers/registry.py` with resolve-time validation.
- **Step 4 — Simli + Cartesia ported** from the `simli` branch, both halves.
- **Step 5 — Anam adapter built**, driven and native modes, both halves.
- **Browser seam.** `public/avatar.js` is now a provider-agnostic shell that
  dynamically imports `browser_module` from `/credentials`; the three vendor
  modules live in `public/providers/`. Vendor SDKs load on demand, so a Simli
  page never downloads LiveKit. New route `GET /providers/{module}.js`,
  filename-constrained.
- **Per-run selection (step 7, backend half).** `video_provider`,
  `tts_provider`, `answer_engine` accepted on `/start` (JSON and form) and
  `/create-live-avatar-bot`, resolved once per run into `run["providers"]`.
  Keepalive and teardown both go through the interface now.

### Partial

- **Step 2 — core extraction.** Only `ParticipantTranscriber` and the Deepgram
  constants moved (`src/core/asr.py`). `live_avatar.py` is 3306 → ~3190 lines.
  Recall/meeting, the run registry, and floor/watchdog/echo are **not** moved.
  There is no `core/turn.py` and no `TurnExecutor` — the speak path still runs
  through `_dispatch_reply` / `_speak_chunk` in `live_avatar.py`.

### Not started

- **`TurnExecutor`.** Without it the `AnswerEngine` interface is defined but not
  on the live path: `_generate_grounded_reply` still calls `query_spiked_rag`
  directly. `SpikedAnswerEngine` wraps it and is registered, but nothing calls
  the wrapper yet, and `AnamNativeAnswerEngine` is not constructed anywhere.
  **This is the next piece of real work** — until it lands, `answer_engine` is
  accepted at `/start` and then ignored.
- **Step 6 — native-mode echo wiring.** The protocol carries
  `avatar_vendor_reply` and anam.js emits it, but no handler feeds it to
  `EchoSuppressor`. Native mode is not safe to run until that exists.
- **Per-client provider config** (the Supabase lookup in §6). Only explicit
  request fields and env defaults resolve today.
- **Step 8.** `live_avatar.py` is not yet a shim.

### Untested against live vendors

Everything Anam is written from the published API and **has never been run
against a real session**: the session-token call, `CUSTOMER_CLIENT_V1`, the
talk-stream chunking, `sendUserMessage`, and the `@anam-ai/js-sdk` import.
The SDK's exact export shape (`createClient`, `AnamEvent`,
`streamToVideoAndAudioElements`, `interruptPersona`) needs checking against the
installed version before the first live run. Simli/Cartesia are ports of code
that ran on the `simli` branch, but not in this shell.

One known design gap in `anam.js` driven mode: Anam has no per-chunk
"finished playing" event, so a chunk is acknowledged when it is accepted into
the talk stream rather than when it has been heard. The backend's
sentence-by-sentence pacing therefore runs open-loop against Anam — sentences
are handed over as fast as they are produced and Anam queues them. This is
fine for correctness (Anam does not overlap its own audio) but means barge-in
mid-answer cuts less precisely than it does on LiveAvatar.
