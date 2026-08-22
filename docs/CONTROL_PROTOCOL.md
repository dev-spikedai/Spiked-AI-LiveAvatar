# Control socket protocol (v1)

The wire contract between the backend and the avatar page. This is the provider
seam: a new video vendor is "a module that speaks this protocol", not a fork of
`avatar.js`. `main` (LiveAvatar) and the `simli` branch converged on this
message set independently, which is the evidence it is the real interface.

Two sockets, deliberately separate:

| socket | audience | slots | may drop? |
|---|---|---|---|
| `/ws/control/{run_id}` | the avatar page | one | no — it is the speech transport |
| `/ws/rep/{run_id}` | the rep's console | many | yes, freely |

Only the control socket is part of the provider contract.

## Backend → page

| type | fields | notes |
|---|---|---|
| `avatar_speak` | `text`, `turn_id`, `chunk_id?` | Text providers render it. Audio providers treat it as "an utterance begins" and wait for frames. |
| `avatar_audio` | `data` (base64 PCM), `turn_id` | Audio providers only. Many per utterance, ordered. |
| `avatar_speak_end` | `turn_id`, `chunk_id?` | Audio providers only. No text equivalent — a text provider's own TTS decides when it has stopped. |
| `avatar_interrupt` | — | Barge-in. Stop now, discard buffered audio. |
| `avatar_user_message` | `text`, `turn_id` | Delegated mode only. Not words to say — the thing the vendor's brain should answer. |
| `heard` | `speaker`, `text`, `reply`, `reason` | Gate verdict for the in-meeting overlay. In avatar mode this is the only path a transcript reaches any frontend. |
| `agent_muted` | `muted_until_epoch_ms`, `seconds` | Epoch deadline, not a duration: the page renders a countdown and a duration would drift by the message's flight time. |
| `agent_unmuted` | `reason` | |

`agent_muted`/`agent_unmuted` go to the control socket as well as the rep
console because the mute countdown is drawn on the meeting video feed, which
every participant sees — not just the rep.

## Page → backend

| type | fields | notes |
|---|---|---|
| `avatar_speak_started` | `turn_id`, `chunk_id?` | Audio is actually playing. Moves the agent to SPEAKING. |
| `avatar_speak_ended` | `turn_id`, `chunk_id?` | **The `chunk_id` is load-bearing** — see below. |
| `avatar_speak_interrupted` | — | Barge-in observed by the page, or the vendor reporting its own interrupt. |
| `avatar_vendor_reply` | `turn_id`, `text` | Delegated mode only. What the vendor's brain actually said. |

Inbound messages carrying a `turn_id` are dropped unless it matches
`run["active_turn_id"]`: a late echo from a superseded turn must not move the
floor.

### Why `chunk_id` on `avatar_speak_ended` matters

- **With** a `chunk_id`: one sentence of a longer answer finished. Unblock the
  dispatch loop waiting on that chunk; say nothing about the floor. The loop —
  not the socket handler — decides when the whole turn is done.
- **Without** one: the whole answer finished. Release the floor and cancel the
  watchdog.

Collapsing the two cases strands the agent: an audio provider that acks without
a `chunk_id` releases the floor while the rest of its answer is still queued.

### `avatar_vendor_reply` is not optional

In delegated mode the core never sees the reply text on its way out. This event
is the only source, and it feeds `EchoSuppressor`. Without it the agent hears
its own voice return through Deepgram and treats it as a new utterance. Arriving
late is a correctness bug, not a cosmetic one.

## Audio format

`avatar_audio` frames are PCM 16-bit signed little-endian, 16 kHz, mono.

Simli's WebRTC endpoint and Anam's audio-passthrough mode independently require
exactly this, and it is what the Cartesia adapter requests, so nothing is
resampled anywhere on the speak path. A provider needing something else declares
it via `AudioFormat` and the registry refuses a TTS pairing that does not match.

## Versioning

`PROTOCOL_VERSION` is bumped only on a breaking change. The page reports the
version it was built against, so a stale cached `avatar.js` against a newer
backend logs a loud mismatch instead of silently never speaking.
