"""The control-socket wire contract between the backend and the avatar page.

This is the provider seam. `main` (LiveAvatar/HeyGen, text-driven) and the
`simli` branch (Simli + Cartesia, audio-driven) were written independently and
converged on the same message set -- which is the evidence that this, not the
Python class layout, is the real interface a video provider implements. Freezing
it here means a new provider is "a module that speaks this protocol" rather than
a fork of avatar.js.

Two sockets, deliberately separate (see avatar_control_endpoint /
rep_console_endpoint):

  /ws/control/{run_id}  the avatar page itself -- speech transport, single slot,
                        provider-facing. Everything under CONTROL_* below.
  /ws/rep/{run_id}      the rep's console -- observability, many slots, may drop
                        at any time without affecting the meeting. REP_* below.
                        NOT part of the provider contract; listed here only so
                        every wire literal in this service has one home.

Nothing in this module imports anything from the rest of the service, so both
the current monolith and the extracted core can depend on it during migration.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Bumped only on a breaking change to the message set below. The page reports
# the version it was built against when it connects, so a stale cached avatar.js
# against a newer backend is a loud mismatch instead of silent dead air.
PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# Audio format -- the contract for providers whose `accepts` is "audio"
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFormat:
    """Raw PCM parameters for the audio path.

    Simli's WebRTC endpoint and Anam's audio-passthrough mode independently
    require the same thing: PCM 16-bit signed little-endian, 16 kHz, mono. That
    is also exactly what the Cartesia adapter requests
    (container=raw, encoding=pcm_s16le), so no resampling happens anywhere on
    the speak path. A provider needing something else declares it here and the
    TTS layer is responsible for matching it.
    """

    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"


PCM16_16K_MONO = AudioFormat()


# ---------------------------------------------------------------------------
# Control socket: backend -> avatar page
# ---------------------------------------------------------------------------

# Text-mode speech. Providers with accepts == "text" render this directly
# (HeyGen avatar.speak_text, Anam streamMessageChunk). Audio-mode providers
# treat it as "an utterance begins" and wait for AVATAR_AUDIO frames.
# `chunk_id` present => one sentence of a streamed answer, and the page must
# echo AVATAR_SPEAK_ENDED carrying that same chunk_id so the dispatch loop can
# release the next sentence. Absent => legacy single-shot whole answer.
CONTROL_AVATAR_SPEAK = "avatar_speak"

# Audio-mode speech: one base64 PCM frame, format per the provider's
# AudioFormat. Many per utterance, ordered.
CONTROL_AVATAR_AUDIO = "avatar_audio"

# Audio-mode end-of-utterance. The text path has no equivalent because the
# provider's own TTS decides when it has finished talking.
CONTROL_AVATAR_SPEAK_END = "avatar_speak_end"

# Barge-in. Stop talking now and discard anything buffered.
CONTROL_AVATAR_INTERRUPT = "avatar_interrupt"

# Delegated mode only: hand a gated transcript to the vendor's own brain
# (Anam sendUserMessage). The vendor composes AND speaks; we are not sending
# it words to say, we are sending it the thing it should answer. Everything
# upstream of this -- whether the turn happens at all -- is still ours.
CONTROL_AVATAR_USER_MESSAGE = "avatar_user_message"

# Gate verdict mirrored onto the in-meeting overlay. In avatar mode this is the
# only path a transcript reaches any frontend at all.
CONTROL_HEARD = "heard"

# Mute countdown overlay -- shown to every meeting participant, not just the rep,
# which is why it goes here and not only to the rep console.
CONTROL_AGENT_MUTED = "agent_muted"
CONTROL_AGENT_UNMUTED = "agent_unmuted"


def avatar_speak(text: str, turn_id: int, chunk_id: Optional[str] = None) -> Dict[str, Any]:
    """Speak `text`. Omit chunk_id for a single-shot whole answer."""
    message: Dict[str, Any] = {"type": CONTROL_AVATAR_SPEAK, "text": text, "turn_id": turn_id}
    if chunk_id is not None:
        message["chunk_id"] = chunk_id
    return message


def avatar_audio(data_b64: str, turn_id: int) -> Dict[str, Any]:
    """One base64-encoded PCM frame of the utterance for `turn_id`."""
    return {"type": CONTROL_AVATAR_AUDIO, "data": data_b64, "turn_id": turn_id}


def avatar_speak_end(turn_id: int) -> Dict[str, Any]:
    """No further audio frames are coming for `turn_id`."""
    return {"type": CONTROL_AVATAR_SPEAK_END, "turn_id": turn_id}


def avatar_interrupt() -> Dict[str, Any]:
    return {"type": CONTROL_AVATAR_INTERRUPT}


def avatar_user_message(text: str, turn_id: int) -> Dict[str, Any]:
    """Delegated mode: the transcript the vendor's brain should answer."""
    return {"type": CONTROL_AVATAR_USER_MESSAGE, "text": text, "turn_id": turn_id}


def heard(speaker: str, text: str, reply: bool, reason: str) -> Dict[str, Any]:
    """The gate's verdict on one finalized, speaker-attributed turn.

    Same payload goes to the control socket and the rep console; the two
    sockets carry it for different audiences, not in different shapes.
    """
    return {
        "type": CONTROL_HEARD,
        "speaker": speaker,
        "text": text,
        "reply": reply,
        "reason": reason,
    }


def agent_muted(muted_until_epoch_ms: int, seconds: int) -> Dict[str, Any]:
    """Epoch-ms deadline, not a duration: the page renders a countdown, and a
    duration would drift by however long the message spent in flight."""
    return {
        "type": CONTROL_AGENT_MUTED,
        "muted_until_epoch_ms": muted_until_epoch_ms,
        "seconds": seconds,
    }


def agent_unmuted(reason: str) -> Dict[str, Any]:
    return {"type": CONTROL_AGENT_UNMUTED, "reason": reason}


# ---------------------------------------------------------------------------
# Control socket: avatar page -> backend
# ---------------------------------------------------------------------------

# Audio actually started playing. Closes the dispatch->speaking timing window
# and moves the agent to SPEAKING.
CONTROL_AVATAR_SPEAK_STARTED = "avatar_speak_started"

# One unit of speech finished. WITH chunk_id: that sentence is done, the turn is
# not -- unblock the dispatch loop and say nothing about the floor. WITHOUT
# chunk_id: the whole answer is done -- release the floor and cancel the
# watchdog. The two cases are load-bearing; collapsing them strands the agent.
CONTROL_AVATAR_SPEAK_ENDED = "avatar_speak_ended"

# Barge-in observed by the page (or the provider reporting its own interrupt,
# e.g. Anam's TALK_STREAM_INTERRUPTED).
CONTROL_AVATAR_SPEAK_INTERRUPTED = "avatar_speak_interrupted"

# Delegated mode only: what the vendor's brain actually said, recovered from its
# transcript event (Anam MESSAGE_HISTORY_UPDATED). The core cannot know this any
# other way, and it must: without it the echo suppressor has no record of the
# bot's own speech and the agent will hear itself come back through Deepgram and
# treat it as a fresh utterance. Arriving late is a correctness problem, not a
# cosmetic one -- see the risk noted in docs/PROVIDER_REFACTOR_PLAN.md §8.
CONTROL_AVATAR_VENDOR_REPLY = "avatar_vendor_reply"

# Every inbound control message carrying a turn_id is dropped unless it matches
# run["active_turn_id"] -- a late echo from a superseded turn must not move the
# floor. Kept as a named set so the extracted handler can assert on it.
CONTROL_INBOUND_TYPES = frozenset({
    CONTROL_AVATAR_SPEAK_STARTED,
    CONTROL_AVATAR_SPEAK_ENDED,
    CONTROL_AVATAR_SPEAK_INTERRUPTED,
    CONTROL_AVATAR_VENDOR_REPLY,
})


# ---------------------------------------------------------------------------
# Rep console socket -- observability only, NOT the provider contract
# ---------------------------------------------------------------------------

REP_HEARD = CONTROL_HEARD
REP_AGENT_SPOKE = "agent_spoke"
REP_AGENT_STATE = "agent_state"
REP_AGENT_MUTED = CONTROL_AGENT_MUTED
REP_AGENT_UNMUTED = CONTROL_AGENT_UNMUTED
REP_AUTOSPEAK_REASONING = "autospeak_reasoning"
REP_INSIGHT_AVAILABLE = "insight_available"


def agent_spoke(text: str, turn_id: int, source: str) -> Dict[str, Any]:
    """`source` ("addressed" | "invoke" | "autonomous") is display/audit only --
    it has never gated anything."""
    return {"type": REP_AGENT_SPOKE, "text": text, "turn_id": turn_id, "source": source}


def agent_state(state: str) -> Dict[str, Any]:
    return {"type": REP_AGENT_STATE, "state": state}


def insight_available(speaker: Optional[str], topic: Optional[str]) -> Dict[str, Any]:
    """Level 1 on the wire: the console learns the agent could contribute, and
    nothing is said in the room unless somebody accepts."""
    return {"type": REP_INSIGHT_AVAILABLE, "speaker": speaker, "topic": topic}
