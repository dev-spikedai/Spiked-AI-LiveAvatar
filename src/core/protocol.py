"""Control-socket wire contract. See docs/CONTROL_PROTOCOL.md."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "pcm_s16le"


PCM16_16K_MONO = AudioFormat()

# Backend -> page
CONTROL_AVATAR_SPEAK = "avatar_speak"
CONTROL_AVATAR_AUDIO = "avatar_audio"
CONTROL_AVATAR_SPEAK_END = "avatar_speak_end"
CONTROL_AVATAR_INTERRUPT = "avatar_interrupt"
CONTROL_AVATAR_USER_MESSAGE = "avatar_user_message"
CONTROL_HEARD = "heard"
CONTROL_AGENT_MUTED = "agent_muted"
CONTROL_AGENT_UNMUTED = "agent_unmuted"

# Page -> backend
CONTROL_AVATAR_SPEAK_STARTED = "avatar_speak_started"
CONTROL_AVATAR_SPEAK_ENDED = "avatar_speak_ended"
CONTROL_AVATAR_SPEAK_INTERRUPTED = "avatar_speak_interrupted"
CONTROL_AVATAR_VENDOR_REPLY = "avatar_vendor_reply"

CONTROL_INBOUND_TYPES = frozenset({
    CONTROL_AVATAR_SPEAK_STARTED,
    CONTROL_AVATAR_SPEAK_ENDED,
    CONTROL_AVATAR_SPEAK_INTERRUPTED,
    CONTROL_AVATAR_VENDOR_REPLY,
})

# Rep console (observability, not the provider contract)
REP_HEARD = CONTROL_HEARD
REP_AGENT_SPOKE = "agent_spoke"
REP_AGENT_STATE = "agent_state"
REP_AUTOSPEAK_REASONING = "autospeak_reasoning"
REP_INSIGHT_AVAILABLE = "insight_available"


def avatar_speak(text: str, turn_id: int, chunk_id: Optional[str] = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"type": CONTROL_AVATAR_SPEAK, "text": text, "turn_id": turn_id}
    if chunk_id is not None:
        message["chunk_id"] = chunk_id
    return message


def avatar_audio(data_b64: str, turn_id: int) -> Dict[str, Any]:
    return {"type": CONTROL_AVATAR_AUDIO, "data": data_b64, "turn_id": turn_id}


def avatar_speak_end(turn_id: int, chunk_id: Optional[str] = None) -> Dict[str, Any]:
    # chunk_id must round-trip; without it the ack reads as single-shot and
    # releases the floor mid-answer.
    message: Dict[str, Any] = {"type": CONTROL_AVATAR_SPEAK_END, "turn_id": turn_id}
    if chunk_id is not None:
        message["chunk_id"] = chunk_id
    return message


def avatar_user_message(text: str, turn_id: int) -> Dict[str, Any]:
    return {"type": CONTROL_AVATAR_USER_MESSAGE, "text": text, "turn_id": turn_id}
