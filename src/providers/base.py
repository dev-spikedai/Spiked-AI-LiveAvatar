"""The three provider contracts: face, voice, brain.

The split exists because the vendors do not agree on where one ends and the
next begins. HeyGen FULL mode is a face *and* a voice. Simli is a face only.
Anam is a face, optionally a voice, and optionally a brain. Modelling all of
them as "an avatar" forces a lowest-common-denominator interface that fits none
of them; modelling them as three independently swappable roles lets each vendor
fill exactly the roles it actually performs.

The core never imports this package's concrete adapters -- only these ABCs and
the registry. See docs/PROVIDER_REFACTOR_PLAN.md.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from src.core.protocol import AudioFormat, PCM16_16K_MONO


# ---------------------------------------------------------------------------
# Context objects -- what a provider is told about the run and the turn
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Everything a provider may need to open a session, and nothing more.

    Deliberately not the `run` dict: adapters must not reach into agent state
    (floor, governor, echo) or they stop being swappable.
    """

    run_id: str
    bot_name: str
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    auth_token: Optional[str] = None
    avatar_id: Optional[str] = None
    persona_prompt: Optional[str] = None


@dataclass
class TurnContext:
    """One gated turn, as handed to an answer engine.

    The turn gate has already decided this turn *should* happen; the engine
    only decides what is said.
    """

    run_id: str
    turn_id: int
    question: str
    speaker: str
    bot_name: str
    company_name: str = ""
    history_text: str = ""
    catalog: List[str] = field(default_factory=list)
    client_id: Optional[str] = None
    auth_token: Optional[str] = None
    intent: str = "company_knowledge"
    reply_word_limit: int = 45


@dataclass
class VideoSession:
    """The handle the backend keeps, and the blob the browser half receives.

    `credentials` stays provider-namespaced (livekit_url / anam_session_token /
    simli_session_token) rather than being flattened into a shared shape --
    a common shape would only be a union of every vendor's fields wearing a
    disguise, and the browser module for a given provider knows exactly which
    keys it needs.
    """

    provider: str
    credentials: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class DelegatedResult:
    """Outcome of a turn the vendor's own brain answered.

    `spoken_text` is what the vendor actually said, recovered after the fact
    from its transcript/message-history event. The core needs it for the echo
    suppressor and the duplicate guard -- without it the agent will hear its
    own voice come back through Deepgram and treat it as a new utterance.
    """

    spoken_text: str
    interrupted: bool = False


# ---------------------------------------------------------------------------
# Video / face
# ---------------------------------------------------------------------------

class VideoProvider(ABC):
    """Renders the talking face and plays its audio into the meeting."""

    #: registry key, and the value echoed to the browser
    name: str

    #: "text" -> the provider does its own TTS; hand it sentences.
    #: "audio" -> the provider only lip-syncs; hand it PCM and pair it with a
    #: TtsProvider. This single flag is what lets one agent loop drive both.
    accepts: Literal["text", "audio"] = "text"

    #: URL of the ES module implementing the browser half of this provider.
    browser_module: str

    #: Required when accepts == "audio"; ignored otherwise.
    audio_format: Optional[AudioFormat] = None

    @abstractmethod
    async def create_session(self, ctx: RunContext) -> VideoSession:
        """Open a vendor session. Raises HTTPException on misconfiguration."""

    async def keepalive(self, session: VideoSession) -> None:
        """Called once per keepalive_interval_s for the life of the run.

        Default is a no-op because most vendors hold the session open on their
        own (Simli via maxIdleTime on the token, Anam via
        maxSessionLengthSeconds). LiveAvatar is the exception -- it closes an
        idle session out from under a healthy run, and Tom is silent unless
        addressed, so quiet meetings are exactly the case that breaks.
        """
        return None

    #: 0 disables the keepalive task entirely.
    keepalive_interval_s: float = 0.0

    async def close(self, session: VideoSession) -> None:
        """Release the vendor session. Must tolerate an already-dead session."""
        return None


# ---------------------------------------------------------------------------
# TTS / voice
# ---------------------------------------------------------------------------

class TtsProvider(ABC):
    """Synthesizes speech for video providers that only lip-sync."""

    name: str

    #: Must match the paired VideoProvider's audio_format, or the executor
    #: refuses the combination at resolve time rather than shipping silence.
    audio_format: AudioFormat = PCM16_16K_MONO

    @abstractmethod
    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM for `text` as it is synthesized.

        An async generator, not a coroutine returning bytes: the whole point is
        to start pushing frames at the avatar before synthesis has finished.
        """


# ---------------------------------------------------------------------------
# Answer engine / brain
# ---------------------------------------------------------------------------

class AnswerEngine(ABC):
    """Decides what the agent says. Never decides whether it speaks."""

    name: str

    #: "stream"    -> yields sentences; the core speaks them through the video
    #:                provider and keeps full control of pacing, the word
    #:                backstop, barge-in and echo suppression.
    #: "delegated" -> the vendor composes AND speaks (Anam's own persona LLM).
    #:                The core still owns the floor: it decides the turn
    #:                happens, hands the transcript over, and waits.
    mode: Literal["stream", "delegated"] = "stream"

    def stream_answer(self, ctx: TurnContext) -> AsyncIterator[str]:
        """Yield the answer one complete sentence at a time. mode == "stream"."""
        raise NotImplementedError

    async def delegate_turn(self, ctx: TurnContext, session: VideoSession) -> DelegatedResult:
        """Hand the turn to the vendor's brain and return once it has spoken.

        mode == "delegated". Takes the VideoSession because in this mode the
        brain and the face are the same vendor -- that coupling is the whole
        reason the mode exists, so the interface admits it rather than
        pretending the two are independent.
        """
        raise NotImplementedError
