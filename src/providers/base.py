"""Provider contracts: face, voice, brain. See docs/PROVIDER_REFACTOR_PLAN.md §4."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from src.core.protocol import AudioFormat, PCM16_16K_MONO


@dataclass
class RunContext:
    """What a provider needs to open a session. Never the run dict — adapters
    that reach into agent state stop being swappable."""

    run_id: str
    bot_name: str
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    auth_token: Optional[str] = None
    avatar_id: Optional[str] = None
    persona_prompt: Optional[str] = None


@dataclass
class TurnContext:
    """One already-gated turn. The engine decides what is said, never whether."""

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
    # credentials stay provider-namespaced; a shared shape would just be the
    # union of every vendor's fields wearing a disguise.
    provider: str
    credentials: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class DelegatedResult:
    spoken_text: str
    interrupted: bool = False


class VideoProvider(ABC):
    name: str

    # "text" -> does its own TTS, hand it sentences.
    # "audio" -> lip-sync only, pair it with a TtsProvider and hand it PCM.
    accepts: Literal["text", "audio"] = "text"

    browser_module: str
    audio_format: Optional[AudioFormat] = None

    # 0 disables the keepalive task. Most vendors bound the session themselves
    # (Simli maxIdleTime, Anam maxSessionLengthSeconds); LiveAvatar does not.
    keepalive_interval_s: float = 0.0

    @abstractmethod
    async def create_session(self, ctx: RunContext) -> VideoSession:
        ...

    async def keepalive(self, session: VideoSession) -> None:
        return None

    async def close(self, session: VideoSession) -> None:
        """Release the vendor session; must tolerate an already-dead one."""
        return None


class TtsProvider(ABC):
    name: str
    audio_format: AudioFormat = PCM16_16K_MONO

    @abstractmethod
    def stream(self, text: str) -> AsyncIterator[bytes]:
        """Async generator, not a coroutine: frames must start flowing before
        synthesis finishes."""


class AnswerEngine(ABC):
    """Decides what the agent says. Never decides whether it speaks."""

    name: str

    # "stream"    -> yields sentences, core keeps pacing/backstop/barge-in.
    # "delegated" -> vendor composes and speaks; core still owns the floor.
    mode: Literal["stream", "delegated"] = "stream"

    def stream_answer(self, ctx: TurnContext) -> AsyncIterator[str]:
        raise NotImplementedError

    async def delegate_turn(self, ctx: TurnContext, session: VideoSession) -> DelegatedResult:
        # Takes the session because in this mode brain and face are one vendor.
        raise NotImplementedError
