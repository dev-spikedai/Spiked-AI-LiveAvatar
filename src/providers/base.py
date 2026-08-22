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
    kyc_id: Optional[str] = None
    source_ids: Optional[List[str]] = None
    source_ids_task: Optional[Any] = None
    timeout_s: float = 12.0


@dataclass
class VideoSession:
    # credentials stay provider-namespaced; a shared shape would just be the
    # union of every vendor's fields wearing a disguise.
    provider: str
    credentials: Dict[str, Any]
    session_id: Optional[str] = None


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

    @abstractmethod
    async def answer(self, ctx: TurnContext, on_sentence: Optional[Any] = None) -> str:
        """Return the full answer, calling on_sentence per complete sentence.

        Push, not a generator: the caller speaks sentences as they land but
        still needs the whole text afterwards for the degraded-answer check.
        """
