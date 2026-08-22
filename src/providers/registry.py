"""Name -> adapter, and which combinations are legal.

Invalid combinations are rejected at /start, not on the first turn: an
audio-only provider with no TTS raises nothing, it just mouths silence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Type

from fastapi import HTTPException

from src.providers.answer.spiked import SpikedAnswerEngine
from src.providers.base import AnswerEngine, TtsProvider, VideoProvider
from src.providers.tts.cartesia import CartesiaTtsProvider
from src.providers.video.anam import AnamVideoProvider
from src.providers.video.liveavatar import LiveAvatarVideoProvider
from src.providers.video.simli import SimliVideoProvider

# Constructed by the executor (needs the run's socket), so it is a marker name.
ANAM_NATIVE = "anam_native"

VIDEO_PROVIDERS: Dict[str, Type[VideoProvider]] = {
    "liveavatar": LiveAvatarVideoProvider,
    "anam": AnamVideoProvider,
    "simli": SimliVideoProvider,
}

TTS_PROVIDERS: Dict[str, Type[TtsProvider]] = {
    "cartesia": CartesiaTtsProvider,
}

ANSWER_ENGINES: Dict[str, Type[AnswerEngine]] = {
    "spiked": SpikedAnswerEngine,
    # "fastembed": deliberately absent. The interface slot is real; the
    # implementation is deferred (docs/PROVIDER_REFACTOR_PLAN.md §1).
}


@dataclass
class ProviderSet:
    """The three adapters a run was launched with. Resolved once."""

    video: VideoProvider
    answer_name: str
    tts: Optional[TtsProvider] = None
    answer: Optional[AnswerEngine] = None

    @property
    def needs_tts(self) -> bool:
        return self.video.accepts == "audio"

    @property
    def is_delegated(self) -> bool:
        return self.answer_name == ANAM_NATIVE


def resolve(
    video: str = "liveavatar",
    tts: Optional[str] = None,
    answer: str = "spiked",
) -> ProviderSet:
    """Build a validated ProviderSet or raise HTTPException(400)."""
    if video not in VIDEO_PROVIDERS:
        raise HTTPException(400, f"Unknown video provider '{video}'. Known: {sorted(VIDEO_PROVIDERS)}")
    if answer != ANAM_NATIVE and answer not in ANSWER_ENGINES:
        raise HTTPException(400, f"Unknown answer engine '{answer}'. Known: {sorted(ANSWER_ENGINES) + [ANAM_NATIVE]}")
    if tts is not None and tts not in TTS_PROVIDERS:
        raise HTTPException(400, f"Unknown TTS provider '{tts}'. Known: {sorted(TTS_PROVIDERS)}")

    native = answer == ANAM_NATIVE
    if native and video != "anam":
        raise HTTPException(
            400,
            f"answer engine '{ANAM_NATIVE}' requires video provider 'anam' -- the vendor's "
            f"brain and its face are the same session, they cannot be mixed (got '{video}')",
        )

    video_provider = (
        AnamVideoProvider(native=True) if native else VIDEO_PROVIDERS[video]()
    )

    tts_provider: Optional[TtsProvider] = TTS_PROVIDERS[tts]() if tts else None

    if video_provider.accepts == "audio":
        if tts_provider is None:
            raise HTTPException(
                400,
                f"video provider '{video}' only lip-syncs and produces no speech of its own; "
                f"it requires a tts_provider (available: {sorted(TTS_PROVIDERS)})",
            )
        want = video_provider.audio_format
        got = tts_provider.audio_format
        if want is not None and (want.sample_rate, want.channels, want.encoding) != (
            got.sample_rate, got.channels, got.encoding
        ):
            raise HTTPException(
                400,
                f"audio format mismatch: '{video}' wants {want}, '{tts}' produces {got}",
            )
    elif tts_provider is not None:
        raise HTTPException(
            400,
            f"video provider '{video}' does its own speech synthesis; passing "
            f"tts_provider='{tts}' would pay for a voice that is never used",
        )

    answer_engine = None if native else ANSWER_ENGINES[answer]()
    return ProviderSet(
        video=video_provider,
        tts=tts_provider,
        answer=answer_engine,
        answer_name=answer,
    )
