"""The registry's job is to make an unusable provider combination fail at
/start, loudly, instead of at the first turn, silently. These tests are about
that boundary -- not about whether any given vendor works.
"""

import pytest
from fastapi import HTTPException

from src.providers import registry
from src.providers.base import AnswerEngine, TtsProvider, VideoProvider


def test_default_combination_is_the_incumbent():
    """No arguments must reproduce exactly what shipped before the refactor."""
    providers = registry.resolve()
    assert providers.video.name == "liveavatar"
    assert providers.video.accepts == "text"
    assert providers.tts is None
    assert providers.answer is not None and providers.answer.name == "spiked"
    assert not providers.needs_tts
    assert not providers.is_delegated


def test_lipsync_only_provider_without_a_voice_is_refused():
    """The failure this exists to prevent: an avatar that mouths silence for
    the length of a meeting, with nothing raised anywhere."""
    with pytest.raises(HTTPException) as exc:
        registry.resolve(video="simli")
    assert exc.value.status_code == 400
    assert "tts_provider" in str(exc.value.detail)


def test_lipsync_only_provider_with_a_voice_resolves():
    providers = registry.resolve(video="simli", tts="cartesia")
    assert providers.needs_tts
    assert providers.video.accepts == "audio"
    assert providers.tts is not None
    # The formats have to actually agree, not merely both be present.
    assert providers.tts.audio_format == providers.video.audio_format


def test_paying_for_a_voice_the_face_already_has_is_refused():
    with pytest.raises(HTTPException) as exc:
        registry.resolve(video="liveavatar", tts="cartesia")
    assert exc.value.status_code == 400
    assert "own speech synthesis" in str(exc.value.detail)


def test_native_brain_cannot_be_mixed_with_another_vendors_face():
    """Anam's brain and Anam's face are one session; the pairing is not a
    preference, it is a constraint of the vendor."""
    with pytest.raises(HTTPException) as exc:
        registry.resolve(video="liveavatar", answer=registry.ANAM_NATIVE)
    assert exc.value.status_code == 400
    assert "requires video provider 'anam'" in str(exc.value.detail)


def test_native_brain_flips_the_anam_adapter_into_native_mode():
    providers = registry.resolve(video="anam", answer=registry.ANAM_NATIVE)
    assert providers.is_delegated
    assert providers.video.native is True
    # CUSTOMER_CLIENT_V1 is the "no brain" sentinel -- in native mode it must
    # NOT be what gets sent, or Anam will stay silent and nothing will answer.
    assert providers.video.llm_id != "CUSTOMER_CLIENT_V1"
    # The engine is built later, by the executor, because it needs the socket.
    assert providers.answer is None


def test_native_anam_without_a_configured_model_fails_at_session_creation():
    """`llm_id` is empty when ANAM_NATIVE_LLM_ID is unset, which is not the
    same as being valid -- native mode must refuse to open a session it knows
    has no brain behind it, rather than starting a mute avatar."""
    import asyncio

    from src.providers.base import RunContext
    from src.providers.video import anam as anam_module

    provider = anam_module.AnamVideoProvider(native=True, llm_id="")
    anam_module.ANAM_API_KEY = "test-key"
    anam_module.ANAM_AVATAR_ID = "test-avatar"
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(provider.create_session(RunContext(run_id="r1", bot_name="Tom")))
    finally:
        anam_module.ANAM_API_KEY = ""
        anam_module.ANAM_AVATAR_ID = ""
    assert "ANAM_NATIVE_LLM_ID" in str(exc.value.detail)


def test_driven_anam_disables_the_vendor_brain():
    providers = registry.resolve(video="anam")
    assert providers.video.native is False
    assert providers.video.llm_id == "CUSTOMER_CLIENT_V1"
    assert providers.video.accepts == "text"
    assert not providers.is_delegated


@pytest.mark.parametrize(
    "kwargs, missing",
    [
        ({"video": "nope"}, "video provider"),
        ({"answer": "nope"}, "answer engine"),
        ({"video": "simli", "tts": "nope"}, "TTS provider"),
    ],
)
def test_unknown_names_are_rejected_by_name(kwargs, missing):
    with pytest.raises(HTTPException) as exc:
        registry.resolve(**kwargs)
    assert exc.value.status_code == 400
    assert missing.lower() in str(exc.value.detail).lower()


def test_fastembed_is_not_registered_yet():
    """Deliberately deferred. If someone implements it, this test should be
    deleted in the same change -- not left passing by accident."""
    assert "fastembed" not in registry.ANSWER_ENGINES


def test_every_registered_adapter_satisfies_its_contract():
    """Cheap guard against a new adapter forgetting a required attribute and
    only failing when someone actually launches a meeting with it."""
    for name, cls in registry.VIDEO_PROVIDERS.items():
        provider = cls()
        assert isinstance(provider, VideoProvider)
        assert provider.name == name
        assert provider.accepts in ("text", "audio")
        assert provider.browser_module.startswith("/providers/")
        if provider.accepts == "audio":
            assert provider.audio_format is not None, f"{name} must declare an audio_format"

    for name, cls in registry.TTS_PROVIDERS.items():
        tts = cls()
        assert isinstance(tts, TtsProvider)
        assert tts.name == name
        assert tts.audio_format.sample_rate > 0

    for name, cls in registry.ANSWER_ENGINES.items():
        engine = cls()
        assert isinstance(engine, AnswerEngine)
        assert engine.name == name
        assert engine.mode in ("stream", "delegated")
