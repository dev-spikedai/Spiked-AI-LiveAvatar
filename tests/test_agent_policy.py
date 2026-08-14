import struct

from src.agent_policy import (
    FinalUtteranceBuffer,
    SustainedSpeechDetector,
    apply_validated_corrections,
    detect_invocation,
    normalize_reply,
    requires_company_knowledge,
)


def result(text, *, is_final=True, speech_final=False):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text, "words": [{"word": text}]}]},
    }


def test_invocation_requires_safe_explicit_name():
    assert detect_invocation("Tom, what does it cost?", "Tom").addressed
    assert detect_invocation("What does it cost, Thom?", "Tom").addressed
    assert not detect_invocation("What time can you meet?", "Tom").addressed
    assert not detect_invocation("Dom, what do you think?", "Tom").addressed
    assert not detect_invocation("We will discuss tomorrow.", "Tom").addressed


def test_invocation_is_required_again_after_an_answer():
    assert detect_invocation("Tom, explain the architecture.", "Tom").addressed
    assert not detect_invocation("And what about security?", "Tom").addressed


def test_final_segments_are_buffered_until_end_of_turn():
    buffer = FinalUtteranceBuffer()
    assert buffer.add_result(result("Tom what", speech_final=False)) is None
    utterance = buffer.add_result(result("does it cost", speech_final=True))
    assert utterance["text"] == "Tom what does it cost"
    assert buffer.flush() is None


def test_interim_results_never_enter_the_final_utterance():
    buffer = FinalUtteranceBuffer()
    assert buffer.add_result(result("Tom what is", is_final=False)) is None
    utterance = buffer.add_result(result("Tom what is SpikedAI", speech_final=True))
    assert utterance["text"] == "Tom what is SpikedAI"


def test_utterance_end_flushes_accumulated_final_segments():
    buffer = FinalUtteranceBuffer()
    buffer.add_result(result("Tom please wait", speech_final=False))
    assert buffer.add_result({"type": "UtteranceEnd"})["text"] == "Tom please wait"


def test_corrections_must_be_verified_and_high_confidence():
    text = apply_validated_corrections(
        "Tom, explain spike ai pricing",
        [
            {"raw": "spike ai", "replacement": "SpikedAI", "confidence": 0.95},
            {"raw": "pricing", "replacement": "Invented Product", "confidence": 0.99},
        ],
        ["SpikedAI", "3CAI"],
    )
    assert text == "Tom, explain SpikedAI pricing"


def test_spoken_reply_is_sanitized_and_bounded():
    reply = normalize_reply(
        "**First sentence.** Second sentence. Third sentence that must be removed.",
        max_words=20,
        max_sentences=2,
    )
    assert reply == "First sentence. Second sentence."


def test_company_fact_signals_force_grounded_route():
    assert requires_company_knowledge("Tom, what does your platform cost?", ["SpikedAI"])
    assert requires_company_knowledge("Tom, explain 3CAI", ["SpikedAI", "3CAI"])
    assert not requires_company_knowledge("Tom, what did Alice say earlier?", ["SpikedAI"])


def test_sustained_speech_requires_configured_duration():
    detector = SustainedSpeechDetector(threshold_ms=700)
    detector._vad = None
    voiced_frame = struct.pack("<320h", *([1000] * 320))
    for _ in range(34):
        assert not detector.feed(voiced_frame)
    assert detector.feed(voiced_frame)


def test_silence_breaks_barge_in_streak():
    detector = SustainedSpeechDetector(threshold_ms=100)
    detector._vad = None
    voiced_frame = struct.pack("<320h", *([1000] * 320))
    silent_frame = struct.pack("<320h", *([0] * 320))
    for _ in range(4):
        assert not detector.feed(voiced_frame)
    assert not detector.feed(silent_frame)
    for _ in range(4):
        assert not detector.feed(voiced_frame)
