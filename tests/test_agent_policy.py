import struct

from src.agent_policy import (
    FinalUtteranceBuffer,
    FloorState,
    SustainedSpeechDetector,
    apply_validated_corrections,
    detect_invocation,
    detect_mute_command,
    evaluate_turn,
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


def test_mute_command_requires_the_wake_name():
    assert detect_mute_command("Tom, stay quiet for 30 seconds", "Tom") == 30
    assert detect_mute_command("stay quiet for 30 seconds", "Tom") is None


def test_mute_command_parses_minutes_and_variants():
    assert detect_mute_command("Tom, be quiet for 2 minutes", "Tom") == 120
    assert detect_mute_command("Tom, stop listening for 45 seconds", "Tom") == 45
    assert detect_mute_command("Tom, don't listen for 1 min", "Tom") == 60


def test_mute_command_covers_natural_phrasings():
    """Real speech rarely matches the narrowest possible phrasing — these are
    variants that were silently missed before the phrase set was widened."""
    assert detect_mute_command("Tom, stop talking for 30 seconds", "Tom") == 30
    assert detect_mute_command("Tom, shut up for a minute", "Tom") is None  # no digit duration
    assert detect_mute_command("Tom, shut up for 1 minute", "Tom") == 60
    assert detect_mute_command("Tom, don't talk for 30 seconds", "Tom") == 30
    assert detect_mute_command("Tom, don't speak for 20 seconds", "Tom") == 20
    assert detect_mute_command("Tom, pause for 30 seconds", "Tom") == 30
    assert detect_mute_command("Tom, hold on for 30 seconds", "Tom") == 30
    assert detect_mute_command("Tom, give me a minute", "Tom") is None  # no digit duration
    assert detect_mute_command("Tom, give me 2 minutes", "Tom") == 120
    assert detect_mute_command("Tom, stand by for 30 seconds", "Tom") == 30


def test_mute_command_parses_spelled_out_durations():
    """A live test caught this: Deepgram transcribes a spoken duration as
    words ("thirty seconds"), never digits, so the digit-only regex silently
    matched nothing and the mute command never fired."""
    assert detect_mute_command("Tom, be quiet for thirty seconds", "Tom") == 30
    assert detect_mute_command("Tom, stay quiet for ten seconds", "Tom") == 10
    assert detect_mute_command("Tom, be quiet for two minutes", "Tom") == 120
    assert detect_mute_command("Tom, be quiet for forty five seconds", "Tom") == 45
    assert detect_mute_command("Tom, be quiet for forty-five seconds", "Tom") == 45


def test_mute_command_clamps_to_sane_bounds():
    assert detect_mute_command("Tom, stay quiet for 1 second", "Tom") == 5
    assert detect_mute_command("Tom, stay quiet for 10000 seconds", "Tom") == 3600


def test_mute_command_parses_hours():
    """A live test caught this: "hours" wasn't a recognized unit at all, so
    "stay quiet for two hours" fell through to classification instead of
    muting (and clamping to the 1hr ceiling, same as any long duration)."""
    assert detect_mute_command("Tom, stay quiet for 2 hours", "Tom") == 3600
    assert detect_mute_command("Tom, stay quiet for two hours", "Tom") == 3600
    assert detect_mute_command("Tom, stay quiet for 1 hr", "Tom") == 3600


def test_mute_command_needs_a_duration():
    assert detect_mute_command("Tom, stay quiet", "Tom") is None


def test_non_mute_commands_are_not_detected():
    assert detect_mute_command("Tom, what does it cost?", "Tom") is None


def _floor(now: float, **overrides) -> FloorState:
    floor = FloorState(last_bot_finished_at=now, last_addressed_participant="p1")
    for key, value in overrides.items():
        setattr(floor, key, value)
    return floor


def test_first_nameless_followup_accepts_a_declarative_answer():
    """No fixed keyword list can enumerate every declarative reply, so the
    very first follow-up right after the agent finishes — the highest-
    confidence case — skips the phrase-shape check entirely."""
    floor = _floor(now=100.0)
    decision = evaluate_turn(
        "Pricing is what I actually care about.", "p1", "Tom", floor, now=101.0,
    )
    assert decision.should_reply
    assert decision.reason == "followup_window"


def test_later_nameless_followup_still_requires_followup_shape():
    """Once the exchange has already continued once, an unrelated declarative
    aside from the same speaker should not keep riding the window."""
    floor = _floor(now=100.0, consecutive_followups=1)
    decision = evaluate_turn(
        "Pricing is what I actually care about.", "p1", "Tom", floor, now=101.0,
    )
    assert not decision.should_reply
    assert decision.reason == "not_followup_shaped"

    floor2 = _floor(now=100.0, consecutive_followups=1)
    decision2 = evaluate_turn("What about pricing?", "p1", "Tom", floor2, now=101.0)
    assert decision2.should_reply


def test_invocation_is_required_again_after_an_answer():
    assert detect_invocation("Tom, explain the architecture.", "Tom").addressed
    assert not detect_invocation("And what about security?", "Tom").addressed


def test_tom_is_a_precise_wake_name():
    assert detect_invocation("Tom, can you answer that?", "Tom").addressed
    assert detect_invocation("What do you think, Tom?", "Tom").addressed
    assert not detect_invocation("Ctom, can you answer that?", "Tom").addressed
    assert not detect_invocation("The larynx is unrelated.", "Tom").addressed


def test_spiked_ai_invocation_accepts_stt_formatting_variants():
    for transcript in (
        "SpikedAI, what do you offer?",
        "Spiked AI, what do you offer?",
        "Hey Spiked A.I., what do you offer?",
        "Spike AI, what do you offer?",
    ):
        assert detect_invocation(transcript, "SpikedAI").addressed

    assert not detect_invocation("Spike, what do you offer?", "SpikedAI").addressed
    assert not detect_invocation("What does the AI offer?", "SpikedAI").addressed


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


def test_brief_silence_is_tolerated_within_the_streak():
    """A single dropped frame (a plosive, a breath) must not wipe the whole
    streak — real continuous speech is not uniform acoustic energy."""
    detector = SustainedSpeechDetector(threshold_ms=100)
    detector._vad = None
    voiced_frame = struct.pack("<320h", *([1000] * 320))
    silent_frame = struct.pack("<320h", *([0] * 320))
    for _ in range(2):
        assert not detector.feed(voiced_frame)
    assert not detector.feed(silent_frame)  # within tolerance, streak survives
    for _ in range(2):
        assert not detector.feed(voiced_frame)
    assert detector.feed(voiced_frame)  # streak still reaches the threshold


def test_sustained_silence_breaks_barge_in_streak():
    detector = SustainedSpeechDetector(threshold_ms=100)
    detector._vad = None
    voiced_frame = struct.pack("<320h", *([1000] * 320))
    silent_frame = struct.pack("<320h", *([0] * 320))
    for _ in range(4):
        assert not detector.feed(voiced_frame)
    # Longer than the tolerance window — a genuine pause, not a dropout.
    for _ in range(SustainedSpeechDetector.UNVOICED_TOLERANCE_FRAMES + 1):
        assert not detector.feed(silent_frame)
    for _ in range(4):
        assert not detector.feed(voiced_frame)
