"""Level 1.5: the autonomous interjection path.

_judge_interjection is the stricter, second classification layered on top of
the Level 1 heuristic. _consider_autospeak is the caller-side discipline that
must fail closed on every check (cap, cooldown, floor, governor) before ever
letting a warmed reply reach _take_floor_and_speak. These tests exercise both
in isolation from the Level 1 prefetch (see test_grounded_reply.py for that),
mocking the pieces this feature composes rather than reimplementing them.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src import live_avatar
from src.agent_policy import AgentState, EchoSuppressor, FloorState, SpeechGovernor


# -- _judge_interjection ------------------------------------------------


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate_content(self, **kwargs):
        return self.responses.pop(0)


def _install_model(monkeypatch, responses):
    monkeypatch.setattr(
        live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=FakeModels(responses)))
    )


def test_judge_interjection_parses_a_structured_response(monkeypatch):
    _install_model(monkeypatch, [
        SimpleNamespace(parsed=live_avatar.InterjectionJudgmentModel(
            worth_interjecting=True, confidence=0.9, reason="contradicts a stated fact",
        ))
    ])

    judgment = asyncio.run(live_avatar._judge_interjection(
        transcript="We don't think you support SSO",
        history_text="",
        bot_name="Tom",
        preferred_model="fake-model",
    ))

    assert judgment.worth_interjecting is True
    assert judgment.confidence == 0.9
    assert judgment.reason == "contradicts a stated fact"


def test_judge_interjection_fails_closed_on_error(monkeypatch):
    """An unparseable or failed judgment call must never be treated as
    permission to take the floor unprompted."""

    class ExplodingModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("upstream blip")

    monkeypatch.setattr(
        live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=ExplodingModels()))
    )

    judgment = asyncio.run(live_avatar._judge_interjection(
        transcript="anything", history_text="", bot_name="Tom", preferred_model="fake-model",
    ))

    assert judgment.worth_interjecting is False
    assert judgment.confidence == 0.0
    assert judgment.reason == "judgment_failed"


# -- _consider_autospeak --------------------------------------------------


def _run_state(**overrides):
    run = {
        "run_id": "r1",
        "bot_name": "Tom",
        "state": AgentState.LISTENING,
        "history": [],
        "turn_counter": 0,
        "active_turn_id": None,
        "control_ws": None,
        "rep_sockets": set(),
        "pending_insight": {"speaker": "Lisa", "question": "q", "reply": "warmed", "created_at": 0},
        "last_insight_at": None,
        "autospeak_enabled": True,
        "autospeak_count": 0,
        "last_autospeak_at": None,
        "intel": None,
        "token": "",
        "client_id": None,
        "user_context": {"company_name": "SpikedAI"},
        "pending_turns": {},
        "floor": FloorState(),
        "echo": EchoSuppressor(similarity_threshold=0.9, tail_seconds=8),
        "governor": SpeechGovernor(cooldown_seconds=2, max_replies_per_window=4, window_seconds=30),
    }
    run.update(overrides)
    return run


class RecordingWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


def _consider(run, monkeypatch, judgment, *, spy_take_floor=None):
    async def fake_judge(**kwargs):
        return judgment

    monkeypatch.setattr(live_avatar, "_judge_interjection", fake_judge)

    if spy_take_floor is not None:
        monkeypatch.setattr(live_avatar, "_take_floor_and_speak", spy_take_floor)

    asyncio.run(live_avatar._consider_autospeak(run, "Lisa", "the transcript", "warmed reply", "history"))


def _worthy_judgment(confidence=0.9):
    return live_avatar.InterjectionJudgment(True, confidence, "closes a real gap")


def _unworthy_judgment():
    return live_avatar.InterjectionJudgment(False, 0.2, "not load-bearing")


def test_speaks_when_judged_worthy_and_confident(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()})
    calls = []

    async def spy(run_, question, speaker, **kwargs):
        calls.append((question, speaker, kwargs))
        return {"accepted": True, "turn_id": 1, "warm": True}

    _consider(run, monkeypatch, _worthy_judgment(), spy_take_floor=spy)

    assert len(calls) == 1
    question, speaker, kwargs = calls[0]
    assert question == "the transcript"
    assert speaker == "Lisa"
    assert kwargs["warm_reply"] == "warmed reply"
    assert kwargs["source"] == "autonomous"
    assert run["autospeak_count"] == 1
    assert run["last_autospeak_at"] is not None
    assert run["pending_insight"] is None  # spoken, not offered
    reasoning = [m for m in next(iter(run["rep_sockets"])).sent if m["type"] == "autospeak_reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["worth_interjecting"] is True


def test_stays_silent_when_judged_not_worthy(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()})
    calls = []

    async def spy(*args, **kwargs):
        calls.append(1)

    _consider(run, monkeypatch, _unworthy_judgment(), spy_take_floor=spy)

    assert calls == []
    assert run["autospeak_count"] == 0
    assert run["pending_insight"] is not None  # Level 1 cue still stands
    reasoning = [m for m in next(iter(run["rep_sockets"])).sent if m["type"] == "autospeak_reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0]["worth_interjecting"] is False


def test_stays_silent_below_confidence_floor_even_if_worth_interjecting(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()})
    calls = []

    async def spy(*args, **kwargs):
        calls.append(1)

    below_floor = live_avatar.AGENT_AUTOSPEAK_MIN_CONFIDENCE - 0.01
    _consider(run, monkeypatch, _worthy_judgment(confidence=below_floor), spy_take_floor=spy)

    assert calls == []
    assert run["autospeak_count"] == 0


def test_respects_the_per_run_cap(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()}, autospeak_count=live_avatar.AGENT_AUTOSPEAK_MAX_PER_RUN)
    calls = []

    async def spy(*args, **kwargs):
        calls.append(1)

    async def fake_judge(**kwargs):
        # Should never even be reached: the cap check short-circuits first.
        calls.append("judged")
        return _worthy_judgment()

    monkeypatch.setattr(live_avatar, "_judge_interjection", fake_judge)
    monkeypatch.setattr(live_avatar, "_take_floor_and_speak", spy)

    asyncio.run(live_avatar._consider_autospeak(run, "Lisa", "text", "warmed", "history"))

    assert calls == []
    assert run["autospeak_count"] == live_avatar.AGENT_AUTOSPEAK_MAX_PER_RUN


def test_respects_the_cooldown(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()}, last_autospeak_at=__import__("time").monotonic())
    calls = []

    async def fake_judge(**kwargs):
        calls.append("judged")
        return _worthy_judgment()

    async def spy(*args, **kwargs):
        calls.append("spoke")

    monkeypatch.setattr(live_avatar, "_judge_interjection", fake_judge)
    monkeypatch.setattr(live_avatar, "_take_floor_and_speak", spy)

    asyncio.run(live_avatar._consider_autospeak(run, "Lisa", "text", "warmed", "history"))

    assert calls == []  # cooldown short-circuits before the judgment call


def test_never_speaks_over_an_active_turn(monkeypatch):
    """If the floor moved on while the (awaited) RAG warm was in flight, the
    state recheck must catch it before the judgment call ever fires."""
    run = _run_state(rep_sockets={RecordingWS()}, state=AgentState.SPEAKING)
    calls = []

    async def fake_judge(**kwargs):
        calls.append("judged")
        return _worthy_judgment()

    async def spy(*args, **kwargs):
        calls.append("spoke")

    monkeypatch.setattr(live_avatar, "_judge_interjection", fake_judge)
    monkeypatch.setattr(live_avatar, "_take_floor_and_speak", spy)

    asyncio.run(live_avatar._consider_autospeak(run, "Lisa", "text", "warmed", "history"))

    assert calls == []


def test_respects_the_governor(monkeypatch):
    run = _run_state(rep_sockets={RecordingWS()})
    run["governor"].note_reply("something", __import__("time").monotonic())  # trips cooldown
    calls = []

    async def fake_judge(**kwargs):
        calls.append("judged")
        return _worthy_judgment()

    async def spy(*args, **kwargs):
        calls.append("spoke")

    monkeypatch.setattr(live_avatar, "_judge_interjection", fake_judge)
    monkeypatch.setattr(live_avatar, "_take_floor_and_speak", spy)

    asyncio.run(live_avatar._consider_autospeak(run, "Lisa", "text", "warmed", "history"))

    assert calls == []


# -- _consider_autospeak_candidate -----------------------------------------


async def _drive_candidate(run, speaker, text, history=None):
    """Fire the function under test, then run whatever background task it
    scheduled to completion, inside the same event loop."""
    live_avatar._consider_autospeak_candidate(run, speaker, text, history or [])
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)


def test_autospeak_candidate_fires_on_question_shaped_turns_too(monkeypatch):
    """Unlike the old design, this no longer defers to _consider_insight for
    question-shaped turns — Level 1's topic gate has real recall gaps (see
    the function's docstring), so Level 1.5 evaluates independently."""
    run = _run_state()
    calls = []

    async def fake_generate(**kwargs):
        calls.append(1)
        return "an answer"

    async def fake_consider(*args, **kwargs):
        pass

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)
    monkeypatch.setattr(live_avatar, "_consider_autospeak", fake_consider)

    asyncio.run(_drive_candidate(run, "Lisa", "What about our security posture?"))

    assert len(calls) == 1


def test_autospeak_candidate_fires_on_off_topic_statements(monkeypatch):
    """No keyword topic gate any more — RAG's own empty-result is the real
    relevance filter (cheap on a miss: retrieval only, no generation)."""
    run = _run_state()
    calls = []

    async def fake_generate(**kwargs):
        calls.append(1)
        return None  # RAG found nothing — the real filter doing its job

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)

    asyncio.run(_drive_candidate(run, "Lisa", "Great, sounds good, let's move on then."))

    assert len(calls) == 1  # attempted, and correctly yielded nothing to say


def test_autospeak_candidate_ignores_incomplete_fragments(monkeypatch):
    run = _run_state()
    calls = []

    async def fake_generate(**kwargs):
        calls.append(1)
        return "should not be reached"

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)

    asyncio.run(_drive_candidate(run, "Lisa", "So for the security requirements, that's um"))

    assert calls == []


def test_autospeak_candidate_off_when_not_enabled(monkeypatch):
    run = _run_state(autospeak_enabled=False)
    calls = []

    async def fake_generate(**kwargs):
        calls.append(1)
        return "should not be reached"

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)

    asyncio.run(_drive_candidate(run, "Lisa", "We don't support SSO for security, unfortunately."))

    assert calls == []


def test_autospeak_candidate_fires_on_a_stated_misconception(monkeypatch):
    """The scenario Level 1's keyword gate structurally missed: a flat
    declarative misstatement of fact, not phrased as a question."""
    run = _run_state()
    generate_calls = []
    consider_calls = []

    async def fake_generate(**kwargs):
        generate_calls.append(kwargs)
        return "We do support SSO via SAML on Enterprise plans."

    async def fake_consider(run_, speaker, transcript, warmed_reply, history_text):
        consider_calls.append((speaker, transcript, warmed_reply))

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)
    monkeypatch.setattr(live_avatar, "_consider_autospeak", fake_consider)

    text = "We don't support single sign-on for security requirements, so you'd need individual logins."
    asyncio.run(_drive_candidate(run, "Sai", text))

    assert len(generate_calls) == 1
    assert len(consider_calls) == 1
    speaker, transcript, warmed_reply = consider_calls[0]
    assert speaker == "Sai"
    assert transcript == text
    assert warmed_reply == "We do support SSO via SAML on Enterprise plans."


def test_autospeak_candidate_fires_on_a_missed_technical_question(monkeypatch):
    """The concrete example that motivated dropping the keyword gate: a real
    technical question phrased in domain language ("Kubernetes", "state
    transfer") that contains none of requires_company_knowledge's hardcoded
    factual_terms and wouldn't fuzzy-match a generic catalog."""
    run = _run_state()
    generate_calls = []
    consider_calls = []

    async def fake_generate(**kwargs):
        generate_calls.append(kwargs)
        return "State is externalized to a shared store; pods are treated as disposable."

    async def fake_consider(run_, speaker, transcript, warmed_reply, history_text):
        consider_calls.append((speaker, transcript, warmed_reply))

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)
    monkeypatch.setattr(live_avatar, "_consider_autospeak", fake_consider)

    text = "How does the state transfer between different machines when orchestrated using Kubernetes?"
    asyncio.run(_drive_candidate(run, "Diptanshu", text))

    assert len(generate_calls) == 1
    assert len(consider_calls) == 1


def test_autospeak_candidate_shares_the_level_1_cooldown_clock(monkeypatch):
    run = _run_state(last_insight_at=__import__("time").monotonic())
    calls = []

    async def fake_generate(**kwargs):
        calls.append(1)
        return "should not be reached"

    monkeypatch.setattr(live_avatar, "_generate_grounded_reply", fake_generate)

    text = "We don't support single sign-on for security requirements."
    asyncio.run(_drive_candidate(run, "Sai", text))

    assert calls == []


# -- _take_floor_and_speak / FloorUnavailable ------------------------------


def test_take_floor_and_speak_raises_when_not_listening():
    run = _run_state(state=AgentState.SPEAKING)
    with pytest.raises(live_avatar.FloorUnavailable):
        asyncio.run(live_avatar._take_floor_and_speak(run, "q", "Lisa"))


def test_take_floor_and_speak_raises_with_no_question():
    run = _run_state()
    with pytest.raises(live_avatar.FloorUnavailable):
        asyncio.run(live_avatar._take_floor_and_speak(run, "", "Lisa"))


def test_take_floor_and_speak_dispatches_with_the_given_source(monkeypatch):
    run = _run_state()
    dispatched = []

    async def fake_dispatch(run_, answer, turn_id, source="addressed"):
        dispatched.append((answer, turn_id, source))
        return True

    monkeypatch.setattr(live_avatar, "_dispatch_reply", fake_dispatch)

    async def drive():
        result = await live_avatar._take_floor_and_speak(
            run, "q", "Lisa", warm_reply="the warmed answer", source="autonomous",
        )
        await run["active_response_task"]
        return result

    result = asyncio.run(drive())

    assert result["accepted"] is True
    assert result["warm"] is True
    assert dispatched == [("the warmed answer", result["turn_id"], "autonomous")]
