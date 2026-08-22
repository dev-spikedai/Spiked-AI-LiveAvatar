"""Cover the three-part spoken reply and the Level 1 / invoke surface.

The older routing tests stub the model with a bare ``SimpleNamespace(text=...)``,
which has no ``parsed`` attribute — so they exercise the raw-text fallback, not
the structured path. These tests drive the structured path directly, because the
closing question is the behavior the persona is defined by and it is exactly what
a whole-reply truncation would silently remove.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src import live_avatar
from src.agent_policy import (
    AgentState,
    EchoSuppressor,
    FloorState,
    SpeechGovernor,
    compose_reply,
)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _install_model(monkeypatch, responses):
    models = FakeModels(responses)
    monkeypatch.setattr(
        live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models))
    )
    return models


def _run_reply(intent="company_knowledge", **overrides):
    kwargs = dict(
        analysis=live_avatar.TurnAnalysis(
            response_action="respond",
            intent=intent,
            resolved_query="what about SOC 2",
            corrections=[],
        ),
        transcript="Tom, what about SOC 2?",
        speaker="Lisa",
        bot_name="Tom",
        company_name="SpikedAI",
        history_text="",
        catalog=["SpikedAI"],
        auth_token="token",
        client_id="client",
        preferred_model="fake-model",
    )
    kwargs.update(overrides)
    return asyncio.run(live_avatar._generate_grounded_reply(**kwargs))


class FakeControlWs:
    """Records every avatar_speak send and instantly resolves the matching
    chunk_events entry, as if the avatar finished speaking immediately —
    lets _speak_chunk's await return without a real avatar.js round trip."""

    def __init__(self, run):
        self.run = run
        self.sent: list = []

    async def send_json(self, message):
        self.sent.append(message)
        if message.get("type") == "avatar_speak":
            chunk_id = message.get("chunk_id")
            event = (self.run.get("chunk_events") or {}).get(chunk_id)
            if event is not None:
                event.set()


def _speaks(run):
    """Every avatar_speak the run sent, filler included."""
    return [m for m in run["control_ws"].sent if m.get("type") == "avatar_speak"]


def _answer_speaks(run):
    """Only the chunks carrying the actual answer.

    The company_knowledge path speaks a filler line ("let me check the docs")
    while the RAG call is still in flight, so the raw send list always leads
    with one extra chunk that is not part of the answer. It is identified by
    its chunk_id rather than by position, so a test that cares about answer
    content stays honest if the filler ever moves or stops firing.
    """
    return [m for m in _speaks(run) if not str(m.get("chunk_id", "")).endswith("-filler")]


def _filler_speaks(run):
    return [m for m in _speaks(run) if str(m.get("chunk_id", "")).endswith("-filler")]


def _streaming_run(**overrides):
    run = {
        "run_id": "r1",
        "bot_name": "Tom",
        "state": live_avatar.AgentState.THINKING,
        "active_turn_id": 1,
        "history": [],
        "echo": EchoSuppressor(),
        "governor": SpeechGovernor(),
        "floor": FloorState(),
        "rep_sockets": set(),
        "chunk_events": {},
        "turn_timing": {},
    }
    run["control_ws"] = FakeControlWs(run)
    run.update(overrides)
    return run


# -- sentence-by-sentence streaming dispatch ---------------------------------


def test_company_knowledge_streams_sentence_by_sentence(monkeypatch):
    """With run+turn_id given, each sentence is spoken as it arrives — not
    buffered and sent as one block after the whole answer is ready."""
    _install_model(monkeypatch, [])
    run = _streaming_run()

    async def streaming_rag(question, *args, on_sentence=None, **kwargs):
        assert on_sentence is not None
        await on_sentence("We hold SOC 2 Type II certification.")
        await on_sentence("Our last audit closed in March.")
        return "We hold SOC 2 Type II certification. Our last audit closed in March."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", streaming_rag)

    reply = _run_reply(run=run, turn_id=1)

    # The filler goes out first and is not part of the answer; everything
    # after it is.
    assert _speaks(run)[0]["chunk_id"] == "1-filler"
    assert _speaks(run)[0]["text"] in live_avatar.COMPANY_KNOWLEDGE_FILLER_PHRASES

    speak_messages = _answer_speaks(run)
    assert len(speak_messages) == 2
    assert speak_messages[0]["text"] == "We hold SOC 2 Type II certification."
    assert speak_messages[1]["text"] == "Our last audit closed in March."
    # Distinct chunk_ids, same turn_id, so the control_ws handler can tell
    # these apart while still knowing they belong to one turn.
    assert speak_messages[0]["chunk_id"] != speak_messages[1]["chunk_id"]
    assert speak_messages[0]["turn_id"] == speak_messages[1]["turn_id"] == 1
    assert run["_streamed_turn_id"] == 1
    assert reply == "We hold SOC 2 Type II certification. Our last audit closed in March."
    # Bookkeeping the caller would otherwise do via _dispatch_reply already
    # happened as a side effect of streaming.
    assert run["history"][-1] == {"speaker": "Tom", "participant_id": "bot", "text": reply}
    assert run["state"] == live_avatar.AgentState.LISTENING  # floor released


def test_filler_precedes_the_answer_and_is_not_part_of_it(monkeypatch):
    """The latency-masking filler is spoken once, before any real chunk, and
    leaves no trace in the reply, the history, or the echo suppressor's view
    of what the bot said."""
    _install_model(monkeypatch, [])
    run = _streaming_run()

    async def streaming_rag(question, *args, on_sentence=None, **kwargs):
        await on_sentence("We hold SOC 2 Type II certification.")
        return "We hold SOC 2 Type II certification."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", streaming_rag)

    reply = _run_reply(run=run, turn_id=1)

    sent = _speaks(run)
    assert len(_filler_speaks(run)) == 1, "exactly one filler per turn"
    assert sent[0]["chunk_id"] == "1-filler", "filler leads, never interleaves"
    assert sent[0]["turn_id"] == 1, "filler rides the turn it is masking"
    # What the meeting hears is the filler; what the transcript records is not.
    assert reply == "We hold SOC 2 Type II certification."
    assert run["history"][-1]["text"] == reply


def test_streaming_never_speaks_a_raw_degraded_first_sentence(monkeypatch):
    """If the very first sentence looks like a backend failure marker,
    nothing is spoken via streaming — it falls through to the existing
    full-buffer degraded-answer handling instead."""
    _install_model(monkeypatch, [])
    run = _streaming_run()

    async def degraded_rag(question, *args, on_sentence=None, **kwargs):
        if on_sentence is not None:
            await on_sentence("An error occurred while accessing the company knowledge base.")
        return "An error occurred while accessing the company knowledge base."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", degraded_rag)

    reply = _run_reply(run=run, turn_id=1)

    # The filler has already been committed to by the time the backend's
    # failure is visible — that is the cost of masking latency, and it is
    # harmless ("let me check the docs" then an apology). What must not
    # happen is the degraded text itself reaching the avatar.
    assert _answer_speaks(run) == []
    assert len(_filler_speaks(run)) == 1
    assert "_streamed_turn_id" not in run
    assert "don" in reply.lower()  # the standard apology fallback


def test_streaming_stops_at_the_word_backstop_on_a_sentence_boundary(monkeypatch):
    """The incremental word budget behaves like the non-streaming backstop:
    it stops at a complete sentence, never mid-clause, and never speaks past
    the budget."""
    _install_model(monkeypatch, [])
    run = _streaming_run()
    budget = live_avatar.AGENT_MAX_REPLY_WORDS + live_avatar.MAX_QUESTION_WORDS
    first = " ".join(["alpha"] * (budget - 5)) + "."
    second = " ".join(["beta"] * 20) + "."

    async def long_rag(question, *args, on_sentence=None, **kwargs):
        if on_sentence is not None:
            await on_sentence(first)
            await on_sentence(second)
        return f"{first} {second}"

    monkeypatch.setattr(live_avatar, "query_spiked_rag", long_rag)

    reply = _run_reply(run=run, turn_id=1)

    speak_messages = _answer_speaks(run)
    assert len(speak_messages) == 1
    assert speak_messages[0]["text"] == first
    assert reply == first
    # The filler is spoken outside the word budget — it is latency masking,
    # not part of the answer, so it must not eat into the backstop.
    assert len(_filler_speaks(run)) == 1


def test_streaming_stops_when_the_turn_is_superseded_mid_answer(monkeypatch):
    """Barge-in / a new turn taking over mid-stream must stop further chunks
    from being spoken, not talk over whatever superseded it."""
    _install_model(monkeypatch, [])
    run = _streaming_run()

    async def interrupting_rag(question, *args, on_sentence=None, **kwargs):
        if on_sentence is not None:
            await on_sentence("First sentence spoken normally.")
            run["active_turn_id"] = 2  # a new turn took the floor mid-stream
            await on_sentence("This one must never be spoken.")
        return "First sentence spoken normally. This one must never be spoken."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", interrupting_rag)

    _run_reply(run=run, turn_id=1)

    speak_messages = _answer_speaks(run)
    assert len(speak_messages) == 1
    assert speak_messages[0]["text"] == "First sentence spoken normally."


# -- the three-part reply ---------------------------------------------------


def test_company_knowledge_is_single_shot_and_never_calls_gemini(monkeypatch):
    """company_knowledge turns speak the backend's answer directly — one
    network call (/ask/regular), zero Gemini round trips. Only a short
    persona one-liner rides along with the question; a full instruction
    paragraph would dominate the backend's retrieval embedding (the same
    `question` field is used for both retrieval and generation) and pull
    different chunks/facts than the console's clean question would."""
    _install_model(monkeypatch, [])  # any consumption attempt raises IndexError

    async def recording_rag(question, *args, **kwargs):
        assert question.startswith("what about SOC 2")
        assert "Solution Architect" in question
        assert len(question.split()) < 30  # one-liner, not a paragraph
        return "We hold SOC 2 Type II. When does your review cycle start?"

    monkeypatch.setattr(live_avatar, "query_spiked_rag", recording_rag)

    reply = _run_reply()

    assert reply == "We hold SOC 2 Type II. When does your review cycle start?"
    assert reply.endswith("?")


def test_reply_without_a_question_is_left_alone(monkeypatch):
    """Greetings and confirmations legitimately end without a question."""
    _install_model(monkeypatch, [
        SimpleNamespace(parsed=live_avatar.GroundedReply(
            answer="Yes, I can hear you.", bridge="", next_question="",
        ))
    ])
    monkeypatch.setattr(live_avatar, "query_spiked_rag", _ok_rag)

    reply = _run_reply(intent="social")

    assert reply == "Yes, I can hear you."


def test_single_shot_strips_markdown_and_source_tags(monkeypatch):
    """The backend's raw text can carry markdown/[1] source tags meant for
    on-screen reading; none of it may reach TTS."""
    _install_model(monkeypatch, [])

    async def markdown_rag(*args, **kwargs):
        return "**We hold** SOC 2 Type II [1]. - bullet leftover\nWho owns compliance on your side?"

    monkeypatch.setattr(live_avatar, "query_spiked_rag", markdown_rag)

    reply = _run_reply()

    assert "*" not in reply and "[1]" not in reply
    assert reply.endswith("Who owns compliance on your side?")


def test_single_shot_backstop_trims_an_overlong_answer(monkeypatch):
    """The model is asked nicely for a word budget; if it ignores that, the
    backstop still caps length rather than speaking an unbounded answer."""
    _install_model(monkeypatch, [])
    overlong = " ".join(["word"] * 200)

    async def overlong_rag(*args, **kwargs):
        return overlong

    monkeypatch.setattr(live_avatar, "query_spiked_rag", overlong_rag)

    reply = _run_reply()

    assert len(reply.split()) <= live_avatar.AGENT_MAX_REPLY_WORDS + live_avatar.MAX_QUESTION_WORDS
    assert reply[-1] in ".!?"


def test_backstop_trims_on_a_sentence_boundary_not_mid_clause(monkeypatch):
    """A document-length backend answer (real sentences, not a word blob)
    must be trimmed at the last complete sentence that fits, never cut off
    mid-thought — a live test caught this trailing off ("...orchestrates.")
    when the trim was a raw word-count slice."""
    _install_model(monkeypatch, [])
    budget = live_avatar.AGENT_MAX_REPLY_WORDS + live_avatar.MAX_QUESTION_WORDS
    # Three real sentences: the first two together stay under budget, the
    # third alone would push the raw word count over it.
    first = " ".join(["alpha"] * (budget - 10)) + "."
    second = " ".join(["beta"] * 5) + "."
    third = " ".join(["gamma"] * 20) + "."
    long_answer = f"{first} {second} {third}"

    async def long_rag(*args, **kwargs):
        return long_answer

    monkeypatch.setattr(live_avatar, "query_spiked_rag", long_rag)

    reply = _run_reply()

    assert reply == f"{first} {second}"
    assert reply.endswith(".")
    assert "gamma" not in reply


def test_coaching_intent_skips_retrieval(monkeypatch):
    """Coaching is about running the call, so it never consults the knowledge base."""
    _install_model(monkeypatch, [
        SimpleNamespace(parsed=live_avatar.GroundedReply(
            answer="Decision criteria are still open.",
            bridge="",
            next_question="Ask who else has to sign off on this",
        ))
    ])
    called = []

    async def tracking_rag(*args, **kwargs):
        called.append(args)
        return "should not be reached"

    monkeypatch.setattr(live_avatar, "query_spiked_rag", tracking_rag)

    reply = _run_reply(intent="coaching")

    assert called == []
    assert reply.endswith("?")


def test_unavailable_retrieval_admits_it(monkeypatch):
    _install_model(monkeypatch, [])

    async def dead_rag(*args, **kwargs):
        return "An error occurred while accessing the company knowledge base."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", dead_rag)

    assert "don" in _run_reply().lower()


async def _ok_rag(*args, **kwargs):
    return "SpikedAI maintains SOC 2 Type II certification."


# -- compose_reply budgets --------------------------------------------------


@pytest.mark.parametrize(
    "question, expected_tail",
    [
        ("Who owns identity today", "Who owns identity today?"),
        ("Who owns identity today?", "Who owns identity today?"),
        ("Right", None),                    # stub, dropped
        (" ".join(["word"] * 25), None),    # over budget, dropped whole
    ],
)
def test_question_budget(question, expected_tail):
    reply = compose_reply(answer="Yes.", next_question=question)
    if expected_tail is None:
        assert reply == "Yes."
    else:
        assert reply.endswith(expected_tail)


def test_answer_is_the_only_part_trimmed_mid_thought():
    reply = compose_reply(
        answer=" ".join(["word"] * 60),
        next_question="And what about pricing then",
        max_answer_words=45,
    )
    assert reply.endswith("And what about pricing then?")
    assert reply.split(".")[0].split().count("word") == 45


# -- Level 1 ----------------------------------------------------------------


def _run_state(**overrides):
    run = {
        "run_id": "r1",
        "bot_name": "Tom",
        "state": AgentState.LISTENING,
        "history": [],
        "turn_counter": 0,
        "control_ws": None,
        "rep_sockets": set(),
        "pending_insight": None,
        "last_insight_at": None,
        "intel": None,
        "token": "",
        "client_id": None,
        "user_context": {
            "company_name": "SpikedAI",
            "products_services": "SOC 2 compliance, SCIM provisioning",
        },
        "pending_turns": {},
        "floor": FloorState(),
        "echo": EchoSuppressor(similarity_threshold=0.9, tail_seconds=8),
        "governor": SpeechGovernor(
            cooldown_seconds=2, max_replies_per_window=4, window_seconds=30
        ),
    }
    run.update(overrides)
    return run


class RecordingWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


def _insight_fired(run, text):
    console = next(iter(run["rep_sockets"]))
    async def drive():
        live_avatar._consider_insight(run, "Lisa", text, [])
        await asyncio.sleep(0)

    asyncio.run(drive())
    return any(m.get("type") == "insight_available" for m in console.sent)


@pytest.mark.parametrize(
    "text, fires",
    [
        ("How does SCIM provisioning handle offboarding?", True),
        ("What does your pricing look like at that volume?", True),
        ("We already use SCIM provisioning internally.", False),
        ("What time is the next meeting?", False),
    ],
)
def test_insight_only_fires_on_answerable_questions(monkeypatch, text, fires):
    monkeypatch.setattr(live_avatar, "gemini_client", None)
    run = _run_state(rep_sockets={RecordingWS()})
    assert _insight_fired(run, text) is fires


def test_insight_never_fires_while_the_agent_holds_the_floor(monkeypatch):
    monkeypatch.setattr(live_avatar, "gemini_client", None)
    run = _run_state(rep_sockets={RecordingWS()}, state=AgentState.SPEAKING)
    assert _insight_fired(run, "How does SCIM provisioning work?") is False


def test_insight_is_rate_limited(monkeypatch):
    monkeypatch.setattr(live_avatar, "gemini_client", None)
    run = _run_state(rep_sockets={RecordingWS()})

    assert _insight_fired(run, "How does SCIM provisioning work?") is True
    next(iter(run["rep_sockets"])).sent.clear()
    assert _insight_fired(run, "And what about SOC 2 compliance?") is False


# -- teardown ---------------------------------------------------------------


def test_teardown_is_idempotent_and_evicts_the_run(monkeypatch):
    """Disconnect is reached when something is already wrong; it must not raise."""
    stopped = []

    async def fake_leave(bot_id):
        stopped.append(("recall", bot_id))
        return 200

    class FakeVideo:
        async def close(self, session):
            stopped.append(("avatar", session.session_id))

    from src.core import runs as runs_module

    monkeypatch.setattr(runs_module, "_leave_recall_call", fake_leave)

    class FakeIntel:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    intel = FakeIntel()
    live_avatar._ACTIVE_RUNS["tear"] = _run_state(
        bot_id="bot-1", session_id="sess-1", intel=intel,
        active_response_task=None, watchdog_task=None,
        providers=SimpleNamespace(video=FakeVideo()),
        video_session=SimpleNamespace(session_id="sess-1"),
    )

    result = asyncio.run(live_avatar._teardown_run("tear"))

    assert result["ok"] is True
    assert intel.stopped is True
    assert ("recall", "bot-1") in stopped
    assert ("avatar", "sess-1") in stopped
    assert "tear" not in live_avatar._ACTIVE_RUNS

    # Second call on an already-released run is a no-op, not an error.
    assert asyncio.run(live_avatar._teardown_run("tear")) == {
        "ok": True,
        "already_gone": True,
    }
