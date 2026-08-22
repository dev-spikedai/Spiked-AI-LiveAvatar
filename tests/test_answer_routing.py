"""The reply pipeline must actually consult the run's providers.

Before this wiring existed, answer_engine was accepted at /start and then
ignored, and a delegated run could never speak at all.
"""

import asyncio
from types import SimpleNamespace


from src import live_avatar
from src.agent_policy import AgentState, EchoSuppressor, FloorState, SpeechGovernor
from src.providers.base import AnswerEngine, TurnContext


class FakeControlWs:
    """Acks each chunk instantly, as if the avatar finished speaking at once."""

    def __init__(self, run=None):
        self.sent = []
        self.run = run

    async def send_json(self, message):
        self.sent.append(message)
        if message.get("type") == "avatar_speak" and self.run is not None:
            event = (self.run.get("chunk_events") or {}).get(message.get("chunk_id"))
            if event is not None:
                event.set()


def _run(providers=None, **overrides):
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
        "providers": providers,
    }
    run["control_ws"] = FakeControlWs(run)
    run.update(overrides)
    return run


def _reply(run, turn_id=1):
    return asyncio.run(live_avatar._generate_grounded_reply(
        analysis=live_avatar.TurnAnalysis(
            response_action="respond",
            intent="company_knowledge",
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
        run=run,
        turn_id=turn_id,
    ))


class RecordingEngine(AnswerEngine):
    name = "recording"

    def __init__(self, answer_text="We hold SOC 2 Type II."):
        self.answer_text = answer_text
        self.contexts = []

    async def answer(self, ctx: TurnContext, on_sentence=None) -> str:
        self.contexts.append(ctx)
        if on_sentence is not None:
            await on_sentence(self.answer_text)
        return self.answer_text


# -- streaming engines ------------------------------------------------------


def test_selected_engine_is_used_instead_of_calling_retrieval_directly(monkeypatch):
    called = []

    async def should_not_run(*args, **kwargs):
        called.append(args)
        return "direct retrieval"

    monkeypatch.setattr(live_avatar, "query_spiked_rag", should_not_run)
    engine = RecordingEngine()
    run = _run(providers=SimpleNamespace(
        is_delegated=False, answer=engine, video=SimpleNamespace(accepts="text"), tts=None,
    ))

    reply = _reply(run)

    assert called == [], "the engine must replace the direct retrieval call"
    assert len(engine.contexts) == 1
    assert reply == "We hold SOC 2 Type II."


def test_engine_receives_the_retrieval_context_it_needs(monkeypatch):
    """kyc_id / source_ids / timeout used to be passed straight to
    query_spiked_rag; routing through an engine must not drop them."""
    monkeypatch.setattr(live_avatar, "query_spiked_rag", lambda *a, **k: None)
    engine = RecordingEngine()
    run = _run(providers=SimpleNamespace(
        is_delegated=False, answer=engine, video=SimpleNamespace(accepts="text"), tts=None,
    ))

    asyncio.run(live_avatar._generate_grounded_reply(
        analysis=live_avatar.TurnAnalysis(
            response_action="respond", intent="company_knowledge",
            resolved_query="what about SOC 2", corrections=[],
        ),
        transcript="Tom, what about SOC 2?", speaker="Lisa", bot_name="Tom",
        company_name="SpikedAI", history_text="", catalog=["SpikedAI"],
        auth_token="token", client_id="client", preferred_model="fake-model",
        kyc_id="kyc-7", source_ids=["s1", "s2"], rag_timeout_s=3.5,
        run=run, turn_id=1,
    ))

    ctx = engine.contexts[0]
    assert ctx.kyc_id == "kyc-7"
    assert ctx.source_ids == ["s1", "s2"]
    assert ctx.timeout_s == 3.5
    assert ctx.client_id == "client"
    assert ctx.auth_token == "token"
    assert ctx.turn_id == 1


def test_a_run_without_providers_still_calls_retrieval_directly(monkeypatch):
    """Runs created before the provider layer must survive a rolling deploy."""
    seen = []

    async def direct(*args, **kwargs):
        seen.append(kwargs)
        if kwargs.get("on_sentence"):
            await kwargs["on_sentence"]("Direct answer.")
        return "Direct answer."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", direct)
    assert _reply(_run(providers=None)) == "Direct answer."
    assert len(seen) == 1


# -- delegated turns --------------------------------------------------------


def test_delegated_turn_hands_the_transcript_to_the_vendor(monkeypatch):
    called = []
    monkeypatch.setattr(
        live_avatar, "query_spiked_rag",
        lambda *a, **k: called.append(a),
    )
    run = _run(providers=SimpleNamespace(
        is_delegated=True, answer=None, video=SimpleNamespace(accepts="text"), tts=None,
    ))

    async def drive():
        loop = asyncio.get_running_loop()
        loop.call_later(
            0.01, live_avatar.speech.resolve_vendor_reply, run, 1, "Anam answered this."
        )
        return await live_avatar._generate_grounded_reply(
            analysis=live_avatar.TurnAnalysis(
                response_action="respond", intent="company_knowledge",
                resolved_query="what about SOC 2", corrections=[],
            ),
            transcript="Tom, what about SOC 2?", speaker="Lisa", bot_name="Tom",
            company_name="SpikedAI", history_text="", catalog=["SpikedAI"],
            auth_token="token", client_id="client", preferred_model="fake-model",
            run=run, turn_id=1,
        )

    reply = asyncio.run(drive())

    assert called == [], "a delegated turn must not run our own retrieval"
    assert reply == "Anam answered this."
    assert run["control_ws"].sent == [
        {"type": "avatar_user_message", "text": "what about SOC 2", "turn_id": 1}
    ]
    # Marked as already-spoken so the caller does not dispatch it a second time.
    assert run["_streamed_turn_id"] == 1
    assert run["state"] == AgentState.LISTENING


def test_delegated_turn_that_times_out_releases_the_floor(monkeypatch):
    monkeypatch.setattr(live_avatar, "DELEGATED_TURN_TIMEOUT_S", 0.05)
    run = _run(providers=SimpleNamespace(
        is_delegated=True, answer=None, video=SimpleNamespace(accepts="text"), tts=None,
    ))

    assert _reply(run) is None
    # The whole point: a silent vendor must not strand the agent in THINKING.
    assert run["state"] == AgentState.LISTENING
    assert run["_streamed_turn_id"] == 1
