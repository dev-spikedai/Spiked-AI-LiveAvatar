import asyncio
import base64
import hashlib
import hmac
from types import SimpleNamespace

from src import live_avatar


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate_content(self, **kwargs):
        return self.responses.pop(0)


def test_self_contained_question_skips_the_classification_round_trip(monkeypatch):
    """A self-contained company question goes straight to RAG: one model call."""
    models = FakeModels([
        SimpleNamespace(text="It uses the verified security controls described in the knowledge base."),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))
    rag_queries = []

    async def fake_rag(query, auth_token, client_id):
        rag_queries.append((query, auth_token, client_id))
        return "Verified security controls are enabled."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", fake_rag)
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="Tom, how do you handle security?",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        client_id="client",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom", "keywords": ["SpikedAI"]},
    ))

    # Raw transcript is the retrieval query, and the analysis response was never consumed.
    assert rag_queries == [("Tom, how do you handle security?", "token", "client")]
    assert models.responses == []
    assert answer == "It uses the verified security controls described in the knowledge base."


def test_turn_needing_context_still_runs_query_repair(monkeypatch):
    """A pronoun means the raw text is a poor query, so classification still runs."""
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                intent="company_knowledge",
                resolved_query="SpikedAI enterprise pricing",
                corrections=[],
            )
        ),
        SimpleNamespace(text="Pricing starts at the published enterprise tier."),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))
    rag_queries = []

    async def fake_rag(query, auth_token, client_id):
        rag_queries.append((query, auth_token, client_id))
        return "Enterprise pricing is published."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", fake_rag)
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="Tom, what is its pricing?",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        client_id="client",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom", "keywords": ["SpikedAI"]},
    ))

    assert rag_queries == [("SpikedAI enterprise pricing", "token", "client")]
    assert answer == "Pricing starts at the published enterprise tier."


def test_rag_failure_returns_short_honest_fallback(monkeypatch):
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                intent="company_knowledge",
                resolved_query="pricing",
                corrections=[],
            )
        ),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))

    async def fake_rag(*args):
        return "No specific documentation found for this query."

    monkeypatch.setattr(live_avatar, "query_spiked_rag", fake_rag)
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="Tom, what is the price?",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom"},
    ))
    assert answer == "I don’t have verified information on that available right now."


def test_llm_response_gate_can_keep_the_agent_silent(monkeypatch):
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                response_action="silent",
                intent="social",
                resolved_query="",
                corrections=[],
            )
        ),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))

    async def fail_if_rag_called(*args):
        raise AssertionError("silent turns must not query RAG")

    monkeypatch.setattr(live_avatar, "query_spiked_rag", fail_if_rag_called)
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="I told Tom about this yesterday.",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom"},
    ))
    assert answer is None


def test_llm_response_gate_can_acknowledge_without_generation(monkeypatch):
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                response_action="acknowledge",
                intent="command",
                resolved_query="wait",
                corrections=[],
            )
        ),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="Tom, please wait a moment.",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom"},
    ))
    assert answer == "Understood."


def make_run(**overrides):
    run = {
        "bot_name": "Tom",
        "history": [],
        "active_response_task": None,
        "watchdog_task": None,
        "pending_turns": {},
        "floor": live_avatar.FloorState(),
        "echo": live_avatar.EchoSuppressor(),
        "governor": live_avatar.SpeechGovernor(),
        "state": live_avatar.AgentState.LISTENING,
    }
    run.update(overrides)
    return run


def test_third_person_mention_cannot_take_the_fast_path(monkeypatch):
    """The fast path must not skip the LLM addressee gate on an ambiguous turn.

    "pricing" satisfies requires_company_knowledge and there is no pronoun, so
    the only thing standing between this turn and an unwanted answer is the
    is_directly_addressed check.
    """
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                response_action="silent",
                intent="meeting_context",
                resolved_query="",
                corrections=[],
            )
        ),
    ])
    monkeypatch.setattr(live_avatar, "gemini_client", SimpleNamespace(aio=SimpleNamespace(models=models)))

    async def fail_if_rag_called(*args):
        raise AssertionError("a third-person mention must not reach RAG")

    monkeypatch.setattr(live_avatar, "query_spiked_rag", fail_if_rag_called)
    answer = asyncio.run(live_avatar.process_transcript_with_gemini(
        transcript="I asked Tom about pricing yesterday.",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Tom", "keywords": ["SpikedAI"]},
    ))
    assert answer is None


def test_acknowledgement_is_rate_limited_like_any_other_reply():
    """`acknowledge` returns text through the normal dispatch, so the governor
    sees it. Otherwise repeated "Tom, wait" would machine-gun "Understood."."""
    governor = live_avatar.SpeechGovernor()
    now = live_avatar.time.monotonic()

    allowed, _ = governor.allows_reply(now)
    assert allowed
    governor.note_reply("Understood.", now)

    # Same acknowledgement moments later is both inside the cooldown and a duplicate.
    allowed, reason = governor.allows_reply(now + 0.5)
    assert not allowed and reason == "cooldown"
    assert governor.is_duplicate("Understood.", now + 5)


def test_unaddressed_turn_is_recorded_but_never_scheduled():
    run = make_run()
    live_avatar._finalize_turn(run, "42", "Alice", "What time can you meet?")
    assert run["active_response_task"] is None
    assert run["history"] == [{
        "speaker": "Alice",
        "participant_id": "42",
        "text": "What time can you meet?",
    }]


def test_self_audio_is_dropped_before_the_gate():
    """The agent hearing its own reply must not reach history or the gate."""
    run = make_run()
    reply = "Tom here, our pricing starts at four hundred dollars per seat."
    run["echo"].note_bot_speech(reply, live_avatar.time.monotonic())

    async def drive():
        # Arrives under a human participant id, as room echo actually does.
        live_avatar._ingest_utterance(run, "42", "Alice", {"text": reply, "words": []})
        await asyncio.sleep(0)

    asyncio.run(drive())
    assert run["history"] == []
    assert run["pending_turns"] == {}


def test_governor_blocks_a_second_reply_inside_the_cooldown():
    run = make_run()
    now = live_avatar.time.monotonic()
    run["governor"].note_reply("Our pricing starts at four hundred dollars.", now)

    live_avatar._finalize_turn(run, "42", "Alice", "Tom, what about support hours?")

    # Turn is still recorded, but no inference task is spawned.
    assert run["active_response_task"] is None
    assert len(run["history"]) == 1


def test_recall_websocket_signature_verification(monkeypatch):
    key = b"recall-test-secret"
    secret = "whsec_" + base64.b64encode(key).decode()
    message_id = "msg_123"
    timestamp = "1234567890"
    digest = hmac.new(key, f"{message_id}.{timestamp}.".encode(), hashlib.sha256).digest()
    headers = {
        "webhook-id": message_id,
        "webhook-timestamp": timestamp,
        "webhook-signature": "v1," + base64.b64encode(digest).decode(),
    }
    monkeypatch.setattr(live_avatar, "RECALL_WEBHOOK_SECRET", secret)
    assert live_avatar.verify_recall_websocket(headers, "fallback", "wrong")
    headers["webhook-signature"] = "v1," + base64.b64encode(b"wrong").decode()
    assert not live_avatar.verify_recall_websocket(headers, "fallback", "fallback")


def test_recall_websocket_uses_per_run_token_without_workspace_secret(monkeypatch):
    monkeypatch.setattr(live_avatar, "RECALL_WEBHOOK_SECRET", "")
    assert live_avatar.verify_recall_websocket({}, "secret-token", "secret-token")
    assert not live_avatar.verify_recall_websocket({}, "secret-token", "wrong")
