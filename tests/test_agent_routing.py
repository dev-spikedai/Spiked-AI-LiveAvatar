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


def test_company_knowledge_route_always_calls_rag(monkeypatch):
    models = FakeModels([
        SimpleNamespace(
            parsed=live_avatar.TurnAnalysis(
                intent="company_knowledge",
                resolved_query="SpikedAI enterprise security",
                corrections=[],
            )
        ),
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

    assert rag_queries == [("SpikedAI enterprise security", "token", "client")]
    assert answer == "It uses the verified security controls described in the knowledge base."


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


def test_llm_response_gate_can_keep_lara_silent(monkeypatch):
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
        transcript="I told Lara about this yesterday.",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Lara"},
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
        transcript="Lara, please wait a moment.",
        speaker="Alice",
        conversation_history=[],
        auth_token="token",
        user_context={"company_name": "SpikedAI", "bot_name": "Lara"},
    ))
    assert answer == "Understood."


def test_unaddressed_turn_is_recorded_but_never_scheduled():
    run = {
        "bot_name": "Tom",
        "history": [],
        "active_response_task": None,
    }
    live_avatar._schedule_completed_turn(
        run,
        participant_id="42",
        participant_name="Alice",
        utterance={"text": "What time can you meet?", "words": []},
    )
    assert run["active_response_task"] is None
    assert run["history"] == [{
        "speaker": "Alice",
        "participant_id": "42",
        "text": "What time can you meet?",
    }]


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
