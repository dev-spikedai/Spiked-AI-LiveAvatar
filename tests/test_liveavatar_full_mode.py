import asyncio

from src.providers.base import RunContext
from src.providers.video import liveavatar


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = "ok"
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/sessions/token"):
            return Response({"data": {"session_token": "session-token"}})
        return Response({
            "data": {
                "session_id": "session-1",
                "livekit_url": "wss://livekit",
                "livekit_client_token": "client-token",
            }
        })


def test_liveavatar_provider_requests_full_push_to_talk(monkeypatch):
    FakeClient.calls = []
    monkeypatch.setattr(liveavatar, "LIVEAVATAR_API_KEY", "api-key")
    monkeypatch.setattr(liveavatar, "LIVEAVATAR_AVATAR_ID", "configured-avatar")
    monkeypatch.setattr(liveavatar.httpx, "AsyncClient", lambda **_kwargs: FakeClient())

    provider = liveavatar.LiveAvatarVideoProvider()
    session = asyncio.run(
        provider.create_session(RunContext(run_id="r1", bot_name="Tom"))
    )

    token_payload = FakeClient.calls[0][1]["json"]
    assert token_payload["mode"] == "FULL"
    assert token_payload["interactivity_type"] == "PUSH_TO_TALK"
    assert token_payload["avatar_persona"] == {}
    assert token_payload["avatar_id"] == "configured-avatar"
    assert session.credentials["mode"] == "FULL"
    assert session.session_id == "session-1"
