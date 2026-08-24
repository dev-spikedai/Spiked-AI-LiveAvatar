"""The frontend end of the provider seam: avatar.js imports whatever
`browser_module` names, so that route has to serve every registered provider
and nothing else.
"""

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

import src.live_avatar as live_avatar
from src.live_avatar import app
from src.providers import registry


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_every_registered_video_provider_has_a_servable_browser_half(client):
    """A provider whose Python half exists but whose JS half does not is a
    404 at meeting time, on a page nobody is watching."""
    for name, cls in registry.VIDEO_PROVIDERS.items():
        module = cls().browser_module
        resp = client.get(module)
        assert resp.status_code == 200, f"{name}: {module} is not served"
        assert "export async function connect" in resp.text, (
            f"{name}: {module} does not implement the provider contract"
        )


def test_unknown_provider_module_is_not_found(client):
    assert client.get("/providers/nope.js").status_code == 404


@pytest.mark.parametrize("attack", ["..%2f..%2f.env", "..%2fmain.py", "sub%2fdir.js"])
def test_provider_route_refuses_paths(client, attack):
    """The module name arrives from a URL handed to the page; it must not be
    able to address anything but a bare .js file in the providers directory."""
    assert client.get(f"/providers/{attack}").status_code == 404


def test_shell_is_served_and_resolves_a_provider(client):
    resp = client.get("/avatar.js")
    assert resp.status_code == 200
    assert "browser_module" in resp.text
    assert "import(moduleUrl)" in resp.text


def test_health_exposes_non_secret_runtime_capabilities(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "warm_rag_configured" in body
    assert "mcp_configured" in body
    assert "liveavatar_full_mode" in body
    assert "persistent_memory_configured" in body
    assert "RECALL_API_KEY" not in str(body)


def test_diagnostics_are_sanitized_and_include_live_timing(client):
    run_id = "run_diagnostics_test"
    live_avatar._ACTIVE_RUNS[run_id] = {
        "run_id": run_id,
        "state": "LISTENING",
        "bot_name": "Tom",
        "providers": SimpleNamespace(video=SimpleNamespace(name="liveavatar")),
        "autospeak_enabled": True,
        "meeting_preferences": {"speak_up_when_helpful": True},
        "turn_timing": {1: {"finalized_at": 1.0, "dispatched_at": 2.0}},
        "live_turns": {"p1": {"speaker": "Client", "updated_at": 3.0, "wake_candidate": True}},
        "proactive_prefetches": {},
        "persistent_memory": [],
        "warm_task": None,
        "active_mcp_context": None,
        "token": "must-not-appear",
        "meeting_url": "must-not-appear",
    }
    try:
        response = client.get(f"/api/runs/{run_id}/diagnostics")
    finally:
        live_avatar._ACTIVE_RUNS.pop(run_id, None)

    assert response.status_code == 200
    body = response.json()
    assert body["turn_timing"]["1"]["dispatched_at"] == 2.0
    assert body["live_turns"]["p1"]["wake_candidate"] is True
    assert "token" not in body
    assert "meeting_url" not in body
