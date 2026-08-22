"""The frontend end of the provider seam: avatar.js imports whatever
`browser_module` names, so that route has to serve every registered provider
and nothing else.
"""

import pytest
from fastapi.testclient import TestClient

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
