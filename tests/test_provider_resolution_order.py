"""Explicit request field beats per-client config beats env default."""

import asyncio
from types import SimpleNamespace

import pytest

from src import supabase_client
from src.live_avatar import pick_provider


def test_explicit_request_field_wins():
    assert pick_provider("anam", "simli", "liveavatar") == "anam"


def test_client_config_wins_over_env_default():
    assert pick_provider(None, "simli", "liveavatar") == "simli"


def test_env_default_is_the_last_resort():
    assert pick_provider(None, None, "liveavatar") == "liveavatar"


def test_no_tts_anywhere_stays_none():
    """None must survive: it is what tells the registry a text provider needs
    no voice, and "" would be looked up as a provider name."""
    assert pick_provider(None, None, None) is None


def test_lookup_without_a_client_id_returns_no_overrides():
    assert asyncio.run(supabase_client.get_client_providers(None)) == {}


def test_lookup_failure_is_not_fatal(monkeypatch):
    """A provider lookup failing must not stop a meeting from starting."""
    class Boom:
        def table(self, *_a, **_k):
            raise RuntimeError("supabase down")

    monkeypatch.setattr(supabase_client, "_supabase_client", Boom())
    monkeypatch.setattr(supabase_client, "_PROVIDER_CACHE", {})
    assert asyncio.run(supabase_client.get_client_providers("c1")) == {}


def test_blank_columns_are_not_treated_as_overrides(monkeypatch):
    """A row with empty strings means "no preference", not "use empty"."""
    class FakeTable:
        def select(self, *_): return self
        def eq(self, *_): return self
        def limit(self, *_): return self
        def execute(self):
            return SimpleNamespace(data=[{"video_provider": "anam", "tts_provider": "", "answer_engine": None}])

    monkeypatch.setattr(supabase_client, "_supabase_client", SimpleNamespace(table=lambda *_: FakeTable()))
    monkeypatch.setattr(supabase_client, "_PROVIDER_CACHE", {})
    assert asyncio.run(supabase_client.get_client_providers("c1")) == {"video_provider": "anam"}
