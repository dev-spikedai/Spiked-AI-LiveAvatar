"""
Verifies the A1 fix in routers/search.py's build_rag_context: the Supabase RPC
fallback call must run in a thread (asyncio.to_thread), not directly on the
event loop, so a slow RPC on one request doesn't stall every other in-flight
request on the process.

Simulates a slow synchronous .execute() call and asserts a concurrently
running coroutine keeps making progress (ticking) while the RPC "runs" --
if the RPC call were blocking the event loop directly, no ticks could occur
during that window.
"""
import asyncio
import time

import numpy as np

import routers.search as search_mod


class _FakeExecuteResult:
    data = []


class _FakeRPCBuilder:
    def __init__(self, delay: float):
        self.delay = delay

    def execute(self):
        # Simulates a slow, synchronous network round trip (as the real
        # supabase-py client's .execute() is).
        time.sleep(self.delay)
        return _FakeExecuteResult()


class _FakeSupabase:
    def __init__(self, delay: float):
        self.delay = delay

    def rpc(self, name, params):
        return _FakeRPCBuilder(self.delay)


def test_rpc_fallback_does_not_block_event_loop(monkeypatch):
    delay = 0.3
    fake_supabase = _FakeSupabase(delay)

    monkeypatch.setattr(search_mod, "get_g_vars", lambda: {"supabase": fake_supabase})
    monkeypatch.setattr(search_mod, "_RAG_CACHE", {})
    monkeypatch.setattr(search_mod.warm_index, "search", lambda *a, **k: None)
    monkeypatch.setattr(search_mod.warm_index, "kick_off_warm_load", lambda user_id: None)

    async def fake_get_embeddings(texts):
        return np.zeros((1, 4), dtype=np.float32)

    monkeypatch.setattr(search_mod, "get_embeddings", fake_get_embeddings)

    async def scenario():
        ticks = {"count": 0}

        async def ticker():
            while True:
                ticks["count"] += 1
                await asyncio.sleep(0.02)

        ticker_task = asyncio.create_task(ticker())
        result = await search_mod.build_rag_context("some question", "user-not-in-demo-set")
        ticker_task.cancel()
        return ticks["count"], result

    ticks, result = asyncio.run(scenario())

    assert result is None  # empty RPC data -> no context, expected
    # delay=0.3s / tick interval=0.02s -> ~15 ticks possible if the event loop
    # was never blocked. A blocking-the-loop regression would produce ~0-1.
    assert ticks >= 8, f"expected the event loop to keep ticking during the RPC call, got {ticks} ticks"
