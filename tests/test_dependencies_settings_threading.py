"""
Verifies the A2 fix in core/dependencies.py's get_current_user_settings: the
Supabase settings-fetch call must run in a thread (asyncio.to_thread), not
directly on the event loop, on a cache miss.

Same methodology as tests/test_search_rpc_threading.py -- simulates a slow
synchronous .execute() call and asserts a concurrently running coroutine
keeps ticking while the fetch "runs".
"""
import asyncio
import time

import core.dependencies as deps_mod


class _FakeExecuteResult:
    data = {
        "bot_name": "SpikedAI",
        "selected_persona": "balanced",
        "custom_prompt": "",
        "answer_styles": [],
        "meeting_domains": [],
        "strategic_keywords": [],
        "executive_snapshot": "",
        "seller_company": "",
        "products_services": "",
        "product_domain": "",
        "client_company": "",
        "seller_name": "",
        "client_names": "",
        "sub_domains": "",
        "company_url": "",
        "seller_linkedin_url": "",
        "seller_job_profile": "",
        "user_industry": "",
    }


class _FakeQuery:
    def __init__(self, delay: float):
        self.delay = delay

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def single(self):
        return self

    def execute(self):
        time.sleep(self.delay)
        return _FakeExecuteResult()


class _FakeSupabase:
    def __init__(self, delay: float):
        self.delay = delay

    def table(self, name):
        return _FakeQuery(self.delay)


def test_settings_fetch_does_not_block_event_loop(monkeypatch):
    delay = 0.3
    fake_supabase = _FakeSupabase(delay)

    monkeypatch.setattr(deps_mod, "get_g_vars", lambda: {"supabase": fake_supabase})
    monkeypatch.setattr(deps_mod, "_SETTINGS_CACHE", {})

    async def scenario():
        ticks = {"count": 0}

        async def ticker():
            while True:
                ticks["count"] += 1
                await asyncio.sleep(0.02)

        ticker_task = asyncio.create_task(ticker())
        settings = await deps_mod.get_current_user_settings(user_id="some-user")
        ticker_task.cancel()
        return ticks["count"], settings

    ticks, settings = asyncio.run(scenario())

    assert settings.bot_name == "SpikedAI"
    assert ticks >= 8, f"expected the event loop to keep ticking during the settings fetch, got {ticks} ticks"
