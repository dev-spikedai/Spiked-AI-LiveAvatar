"""
Verifies the B1 fix in core/kyc_database.py: get_client_kyc_config must cache
its Supabase lookup (WARM_PATH_REPORT.md claimed this was already fixed --
it wasn't, as of this codebase state), and the write paths
(upsert_client_kyc_config, upsert_client_context) must invalidate the cache
entry they touch so an edit is visible on the next read immediately, not
after the TTL expires.
"""
import asyncio

import core.kyc_database as kyc_mod


class _FakeSelectQuery:
    def __init__(self, rows_by_key, call_log):
        self._rows_by_key = rows_by_key
        self._call_log = call_log
        self._filters = {}

    def select(self, *a, **k):
        return self

    def eq(self, field, value):
        self._filters[field] = value
        return self

    def maybe_single(self):
        return self

    def execute(self):
        self._call_log.append(dict(self._filters))
        key = (self._filters.get("user_id"), self._filters.get("client_id"), self._filters.get("kyc_id"))
        row = self._rows_by_key.get(key)

        class _Resp:
            data = row

        return _Resp()


class _FakeUpsertQuery:
    def __init__(self, rows_by_key):
        self._rows_by_key = rows_by_key
        self._record = None

    def upsert(self, record, on_conflict=None):
        self._record = record
        return self

    def execute(self):
        key = (self._record["user_id"], self._record["client_id"], self._record["kyc_id"])
        self._rows_by_key[key] = dict(self._record)

        class _Resp:
            data = [self._record]

        return _Resp()


class _FakeSupabase:
    def __init__(self):
        self.rows_by_key = {}
        self.select_calls = []

    def table(self, name):
        assert name == "client_kyc_configs"
        return _DualQuery(self.rows_by_key, self.select_calls)


class _DualQuery:
    """Returns the right fake builder depending on whether .select or .upsert is called next."""

    def __init__(self, rows_by_key, call_log):
        self._rows_by_key = rows_by_key
        self._call_log = call_log

    def select(self, *a, **k):
        return _FakeSelectQuery(self._rows_by_key, self._call_log).select(*a, **k)

    def upsert(self, record, on_conflict=None):
        return _FakeUpsertQuery(self._rows_by_key).upsert(record, on_conflict)


def test_get_client_kyc_config_is_cached_across_calls(monkeypatch):
    fake_supabase = _FakeSupabase()
    key = ("user-1", "client-1", "kyc-1")
    fake_supabase.rows_by_key[key] = {"seller_name": "Alice", "client_company": "Acme"}

    monkeypatch.setattr(kyc_mod, "get_g_vars", lambda: {"supabase": fake_supabase})
    monkeypatch.setattr(kyc_mod, "_KYC_CONFIG_CACHE", {})

    async def scenario():
        first = await kyc_mod.get_client_kyc_config(*key)
        second = await kyc_mod.get_client_kyc_config(*key)
        return first, second

    first, second = asyncio.run(scenario())

    assert first["seller_name"] == "Alice"
    assert second == first
    assert len(fake_supabase.select_calls) == 1, (
        f"expected exactly 1 Supabase round trip (2nd call should hit cache), got {len(fake_supabase.select_calls)}"
    )


def test_upsert_invalidates_the_cache_entry(monkeypatch):
    fake_supabase = _FakeSupabase()
    key = ("user-1", "client-1", "kyc-1")
    fake_supabase.rows_by_key[key] = {"seller_name": "Alice", "client_company": "Acme"}

    monkeypatch.setattr(kyc_mod, "get_g_vars", lambda: {"supabase": fake_supabase})
    monkeypatch.setattr(kyc_mod, "_KYC_CONFIG_CACHE", {})

    async def scenario():
        before = await kyc_mod.get_client_kyc_config(*key)
        await kyc_mod.upsert_client_context(
            user_id="user-1", client_id="client-1", kyc_id="kyc-1",
            client_names="Bob", client_company="NewCo",
        )
        after = await kyc_mod.get_client_kyc_config(*key)
        return before, after

    before, after = asyncio.run(scenario())

    assert before["client_company"] == "Acme"
    # After the upsert, the cache entry must have been evicted so this read
    # reflects the write immediately rather than serving the stale "Acme" row
    # for up to _KYC_CONFIG_CACHE_TTL.
    assert after["client_company"] == "NewCo"
    assert len(fake_supabase.select_calls) == 2, (
        f"expected a fresh Supabase read after the upsert invalidated the cache, got {len(fake_supabase.select_calls)} reads"
    )
