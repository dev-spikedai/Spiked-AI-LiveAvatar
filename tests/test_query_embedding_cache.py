"""
Verifies the B2 fix in routers/search.py: _get_query_embedding must cache the
embedding vector for a (user_id, question, query_augment) key so the live
retrieval call and the cognitive wide-retrieve call for the identical
question -- which use different top_k and therefore different _RAG_CACHE
entries -- don't each pay their own embedding model call.
"""
import asyncio

import numpy as np

import routers.search as search_mod


def test_repeated_call_same_key_hits_cache(monkeypatch):
    monkeypatch.setattr(search_mod, "_QUERY_EMBEDDING_CACHE", {})
    calls = {"count": 0}

    async def fake_get_embeddings(texts):
        calls["count"] += 1
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(search_mod, "get_embeddings", fake_get_embeddings)

    async def scenario():
        first = await search_mod._get_query_embedding("user-1", "what products?", "", "query: what products?")
        second = await search_mod._get_query_embedding("user-1", "what products?", "", "query: what products?")
        return first, second

    first, second = asyncio.run(scenario())

    assert calls["count"] == 1, f"expected 1 embedding call (2nd should hit cache), got {calls['count']}"
    assert np.array_equal(first, second)


def test_different_query_augment_is_a_cache_miss(monkeypatch):
    """query_augment changes the actual embedded text (KYC/persona steering) --
    a differently-augmented embedding must never be served across contexts."""
    monkeypatch.setattr(search_mod, "_QUERY_EMBEDDING_CACHE", {})
    calls = {"count": 0}

    async def fake_get_embeddings(texts):
        calls["count"] += 1
        return np.ones((1, 4), dtype=np.float32)

    monkeypatch.setattr(search_mod, "get_embeddings", fake_get_embeddings)

    async def scenario():
        await search_mod._get_query_embedding("user-1", "what products?", "", "query: what products?")
        await search_mod._get_query_embedding("user-1", "what products?", "enterprise buyer", "query: what products?\nenterprise buyer")

    asyncio.run(scenario())

    assert calls["count"] == 2, f"expected 2 embedding calls (different augment = different key), got {calls['count']}"
