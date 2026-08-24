"""
Verifies the D1 fix in routers/search.py: _grounding_fraction (via
_resolve_chunk_embeddings) must reuse precomputed chunk embeddings instead of
re-embedding chunk content that already has a vector from retrieval -- and
must NOT reuse a precomputed vector whose dimensionality doesn't match the
sentence embeddings' (the locked-down small-model demo path's chunks carry a
384-dim vector vs the normal 1024-dim space; reusing that would silently
corrupt the cosine comparison, not just be slower).
"""
import asyncio

import numpy as np

import routers.search as search_mod


def _fake_get_embeddings_counting(calls, dim=8):
    async def fake(texts):
        calls.append(list(texts))
        return np.ones((len(texts), dim), dtype=np.float32)
    return fake


def test_all_precomputed_embeddings_valid_skips_recompute(monkeypatch):
    calls = []
    monkeypatch.setattr(search_mod, "get_embeddings", _fake_get_embeddings_counting(calls, dim=8))

    chunk_texts = ["chunk one", "chunk two", "chunk three"]
    chunk_embeddings = [np.zeros(8, dtype=np.float32) for _ in chunk_texts]

    result = asyncio.run(search_mod._resolve_chunk_embeddings(chunk_texts, chunk_embeddings, expected_dim=8))

    assert result.shape == (3, 8)
    assert calls == [], f"expected zero get_embeddings calls (all precomputed valid), got {len(calls)} call(s)"


def test_missing_embeddings_are_recomputed_only_for_those_entries(monkeypatch):
    calls = []
    monkeypatch.setattr(search_mod, "get_embeddings", _fake_get_embeddings_counting(calls, dim=8))

    chunk_texts = ["has embedding", "missing embedding", "also has embedding"]
    chunk_embeddings = [np.zeros(8, dtype=np.float32), None, np.zeros(8, dtype=np.float32)]

    result = asyncio.run(search_mod._resolve_chunk_embeddings(chunk_texts, chunk_embeddings, expected_dim=8))

    assert result.shape == (3, 8)
    assert len(calls) == 1
    assert calls[0] == ["missing embedding"], f"expected only the missing chunk to be recomputed, got {calls[0]}"


def test_dimension_mismatch_is_treated_as_missing_not_reused(monkeypatch):
    """The critical safety property: a 384-dim (small-model) embedding must
    never be silently used where 1024-dim (large-model) sentence embeddings
    are expected -- that would corrupt the cosine similarity, not just skip
    an optimization."""
    calls = []
    monkeypatch.setattr(search_mod, "get_embeddings", _fake_get_embeddings_counting(calls, dim=8))

    chunk_texts = ["wrong dim chunk"]
    wrong_dim_embedding = np.zeros(4, dtype=np.float32)  # expected_dim will be 8

    result = asyncio.run(
        search_mod._resolve_chunk_embeddings(chunk_texts, [wrong_dim_embedding], expected_dim=8)
    )

    assert result.shape == (1, 8)
    assert len(calls) == 1 and calls[0] == ["wrong dim chunk"], "mismatched-dim embedding must be recomputed, not reused"


def test_no_precomputed_embeddings_falls_back_to_full_recompute(monkeypatch):
    calls = []
    monkeypatch.setattr(search_mod, "get_embeddings", _fake_get_embeddings_counting(calls, dim=8))

    chunk_texts = ["a", "b"]
    result = asyncio.run(search_mod._resolve_chunk_embeddings(chunk_texts, None, expected_dim=8))

    assert result.shape == (2, 8)
    assert len(calls) == 1 and calls[0] == ["a", "b"]


def test_warm_index_search_result_includes_matching_embedding():
    """services/warm_index.py's _search_index must attach each hit's own
    embedding (row i of the bucket's embeddings matrix lines up with
    chunks[i])."""
    from services.warm_index import _build_index, _search_index

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    chunks = [
        {"source_id": "s0", "filename": "f0", "content": "c0"},
        {"source_id": "s1", "filename": "f1", "content": "c1"},
        {"source_id": "s2", "filename": "f2", "content": "c2"},
    ]
    index = _build_index(embeddings)

    results = _search_index(index, chunks, embeddings, np.array([1.0, 0.0], dtype=np.float32), None, top_k=1)

    assert len(results) == 1
    assert results[0]["source_id"] == "s0"
    assert np.array_equal(results[0]["embedding"], embeddings[0])
