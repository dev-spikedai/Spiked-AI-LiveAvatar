"""
Verifies the C1 fix in services/warm_index.py: search() must serve a stale
(past WARM_INDEX_TTL_SECONDS) bucket's in-memory data instead of discarding
it and forcing the caller onto the slow RPC fallback path.
"""
import time

import numpy as np

from services.warm_index import _WarmIndex, _build_index


def _make_stale_bucket(loaded_at_offset_seconds: float):
    chunks = [
        {"source_id": "s1", "filename": "doc1.pdf", "content": "hello world"},
        {"source_id": "s2", "filename": "doc2.pdf", "content": "goodbye world"},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return {
        "embeddings": embeddings,
        "chunks": chunks,
        "index": _build_index(embeddings),
        "loaded_at": time.monotonic() - loaded_at_offset_seconds,
    }


def test_stale_bucket_still_serves_results_instead_of_none(monkeypatch):
    index = _WarmIndex()
    # WARM_INDEX_TTL_SECONDS defaults to 300; put this bucket well past that.
    index._by_user["u"] = _make_stale_bucket(loaded_at_offset_seconds=600)

    kicked_off = []
    monkeypatch.setattr(index, "kick_off_warm_load", lambda user_id: kicked_off.append(user_id))

    result = index.search("u", np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert result is not None, "stale bucket must still be searched, not discarded"
    assert len(result) == 2
    assert result[0]["source_id"] == "s1"  # closest match to [1.0, 0.0]
    # A background refresh must still be kicked off even though we served stale data.
    assert kicked_off == ["u"]


def test_fresh_bucket_does_not_trigger_a_reload(monkeypatch):
    index = _WarmIndex()
    index._by_user["u"] = _make_stale_bucket(loaded_at_offset_seconds=1)  # well within TTL

    kicked_off = []
    monkeypatch.setattr(index, "kick_off_warm_load", lambda user_id: kicked_off.append(user_id))

    result = index.search("u", np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert result is not None
    assert kicked_off == []  # no reload needed for fresh data
