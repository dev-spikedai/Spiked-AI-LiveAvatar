import asyncio
import time

import numpy as np

from services.warm_index import _WarmIndex


def test_loaded_empty_corpus_is_a_warm_empty_result():
    index = _WarmIndex()
    index._by_user["u"] = {
        "embeddings": np.zeros((0, 1), dtype=np.float32),
        "chunks": [],
        "index": None,
        "loaded_at": time.monotonic(),
    }

    result = index.search("u", np.array([1.0], dtype=np.float32))

    assert result == []


def test_concurrent_authenticated_warm_requests_share_one_load():
    async def scenario():
        index = _WarmIndex()
        calls = 0

        async def fake_load(user_id):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return 7

        index.load_user = fake_load
        results = await asyncio.gather(
            index.ensure_user_loaded("u"),
            index.ensure_user_loaded("u"),
        )
        return calls, results

    calls, results = asyncio.run(scenario())

    assert calls == 1
    assert results == [7, 7]
