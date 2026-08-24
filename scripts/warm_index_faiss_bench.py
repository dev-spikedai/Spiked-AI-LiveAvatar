"""
Old numpy argpartition top-k (what search()/search_small() did before) vs the
FAISS IndexFlatIP replacement (services/warm_index.py), at realistic per-user
corpus sizes. Synthetic embeddings -- this isn't a retrieval-quality check
(the two implementations are numerically identical, see the parity check in
services/warm_index.py's own smoke test), only a latency comparison.

Usage:
    virtualenv/Scripts/python scripts/warm_index_faiss_bench.py
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from services.warm_index import _build_index, _search_index

DIM = 1024  # e5-large-v2's dimension, the production embedding model
CORPUS_SIZES = [100, 500, 2000, 8000]
N_QUERY_RUNS = 50
TOP_K = 10
FILTER_FRACTION = 0.2  # source_ids narrows to ~20% of the corpus


def _old_search(embeddings, chunks, query, source_ids, top_k):
    if source_ids:
        keep = set(source_ids)
        idx = [i for i, c in enumerate(chunks) if c["source_id"] in keep]
        if not idx:
            return []
        embeddings = embeddings[idx]
        chunks = [chunks[i] for i in idx]

    sims = embeddings @ query
    k = min(top_k, len(chunks))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(chunks[i]["content"], float(sims[i])) for i in top_idx]


def _time_runs(fn, n=N_QUERY_RUNS):
    # One untimed warm-up call so the first-call cost (allocation, cache
    # miss) doesn't skew the mean of what's actually a hot loop in prod.
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000  # ms/call


def main():
    rng = np.random.default_rng(0)
    print(f"{'corpus':>8} | {'old unfiltered':>15} | {'faiss unfiltered':>17} | "
          f"{'old filtered':>13} | {'faiss filtered':>15} | {'build':>8}")
    print("-" * 90)

    for n in CORPUS_SIZES:
        embeddings = rng.random((n, DIM), dtype=np.float32)
        n_sources = max(1, n // 20)
        chunks = [
            {"source_id": f"s{i % n_sources}", "filename": f"f{i}.pdf", "content": f"chunk {i}"}
            for i in range(n)
        ]
        query = rng.random(DIM, dtype=np.float32)
        filter_sources = [f"s{i}" for i in range(max(1, int(n_sources * FILTER_FRACTION)))]

        t_build0 = time.perf_counter()
        index = _build_index(embeddings)
        build_ms = (time.perf_counter() - t_build0) * 1000

        old_unfiltered_ms = _time_runs(lambda: _old_search(embeddings, chunks, query, None, TOP_K))
        faiss_unfiltered_ms = _time_runs(lambda: _search_index(index, chunks, query, None, TOP_K))
        old_filtered_ms = _time_runs(lambda: _old_search(embeddings, chunks, query, filter_sources, TOP_K))
        faiss_filtered_ms = _time_runs(lambda: _search_index(index, chunks, query, filter_sources, TOP_K))

        print(f"{n:>8} | {old_unfiltered_ms:>12.3f}ms | {faiss_unfiltered_ms:>14.3f}ms | "
              f"{old_filtered_ms:>10.3f}ms | {faiss_filtered_ms:>12.3f}ms | {build_ms:>6.2f}ms")


if __name__ == "__main__":
    main()
