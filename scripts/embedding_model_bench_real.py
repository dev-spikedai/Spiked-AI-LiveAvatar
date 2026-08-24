"""
Same comparison as embedding_model_bench.py (e5-large-v2 vs e5-base-v2 vs
e5-small-v2: query latency + retrieval agreement), but against a real
account's ingested chunks instead of a hand-written corpus.

Reuses the *already-stored* large-v2 embeddings from Supabase for the
baseline corpus side (no need to recompute -- that's what ingestion wrote),
and only computes fresh embeddings for the candidate models plus the query
side for all three models.

Usage:
    virtualenv/Scripts/python scripts/embedding_model_bench_real.py [user_id] [filename_filter]

Defaults to core.config.DEFAULT_TEST_USER_ID if no user_id is given, and no
filename filter (all completed chunks) if none is given. filename_filter is a
case-insensitive substring match against sources.filename, e.g. "spiked" to
scope down to only SpikedAI's own docs in a mixed test account.
"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from core.config import DEFAULT_TEST_USER_ID, get_g_vars, init_supabase_client

MODELS = [
    ("intfloat/e5-large-v2", "current"),
    ("intfloat/e5-base-v2", "candidate"),
    ("intfloat/e5-small-v2", "candidate"),
]

N_QUERY_RUNS = 8
TOP_K = 3
MAX_CHUNKS = 60

# Generic handsfree-style questions -- content-agnostic on purpose so this
# runs against whatever real account/docs you point it at without hand-tuning.
QUERIES = [
    "what does the pricing include",
    "how does the integration work",
    "is our data secure",
    "what does the product do",
    "how do we get started",
    "what is the support process",
]


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def fetch_real_chunks(user_id: str, filename_filter: str | None = None):
    init_supabase_client()
    supabase = get_g_vars()["supabase"]
    resp = (
        supabase.table("chunks")
        .select("id, source_id, content, embedding, sources!inner(user_id, filename, ingestion_status)")
        .eq("sources.ingestion_status", "COMPLETED")
        .eq("sources.user_id", user_id)
        .order("id")
        .limit(500)  # over-fetch, then filter by filename client-side below
        .execute()
    )
    rows = resp.data or []
    corpus, stored_embeddings = [], []
    needle = filename_filter.lower() if filename_filter else None
    for row in rows:
        filename = (row.get("sources") or {}).get("filename") or row["source_id"]
        if needle and needle not in filename.lower():
            continue
        emb = _parse_embedding(row.get("embedding"))
        if emb is None:
            continue
        corpus.append((filename, row["content"]))
        stored_embeddings.append(emb)
        if len(corpus) >= MAX_CHUNKS:
            break
    return corpus, np.asarray(stored_embeddings, dtype=np.float32)


def load_model(name: str):
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    return tokenizer, model, load_s, n_params


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(tokenizer, model, texts: list[str]) -> np.ndarray:
    batch = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    emb = average_pool(out.last_hidden_state, batch["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def top_k_indices(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> list[int]:
    k = min(k, len(corpus_emb))
    sims = corpus_emb @ query_emb
    idx = np.argpartition(-sims, k - 1)[:k]
    return list(idx[np.argsort(-sims[idx])])


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST_USER_ID
    filename_filter = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Fetching real completed chunks for user_id={user_id}"
          f"{f', filename contains {filename_filter!r}' if filename_filter else ''} ...")
    corpus, stored_large_embeddings = fetch_real_chunks(user_id, filename_filter)
    if not corpus:
        print("No completed chunks matched. Pass a different user_id/filter, "
              "or check APP_ENV/.env point at the right project.")
        sys.exit(1)
    print(f"Got {len(corpus)} real chunks from {len({f for f, _ in corpus})} source file(s).\n")

    passages = [f"passage: {text}" for _, text in corpus]
    baseline_top: dict[str, list[int]] = {}
    top_hits_by_model: dict[str, list[list[int]]] = {}
    rows = []

    for model_name, tag in MODELS:
        print(f"--- {model_name} ({tag}) ---")
        tokenizer, model, load_s, n_params = load_model(model_name)
        dim = model.config.hidden_size
        print(f"  load time: {load_s:.2f}s  params: {n_params/1e6:.0f}M  dim: {dim}")

        if tag == "current":
            # Reuse what ingestion already wrote instead of recomputing --
            # this is exactly what's live in Supabase today.
            corpus_emb = stored_large_embeddings
            corpus_embed_s = 0.0
            print(f"  corpus embed: reused {len(corpus)} stored embeddings from Supabase (0.000s)")
        else:
            t0 = time.perf_counter()
            corpus_emb = embed(tokenizer, model, passages)
            corpus_embed_s = time.perf_counter() - t0
            print(f"  corpus embed ({len(corpus)} chunks): {corpus_embed_s:.3f}s")

        query_latencies = []
        per_query_top = []
        for q in QUERIES:
            qtext = [f"query: {q}"]
            embed(tokenizer, model, qtext)  # warm-up
            times = []
            for _ in range(N_QUERY_RUNS):
                t0 = time.perf_counter()
                qemb = embed(tokenizer, model, qtext)[0]
                times.append(time.perf_counter() - t0)
            query_latencies.extend(times)
            per_query_top.append(top_k_indices(qemb, corpus_emb, TOP_K))

        avg_ms = 1000 * sum(query_latencies) / len(query_latencies)
        p95_ms = 1000 * sorted(query_latencies)[int(0.95 * len(query_latencies)) - 1]
        print(f"  query embed latency: avg {avg_ms:.1f}ms  p95 {p95_ms:.1f}ms")

        if tag == "current":
            baseline_top = dict(zip(QUERIES, per_query_top))
            overlap_note = "baseline"
        else:
            overlaps = []
            for q, top in zip(QUERIES, per_query_top):
                base = set(baseline_top.get(q, []))
                overlaps.append(len(base & set(top)) / max(1, min(TOP_K, len(corpus))))
            overlap_note = f"{100*sum(overlaps)/len(overlaps):.0f}% avg top-{TOP_K} overlap vs stored large-v2"
        print(f"  retrieval agreement: {overlap_note}\n")

        rows.append((model_name, tag, n_params, dim, load_s, corpus_embed_s, avg_ms, p95_ms, overlap_note))
        top_hits_by_model[model_name] = per_query_top

    print("\n=== Summary ===")
    print(f"{'model':<24}{'tag':<11}{'params':>8}{'dim':>6}{'load_s':>8}{'avg_ms':>9}{'p95_ms':>9}   agreement")
    for name, tag, n_params, dim, load_s, _, avg_ms, p95_ms, note in rows:
        print(f"{name:<24}{tag:<11}{n_params/1e6:>6.0f}M{dim:>6}{load_s:>8.2f}{avg_ms:>9.1f}{p95_ms:>9.1f}   {note}")

    print("\nPer-query top hits (filenames, for manual eyeballing):")
    for qi, q in enumerate(QUERIES):
        print(f"\n  Q: {q}")
        for model_name, tag in MODELS:
            top = top_hits_by_model[model_name][qi]
            hits = ", ".join(corpus[i][0] for i in top)
            print(f"    [{tag:<9}] {model_name:<24} -> {hits}")


if __name__ == "__main__":
    main()
