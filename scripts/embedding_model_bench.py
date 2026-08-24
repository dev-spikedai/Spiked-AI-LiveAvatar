"""
Compares e5-large-v2 (current, core/config.py EMBEDDING_MODEL_NAME) against
smaller E5-family candidates on CPU: query-embedding latency (the dominant
per-request cost identified in build_rag_context -- every /ask/regular call
pays this before the warm/cold retrieval branch even runs) and retrieval
quality (top-k agreement against the large-v2 baseline).

No live Supabase access from this environment (no .env here), so this uses a
small hand-written "cherry-picked" corpus of realistic sales/product-doc
sentences instead of a live account's chunks -- same approach classify_bench.py
already uses in this scripts/ dir when live data isn't available. Swap
CORPUS/QUERIES for real cherry-picked doc text before trusting the quality
numbers for an actual demo account.

Usage:
    virtualenv/Scripts/python scripts/embedding_model_bench.py
"""
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODELS = [
    ("intfloat/e5-large-v2", "current"),
    ("intfloat/e5-base-v2", "candidate"),
    ("intfloat/e5-small-v2", "candidate"),
]

N_QUERY_RUNS = 8  # repeat each query embed this many times for a stable avg

# Cherry-picked, hand-written to mimic ingested sales-doc chunks (product
# descriptions, pricing, integrations, SLAs) -- the kind of content
# build_rag_context retrieves against. Replace with real chunk text pulled
# from a demo account for a trustworthy quality read.
CORPUS = [
    ("pricing.pdf", "SpikedAI's Growth plan is $499/month and includes up to 10 seats, "
                     "unlimited meeting minutes, and CRM sync with Salesforce and HubSpot."),
    ("pricing.pdf", "Enterprise pricing is custom and includes SSO, a dedicated success "
                     "manager, and a 99.9% uptime SLA with financial penalties for breach."),
    ("integrations.pdf", "SpikedAI integrates natively with Salesforce, HubSpot, and Zoom, "
                          "syncing meeting notes and action items directly into the CRM record."),
    ("integrations.pdf", "Slack integration posts a meeting summary to a configurable channel "
                          "within 60 seconds of the call ending, including next-step owners."),
    ("security.pdf", "All customer data is encrypted at rest with AES-256 and in transit with "
                      "TLS 1.3. SpikedAI is SOC 2 Type II certified as of this year."),
    ("security.pdf", "Data retention defaults to 90 days for meeting recordings; customers on "
                      "Enterprise can configure retention down to 30 days or up to 2 years."),
    ("product_overview.pdf", "Tom, SpikedAI's live meeting avatar, joins Zoom, Teams, and Meet "
                              "as a participant and answers grounded questions about the seller's "
                              "product only when explicitly addressed by name."),
    ("product_overview.pdf", "The RAG pipeline retrieves from the seller's uploaded knowledge base "
                              "-- pricing sheets, security docs, integration guides -- so Tom never "
                              "answers from general knowledge alone."),
    ("onboarding.pdf", "New accounts complete onboarding by uploading at least one knowledge "
                        "document and connecting a calendar; first meeting can be live within 10 minutes."),
    ("onboarding.pdf", "A dedicated Slack channel with the SpikedAI team is provisioned for every "
                        "Enterprise customer during the first 30 days post-signup."),
]

# Realistic handsfree-style questions, matching the flavor of scripts/benchmark.py's QUESTIONS.
QUERIES = [
    "what does the enterprise plan include",
    "how does the CRM integration work",
    "is our data encrypted",
    "what is Tom and how does it answer questions",
    "how long are meeting recordings kept",
    "how fast can we get started",
]

TOP_K = 3


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
    sims = corpus_emb @ query_emb
    idx = np.argpartition(-sims, k - 1)[:k]
    return list(idx[np.argsort(-sims[idx])])


def main():
    passages = [f"passage: {text}" for _, text in CORPUS]
    baseline_top: dict[str, list[int]] = {}

    print(f"Corpus: {len(CORPUS)} cherry-picked chunks, {len(QUERIES)} queries, "
          f"top-{TOP_K} retrieval per query.\n")

    rows = []
    top_hits_by_model: dict[str, list[list[int]]] = {}
    for model_name, tag in MODELS:
        print(f"--- {model_name} ({tag}) ---")
        tokenizer, model, load_s, n_params = load_model(model_name)
        dim = model.config.hidden_size
        print(f"  load time: {load_s:.2f}s  params: {n_params/1e6:.0f}M  dim: {dim}")

        t0 = time.perf_counter()
        corpus_emb = embed(tokenizer, model, passages)
        corpus_embed_s = time.perf_counter() - t0
        print(f"  corpus embed ({len(CORPUS)} chunks): {corpus_embed_s:.3f}s")

        query_latencies = []
        per_query_top = []
        for q in QUERIES:
            qtext = [f"query: {q}"]
            # Warm up once (first call sometimes pays extra allocator cost), then time.
            embed(tokenizer, model, qtext)
            times = []
            for _ in range(N_QUERY_RUNS):
                t0 = time.perf_counter()
                qemb = embed(tokenizer, model, qtext)[0]
                times.append(time.perf_counter() - t0)
            query_latencies.extend(times)
            per_query_top.append(top_k_indices(qemb, corpus_emb, TOP_K))

        avg_ms = 1000 * sum(query_latencies) / len(query_latencies)
        p95_ms = 1000 * sorted(query_latencies)[int(0.95 * len(query_latencies)) - 1]
        print(f"  query embed latency: avg {avg_ms:.1f}ms  p95 {p95_ms:.1f}ms  "
              f"(n={len(query_latencies)} across {len(QUERIES)} queries x {N_QUERY_RUNS} runs)")

        if tag == "current":
            baseline_top = dict(zip(QUERIES, per_query_top))
            overlap_note = "baseline"
        else:
            overlaps = []
            for q, top in zip(QUERIES, per_query_top):
                base = set(baseline_top.get(q, []))
                overlaps.append(len(base & set(top)) / TOP_K)
            overlap_note = f"{100*sum(overlaps)/len(overlaps):.0f}% avg top-{TOP_K} overlap vs large-v2"
        print(f"  retrieval agreement: {overlap_note}\n")

        rows.append((model_name, tag, n_params, dim, load_s, avg_ms, p95_ms, overlap_note))
        top_hits_by_model[model_name] = per_query_top

    print("\n=== Summary ===")
    print(f"{'model':<24}{'tag':<11}{'params':>8}{'dim':>6}{'load_s':>8}{'avg_ms':>9}{'p95_ms':>9}   agreement")
    for name, tag, n_params, dim, load_s, avg_ms, p95_ms, note in rows:
        print(f"{name:<24}{tag:<11}{n_params/1e6:>6.0f}M{dim:>6}{load_s:>8.2f}{avg_ms:>9.1f}{p95_ms:>9.1f}   {note}")

    print("\nPer-query top hits (for manual eyeballing of quality, not just overlap %):")
    for qi, q in enumerate(QUERIES):
        print(f"\n  Q: {q}")
        for model_name, tag in MODELS:
            top = top_hits_by_model[model_name][qi]
            hits = ", ".join(CORPUS[i][0] for i in top)
            print(f"    [{tag:<9}] {model_name:<24} -> {hits}")


if __name__ == "__main__":
    main()
