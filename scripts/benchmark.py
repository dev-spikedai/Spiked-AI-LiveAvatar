"""
Compares /ask/regular latency between this local warm-index service and prod
backend-one, for the same user/questions. Not a load test -- one request at a
time, meant to produce numbers directly comparable to the Tom Latency
Benchmark artifact (RAG ttfb was 1.39-3.78s avg 2.89s there).

Usage:
    SPIKED_JWT=<bearer token> python scripts/benchmark.py
    -- or put SPIKED_JWT=<token> in this directory's .env, now loaded below.

Get the JWT from browser devtools: Network tab, any /ask/regular request,
copy the "Authorization: Bearer ..." header value (just the token part).
"""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

LOCAL_URL = "http://127.0.0.1:8000"
PROD_URL = "https://spikedai-production-application-409019309412.us-central1.run.app"

# From the logs/curl already seen in this session -- edit if testing a different account.
CLIENT_ID = "4731fe24-1b6d-49c4-b226-ffd92b2dbc61"
KYC_ID = "00000000-0000-0000-0000-000000000000"
# Required: build_rag_context fail-closed semantics mean client_id present +
# source_ids empty/missing returns None (the "no docs" branch) unconditionally
# -- that's what happened the first run (bytes=36 on every request = the
# fallback message, not a real answer).
SOURCE_IDS = ["be9fe236-7ecb-4f76-b1ad-b23cc5b13fe8"]

QUESTIONS = [
    "what is spikedai",
    "why use spikedai",
    "what is your integrations",
    "how does your integrations work",
]

# Groq's rate limit is shared between this service and prod (same key) --
# firing all 8 requests back to back trips it, and the resulting 429 falls
# back to a short "Error: Service unavailable." (27 bytes) that looks like a
# suspiciously fast "success" if you're not checking response size. Space
# requests out and flag anything implausibly short instead of averaging it in.
REQUEST_DELAY_SECONDS = 10
MIN_PLAUSIBLE_ANSWER_BYTES = 100


def time_request(base_url: str, question: str, token: str) -> dict:
    payload = {
        "question": question,
        "client_id": CLIENT_ID,
        "kyc_id": KYC_ID,
        "source_ids": SOURCE_IDS,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    t0 = time.perf_counter()
    ttfb = None
    total_bytes = 0
    with httpx.stream("POST", f"{base_url}/ask/regular", json=payload, headers=headers, timeout=60) as resp:
        status = resp.status_code
        for chunk in resp.iter_bytes():
            if ttfb is None:
                ttfb = time.perf_counter() - t0
            total_bytes += len(chunk)
    total = time.perf_counter() - t0

    return {"status": status, "ttfb": ttfb, "total": total, "bytes": total_bytes}


def main():
    token = os.getenv("SPIKED_JWT")
    if not token:
        print("Set SPIKED_JWT (see docstring for how to grab one).", file=sys.stderr)
        sys.exit(1)

    targets = [("local (warm-index)", LOCAL_URL), ("prod (backend-one)", PROD_URL)]

    rows = []
    for label, base_url in targets:
        for q in QUESTIONS:
            time.sleep(REQUEST_DELAY_SECONDS)
            try:
                r = time_request(base_url, q, token)
            except Exception as e:
                rows.append((label, q, None, None, False, f"ERROR: {e}"))
                continue
            plausible = r["status"] == 200 and r["bytes"] >= MIN_PLAUSIBLE_ANSWER_BYTES
            flag = "" if plausible else "  <-- SUSPECT (too short, likely an error body)"
            rows.append((label, q, r["ttfb"], r["total"], plausible,
                         f"status={r['status']} bytes={r['bytes']}{flag}"))

    print(f"\n{'target':<22} {'question':<32} {'ttfb':>8} {'total':>8}   note")
    print("-" * 90)
    for label, q, ttfb, total, plausible, note in rows:
        ttfb_s = f"{ttfb:.3f}s" if ttfb is not None else "  -   "
        total_s = f"{total:.3f}s" if total is not None else "  -   "
        print(f"{label:<22} {q[:30]:<32} {ttfb_s:>8} {total_s:>8}   {note}")

    for label, _ in targets:
        ttfbs = [t for l, _, t, _, p, _ in rows if l == label and p and t is not None]
        excluded = sum(1 for l, _, t, _, p, _ in rows if l == label and t is not None and not p)
        if ttfbs:
            avg = sum(ttfbs) / len(ttfbs)
            note = f" ({excluded} excluded as suspect)" if excluded else ""
            print(f"\n{label}: avg ttfb over {len(ttfbs)} plausible requests = {avg:.3f}s{note}")
        elif excluded:
            print(f"\n{label}: all {excluded} requests were suspect -- no plausible average")


if __name__ == "__main__":
    main()
