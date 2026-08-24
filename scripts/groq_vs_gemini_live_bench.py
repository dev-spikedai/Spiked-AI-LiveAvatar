"""
Answers "why Groq over Gemini for the LIVE streamed answer" with real
numbers for the actual call shape ask_regular() makes: a persuasive grounded
sales answer, streamed, where time-to-first-token (TTFT) is what the user
actually perceives as latency -- not total completion time, since the
frontend renders tokens as they arrive.

Compares:
  - Groq openai/gpt-oss-120b (core.config.GROQ_MODEL, what /ask/regular
    actually uses today via stream_groq_response)
  - Gemini 3 Flash preview, streamed (same model call shape as
    call_gemini_flash_llm uses non-streamed for the cognitive background
    path -- here run via the streaming API for a fair TTFT comparison)

Uses a synthetic system prompt + context matching build_cognitive_prompt's
real shape/length (sales grounding rules + a few retrieved chunks) since we
don't want to import routers/search.py's full settings-dependent prompt
builder here.

Usage:
    virtualenv/Scripts/python scripts/groq_vs_gemini_live_bench.py
"""
import asyncio
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

N_RUNS = 3
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3-flash-preview"

SYSTEM_PROMPT = """You are SpikedAI -- the seller (SpikedAI) speaking directly to the buyer
(NovaTech) in a strategic sales conversation. Tone: natural, confident,
persuasive, concise -- never robotic or formal, never open with "Hello"/"Hi".

GROUNDING RULE: Answer only from the provided documents. If they don't
contain enough to answer, say so plainly and stop. Never invent facts,
features, pricing, or timelines not explicitly supported by the context."""

CONTEXT = """Source File: SPIKEDAI_PRODUCT_MASTER.pdf
Content: SpikedAI's Growth plan is $499/month and includes up to 10 seats,
unlimited meeting minutes, and CRM sync with Salesforce and HubSpot.
Enterprise pricing is custom and includes SSO, a dedicated success manager,
and a 99.9% uptime SLA. Tom, SpikedAI's live meeting avatar, joins Zoom,
Teams, and Meet as a participant and answers grounded questions about the
seller's product only when explicitly addressed by name. The RAG pipeline
retrieves from the seller's uploaded knowledge base so Tom never answers
from general knowledge alone. All customer data is encrypted at rest with
AES-256 and in transit with TLS 1.3. SpikedAI is SOC 2 Type II certified."""

QUESTION = "What does the enterprise plan include and how does the avatar make sure it doesn't make things up?"

USER_PROMPT = f"Context:\n{CONTEXT}\n\nQuestion: {QUESTION}"


async def bench_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            ttft = None
            n_chunks = 0
            async with client.stream(
                "POST", url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                    ],
                    "temperature": 0.2,
                    "stream": True,
                },
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    if ttft is None and chunk:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
            total = time.perf_counter() - t0
            results.append((ttft, total))
            print(f"  [groq run {i+1}] ttft={ttft:.3f}s total={total:.3f}s")
    return results


async def bench_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    from google import genai

    client = genai.Client(api_key=api_key)
    results = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        ttft = None
        async for chunk in await client.aio.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=USER_PROMPT,
            config={"system_instruction": SYSTEM_PROMPT, "temperature": 0.2, "max_output_tokens": 500},
        ):
            if ttft is None and (chunk.text or ""):
                ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
        results.append((ttft, total))
        print(f"  [gemini run {i+1}] ttft={ttft:.3f}s total={total:.3f}s")
    return results


def summarize(label, results):
    if not results:
        print(f"{label}: SKIPPED (no API key)")
        return
    ttfts = [r[0] for r in results if r[0] is not None]
    totals = [r[1] for r in results]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
    avg_total = sum(totals) / len(totals)
    print(f"{label}: avg ttft={avg_ttft:.3f}s  avg total={avg_total:.3f}s  (n={len(results)})")


async def main():
    print(f"Live-answer shape: system+context ~{len(SYSTEM_PROMPT)+len(CONTEXT)} chars, "
          f"streamed, {N_RUNS} runs each.\n")

    print("--- Groq (production model) ---")
    groq_results = await bench_groq()
    print("\n--- Gemini 3 Flash preview (streamed) ---")
    gemini_results = await bench_gemini()

    print("\n=== Summary ===")
    summarize(f"Groq {GROQ_MODEL}", groq_results)
    summarize(f"Gemini {GEMINI_MODEL}", gemini_results)


if __name__ == "__main__":
    asyncio.run(main())
