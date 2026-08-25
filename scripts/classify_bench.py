"""
Isolated Gemini-vs-Groq latency test for the orchestrator's turn
classify-and-draft-reply call (Spiked-AI-LiveAvatar/src/live_avatar.py
~line 710) -- NOT a live-answer RAG test, this is specifically sizing
whether swapping that call to Groq is worth doing.

Uses a synthetic prompt matching the real one's shape (company context,
conversation history, structured JSON output, ~300 output tokens) since we
don't have live meeting data to replay. Real prompt content will vary this
some, but the shape (input size, JSON structure, output length) is what
drives latency, not the specific words.

Usage:
    virtualenv/Scripts/python scripts/classify_bench.py
"""
import asyncio
import json
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

N_RUNS = 5

SYNTHETIC_PROMPT = """Decide whether Tom should respond, then classify and normalize this wake-name-matched meeting turn. If the turn also needs a spoken reply and isn't a company_knowledge question, draft that reply in the same response.
Company: SpikedAI
Offerings: AI meeting copilot, CRM integrations, sales intelligence
Verified entity candidates: SpikedAI, Salesforce, HubSpot
Recent finalized conversation:
Sai: So we've been looking at a few vendors for this.
Tom: Happy to help however I can.
Sai: What sets you apart from the others we've seen?
Current speaker: Sai Bhuwan
Raw ASR: Hey Tom what are the CRM integrations you support

Set response_action to: respond, acknowledge, or silent.
Use company_knowledge for company/product/features/pricing/security/SLA/integration questions.
Use meeting_context for questions about what meeting participants said or discussed.
Use coaching when the rep asks Tom for help running the call itself.
Use social for greetings/identity questions. Use command for stop/wait/repeat.
Resolve pronouns in resolved_query. Reply fields (answer/bridge/next_question): leave empty if intent is company_knowledge, otherwise fill them for spoken delivery, answer at most 90 words, bridge at most one clause, next_question at most 20 words."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "response_action": {"type": "string", "enum": ["respond", "acknowledge", "silent"]},
        "intent": {"type": "string", "enum": ["company_knowledge", "meeting_context", "coaching", "social", "command"]},
        "resolved_query": {"type": "string"},
        "answer": {"type": "string"},
        "bridge": {"type": "string"},
        "next_question": {"type": "string"},
    },
    "required": ["response_action", "intent", "resolved_query"],
}


async def bench_gemini():
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        await client.aio.models.generate_content(
            model=model,
            contents=SYNTHETIC_PROMPT,
            config=types.GenerateContentConfig(
                max_output_tokens=320,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        times.append(time.perf_counter() - t0)
    return times


async def bench_groq(model: str):
    import httpx

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    times = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "Respond only with valid JSON matching this schema: " + json.dumps(RESPONSE_SCHEMA)},
                            {"role": "user", "content": SYNTHETIC_PROMPT},
                        ],
                        "max_tokens": 1024,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"  [{model}] ERROR: {e}")
                return None
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            if i == 0:
                print(f"  [{model} sample output] {content[:200]!r}")
    return times


def report(label, times):
    if times is None:
        print(f"{label}: SKIPPED (no API key)")
        return
    avg = sum(times) / len(times)
    print(f"{label}: {[f'{t:.2f}s' for t in times]} -> avg {avg:.2f}s")


async def main():
    print(f"Synthetic classify-and-draft-reply call, {N_RUNS} runs each\n")
    gemini_times = await bench_gemini()
    report("Gemini 3.5 Flash-Lite (current)", gemini_times)
    for model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/compound-mini"]:
        times = await bench_groq(model)
        report(f"Groq {model}", times)


if __name__ == "__main__":
    asyncio.run(main())
