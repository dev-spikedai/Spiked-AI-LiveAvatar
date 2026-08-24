import os
import asyncio
import json
import logging
import re
from typing import List, AsyncGenerator, Any, cast, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from openai._exceptions import APIError, RateLimitError, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam
import torch
from fastapi import HTTPException
from google import genai

from core.config import get_g_vars, GROQ_API_URL, GROQ_API_KEY, GROQ_MODEL, OPENAI_CHAT_MODEL

logger = logging.getLogger(__name__)

# --- Production-Grade Configuration ---
STREAMING_TIMEOUT = 30.0
NON_STREAMING_TIMEOUT = 30.0
CONNECT_TIMEOUT = 5.0
MAX_KEEPALIVE = 50
MAX_CONNECTIONS = 200

# --- Global Clients (Singleton Pattern with Thread Safety) ---
_clients_lock = asyncio.Lock()
_async_httpx_client: Optional[httpx.AsyncClient] = None
_openai_async_client: Optional[AsyncOpenAI] = None
_gemini_client: Optional[genai.Client] = None


async def get_httpx_client() -> httpx.AsyncClient:
    """Get or create global HTTP client with optimized connection pooling."""
    global _async_httpx_client
    async with _clients_lock:
        if _async_httpx_client is None:
            _async_httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=STREAMING_TIMEOUT,
                    connect=CONNECT_TIMEOUT,
                    read=STREAMING_TIMEOUT,
                    write=10.0,
                    pool=5.0
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=MAX_KEEPALIVE,
                    max_connections=MAX_CONNECTIONS,
                    keepalive_expiry=30.0
                ),
                http2=True,  # Enable HTTP/2 for better performance
                follow_redirects=True
            )
            logger.info("Initialized global HTTPX client with HTTP/2 and connection pooling")
    return _async_httpx_client


async def get_openai_async_client() -> AsyncOpenAI:
    """Get or create global OpenAI async client with retry logic."""
    global _openai_async_client
    async with _clients_lock:
        if _openai_async_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
            _openai_async_client = AsyncOpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(
                    timeout=NON_STREAMING_TIMEOUT,
                    connect=CONNECT_TIMEOUT
                ),
                max_retries=2  # Built-in retry logic
            )
            logger.info("Initialized global OpenAI async client with retry logic")
    return _openai_async_client

async def get_gemini_client() -> genai.Client:
    """Get or create global Gemini client."""
    global _gemini_client

    async with _clients_lock:
        if _gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set")

            _gemini_client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized")

    return _gemini_client

async def cleanup_clients():
    """Cleanup function for graceful shutdown of all API clients."""
    global _async_httpx_client, _openai_async_client
    
    if _async_httpx_client:
        await _async_httpx_client.aclose()
        _async_httpx_client = None
        logger.info("Closed HTTPX client")
    
    if _openai_async_client:
        await _openai_async_client.close()
        _openai_async_client = None
        logger.info("Closed OpenAI async client")
        
    logger.info("All API clients cleaned up successfully")


def average_pool(last_hidden_states, attention_mask):
    """Average pooling for embeddings."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


async def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using the configured model."""
    g_vars = get_g_vars()
    embedding_tokenizer = g_vars["embedding_tokenizer"]
    embedding_model = g_vars["embedding_model"]
    device = g_vars["device"]

    if not texts:
        return np.array([])
    
    prefixed_texts = [f"passage: {text.strip()}" for text in texts if text and text.strip()]
    if not prefixed_texts:
        return np.array([])
    
    batch_dict = embedding_tokenizer(
        prefixed_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    def compute_embeddings_sync(model, batch):
        with torch.no_grad():
            return model(**batch)
            
    outputs = await asyncio.to_thread(compute_embeddings_sync, embedding_model, batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype(np.float32)


async def match_help_documents(
    query: str,
    supabase,
    *,
    match_threshold: float = 0.75,
    match_count: int = 3,
):
    """
    Semantic vector search over help_documents using Supabase RPC.
    """
    embeddings = await get_embeddings([f"query: {query}"])
    if embeddings.size == 0:
        return []

    rpc_params = {
        "query_embedding": embeddings[0].tolist(),
        "match_threshold": match_threshold,
        "match_count": match_count,
    }

    try:
        result = supabase.rpc("match_help_documents", rpc_params).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"match_help_documents failed: {e}", exc_info=True)
        return []


async def stream_gemini_response(messages: List[dict]) -> AsyncGenerator[str, None]:
    """Fallback for the live streamed answer when Groq is rate-limited.

    Slower time-to-first-token than Groq for this call shape (see
    WARM_PATH_REPORT.md's TTFB comparison) so it stays a fallback, not the
    default -- but it keeps /ask/regular answering instead of erroring out
    during a Groq 429.
    """
    client = await get_gemini_client()

    system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), None)
    user_content = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")

    try:
        stream = await client.aio.models.generate_content_stream(
            model="gemini-3-flash-preview",
            contents=user_content,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.2,
                "max_output_tokens": 2000,
            },
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        logger.error(f"Gemini fallback stream failed: {e}", exc_info=True)
        yield "Error: Failed to get response."


async def stream_groq_response(messages: List[dict]) -> AsyncGenerator[str, None]:
    client = await get_httpx_client()

    yield ""
    await asyncio.sleep(0)

    try:
        async with client.stream(
            "POST",
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
                "Connection": "keep-alive",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "stream": True,
                "stream_options": {"include_usage": False},
            },
            timeout=STREAMING_TIMEOUT,
        ) as response:
            if response.status_code == 429:
                logger.warning("Groq rate-limited on live answer; falling back to Gemini")
                async for chunk in stream_gemini_response(messages):
                    yield chunk
                return

            response.raise_for_status()

            buffer = b""

            async for chunk in response.aiter_bytes():
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    if not line.startswith(b"data:"):
                        continue

                    payload = line[5:].strip()

                    if payload == b"[DONE]":
                        return

                    try:
                        data = json.loads(payload)
                        content = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if content:
                            yield content
                    except Exception:
                        continue

    except httpx.TimeoutException:
        logger.error("Groq request timeout")
        yield "Error: Request timed out."
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq HTTP error: {e.response.status_code}")
        yield "Error: Service unavailable."
    except Exception as e:
        logger.error(f"Groq stream failed: {e}", exc_info=True)
        yield "Error: Failed to get response."


async def call_groq_llm(prompt: str, system_prompt: Optional[str] = None, 
                       max_tokens: int = 2000, is_json: bool = False) -> str:
    """Optimized Groq non-streaming call with proper error handling."""
    client = await get_httpx_client()
    
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = await client.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=NON_STREAMING_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
        
    except httpx.TimeoutException:
        logger.error("Groq LLM call timeout")
        raise HTTPException(status_code=504, detail="The request to the AI model timed out.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq HTTP error: {e.response.status_code}")
        raise HTTPException(status_code=502, detail="The AI model service returned an error.")
    except Exception as e:
        logger.error(f"Groq LLM call failed: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"The AI model service returned an error: {e}")


def parse_json_from_llm_response(response: str) -> dict:
    """Extract a JSON object from an LLM response.

    LLMs occasionally wrap JSON in ```json fences, add a sentence of prose,
    use smart quotes ("curly" instead of straight), or leave a trailing comma.
    We try a series of normalisations before giving up.
    """
    def _try_load(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 1. Markdown fence — most common Gemini output
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
    candidate = json_match.group(1) if json_match else None

    # 2. Outer-brace slice — handles prose before/after JSON
    if candidate is None:
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = response[start:end + 1]
        else:
            candidate = response

    # Try as-is first
    parsed = _try_load(candidate)
    if parsed is not None:
        return parsed

    # 3. Repair common issues and retry: smart quotes, trailing commas
    repaired = (candidate
                .replace("“", '"').replace("”", '"')   # curly double quotes
                .replace("‘", "'").replace("’", "'"))  # curly single quotes
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)            # trailing commas
    parsed = _try_load(repaired)
    if parsed is not None:
        return parsed

    logger.error("Failed to parse JSON from LLM response. Response (first 500 chars): %s", response[:500])
    raise ValueError("The AI model returned a response that could not be parsed as JSON.")


async def call_openai_llm(prompt: str, system_prompt: Optional[str] = None, 
                         max_tokens: int = 2000, is_json: bool = False) -> str:
    """Optimized OpenAI LLM call using async client."""
    client = await get_openai_async_client()
    
    try:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        kwargs = {
            "model": "gpt-4o",
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": NON_STREAMING_TIMEOUT
        }
        
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        if content is None:
            logger.error("OpenAI LLM returned empty content.")
            raise HTTPException(status_code=502, detail="The AI model returned no content.")
        return content
        
    except APIConnectionError:
        logger.error("OpenAI LLM call timeout")
        raise HTTPException(status_code=504, detail="The request to the AI model timed out.")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"The AI model service returned an error: {e}")
    except Exception as e:
        logger.error(f"OpenAI LLM call failed: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"The AI model service returned an error: {e}")


async def stream_openai_response(
    messages: List[ChatCompletionMessageParam],
    model: str = "gpt-4o-mini"
) -> AsyncGenerator[str, None]:
    """Stream OpenAI response using async client (SDK v2 compatible)."""
    client = await get_openai_async_client()

    try:
        async with client.chat.completions.stream(
            model=model,
            messages=messages,
            timeout=STREAMING_TIMEOUT,
        ) as stream:
            async for event in stream:
                if event.type == "content.delta":
                    yield event.delta

    except APIConnectionError:
        logger.error("OpenAI stream timeout")
        yield "Error: Request timed out."

    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        yield "Error: Service unavailable."

    except Exception as e:
        logger.error(f"OpenAI stream failed: {e}", exc_info=True)
        yield "Error: Failed to get response."


# --- Map-Reduce Product Extraction ---

# Tightened extraction prompt: targets proper product/offering names,
# rejects generic verbs/features. Used by the map phase (per-chunk).
_MAP_EXTRACTION_PROMPT = """
You are an information extraction system specialized in identifying
concrete product and service NAMES from business documents.

Extract only items that are:
- Explicitly NAMED in the text (proper nouns, branded names, SKU-like identifiers,
  or formally titled offerings — e.g. "Cloud ERP Platform", "Acme Analytics Suite",
  "Premium Support Plan").
- Products, services, subscriptions, platforms, modules, APIs, or professional
  services that the author sells, offers, or provides.

Strict exclusions — do NOT extract:
- Generic features, capabilities, or verbs (e.g. "analytics", "reporting",
  "consulting", "integration", "monitoring", "data processing").
- Industry categories (e.g. "cloud software", "enterprise solutions").
- Company names alone, unless the company name is part of the offering name.
- Implied or inferred products — only what is NAMED.
- Summaries or descriptions — emit the name as it appears.

Preserve the exact wording used in the document.

Output format: Return ONLY a JSON array of strings. No markdown fences, no prose.
Each element is one product/service name. If none, return [].
""".strip()


def _chunk_text(text: str, chunk_size: int = 8000, overlap: int = 400) -> List[str]:
    """Split text into overlapping chunks for map-phase extraction.
    Overlap preserves product names that straddle chunk boundaries."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks


async def _extract_products_from_chunk(chunk: str) -> List[str]:
    """Single chunk -> list of product name strings. Tries Groq first, falls back to OpenAI on error."""
    prompt = f"Document excerpt:\n\n{chunk}"
    try:
        response = await call_groq_llm(
            prompt=prompt,
            system_prompt=_MAP_EXTRACTION_PROMPT,
            max_tokens=800,
            is_json=False,
        )
    except Exception as e:
        logger.warning(f"Groq map-phase failed, falling back to OpenAI: {e}")
        try:
            response = await call_openai_llm(
                prompt=prompt,
                system_prompt=_MAP_EXTRACTION_PROMPT,
                max_tokens=800,
            )
        except Exception as e2:
            logger.error(f"OpenAI map-phase fallback also failed: {e2}")
            return []

    try:
        # Try JSON array first; fall back to parse_json_from_llm_response for object wrappers.
        stripped = response.strip()
        if stripped.startswith("["):
            data = json.loads(stripped)
        else:
            data = parse_json_from_llm_response(response)
        items = data if isinstance(data, list) else data.get("products", data.get("topics", []))
        return [s.strip() for s in items if isinstance(s, str) and s.strip()]
    except Exception as e:
        logger.error(f"Failed to parse map-phase output: {e}. Raw: {response[:300]}")
        return []


async def extract_products_map(text: str, max_concurrency: int = 8) -> List[tuple[str, int]]:
    """MAP phase: chunk the doc, extract per-chunk in parallel, return
    (product_name, mention_count) tuples with case-insensitive dedupe within the doc.
    No truncation — full document is scanned."""
    if not text or not text.strip():
        return []

    chunks = _chunk_text(text)
    sem = asyncio.Semaphore(max_concurrency)

    async def _bounded(chunk: str) -> List[str]:
        async with sem:
            return await _extract_products_from_chunk(chunk)

    per_chunk = await asyncio.gather(*[_bounded(c) for c in chunks], return_exceptions=False)

    # Aggregate within doc: case-insensitive key -> (first-seen casing, count)
    agg: dict[str, tuple[str, int]] = {}
    for items in per_chunk:
        for raw in items:
            name = raw.strip()
            if not name:
                continue
            key = name.lower()
            if key in agg:
                display, count = agg[key]
                agg[key] = (display, count + 1)
            else:
                agg[key] = (name, 1)

    return [(display, count) for (display, count) in agg.values()]


async def canonicalize_and_rank_products(
    candidates: List[dict], top_n: int = 25
) -> List[str]:
    """REDUCE phase: merge variant spellings, rank by cross-doc mention signal, return top N.
    `candidates` is a list of {product_name, mention_count, source_id} rows from raw_product_extractions.

    Strategy:
    1. Pre-aggregate by case-insensitive product name -> {display, total_mentions, doc_count}.
    2. If <= top_n * 2 candidates, skip LLM canonicalization (not worth the cost/latency).
    3. Otherwise, send candidate list to LLM to merge variants (e.g. "Cloud ERP" + "Cloud ERP Module"
       -> canonical "Cloud ERP Module"), keeping doc_count as the primary ranking signal.
    4. Rank by: doc_count DESC, total_mentions DESC; return top N display names.
    """
    if not candidates:
        return []

    # Step 1: pre-aggregate by lowercase key
    by_key: dict[str, dict] = {}
    for row in candidates:
        name = (row.get("product_name") or "").strip()
        if not name:
            continue
        key = name.lower()
        mc = int(row.get("mention_count") or 1)
        sid = row.get("source_id")
        if key not in by_key:
            by_key[key] = {
                "display": name,
                "total_mentions": mc,
                "source_ids": {sid} if sid else set(),
            }
        else:
            entry = by_key[key]
            entry["total_mentions"] += mc
            if sid:
                entry["source_ids"].add(sid)

    aggregated = [
        {
            "display": v["display"],
            "total_mentions": v["total_mentions"],
            "doc_count": len(v["source_ids"]),
        }
        for v in by_key.values()
    ]

    # Step 2: if small candidate pool, skip LLM canonicalization
    if len(aggregated) <= max(top_n * 2, 20):
        aggregated.sort(key=lambda x: (-x["doc_count"], -x["total_mentions"], x["display"].lower()))
        return [a["display"] for a in aggregated[:top_n]]

    # Step 3: LLM-based variant merging on the full candidate list
    reduce_system_prompt = """
You are a product catalog deduplication system.

You will receive a JSON list of candidate product/service names extracted from
business documents, each with a mention count and doc_count (how many distinct
documents mention it).

Your task: merge obvious variant spellings of the same product into ONE canonical
name. For example:
  "Cloud ERP", "Cloud ERP Module", "Cloud ERP Platform" -> "Cloud ERP Platform"
  "Acme CRM v2", "Acme CRM" -> "Acme CRM"

Rules:
- Only merge when variants clearly refer to the SAME product. When in doubt, keep
  them separate.
- Do NOT merge distinct products from the same company (e.g. "Acme ERP" and
  "Acme CRM" are different products — keep both).
- Choose the most complete, specific, proper-cased canonical name.
- Sum the mention_count and doc_count across merged variants.

Output format: Return ONLY a JSON array. Each element:
  {"name": "<canonical name>", "doc_count": <int>, "total_mentions": <int>}

No markdown fences, no prose.
""".strip()

    payload = json.dumps(
        [
            {
                "name": a["display"],
                "doc_count": a["doc_count"],
                "total_mentions": a["total_mentions"],
            }
            for a in aggregated
        ]
    )

    try:
        response = await call_openai_llm(
            prompt=f"Candidates:\n{payload}",
            system_prompt=reduce_system_prompt,
            max_tokens=4000,
        )
        stripped = response.strip()
        if stripped.startswith("["):
            merged = json.loads(stripped)
        else:
            merged = parse_json_from_llm_response(response)
            if isinstance(merged, dict):
                merged = merged.get("products", [])
        if not isinstance(merged, list):
            raise ValueError("Canonicalizer did not return a list")
    except Exception as e:
        logger.error(f"Reduce canonicalization failed, falling back to raw aggregation: {e}")
        aggregated.sort(key=lambda x: (-x["doc_count"], -x["total_mentions"], x["display"].lower()))
        return [a["display"] for a in aggregated[:top_n]]

    # Step 4: rank merged list
    cleaned = []
    for m in merged:
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or "").strip()
        if not name:
            continue
        cleaned.append({
            "name": name,
            "doc_count": int(m.get("doc_count") or 0),
            "total_mentions": int(m.get("total_mentions") or 0),
        })

    cleaned.sort(key=lambda x: (-x["doc_count"], -x["total_mentions"], x["name"].lower()))
    return [c["name"] for c in cleaned[:top_n]]


async def extract_topics_from_text(text: str) -> List[str]:
    """Backwards-compatible wrapper: returns a flat list of product names (max 6 by convention).
    Internally delegates to the map phase. Kept so any other callers keep working.

    For the primary client-scoped product view, prefer extract_products_map +
    canonicalize_and_rank_products (the full map-reduce pipeline)."""
    pairs = await extract_products_map(text)
    pairs.sort(key=lambda x: -x[1])  # most-mentioned within doc first
    return [name for (name, _count) in pairs[:6]]
    
async def call_gemini_flash_llm(prompt: str, system_prompt: str | None = None) -> str:
    client = await get_gemini_client()

    try:
        # The new SDK uses a different call structure
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3-flash-preview",
            contents=prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 2000
            }
        )

        if not response or not response.text:
            raise HTTPException(status_code=502, detail="Gemini returned empty response")

        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini Flash call failed: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Gemini API error")