import os
import asyncio
import json
import logging
import re
from typing import List, AsyncGenerator, Any, cast, Optional

import httpx
import numpy as np
import torch
from fastapi import HTTPException
from google import genai

from core.config import get_g_vars, GROQ_API_URL, GROQ_API_KEY, GROQ_MODEL

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
    global _async_httpx_client
    
    if _async_httpx_client:
        await _async_httpx_client.aclose()
        _async_httpx_client = None
        logger.info("Closed HTTPX client")

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
            model="gemini-3.5-flash-lite",
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