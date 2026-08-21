"""
RAG Client for Live Avatar Bot.

Provides two modes:
  1. HTTP mode: Authenticates as a bot service user and calls /ask/handsfree
     (requires documents uploaded to the bot user's Supabase account).
  2. Direct mode: Uses the service-role key to query Supabase sources/chunks
     directly, builds context, and lets Gemini generate the answer.

Direct mode is the primary path — it works without needing a user JWT
and doesn't require the e5-large-v2 embedding model locally.
"""

import os
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any

import httpx

logger = logging.getLogger("LiveAvatar-RAG")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SPIKED_BACKEND_URL = os.getenv("SPIKED_BACKEND_URL", "").rstrip("/")
BOT_EMAIL = os.getenv("BOT_SERVICE_EMAIL", "liveavatar-bot@spiked.ai")
BOT_PASSWORD = os.getenv("BOT_SERVICE_PASSWORD", "SpikedLiveAvatar2024!")

_jwt_token: Optional[str] = None
_jwt_expiry: float = 0.0
_bot_user_id: Optional[str] = None

_SOURCE_CACHE: Dict[str, Dict[str, Any]] = {}
_SOURCE_CACHE_TTL = 300  # 5 minutes


async def _ensure_bot_user() -> str:
    """Create the bot service user in Supabase (idempotent), return user_id."""
    global _bot_user_id
    if _bot_user_id:
        return _bot_user_id

    headers_admin = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/admin/users",
            headers=headers_admin,
            json={
                "email": BOT_EMAIL,
                "password": BOT_PASSWORD,
                "email_confirm": True,
            },
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            _bot_user_id = data.get("id")
            logger.info("[RAG] Created bot user id=%s", _bot_user_id)
        elif resp.status_code == 400 and "already" in resp.text.lower():
            resp2 = await client.get(
                f"{SUPABASE_URL}/auth/v1/admin/users",
                headers=headers_admin,
                params={"email": BOT_EMAIL},
            )
            users = resp2.json().get("users", [])
            if users:
                _bot_user_id = users[0]["id"]
                logger.info("[RAG] Bot user already exists id=%s", _bot_user_id)
        else:
            logger.warning("[RAG] Bot user creation returned %s: %s", resp.status_code, resp.text[:200])

    return _bot_user_id or ""


async def get_jwt() -> str:
    """Sign in as the bot user and return a valid Supabase JWT (cached)."""
    global _jwt_token, _jwt_expiry
    now = time.time()
    if _jwt_token and now < _jwt_expiry - 60:
        return _jwt_token

    await _ensure_bot_user()

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json",
            },
            json={"email": BOT_EMAIL, "password": BOT_PASSWORD},
        )
        if resp.status_code == 200:
            data = resp.json()
            _jwt_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)
            _jwt_expiry = now + expires_in
            logger.info("[RAG] Signed in as bot user, token expires in %ds", expires_in)
            return _jwt_token or ""
        else:
            logger.error("[RAG] Bot sign-in failed %s: %s", resp.status_code, resp.text[:200])
            return ""


async def fetch_source_ids(client_id: str, user_id: Optional[str] = None) -> List[str]:
    """Fetch completed source IDs for a client from the sources table."""
    cache_key = f"{client_id}:{user_id or 'any'}"
    cached = _SOURCE_CACHE.get(cache_key)
    if cached and time.time() - cached["ts"] < _SOURCE_CACHE_TTL:
        return cached["ids"]

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    params: Dict[str, Any] = {
        "select": "id",
        "ingestion_status": "eq.COMPLETED",
    }
    if client_id:
        params["client_id"] = f"eq.{client_id}"
    if user_id:
        params["user_id"] = f"eq.{user_id}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/sources",
                headers=headers,
                params=params,
            )
            if resp.status_code == 200:
                rows = resp.json()
                ids = [r["id"] for r in rows if r.get("id")]
                _SOURCE_CACHE[cache_key] = {"ids": ids, "ts": time.time()}
                logger.info("[RAG] Fetched %d source_ids for client=%s", len(ids), client_id)
                return ids
            logger.warning("[RAG] sources query returned %s", resp.status_code)
    except Exception as e:
        logger.error("[RAG] Error fetching source_ids: %s", e)
    return []


async def _fetch_chunks_for_sources(source_ids: List[str], max_chunks: int = 20) -> List[Dict[str, str]]:
    """Fetch chunk content for the given source IDs via Supabase REST."""
    if not source_ids:
        return []

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }

    all_chunks: List[Dict[str, str]] = []
    batch_size = 10
    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i : i + batch_size]
        id_filter = ",".join(batch)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{SUPABASE_URL}/rest/v1/chunks",
                    headers=headers,
                    params={
                        "select": "source_id,content,sources!inner(filename)",
                        "source_id": f"in.({id_filter})",
                        "order": "created_at.asc",
                        "limit": max_chunks,
                    },
                )
                if resp.status_code == 200:
                    rows = resp.json()
                    for row in rows:
                        src = row.get("sources") or {}
                        all_chunks.append({
                            "filename": src.get("filename", "unknown"),
                            "content": row.get("content", ""),
                        })
        except Exception as e:
            logger.error("[RAG] Error fetching chunks batch: %s", e)

    logger.info("[RAG] Fetched %d chunks from %d sources", len(all_chunks), len(source_ids))
    return all_chunks


def _build_context_from_chunks(chunks: List[Dict[str, str]], max_chars: int = 8000) -> str:
    """Build a context string from chunks, respecting a character budget."""
    if not chunks:
        return ""
    parts: List[str] = []
    total = 0
    for ch in chunks:
        entry = f"Source: {ch['filename']}\n{ch['content']}\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)
    return "\n---\n".join(parts)


async def query_rag_direct(
    question: str,
    client_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Query RAG by fetching chunks directly from Supabase (no embedding model needed)."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.warning("[RAG] Supabase credentials not configured")
        return ""

    source_ids = await fetch_source_ids(client_id, user_id)
    if not source_ids:
        logger.info("[RAG] No source_ids for client=%s — empty KB", client_id)
        return ""

    chunks = await _fetch_chunks_for_sources(source_ids)
    if not chunks:
        logger.info("[RAG] No chunks found for source_ids")
        return ""

    context = _build_context_from_chunks(chunks)
    return context


async def query_rag_http(
    question: str,
    client_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
) -> str:
    """Query the SpikedAI /ask/handsfree endpoint with a valid bot JWT + source_ids."""
    token = await get_jwt()
    if not token:
        logger.warning("[RAG] No JWT — falling back to direct mode")
        return ""

    payload: Dict[str, Any] = {"question": question}
    if client_id:
        payload["client_id"] = client_id
    if source_ids:
        payload["source_ids"] = source_ids

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SPIKED_BACKEND_URL}/ask/handsfree",
                headers=headers,
                json=payload,
            )
            if resp.status_code != 200:
                logger.warning("[RAG] HTTP endpoint returned %s", resp.status_code)
                return ""

            full_text = ""
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    chunk = line.replace("data:", "").strip()
                    full_text += chunk + " "
                elif line:
                    full_text += line + " "

            result = full_text.strip()
            if "no relevant information" in result.lower():
                logger.info("[RAG] HTTP endpoint returned empty-kb marker")
                return ""
            return result
    except Exception as e:
        logger.error("[RAG] HTTP endpoint error: %s", e)
        return ""


async def query_rag(
    question: str,
    client_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Main RAG entry point. Tries HTTP endpoint first (with source_ids),
    then falls back to direct Supabase chunk retrieval.
    Returns context text for Gemini, or empty string if nothing found.
    """
    source_ids = await fetch_source_ids(client_id, user_id) if client_id else []

    if SPIKED_BACKEND_URL and SUPABASE_ANON_KEY:
        result = await query_rag_http(question, client_id, source_ids)
        if result:
            return result

    return await query_rag_direct(question, client_id, user_id)
