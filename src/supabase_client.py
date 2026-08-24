import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from supabase import create_client, Client

logger = logging.getLogger("SpikedMeetingAgent")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://api.spiked.ai")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_KEY_TRANSCRIPT")

_supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("[Supabase] Initialized client successfully")
    except Exception as e:
        logger.warning(f"[Supabase] Could not initialize client: {e}")

# In-Memory Keyword & Config Cache (10-minute TTL)
_KEYWORD_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 600

# Default foundational terms for SpikedAI ecosystem
DEFAULT_CORE_KEYWORDS: List[str] = [
    "Tom",
    "Spiked",
    "SpikedAI",
    "Spiked AI",
    "LiveAvatar",
    "Handsfree",
    "Recall",
    "Recall.ai",
    "Deepgram",
    "Nova-3",
    "Nova-2",
    "Gemini",
    "SLA",
    "SLAs",
    "CRM",
    "RAG",
    "Supabase",
    "LiveKit",
    "WebRTC"
]

async def get_user_keywords_and_products(
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    auth_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves and caches domain keywords, product names, and company profile from Supabase.
    Returns:
        {
            "keywords": List[str],
            "company_name": str,
            "products_services": str,
            "bot_name": str,
            "product_domain": str
        }
    """
    cache_key = f"{user_id or 'anon'}:{client_id or 'none'}"
    now = time.time()
    
    if cache_key in _KEYWORD_CACHE:
        cached = _KEYWORD_CACHE[cache_key]
        if now - cached.get("cached_at", 0) < _CACHE_TTL_SECONDS:
            return cached["data"]

    keywords_set: Set[str] = set(DEFAULT_CORE_KEYWORDS)
    company_name = "SpikedAI"
    products_services = ""
    bot_name = "Tom"
    product_domain = "Enterprise Sales & AI Meeting Automation"

    client_to_use = _supabase_client
    if not client_to_use and SUPABASE_URL and auth_token:
        try:
            client_to_use = create_client(SUPABASE_URL, auth_token)
        except Exception:
            pass

    if client_to_use and user_id and user_id != "unknown_user":
        try:
            logger.info(f"[Supabase] Querying user_configs for user_id={user_id}")
            # 1. Fetch user_configs row
            res = await asyncio.to_thread(
                lambda: client_to_use.table("user_configs")
                .select("bot_name, seller_company, products_services, product_domain, strategic_keywords, custom_prompt")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            
            if res.data and len(res.data) > 0:
                cfg = res.data[0]
                bot_name = (cfg.get("bot_name") or "Tom").strip()
                company_name = cfg.get("seller_company") or company_name
                products_services = cfg.get("products_services") or ""
                product_domain = cfg.get("product_domain") or product_domain
                logger.info(f"[Supabase] Loaded user_configs: seller_company='{company_name}', products='{products_services[:60]}...'")
                
                # Add strategic keywords
                strat_kw = cfg.get("strategic_keywords")
                if isinstance(strat_kw, list):
                    for kw in strat_kw:
                        if isinstance(kw, str) and kw.strip():
                            keywords_set.add(kw.strip())
                            
                # Extract words from products_services
                if products_services:
                    for line in products_services.replace(",", "\n").split("\n"):
                        clean_item = line.strip()
                        if clean_item and len(clean_item) < 40:
                            keywords_set.add(clean_item)
            else:
                logger.warning(f"[Supabase] No user_configs row found for user_id={user_id}")

            # 2. Fetch source documents titles
            sources_query = client_to_use.table("sources").select("filename, description").eq("user_id", user_id)
            if client_id:
                sources_query = sources_query.eq("client_id", client_id)
            sources_res = await asyncio.to_thread(lambda: sources_query.limit(25).execute())
            
            if sources_res.data:
                for src in sources_res.data:
                    fname = src.get("filename") or ""
                    clean_fname = os.path.splitext(fname)[0].replace("_", " ").replace("-", " ")
                    if clean_fname and len(clean_fname) < 40:
                        keywords_set.add(clean_fname)

        except Exception as e:
            logger.warning(f"[Supabase] Error querying user keywords: {e}")

    # Deduplicate and sort keywords
    all_keywords = sorted(list(keywords_set), key=lambda x: -len(x))
    
    result_data = {
        "keywords": all_keywords,
        "company_name": company_name,
        "products_services": products_services,
        "bot_name": bot_name,
        "product_domain": product_domain
    }
    
    _KEYWORD_CACHE[cache_key] = {
        "cached_at": now,
        "data": result_data
    }
    
    logger.info(f"[Supabase] Loaded {len(all_keywords)} boosted keywords for user={user_id}")
    return result_data


async def get_completed_source_ids(
    user_id: Optional[str],
    client_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> List[str]:
    """Resolve completed source IDs directly from Supabase.

    The standalone warm-RAG service intentionally exposes only the answer and
    warm routes; it does not carry the legacy backend's ``/documents`` and
    ``/websites`` endpoints. Keeping this lookup here lets the orchestrator
    pass client-scoped source IDs to either backend without depending on those
    legacy routes.
    """
    if not user_id or user_id == "unknown_user":
        return []

    client_to_use = _supabase_client
    if not client_to_use and SUPABASE_URL and auth_token:
        try:
            client_to_use = create_client(SUPABASE_URL, auth_token)
        except Exception:
            client_to_use = None
    if not client_to_use:
        return []

    try:
        query = (
            client_to_use.table("sources")
            .select("id")
            .eq("user_id", user_id)
            .eq("ingestion_status", "COMPLETED")
        )
        if client_id:
            query = query.eq("client_id", client_id)
        response = await asyncio.to_thread(query.execute)
        return [
            str(row["id"])
            for row in (response.data or [])
            if isinstance(row, dict) and row.get("id")
        ]
    except Exception as exc:
        logger.info("[Supabase] Source ID lookup unavailable: %s", exc)
        return []


# Per-client provider overrides. Cached alongside the keyword config because it
# changes on the same timescale (rarely) and is read on the /start path.
_PROVIDER_CACHE: Dict[str, Dict[str, Any]] = {}
PROVIDER_TABLE = os.getenv("SUPABASE_PROVIDER_TABLE", "client_provider_configs")

MEMORY_TABLE = os.getenv("SUPABASE_AGENT_MEMORY_TABLE", "agent_memory")


def _is_active_memory_row(row: Dict[str, Any]) -> bool:
    """Fail closed for malformed expiry values; never hydrate expired memory."""
    expires_at = row.get("expires_at")
    if not expires_at:
        return True
    try:
        value = expires_at if isinstance(expires_at, datetime) else datetime.fromisoformat(
            str(expires_at).replace("Z", "+00:00")
        )
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value > datetime.now(timezone.utc)
    except (TypeError, ValueError):
        logger.warning("[Memory] Ignoring row with malformed expires_at")
        return False


async def load_agent_memory(
    user_id: Optional[str],
    client_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load durable, explicitly scoped agent memory for a meeting.

    The table is optional during rollout. A missing table or unavailable
    Supabase connection returns an empty list, leaving the live meeting
    functional with working and ephemeral memory only.
    """
    if not user_id or user_id == "unknown_user":
        return []
    client_to_use = _supabase_client
    if not client_to_use and SUPABASE_URL and auth_token:
        try:
            client_to_use = create_client(SUPABASE_URL, auth_token)
        except Exception:
            client_to_use = None
    if not client_to_use:
        return []

    try:
        query = (
            client_to_use.table(MEMORY_TABLE)
            .select("memory_type, memory_key, memory_value, confidence, expires_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(100)
        )
        if client_id:
            query = query.or_(f"client_id.eq.,client_id.eq.{client_id}")
        else:
            query = query.eq("client_id", "")
        response = await asyncio.to_thread(query.execute)
        return [
            row for row in (response.data or [])
            if isinstance(row, dict) and _is_active_memory_row(row)
        ]
    except Exception as exc:
        logger.info("[Memory] Persistent memory unavailable: %s", exc)
        return []


async def save_agent_memory(
    user_id: Optional[str],
    memory_type: str,
    memory_key: str,
    memory_value: Any,
    client_id: Optional[str] = None,
    confidence: float = 0.9,
    source: str = "meeting_instruction",
) -> bool:
    """Upsert one durable memory item; failures never affect speech."""
    if not user_id or user_id == "unknown_user" or not memory_key.strip():
        return False
    client_to_use = _supabase_client
    if not client_to_use:
        return False
    row = {
        "user_id": user_id,
        "client_id": client_id or "",
        "memory_type": memory_type,
        "memory_key": memory_key.strip(),
        "memory_value": memory_value,
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "source": source,
    }
    try:
        await asyncio.to_thread(
            lambda: client_to_use.table(MEMORY_TABLE)
            .upsert(row, on_conflict="user_id,client_id,memory_type,memory_key")
            .execute()
        )
        return True
    except Exception as exc:
        logger.info("[Memory] Persistent write unavailable: %s", exc)
        return False


async def get_client_providers(client_id: Optional[str]) -> Dict[str, Optional[str]]:
    """Return {video_provider, tts_provider, answer_engine} for a client.

    Missing keys mean "no override" -- the caller falls back to whatever the
    request or the env default says. Never raises: a provider lookup failing
    must not stop a meeting from starting.
    """
    empty: Dict[str, Optional[str]] = {}
    if not client_id or not _supabase_client:
        return empty

    now = time.time()
    cached = _PROVIDER_CACHE.get(client_id)
    if cached and now - cached.get("cached_at", 0) < _CACHE_TTL_SECONDS:
        return cached["data"]

    try:
        res = await asyncio.to_thread(
            lambda: _supabase_client.table(PROVIDER_TABLE)
            .select("video_provider,tts_provider,answer_engine")
            .eq("client_id", client_id)
            .limit(1)
            .execute()
        )
        row = (res.data or [None])[0] or {}
        data = {k: v for k, v in row.items() if v}
    except Exception as exc:
        # An absent table is the normal case until someone configures one.
        logger.info("[Supabase] No provider overrides for client_id=%s (%s)", client_id, exc)
        data = empty

    _PROVIDER_CACHE[client_id] = {"cached_at": now, "data": data}
    return data
