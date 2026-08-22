import os
import time
import asyncio
import logging
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
