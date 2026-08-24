"""
Database helper functions for KYC profile Supabase operations.
"""
import logging
import asyncio
import time
from typing import Optional, Dict, Any
from fastapi import HTTPException
from datetime import datetime

from core.config import get_g_vars

# Sentinel kyc_id for the per-client buyer context that is set manually (via the
# Client Context modal) rather than tied to a generated/selected KYC document.
# Lets a client carry a buyer name without any KYC selected. Must be a valid UUID
# because client_kyc_configs.kyc_id is `uuid NOT NULL`; the nil UUID is safe since
# the column has no foreign key to a real KYC row.
MANUAL_KYC_ID = "00000000-0000-0000-0000-000000000000"

logger = logging.getLogger(__name__)

# get_client_kyc_config is called on every /ask/* request that carries a
# client_id (routers/search.py's _resolve_effective_settings), ahead of
# retrieval -- an uncached hit here adds a full Supabase round trip to every
# such request's time-to-first-byte. KYC config changes far less often than
# per-request, so a longer TTL than the analogous _SETTINGS_CACHE
# (core/dependencies.py, 60s) is safe. Invalidated explicitly on write (see
# upsert_client_kyc_config/upsert_client_context below) so edits take effect
# immediately rather than waiting out the TTL.
_KYC_CONFIG_CACHE: Dict[tuple, Dict[str, Any]] = {}
_KYC_CONFIG_CACHE_TTL = 300  # seconds


async def save_profile_to_db(
    user_id: str,
    job_id: str,
    sections: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save the completed V2 profile sections to Supabase.

    Args:
        user_id: The authenticated user's ID
        job_id: The job ID from the profile generation
        sections: Dictionary containing all completed sections

    Returns:
        Dict with the saved record including id and timestamps

    Raises:
        Exception if save fails
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    try:
        # Filter to only include completed sections with content
        completed_sections = {}
        for section_key, section_data in sections.items():
            if isinstance(section_data, dict):
                status = section_data.get("status")
                content = section_data.get("content")
                if status == "completed" and content:
                    completed_sections[section_key] = {
                        "status": status,
                        "content": content,
                        # Include any other metadata from the section
                        **{k: v for k, v in section_data.items()
                           if k not in ["status", "content"]}
                    }

        # Prepare the data to insert
        record = {
            "user_id": user_id,
            "job_id": job_id,
            "job_data": completed_sections,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Insert into Supabase
        logger.info(f"Saving profile to database for job_id: {job_id}, user_id: {user_id}")

        response = await asyncio.to_thread(
            lambda: supabase.table("profiles_history").insert(record).execute()
        )

        if response.data:
            logger.info(f"Successfully saved profile. Record ID: {response.data[0].get('id')}")
            return response.data[0]
        else:
            raise Exception("No data returned from insert operation")

    except Exception as e:
        logger.error(f"Failed to save profile to database: {e}", exc_info=True)
        raise Exception(f"Database save failed: {str(e)}")


async def get_user_profiles(
    user_id: str,
    limit: int = 10,
    offset: int = 0
) -> list:
    """
    Retrieve user's profile history from Supabase.

    Args:
        user_id: The authenticated user's ID
        limit: Maximum number of records to return
        offset: Number of records to skip (for pagination)

    Returns:
        List of profile records
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    try:
        logger.info(f"Fetching profiles for user_id: {user_id}, limit: {limit}, offset: {offset}")

        response = await asyncio.to_thread(
            lambda: supabase.table("profiles_history")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .offset(offset)
            .execute()
        )

        return response.data

    except Exception as e:
        logger.error(f"Failed to fetch profiles from database: {e}", exc_info=True)
        raise Exception(f"Database fetch failed: {str(e)}")


async def get_profile_by_job_id(
    user_id: str,
    job_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific profile by job_id for a user.

    Args:
        user_id: The authenticated user's ID
        job_id: The job ID to lookup

    Returns:
        Profile record or None if not found
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    try:
        logger.info(f"Fetching profile for user_id: {user_id}, job_id: {job_id}")

        response = await asyncio.to_thread(
            lambda: supabase.table("profiles_history")
            .select("*")
            .eq("user_id", user_id)
            .eq("job_id", job_id)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]
        return None

    except Exception as e:
        logger.error(f"Failed to fetch profile from database: {e}", exc_info=True)
        raise Exception(f"Database fetch failed: {str(e)}")


async def upsert_user_config(
    user_id: str,
    seller_name: str,
    seller_company: str,
    seller_website: Optional[str],
    buyer_name: str,
    buyer_company: str
) -> Dict[str, Any]:
    """
    Upsert user configuration to Supabase user_configs table.
    Inserts if user_id doesn't exist, updates if it does.

    Args:
        user_id: The authenticated user's ID
        seller_name: Seller's name
        seller_company: Seller's company
        seller_website: Seller's website URL
        buyer_name: Buyer's name (saved to client_names)
        buyer_company: Buyer's company (saved to client_company)

    Returns:
        Dict with the upserted record

    Raises:
        Exception if upsert fails
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    try:
        # Prepare the data to upsert
        record = {
            "user_id": user_id,
            "seller_name": seller_name,
            "seller_company": seller_company,
            "company_url": seller_website,
            "client_names": buyer_name,
            "client_company": buyer_company,
            "updated_at": datetime.utcnow().isoformat()
        }

        logger.info(f"Upserting user_configs for user_id: {user_id}")

        # Use upsert to insert or update
        response = await asyncio.to_thread(
            lambda: supabase.table("user_configs")
            .upsert(record, on_conflict="user_id")
            .execute()
        )

        if response.data:
            logger.info(f"Successfully upserted user_configs. Record: {response.data[0].get('user_id')}")
            return response.data[0]
        else:
            raise Exception("No data returned from upsert operation")

    except Exception as e:
        logger.error(f"Failed to upsert user_configs: {e}", exc_info=True)
        raise Exception(f"User configs upsert failed: {str(e)}")


def load_plans_config() -> dict:
    import os
    import json
    try:
        # plans.json is at the root of the project, one level up from core/kyc_database.py
        plans_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plans.json")
        with open(plans_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading plans.json: {e}")
        # Default fallback config matching plans.json
        return {
            "plans": {
                "starter": {
                    "simulator_limit": 3,
                    "kyc_limit": 3,
                    "minutes_limit": 540
                },
                "business_pro": {
                    "simulator_limit": 15,
                    "kyc_limit": 15,
                    "minutes_limit": 1440
                }
            },
            "subscription_status_overrides": {
                "exempted": {
                    "unlimited": True
                }
            }
        }


async def check_and_increment_kyc_limit(user_id: str):
    """
    Checks if the user has reached their KYC generation limit based on plans.json and subscriptions.
    If not exceeded, increments the user's kyc_count in user_configs.
    Raises HTTPException (403 Forbidden) if the limit is exceeded.
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    
    if not supabase:
        logger.warning("Supabase client not initialized, skipping limit check")
        return

    # 1. Fetch user's subscription
    try:
        sub_response = await asyncio.to_thread(
            lambda: supabase.table("subscriptions").select("tier", "status").eq("user_id", user_id).execute()
        )
        subscriptions = sub_response.data if sub_response else []
    except Exception as e:
        logger.error(f"Error fetching subscription for user {user_id}: {e}")
        subscriptions = []

    plans_config = load_plans_config()

    # 2. Check for overrides (like exempted) or active tier
    user_status = None
    user_tier = None
    
    is_unlimited = False
    if subscriptions:
        # Check if any subscription has an override first (e.g., exempted)
        for sub in subscriptions:
            status_val = sub.get("status")
            if status_val:
                overrides = plans_config.get("subscription_status_overrides", {})
                if status_val in overrides and overrides[status_val].get("unlimited", False):
                    logger.info(f"User {user_id} has unlimited KYC access via subscription status: {status_val}")
                    is_unlimited = True
                    break
        
        if not is_unlimited:
            # Look for an active or trialing subscription to get the tier
            for sub in subscriptions:
                status_val = sub.get("status")
                if status_val in ["active", "trialing"]:
                    user_tier = sub.get("tier")
                    user_status = status_val
                    break
            
            # Fallback to the first one's tier if none is active/trialing
            if not user_tier and subscriptions:
                user_tier = subscriptions[0].get("tier")
                user_status = subscriptions[0].get("status")

    # 3. Determine KYC limit
    if is_unlimited:
        limit = None
    else:
        plans = plans_config.get("plans", {})
        if not user_tier or user_tier not in plans:
            user_tier = "starter"
            
        plan_details = plans.get(user_tier, {})
        limit = plan_details.get("kyc_limit", 3) # Fallback to 3 if missing
        logger.info(f"User {user_id} subscription tier: {user_tier} (status: {user_status}), KYC limit: {limit}")

    # 4. Fetch user's current kyc_count
    current_count = 0
    try:
        config_response = await asyncio.to_thread(
            lambda: supabase.table("user_configs").select("kyc_count").eq("user_id", user_id).single().execute()
        )
        if config_response and config_response.data:
            current_count = config_response.data.get("kyc_count", 0) or 0
    except Exception as e:
        logger.error(f"Error fetching kyc_count for user {user_id}: {e}")

    # 5. Check if limit is exceeded
    if limit is not None and current_count >= limit:
        raise HTTPException(
            status_code=403,
            detail=f"KYC profile generation limit exceeded ({limit} profiles). Please upgrade your tier to continue."
        )

    # 6. Increment kyc_count in user_configs
    new_count = current_count + 1
    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_configs")
            .update({"kyc_count": new_count, "updated_at": datetime.utcnow().isoformat()})
            .eq("user_id", user_id)
            .execute()
        )
        logger.info(f"Incremented KYC count for user {user_id} from {current_count} to {new_count}")
    except Exception as e:
        logger.error(f"Error incrementing kyc_count for user {user_id}: {e}")
async def upsert_client_kyc_config(
    user_id: str,
    client_id: str,
    kyc_id: str,
    job_id: Optional[str],
    seller_name: Optional[str],
    seller_company: Optional[str],
    company_url: Optional[str],
    client_names: Optional[str],
    client_company: Optional[str],
) -> Dict[str, Any]:
    """
    Upsert a per-(user, client, kyc) overlay row used by /ask/* personalization.
    Mirrors the KYC subset of user_configs but is addressable by (client_id, kyc_id),
    so multiple KYCs per client can coexist without overwriting each other.
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    record = {
        "user_id": user_id,
        "client_id": client_id,
        "kyc_id": kyc_id,
        "job_id": job_id,
        "seller_name": seller_name,
        "seller_company": seller_company,
        "company_url": company_url,
        "client_names": client_names,
        "client_company": client_company,
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("client_kyc_configs")
            .upsert(record, on_conflict="user_id,client_id,kyc_id")
            .execute()
        )
        if response.data:
            logger.info(
                f"Upserted client_kyc_configs user={user_id} client={client_id} kyc={kyc_id}"
            )
            _KYC_CONFIG_CACHE.pop((user_id, client_id, kyc_id), None)
            return response.data[0]
        raise Exception("No data returned from client_kyc_configs upsert")
    except Exception as e:
        logger.error(f"Failed to upsert client_kyc_configs: {e}", exc_info=True)
        raise


async def upsert_client_context(
    user_id: str,
    client_id: str,
    kyc_id: str,
    client_names: Optional[str],
    client_company: Optional[str],
) -> Dict[str, Any]:
    """
    Partial upsert of just the buyer name/company onto a per-(user, client, kyc)
    overlay row. Unlike upsert_client_kyc_config, this only writes the
    client_names/client_company columns, so editing the buyer name on an existing
    KYC row never clobbers its seller fields, executive snapshot, or keywords.
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        raise Exception("Supabase client not initialized")

    record = {
        "user_id": user_id,
        "client_id": client_id,
        "kyc_id": kyc_id,
        "client_names": client_names,
        "client_company": client_company,
        "updated_at": datetime.utcnow().isoformat(),
    }

    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("client_kyc_configs")
            .upsert(record, on_conflict="user_id,client_id,kyc_id")
            .execute()
        )
        if response.data:
            logger.info(
                f"Upserted client context user={user_id} client={client_id} kyc={kyc_id}"
            )
            _KYC_CONFIG_CACHE.pop((user_id, client_id, kyc_id), None)
            return response.data[0]
        raise Exception("No data returned from client context upsert")
    except Exception as e:
        logger.error(f"Failed to upsert client context: {e}", exc_info=True)
        raise


async def get_client_kyc_config(
    user_id: str,
    client_id: str,
    kyc_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch one client_kyc_configs row. Returns None if not found (caller falls
    back to user_configs to preserve legacy behavior).
    """
    cache_key = (user_id, client_id, kyc_id)
    now = time.time()
    cached = _KYC_CONFIG_CACHE.get(cache_key)
    if cached and now - cached["ts"] < _KYC_CONFIG_CACHE_TTL:
        return cached["data"]

    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    if not supabase:
        return None

    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("client_kyc_configs")
            .select(
                "seller_name, seller_company, client_company, client_names, "
                "company_url, products_services, product_domain, sub_domains, "
                "executive_snapshot, strategic_keywords"
            )
            .eq("user_id", user_id)
            .eq("client_id", client_id)
            .eq("kyc_id", kyc_id)
            .maybe_single()
            .execute()
        )
        data = response.data if response and response.data else None
        _KYC_CONFIG_CACHE[cache_key] = {"data": data, "ts": now}
        return data
    except Exception as e:
        logger.warning(f"client_kyc_configs lookup failed (falling back): {e}")
        return None


def load_plans_config() -> dict:
    import os
    import json
    try:
        # plans.json is at the root of the project, one level up from core/kyc_database.py
        plans_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plans.json")
        with open(plans_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading plans.json: {e}")
        # Default fallback config matching plans.json
        return {
            "plans": {
                "starter": {
                    "simulator_limit": 3,
                    "kyc_limit": 3,
                    "minutes_limit": 540
                },
                "pro": {
                    "simulator_limit": 15,
                    "kyc_limit": 15,
                    "minutes_limit": 1440
                }
            },
            "subscription_status_overrides": {
                "exempted": {
                    "unlimited": True
                }
            }
        }


async def check_and_increment_kyc_limit(user_id: str):
    """
    Checks if the user has reached their KYC generation limit based on plans.json and subscriptions.
    If not exceeded, increments the user's kyc_count in user_configs.
    Raises HTTPException (403 Forbidden) if the limit is exceeded.
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    
    if not supabase:
        logger.warning("Supabase client not initialized, skipping limit check")
        return

    # 1. Fetch user's subscription
    try:
        sub_response = await asyncio.to_thread(
            lambda: supabase.table("subscriptions").select("tier", "status").eq("user_id", user_id).execute()
        )
        subscriptions = sub_response.data if sub_response else []
    except Exception as e:
        logger.error(f"Error fetching subscription for user {user_id}: {e}")
        subscriptions = []

    plans_config = load_plans_config()

    # 2. Check for overrides (like exempted) or active tier
    user_status = None
    user_tier = None
    
    is_unlimited = False
    if subscriptions:
        for sub in subscriptions:
            status_val = sub.get("status")
            if status_val:
                overrides = plans_config.get("subscription_status_overrides", {})
                if status_val in overrides and overrides[status_val].get("unlimited", False):
                    logger.info(f"User {user_id} has unlimited KYC access via subscription status: {status_val}")
                    is_unlimited = True
                    break
        
        if not is_unlimited:
            for sub in subscriptions:
                status_val = sub.get("status")
                if status_val in ["active", "trialing"]:
                    user_tier = sub.get("tier")
                    user_status = status_val
                    break
            
            if not user_tier and subscriptions:
                user_tier = subscriptions[0].get("tier")
                user_status = subscriptions[0].get("status")

    # 3. Determine KYC limit
    if is_unlimited:
        limit = None
    else:
        plans = plans_config.get("plans", {})
        if not user_tier or user_tier not in plans:
            user_tier = "starter"
            
        plan_details = plans.get(user_tier, {})
        limit = plan_details.get("kyc_limit", 3)
        logger.info(f"User {user_id} subscription tier: {user_tier} (status: {user_status}), KYC limit: {limit}")

    # 4. Fetch user's current kyc_count
    current_count = 0
    try:
        config_response = await asyncio.to_thread(
            lambda: supabase.table("user_configs").select("kyc_count").eq("user_id", user_id).single().execute()
        )
        if config_response and config_response.data:
            current_count = config_response.data.get("kyc_count", 0) or 0
    except Exception as e:
        logger.error(f"Error fetching kyc_count for user {user_id}: {e}")

    # 5. Check if limit is exceeded
    if limit is not None and current_count >= limit:
        raise HTTPException(
            status_code=403,
            detail=f"KYC profile generation limit exceeded ({limit} profiles). Please upgrade your tier to continue."
        )

    # 6. Increment kyc_count in user_configs
    new_count = current_count + 1
    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_configs")
            .update({"kyc_count": new_count, "updated_at": datetime.utcnow().isoformat()})
            .eq("user_id", user_id)
            .execute()
        )
        logger.info(f"Incremented KYC count for user {user_id} from {current_count} to {new_count}")
    except Exception as e:
        logger.error(f"Error incrementing kyc_count for user {user_id}: {e}")


async def check_user_simulator_limit(user_id: str, mode: str):
    """
    Checks if the user has reached their simulator limit (roleplay/focus sessions).
    Raises HTTPException (403 Forbidden) if the limit is exceeded.
    """
    if mode not in ["roleplay", "focus"]:
        return

    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    
    if not supabase:
        logger.warning("Supabase client not initialized, skipping limit check")
        return

    # 1. Fetch user's subscription
    try:
        sub_response = await asyncio.to_thread(
            lambda: supabase.table("subscriptions").select("tier", "status").eq("user_id", user_id).execute()
        )
        subscriptions = sub_response.data if sub_response else []
    except Exception as e:
        logger.error(f"Error fetching subscription for user {user_id}: {e}")
        subscriptions = []

    plans_config = load_plans_config()

    # 2. Check for overrides (like exempted) or active tier
    user_status = None
    user_tier = None
    
    is_unlimited = False
    if subscriptions:
        for sub in subscriptions:
            status_val = sub.get("status")
            if status_val:
                overrides = plans_config.get("subscription_status_overrides", {})
                if status_val in overrides and overrides[status_val].get("unlimited", False):
                    logger.info(f"User {user_id} has unlimited simulator access via subscription status: {status_val}")
                    is_unlimited = True
                    break
        
        if not is_unlimited:
            for sub in subscriptions:
                status_val = sub.get("status")
                if status_val in ["active", "trialing"]:
                    user_tier = sub.get("tier")
                    user_status = status_val
                    break
            
            if not user_tier and subscriptions:
                user_tier = subscriptions[0].get("tier")
                user_status = subscriptions[0].get("status")

    # 3. Determine simulator limit
    if is_unlimited:
        limit = None
    else:
        plans = plans_config.get("plans", {})
        if not user_tier or user_tier not in plans:
            user_tier = "starter"
            
        plan_details = plans.get(user_tier, {})
        limit = plan_details.get("simulator_limit", 3)
        logger.info(f"User {user_id} subscription tier: {user_tier} (status: {user_status}), simulator limit: {limit}")

    # 4. Fetch user's current simulator count
    current_count = 0
    try:
        config_response = await asyncio.to_thread(
            lambda: supabase.table("user_configs").select("simulator").eq("user_id", user_id).single().execute()
        )
        if config_response and config_response.data:
            current_count = config_response.data.get("simulator", 0) or 0
    except Exception as e:
        logger.error(f"Error fetching simulator count for user {user_id}: {e}")

    # 5. Check if limit is exceeded
    if limit is not None and current_count >= limit:
        raise HTTPException(
            status_code=403,
            detail=f"Simulator session limit exceeded ({limit} sessions). Please upgrade your tier to continue."
        )

    # 6. Increment simulator count in user_configs
    new_count = current_count + 1
    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_configs")
            .update({"simulator": new_count, "updated_at": datetime.utcnow().isoformat()})
            .eq("user_id", user_id)
            .execute()
        )
        logger.info(f"Incremented simulator count for user {user_id} from {current_count} to {new_count}")
    except Exception as e:
        logger.error(f"Error incrementing simulator count for user {user_id}: {e}")


async def increment_proposal_count(user_id: str) -> None:
    """Record one Proposal Agent generation against the user's billing period.

    Called after a proposal is successfully generated — never before, so a
    failed generation is not charged for. Unlike check_user_simulator_limit
    this does NOT raise at the allowance: proposal generations beyond the plan
    limit are billed as overage ($1 each), not blocked. The billing service
    derives the overage from this counter, and the counter is zeroed each
    period by its invoice.payment_succeeded handler.

    Best effort by design: a counter write must never fail a generation the
    user has already received. Failures are logged and swallowed.

    If the current value cannot be read we skip the increment entirely rather
    than writing 1 — deliberately unlike check_user_simulator_limit, which on a
    read failure falls through and writes 1, silently resetting a user who was
    at 12. Losing one billable unit beats corrupting the period's counter.

    Known limitation: read-modify-write, so two generations racing for the
    same user can under-count by one. This matches the existing meeting/KYC/
    simulator counters; an atomic Postgres increment would fix all four
    together and is tracked as a follow-up rather than diverging here.
    """
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not supabase:
        logger.warning("Supabase client not initialized, skipping proposal count")
        return

    current_count = 0
    try:
        config_response = await asyncio.to_thread(
            lambda: supabase.table("user_configs").select("proposal").eq("user_id", user_id).single().execute()
        )
        if config_response and config_response.data:
            current_count = config_response.data.get("proposal", 0) or 0
    except Exception as e:
        logger.error(f"Error fetching proposal count for user {user_id}: {e}")
        return

    new_count = current_count + 1
    try:
        await asyncio.to_thread(
            lambda: supabase.table("user_configs")
            .update({"proposal": new_count, "updated_at": datetime.utcnow().isoformat()})
            .eq("user_id", user_id)
            .execute()
        )
        logger.info(f"Incremented proposal count for user {user_id} from {current_count} to {new_count}")
    except Exception as e:
        logger.error(f"Error incrementing proposal count for user {user_id}: {e}")
