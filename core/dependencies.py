import asyncio
import logging
import time
import httpx
from fastapi import Request, Depends, HTTPException
from jose.exceptions import ExpiredSignatureError, JWTError

from core.config import (
    get_g_vars,
    DEFAULT_TEST_USER_ID,
    APP_ENV,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)
from core.auth import get_jwt_validator, get_token_cache
from models.schemas import SettingsModel

logger = logging.getLogger(__name__)

_MFA_ENROLLMENT_CACHE = {}
_MFA_ENROLLMENT_CACHE_TTL = 300


async def _user_has_verified_totp(user_id: str) -> bool:
    """
    Checks whether the user has a verified TOTP factor enrolled in Supabase Auth.
    Uses a short in-memory cache to avoid repeated admin API calls.
    """
    now = time.time()
    cached = _MFA_ENROLLMENT_CACHE.get(user_id)
    if cached and now - cached["ts"] < _MFA_ENROLLMENT_CACHE_TTL:
        return cached["has_totp"]

    # Skip MFA check in development to avoid slow admin API calls
    if APP_ENV == "development":
        return False

    # Fail open when admin credentials are not available
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
                headers={
                    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                    "apikey": SUPABASE_SERVICE_ROLE_KEY,
                },
            )
            response.raise_for_status()

        user_payload = response.json() or {}
        factors = user_payload.get("factors", [])
        has_totp = any(
            (factor.get("factor_type") == "totp" or factor.get("type") == "totp")
            and factor.get("status") == "verified"
            for factor in factors
        )

        _MFA_ENROLLMENT_CACHE[user_id] = {"has_totp": has_totp, "ts": now}
        return has_totp
    except Exception as e:
        logger.warning(f"Could not determine MFA enrollment for user {user_id}: {e}")
        return False

async def get_user_id_from_token(request: Request) -> str:
    """
    Validates the bearer token using JWT validation (no Supabase call needed).
    Supports both ES256 (ECC) and HS256 (legacy) tokens.
    Uses in-memory cache for previously validated tokens.
    Falls back to test user in development mode if authentication fails.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        if APP_ENV == "development":
            logger.warning("No Authorization header. Using default test user.")
            return DEFAULT_TEST_USER_ID
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme.")

        # Validate JWT locally and inspect current assurance level.
        jwt_validator = get_jwt_validator()
        payload = await jwt_validator.decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing user ID")

        aal = payload.get("aal", "aal1")

        # Optional rollout: only require MFA if this user already has a verified TOTP factor.
        if aal != "aal2":
            has_verified_totp = await _user_has_verified_totp(user_id)
            if has_verified_totp:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "MFA_REQUIRED",
                        "message": "MFA verification required for this session.",
                    },
                )
        
        # Cache the validated token
        token_cache = get_token_cache()
        token_cache.set(token, user_id)
        logger.debug(f"Token validated and cached for user: {user_id}")
        
        return user_id
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ExpiredSignatureError:
        # Expected user-state event (session past TTL) — don't pollute error logs.
        logger.info("Rejected expired token; client should refresh and retry.")
        if APP_ENV == "development":
            return DEFAULT_TEST_USER_ID
        # Keep detail as a string (frontend does `err.detail || fallback`); signal
        # the specific reason via a header so a fetch interceptor can refresh the session.
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please refresh.",
            headers={"X-Auth-Error": "TOKEN_EXPIRED"},
        )
    except JWTError as e:
        # Malformed / wrong-signature token — likely client bug, log at warning.
        logger.warning(f"Rejected invalid token: {e}")
        if APP_ENV == "development":
            return DEFAULT_TEST_USER_ID
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
            headers={"X-Auth-Error": "TOKEN_INVALID"},
        )
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        if APP_ENV == "development":
            logger.warning("Authentication failed. Falling back to default test user.")
            return DEFAULT_TEST_USER_ID
        raise HTTPException(status_code=401, detail="Could not validate credentials.")


async def get_user_id_from_token_legacy(request: Request) -> str:
    """
    LEGACY: Original Supabase-based authentication (makes API call every time).
    Kept for reference or fallback purposes.
    This should NOT be used in production due to high latency and rate limits.
    """
    auth_header = request.headers.get("Authorization")
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    if not auth_header:
        if APP_ENV == "development":
            logger.warning("No Authorization header. Using default test user.")
            return DEFAULT_TEST_USER_ID
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    try:
        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme.")

        max_retries = 3
        user_response = None
        
        for attempt in range(max_retries):
            try:
                user_response = await asyncio.to_thread(supabase.auth.get_user, token)
                break  
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Auth connection dropped (attempt {attempt+1}/{max_retries}). Retrying...")
                    await asyncio.sleep(0.2)
                    continue
                else:
                    raise e

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token.")
        
        return user_response.user.id
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        if APP_ENV == "development":
            logger.warning("Authentication failed. Falling back to default test user.")
            return DEFAULT_TEST_USER_ID
        raise HTTPException(status_code=401, detail="Could not validate credentials.")


from functools import lru_cache
import time

_SETTINGS_CACHE = {}
_SETTINGS_CACHE_TTL = 60  # seconds

async def get_current_user_settings(user_id: str = Depends(get_user_id_from_token)) -> SettingsModel:
    now = time.time()

    cached = _SETTINGS_CACHE.get(user_id)
    if cached and now - cached["ts"] < _SETTINGS_CACHE_TTL:
        return cached["data"]

    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    try:
        response = supabase.table("user_configs").select(
            "bot_name, selected_persona, custom_prompt, answer_styles, meeting_domains, "
            "strategic_keywords, executive_snapshot, "
            "seller_company, products_services, product_domain, "
            "client_company, seller_name, client_names, sub_domains, company_url, "
            "seller_linkedin_url, seller_job_profile, user_industry"
        ).eq("user_id", user_id).single().execute()

        settings = SettingsModel(**{
            "botName": response.data.get("bot_name", "SpikedAI"),
            "selectedPersona": response.data.get("selected_persona", "balanced"),
            "customPrompt": response.data.get("custom_prompt", ""),
            "selectedAnswerStyles": response.data.get("answer_styles", []),
            "meetingDomains": response.data.get("meeting_domains", []),
            "strategic_keywords": response.data.get("strategic_keywords", []),
            "executive_snapshot": response.data.get("executive_snapshot", ""),
            "seller_company": response.data.get("seller_company", ""),
            "products_services": response.data.get("products_services", ""),
            "product_domain": response.data.get("product_domain", ""),
            "client_company": response.data.get("client_company", ""),
            "seller_name": response.data.get("seller_name", ""),
            "client_names": response.data.get("client_names", ""),
            "sub_domains": response.data.get("sub_domains", ""),
            "company_url": response.data.get("company_url", ""),
            "seller_linkedin_url": response.data.get("seller_linkedin_url", ""),
            "seller_job_profile": response.data.get("seller_job_profile", ""),
            "user_industry": response.data.get("user_industry", "")
        })

        _SETTINGS_CACHE[user_id] = {"data": settings, "ts": now}
        return settings

    except Exception:
        return SettingsModel()
