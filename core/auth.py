import logging
import time
from typing import Optional, Dict, Any
from jose import jwt, JWTError, jwk
from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)


class JWTValidator:
    """
    JWT validator that supports ECC (ES256) keys via JWKS.
    Fetches public keys from Supabase's JWKS endpoint and caches them.
    """
    
    def __init__(self, supabase_url: str, jwt_secret: Optional[str] = None):
        """
        Args:
            supabase_url: Your Supabase project URL
            jwt_secret: Optional legacy JWT secret for HS256 backward compatibility
        """
        self.supabase_url = supabase_url
        self.jwt_secret = jwt_secret  # Optional for legacy HS256 support
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time = 0
        self._jwks_cache_ttl = 3600  # Cache JWKS for 1 hour
        self._public_keys: Dict[str, Any] = {}  # Cache parsed public keys by kid
        
    async def _get_jwks(self) -> Dict[str, Any]:
        """Fetch JWKS (JSON Web Key Set) from Supabase with caching."""
        current_time = time.time()
        
        # Return cached JWKS if still valid
        if self._jwks_cache and (current_time - self._jwks_cache_time) < self._jwks_cache_ttl:
            return self._jwks_cache
        
        # Fetch fresh JWKS
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/auth/v1/.well-known/jwks.json",
                    timeout=5.0
                )
                response.raise_for_status()
                jwks_data = response.json()
                self._jwks_cache = jwks_data
                self._jwks_cache_time = current_time
                
                # Clear old public keys cache when JWKS refreshes
                self._public_keys = {}
                
                logger.info("JWKS refreshed from Supabase")
                return jwks_data
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            # Return cached version even if expired, better than failing
            if self._jwks_cache:
                logger.warning("Using expired JWKS cache due to fetch failure")
                return self._jwks_cache
            raise HTTPException(status_code=503, detail="Unable to fetch JWT signing keys")
    
    async def _get_signing_key(self, token: str) -> Any:
        """Get the appropriate signing key for token validation from JWKS."""
        try:
            # Decode header without verification to get the key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            
            if not kid:
                raise JWTError("Token missing 'kid' in header")
            
            # Check if we already have this key cached
            if kid in self._public_keys:
                return self._public_keys[kid]
            
            # Fetch JWKS and find the matching key
            jwks = await self._get_jwks()
            
            for key_data in jwks.get("keys", []):
                if key_data.get("kid") == kid:
                    # Convert JWK to usable format for python-jose
                    public_key = jwk.construct(key_data)
                    self._public_keys[kid] = public_key
                    logger.info(f"Loaded signing key: {kid}")
                    return public_key
            
            raise JWTError(f"Unable to find signing key with kid: {kid}")
            
        except Exception as e:
            logger.error(f"Error getting signing key: {e}")
            raise
    
    async def decode_token(self, token: str) -> dict:
        """
        Decode and validate JWT token.
        Supports both ES256 (ECC keys via JWKS) and HS256 (legacy secret).
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload containing user info
            
        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            # Determine which algorithm to use
            unverified_header = jwt.get_unverified_header(token)
            algorithm = unverified_header.get("alg", "HS256")
            
            if algorithm == "ES256":
                # Modern ECC signing keys validation
                signing_key = await self._get_signing_key(token)
                
                payload = jwt.decode(
                    token,
                    signing_key,
                    algorithms=["ES256"],
                    options={
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_aud": False,  # Supabase tokens may not have aud
                        "require_exp": True,
                    }
                )
                return payload
                
            elif algorithm == "HS256" and self.jwt_secret:
                # Legacy JWT secret validation (backward compatibility)
                payload = jwt.decode(
                    token,
                    self.jwt_secret,
                    algorithms=["HS256"],
                    options={
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_aud": False,
                        "require_exp": True,
                    }
                )
                return payload
            else:
                raise JWTError(f"Unsupported algorithm: {algorithm}")
                
        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during JWT validation: {e}")
            raise JWTError(f"Token validation error: {str(e)}")
    
    async def get_user_id_from_token(self, token: str) -> str:
        """
        Extract user ID from JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            User ID string
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = await self.decode_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token: missing user ID")
            
            return user_id
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid or expired token: {str(e)}")


class TokenCache:
    """
    In-memory cache for validated tokens to further reduce validation overhead.
    Useful for high-frequency requests from the same user.
    """
    
    def __init__(self, ttl: int = 300):
        """
        Args:
            ttl: Time-to-live for cached tokens in seconds (default: 5 minutes)
        """
        self._cache = {}
        self._ttl = ttl
    
    def get(self, token: str) -> Optional[str]:
        """Get cached user ID for a token if still valid."""
        if token in self._cache:
            user_id, timestamp = self._cache[token]
            if time.time() - timestamp < self._ttl:
                return user_id
            else:
                # Token expired in cache
                del self._cache[token]
        return None
    
    def set(self, token: str, user_id: str):
        """Cache a validated token and its user ID."""
        self._cache[token] = (user_id, time.time())
    
    def invalidate(self, token: str):
        """Remove a token from cache."""
        if token in self._cache:
            del self._cache[token]
    
    def clear_expired(self):
        """Remove all expired tokens from cache."""
        current_time = time.time()
        expired_tokens = [
            token for token, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl
        ]
        for token in expired_tokens:
            del self._cache[token]


# Global instances
_jwt_validator: Optional[JWTValidator] = None
_token_cache: Optional[TokenCache] = None


def init_jwt_auth(supabase_url: str, jwt_secret: Optional[str] = None, cache_ttl: int = 300):
    """
    Initialize JWT authentication system.
    
    Args:
        supabase_url: Supabase project URL
        jwt_secret: Optional legacy JWT secret for HS256 backward compatibility
        cache_ttl: Cache TTL in seconds (default: 5 minutes)
    """
    global _jwt_validator, _token_cache
    _jwt_validator = JWTValidator(supabase_url, jwt_secret)
    _token_cache = TokenCache(ttl=cache_ttl)
    
    if jwt_secret:
        logger.info("JWT authentication initialized (supports ES256 + legacy HS256)")
    else:
        logger.info("JWT authentication initialized (ES256 via JWKS)")


def get_jwt_validator() -> JWTValidator:
    """Get the global JWT validator instance."""
    if _jwt_validator is None:
        raise RuntimeError("JWT validator not initialized. Call init_jwt_auth() first.")
    return _jwt_validator


def get_token_cache() -> TokenCache:
    """Get the global token cache instance."""
    if _token_cache is None:
        raise RuntimeError("Token cache not initialized. Call init_jwt_auth() first.")
    return _token_cache