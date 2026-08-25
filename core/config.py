import logging
import os
import sys

import torch
from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

# --- Basic Setup ---
# override=True so the .env file always wins over any stale value already in the
# shell environment (e.g. a var exported by a prior `set -a && . ./.env`, which
# can carry a trailing \r from CRLF line endings and corrupt URLs/keys).
load_dotenv(override=True)

# --- Configuration ---
CHUNK_SIZE = 300
OVERLAP_SIZE = 50
TOP_K = 6
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
hf_token = os.getenv("HF_TOKEN", "").strip()
hf_token = hf_token if hf_token else None
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".ppt", ".html", ".txt", ".md"}
# Per-user hard storage quota (bytes). Sum of file_size across a user's document
# sources may not exceed this. Overridable via env for higher-tier plans later.
STORAGE_LIMIT_BYTES = int(os.getenv("STORAGE_LIMIT_BYTES", str(200 * 1024 ** 2)))  # 200 MB
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "").strip()
GCS_DOCS_PREFIX = "documents/"
GEMINI_FLASH_MODEL = "gemini-1.5-flash-latest"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
BASE_URL = os.getenv("BASE_URL", "https://spikedai-production-application-409019309412.us-central1.run.app").strip()
CHUNK_INSERT_BATCH_SIZE = 100
DEFAULT_TEST_USER_ID = "fd3ff615-b248-4e8f-84f1-ff458bf30d48"
APP_ENV = os.getenv("APP_ENV", "production")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)

# JWT Configuration
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip() or None
JWT_CACHE_TTL = int(os.getenv("JWT_CACHE_TTL", "300"))  # 5 minutes default


# --- Global State & Clients ---
g_vars = {
    "supabase": None,
    "embedding_tokenizer": None,
    "embedding_model": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def get_g_vars():
    return g_vars

def init_supabase_client():
    """Initializes the Supabase client."""
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set.")
    g_vars["supabase"] = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized.")

import asyncio

async def init_embedding_model():
    """Loads the embedding model and tokenizer."""
    try:
        device = g_vars["device"]
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' onto device '{device}'")

        # Offload blocking CPU/IO work to a thread to keep the event loop responsive
        g_vars["embedding_tokenizer"] = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, EMBEDDING_MODEL_NAME, token=hf_token
        )

        def load_model():
            return AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, token=hf_token).to(device).eval()

        g_vars["embedding_model"] = await asyncio.to_thread(load_model)
        logger.info("Embedding model loaded successfully.")
    except Exception:
        logger.exception("FATAL: embedding model load failed")
        raise

def init_jwt_authentication():
    """
    Initialize JWT authentication system.
    Supports ECC (ES256) keys via JWKS and optionally legacy HS256.
    """
    from core.auth import init_jwt_auth
    
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    if not supabase_url:
        logger.warning("JWT authentication not initialized: Missing SUPABASE_URL")
        return
    
    init_jwt_auth(supabase_url, SUPABASE_JWT_SECRET, JWT_CACHE_TTL)
    logger.info("JWT authentication initialized.")

def close_clients():
    """Placeholder for any client cleanup logic."""
    pass
