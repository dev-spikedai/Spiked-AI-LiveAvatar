# main.py
#
# feat/fastembeddings: stripped fork of backend-one carrying only the
# /ask/regular answer-generation path, run as a standalone hot-load RAG
# service. See routers/search.py for the warm in-memory retrieval path.

import logging
import os
import sys
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import faiss
import torch
from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware

from services.ai_helpers import cleanup_clients

# Both torch and faiss default their CPU thread counts to os.cpu_count(),
# which can oversubscribe a CPU-limited deployment container (e.g. Cloud Run
# with --cpu=1 or 2) -- multiple intra-op threads competing for the same
# throttled vCPU makes individual embedding/search calls slower, not faster.
# Env-overridable so the numbers can be tuned per deployment without a code
# change; set as early as possible, before any model/index work happens.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
faiss.omp_set_num_threads(int(os.getenv("FAISS_NUM_THREADS", "1")))

from core.config import (
    init_jwt_authentication,
    init_supabase_client,
    init_embedding_model,
    close_clients,
)

from routers import search
from services.warm_index import warm_index
from core.dependencies import get_user_id_from_token

# Full detail goes to a rotating log file; the console only gets our own
# app-level messages, not httpx's per-request spam (every warm-index refresh
# page, every Supabase call) that was drowning out real signal during dev.
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

_file_handler = RotatingFileHandler(
    _LOG_DIR / "app.log", maxBytes=10 * 1024 * 1024, backupCount=3
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_file_handler.setLevel(logging.INFO)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])

# httpx logs every single request at INFO -- fine for the file, way too
# noisy for the console during a 16-page warm-index refresh.
_console_handler.addFilter(lambda record: record.name != "httpx")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Lifespan: Startup sequence initiated.")
    try:
        init_jwt_authentication()
        init_supabase_client()

        # Load embedding model before accepting traffic so /ask/regular never
        # hits a None tokenizer.
        await init_embedding_model()

        # Warm-load whatever's in the on-disk dev cache (near-instant) plus
        # the small-model demo account, then let coverage grow lazily from
        # there -- warm_index has no fixed account list anymore. Each real
        # user's first request misses (served via the RPC fallback) and
        # kicks off a background load for that user_id; every request after
        # that from the same user is warm. Set WARM_INDEX_AUTOLOAD=false to
        # skip even the disk-cache/demo-account load and start fully cold,
        # or use POST /debug/warm/{user_id} to force-warm one account early.
        if os.getenv("WARM_INDEX_AUTOLOAD", "true").strip().lower() != "false":
            warm_index.load_all()
        else:
            logger.info("WARM_INDEX_AUTOLOAD=false: starting cold, use /debug/warm/{user_id}")

        logger.info("API Lifespan: Core clients ready, warm index loading in background. Accepting requests.")

        yield

    except Exception as e:
        logger.critical(f"API Lifespan: FATAL STARTUP ERROR: {e}", exc_info=True)
        raise
    finally:
        logger.info("API Lifespan: Initiating shutdown.")
        await cleanup_clients()
        close_clients()
        logger.info("API Lifespan: Shutdown complete.")


app = FastAPI(
    title="Fast-Embeddings RAG Service",
    version="0.1.0",
    description="Stripped backend-one fork: hot-loaded in-memory RAG for /ask/regular.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sources, X-Context-Hash, X-Cognitive-Key"],
)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "warm_bots": warm_index.loaded_count(),
    }


app.include_router(search.router)


@app.post("/debug/warm/{user_id}", tags=["System"])
async def debug_warm_user(user_id: str):
    """Warm-load a single user's chunks on demand -- for testing without
    waiting on the full-corpus refresh timer. Read-only against Supabase."""
    count = await warm_index.ensure_user_loaded(user_id)
    return {"user_id": user_id, "chunks_loaded": count}


@app.post("/warm", tags=["System"])
async def warm_authenticated_user(user_id: str = Depends(get_user_id_from_token)):
    """Warm the authenticated user's complete document corpus.

    This is the meeting-start path: the caller supplies the same bearer token
    used for RAG, and the service derives the user id from that token rather
    than trusting a user id in the request body or URL. The in-memory index is
    still filtered by client/source scope at query time, so warming a user's
    corpus does not weaken tenant isolation.
    """
    count = await warm_index.ensure_user_loaded(user_id)
    return {"user_id": user_id, "chunks_loaded": count, "warm": True}
