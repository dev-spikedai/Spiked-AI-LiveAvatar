import asyncio
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from core.config import get_g_vars

logger = logging.getLogger(__name__)

# --- Locked-down small-embedding-model demo path (sai@spiked.ai only) ---
#
# Benchmarked in scripts/embedding_model_bench_real.py against this exact
# account's SPIKEDAI_PRODUCT_MASTER.pdf chunks: e5-small-v2 cuts query-embed
# latency ~8x (110ms -> 13ms) vs the production e5-large-v2, with no wrong-
# document top-1 misses on the real corpus. Demo-scoped on purpose: the
# stored `chunks.embedding` column is a fixed public.vector(1024) (confirmed
# read-only via the Supabase PostgREST OpenAPI schema), so a 384-dim model
# can never share that column -- this path re-embeds chunk *content* with
# the small model entirely in memory instead, and never writes anything back
# to Supabase. Every other account is completely unaffected: they still read
# the stored e5-large-v2 embeddings via the existing warm/RPC paths below.
SMALL_MODEL_NAME = "intfloat/e5-small-v2"
SMALL_MODEL_DEMO_USER_IDS = {
    "ac73135a-7d06-4182-b67f-59a8db613265",  # sai@spiked.ai
}

_small_tokenizer = None
_small_model = None
_small_model_lock = asyncio.Lock()


async def _get_small_model():
    """Lazy singleton load, so the small model's weights aren't pulled/loaded
    until the first demo-account request actually needs them."""
    global _small_tokenizer, _small_model
    if _small_model is not None:
        return _small_tokenizer, _small_model
    async with _small_model_lock:
        if _small_model is None:
            def _load():
                tok = AutoTokenizer.from_pretrained(SMALL_MODEL_NAME)
                mdl = AutoModel.from_pretrained(SMALL_MODEL_NAME).eval()
                return tok, mdl
            _small_tokenizer, _small_model = await asyncio.to_thread(_load)
            logger.info(f"[WarmIndex] small demo model loaded: {SMALL_MODEL_NAME}")
    return _small_tokenizer, _small_model


def _average_pool_small(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


async def _embed_small(texts: list[str]) -> np.ndarray:
    """Embed already-prefixed ("passage: "/"query: ") texts with the small
    demo model. Self-contained (own tokenizer/model/pooling) -- deliberately
    not sharing core.config's g_vars embedding model, since that one stays
    pinned to e5-large-v2 for every other account's retrieval and the RPC
    fallback's expected vector(1024) parameter."""
    tokenizer, model = await _get_small_model()

    def _run():
        batch = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**batch)
        emb = _average_pool_small(out.last_hidden_state, batch["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    return await asyncio.to_thread(_run)

_PAGE_SIZE = 500
WARM_INDEX_TTL_SECONDS = float(os.getenv("WARM_INDEX_TTL_SECONDS", "300"))

# Local disk cache so a `--reload` restart (every file save during dev) isn't
# a fresh Supabase round trip before anything is fast again -- whichever
# users were warm before the restart come back warm instantly, and every
# lazy load after that rewrites this file with the latest state. Dev
# convenience only -- gitignored, and irrelevant in a real multi-instance
# deployment (each instance would need its own warm data anyway).
_CACHE_PATH = Path(__file__).resolve().parent.parent / ".warm_index_cache.pkl"

_SELECT = "id, source_id, content, embedding, sources!inner(user_id, filename, ingestion_status)"


def _build_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """Exact inner-product index over one user's chunk embeddings.

    IndexFlatIP, not an approximate index (IVF/HNSW): per-user corpora here
    are chunk counts in the hundreds-to-low-thousands, where a flat scan is
    already sub-millisecond and exact -- an approximate index would trade
    away correctness to speed up something that isn't the bottleneck. What
    FAISS buys at this scale is the C++ SIMD matmul and top-k in one call,
    replacing the numpy `@` + argpartition pair search() used to do by hand,
    plus IDSelector for the source_ids filter instead of slicing the matrix
    down before searching it.
    """
    if embeddings.size == 0:
        return None
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def _search_index(index: faiss.Index, chunks: list, query_embedding: np.ndarray, source_ids, top_k):
    """Shared by search() and search_small(): FAISS top-k, optionally
    restricted to source_ids via an IDSelector so filtering happens inside
    the C++ search rather than by pre-slicing the embedding matrix."""
    if source_ids:
        keep = set(source_ids)
        ids = np.array([i for i, c in enumerate(chunks) if c["source_id"] in keep], dtype=np.int64)
        if ids.size == 0:
            return []
        k = min(top_k, ids.size)
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids))
    else:
        k = min(top_k, len(chunks))
        params = None

    query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
    sims, idx = index.search(query, k, params=params)

    return [
        {
            "source_id": chunks[i]["source_id"],
            "filename": chunks[i]["filename"],
            "content": chunks[i]["content"],
            "similarity": float(s),
        }
        for s, i in zip(sims[0], idx[0])
        if i != -1  # FAISS pads short result rows with -1, not a real hit
    ]


def _parse_embedding(raw) -> Optional[list]:
    """PostgREST returns pgvector columns as a string like "[0.03,-0.01,...]",
    not a JSON array -- json.loads handles that syntax directly."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("[WarmIndex] failed to parse embedding value, skipping row")
        return None


class _WarmIndex:
    """In-memory per-user chunk embeddings. Turns retrieval from a
    network+DB round trip into an in-process matmul.

    Coverage is lazy and user-driven, not a fixed demo account list: the
    first request from any given user_id misses (search() returns None),
    the caller (build_rag_context) kicks off a background load for that
    user via kick_off_warm_load(), and every request after that from the
    same user hits the warm path. No account is special-cased -- this is
    what replaced the old hardcoded TARGET_EMAILS/_resolve_user_ids_by_email
    demo-scoping (email->user_id admin-API resolution isn't needed at all
    now that loading is keyed directly off the user_id already on every
    request).

    Scope note: this replicates only the *dense* arm of build_rag_context's
    hybrid (dense + BM25/RRF) retrieval in backend-one. The sparse arm and
    RRF fusion are not reproduced -- warm results are dense-cosine top-k.
    Good enough to validate the latency win; a local BM25 pass (e.g.
    rank_bm25) is needed before this fully replaces the RPC for
    quality-sensitive traffic.

    Pagination is keyset (order by id, id > last_seen), not OFFSET -- OFFSET
    against a filtered/joined query gets slower every page (Postgres re-scans
    and re-sorts everything before the cursor) and reliably hit
    `statement timeout` past ~3000 rows on a full-corpus crawl.
    """

    def __init__(self):
        self._by_user: dict[str, dict] = {}
        self._small_by_user: dict[str, dict] = {}
        self._loading: set[str] = set()
        self._loading_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def load_all(self):
        """Load instantly from the on-disk cache if one exists (dev
        restarts) -- whichever users were warm last session start warm
        again immediately. No eager account list beyond that: every other
        user warms lazily via kick_off_warm_load() on their first request."""
        self._load_from_disk()
        asyncio.create_task(self._load_small_demo_accounts())

    def kick_off_warm_load(self, user_id: str) -> None:
        """Fire-and-forget background warm-load for a user who just missed
        the warm path. Idempotent per user_id -- a second call while a load
        is already in flight (e.g. a burst of requests from the same user
        before the first load lands) is a no-op, not a duplicate fetch."""
        if user_id in self._by_user and self.is_fresh(user_id):
            return
        if user_id in self._loading_tasks:
            return
        self._loading.add(user_id)
        task = asyncio.create_task(self._load_user_tracked(user_id))
        self._loading_tasks[user_id] = task

    async def ensure_user_loaded(self, user_id: str) -> int:
        """Await one shared load for the user instead of starting duplicates."""
        if user_id in self._by_user and self.is_fresh(user_id):
            return len(self._by_user[user_id].get("chunks") or [])
        task = self._loading_tasks.get(user_id)
        if task is None:
            self._loading.add(user_id)
            task = asyncio.create_task(self._load_user_tracked(user_id))
            self._loading_tasks[user_id] = task
        return await task

    async def _load_user_tracked(self, user_id: str) -> int:
        try:
            count = await self.load_user(user_id)
            logger.info(f"[WarmIndex] lazily warm-loaded {user_id}: {count} chunks")
            return count
        except Exception:
            logger.exception(f"[WarmIndex] lazy warm-load failed for {user_id}")
            return 0
        finally:
            self._loading.discard(user_id)
            self._loading_tasks.pop(user_id, None)

    async def _load_small_demo_accounts(self):
        """Locked-down small-model demo path: re-embed SMALL_MODEL_DEMO_USER_IDS'
        chunk content with e5-small-v2, purely in memory. See module docstring
        above -- separate vector space from self._by_user, never touches
        Supabase's stored (1024-dim) embedding column."""
        for user_id in SMALL_MODEL_DEMO_USER_IDS:
            try:
                count = await self.load_user_small(user_id)
                logger.info(f"[WarmIndex] small-model demo-loaded {user_id}: {count} chunks")
            except Exception:
                logger.exception(f"[WarmIndex] small-model demo load failed for {user_id}")

    async def load_user_small(self, user_id: str) -> int:
        """Same keyset-paginated read as load_user() below, but only reads
        `content` (never the stored `embedding` column) and re-embeds it
        in-process with the small demo model."""
        g_vars = get_g_vars()
        supabase = g_vars["supabase"]
        if not supabase:
            return 0

        contents, chunks = [], []
        last_id = None
        while True:
            def _query(after=last_id):
                q = (
                    supabase.table("chunks")
                    .select("id, source_id, content, sources!inner(user_id, filename, ingestion_status)")
                    .eq("sources.ingestion_status", "COMPLETED")
                    .eq("sources.user_id", user_id)
                    .order("id")
                    .limit(_PAGE_SIZE)
                )
                if after is not None:
                    q = q.gt("id", after)
                return q.execute()

            resp = await asyncio.to_thread(_query)
            rows = resp.data or []
            if not rows:
                break
            for row in rows:
                content = row.get("content") or ""
                if not content.strip():
                    continue
                contents.append(content)
                chunks.append({
                    "source_id": row["source_id"],
                    "filename": (row.get("sources") or {}).get("filename"),
                    "content": content,
                })
            last_id = rows[-1]["id"]
            if len(rows) < _PAGE_SIZE:
                break

        if not chunks:
            async with self._lock:
                self._small_by_user[user_id] = {
                    "embeddings": np.zeros((0, 1)),
                    "chunks": [],
                    "index": None,
                    "loaded_at": time.monotonic(),
                }
            return 0

        embeddings = await _embed_small([f"passage: {c}" for c in contents])

        async with self._lock:
            self._small_by_user[user_id] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "index": _build_index(embeddings),
                "loaded_at": time.monotonic(),
            }
        return len(chunks)

    async def embed_query_small(self, question: str) -> np.ndarray:
        emb = await _embed_small([f"query: {question}"])
        return emb[0]

    def search_small(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        source_ids: Optional[list] = None,
        top_k: int = 10,
    ):
        """Small-model counterpart to search() below -- same FAISS top-k,
        reads self._small_by_user instead. Returns None if this user isn't
        in the small-model demo set (or not loaded yet), same fallback
        contract as search()."""
        bucket = self._small_by_user.get(user_id)
        if bucket is None:
            return None
        if bucket["embeddings"].size == 0:
            return []
        return _search_index(bucket["index"], bucket["chunks"], query_embedding, source_ids, top_k)

    def _load_from_disk(self):
        if not _CACHE_PATH.exists():
            return
        try:
            with open(_CACHE_PATH, "rb") as f:
                self._by_user = pickle.load(f)
            # A cache written before FAISS was added to this file has
            # "embeddings"/"chunks" but no "index" -- rebuild it in memory
            # rather than invalidating the whole cache over one missing key.
            for bucket in self._by_user.values():
                if "index" not in bucket:
                    bucket["index"] = _build_index(bucket["embeddings"])
                bucket.setdefault("loaded_at", time.monotonic())
            logger.info(
                f"[WarmIndex] loaded {len(self._by_user)} users from disk cache "
                f"({_CACHE_PATH.name})"
            )
        except Exception as e:
            logger.warning(f"[WarmIndex] disk cache load failed ({e}); starting cold")

    def _save_to_disk(self):
        try:
            tmp = _CACHE_PATH.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self._by_user, f)
            tmp.replace(_CACHE_PATH)
        except Exception as e:
            logger.warning(f"[WarmIndex] disk cache write failed ({e})")

    async def load_user(self, user_id: str):
        """Warm-load (or refresh) a single user's chunks."""
        g_vars = get_g_vars()
        supabase = g_vars["supabase"]
        if not supabase:
            return 0

        vecs, chunks = [], []
        last_id = None
        while True:
            def _query(after=last_id):
                q = (
                    supabase.table("chunks")
                    .select(_SELECT)
                    .eq("sources.ingestion_status", "COMPLETED")
                    .eq("sources.user_id", user_id)
                    .order("id")
                    .limit(_PAGE_SIZE)
                )
                if after is not None:
                    q = q.gt("id", after)
                return q.execute()

            resp = await asyncio.to_thread(_query)
            rows = resp.data or []
            if not rows:
                break
            for row in rows:
                embedding = _parse_embedding(row.get("embedding"))
                if embedding is None:
                    continue
                vecs.append(embedding)
                chunks.append({
                    "source_id": row["source_id"],
                    "filename": (row.get("sources") or {}).get("filename"),
                    "content": row["content"],
                })
            last_id = rows[-1]["id"]
            if len(rows) < _PAGE_SIZE:
                break

        embeddings = np.asarray(vecs, dtype=np.float32)
        async with self._lock:
            self._by_user[user_id] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "index": _build_index(embeddings),
                "loaded_at": time.monotonic(),
            }
            self._save_to_disk()
        logger.info(f"[WarmIndex] loaded user {user_id}: {len(chunks)} chunks")
        return len(chunks)

    def loaded_count(self) -> int:
        return len(self._by_user)

    def is_fresh(self, user_id: str) -> bool:
        bucket = self._by_user.get(user_id)
        if bucket is None:
            return False
        return time.monotonic() - bucket.get("loaded_at", 0) <= WARM_INDEX_TTL_SECONDS

    def search(
        self,
        user_id: str,
        query_embedding: np.ndarray,
        source_ids: Optional[list] = None,
        top_k: int = 10,
    ):
        """Dense top-k against the warm index (FAISS IndexFlatIP -- exact
        inner product, see _build_index). Returns None if this user isn't
        warm yet -- caller should fall back to the DB RPC."""
        bucket = self._by_user.get(user_id)
        if bucket is None:
            return None
        if not self.is_fresh(user_id):
            self.kick_off_warm_load(user_id)
            return None
        if bucket["embeddings"].size == 0:
            return []
        return _search_index(bucket["index"], bucket["chunks"], query_embedding, source_ids, top_k)


warm_index = _WarmIndex()
