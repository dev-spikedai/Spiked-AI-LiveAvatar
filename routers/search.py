import asyncio
import json
import logging
import hashlib
import re
from urllib.parse import quote
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from openai._exceptions import APIError, RateLimitError, APIConnectionError
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from core.config import get_g_vars, TOP_K, BASE_URL, OPENAI_CHAT_MODEL
from core.dependencies import get_user_id_from_token, get_current_user_settings
from core.kyc_database import get_client_kyc_config, MANUAL_KYC_ID
from core.persona_styles import render_persona, render_styles
from core.prompts import SEARCH_SYSTEM_PROMPT
from models.schemas import AskRequest, SettingsModel
from services.ai_helpers import (
    get_embeddings,
    stream_groq_response,
    call_gemini_flash_llm
)
from services.rag_classifier import classify_and_log
from services.warm_index import warm_index, SMALL_MODEL_DEMO_USER_IDS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Search AI"])


@router.get("/documents")
async def list_user_documents(
    user_id: str = Depends(get_user_id_from_token),
    client_id: str | None = None,
):
    """Minimal stand-in for backend-one's /documents: the orchestrator's
    resolve_source_ids() only reads `id` and `status` per item to build the
    source_ids list for /ask/regular. It caches non-empty results itself
    (AGENT_SOURCE_IDS_CACHE_TTL_S, 300s) -- that cache never engaged while
    this endpoint 404'd, which is why every turn was re-hitting it."""
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    query = (
        supabase.table("sources")
        .select("id, ingestion_status")
        .eq("user_id", user_id)
        .eq("source_type", "document")
        .not_.is_("client_id", "null")
    )
    if client_id:
        query = query.eq("client_id", client_id)
    resp = await asyncio.to_thread(query.execute)
    return [{"id": row["id"], "status": row["ingestion_status"]} for row in (resp.data or [])]


@router.get("/websites")
async def list_user_websites(
    user_id: str = Depends(get_user_id_from_token),
    client_id: str | None = None,
):
    """Same as /documents above, filtered to source_type='website'."""
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    query = (
        supabase.table("sources")
        .select("id, ingestion_status")
        .eq("user_id", user_id)
        .eq("source_type", "website")
        .not_.is_("client_id", "null")
    )
    if client_id:
        query = query.eq("client_id", client_id)
    resp = await asyncio.to_thread(query.execute)
    return [{"id": row["id"], "status": row["ingestion_status"]} for row in (resp.data or [])]


_CONTEXT_CACHE = {}

# Cognitive answers are persisted in Supabase (see cognitive_answers table) so a
# /cognitive poll survives TTL eviction, instance restarts, and multi-instance
# routing. Only the task handles stay in-memory — they merely dedupe duplicate
# generations within a single process; a redundant generation on another instance
# is harmless (last write wins).
_COGNITIVE_TASKS = {}
_COGNITIVE_LOCK = asyncio.Lock()

COGNITIVE_TTL_SECONDS = 3600          # rows older than this are purged
COGNITIVE_STALE_PENDING_SECONDS = 90  # a 'pending' row older than this = a dead task -> report failed
_COG_CLEANUP_INTERVAL = 300           # throttle the DB cleanup to at most once / 5 min
_last_cog_cleanup = 0.0


async def _cog_write(cognitive_key: str, status: str, answer: str | None = None, error: str | None = None):
    """Upsert a cognitive answer row. Best-effort: a DB hiccup must never crash
    the background generation task."""
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    if not supabase:
        return
    row = {
        "cognitive_key": cognitive_key,
        "status": status,
        "answer": answer,
        "error": error,
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        await asyncio.to_thread(
            lambda: supabase.table("cognitive_answers")
            .upsert(row, on_conflict="cognitive_key")
            .execute()
        )
    except Exception as e:
        logger.warning(f"[COG] cache write failed for {cognitive_key}: {e}")


async def _cog_read(cognitive_key: str):
    """Return {status, answer, error, updated_at} for a key, or None if absent."""
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    if not supabase:
        return None
    try:
        res = await asyncio.to_thread(
            lambda: supabase.table("cognitive_answers")
            .select("status, answer, error, updated_at")
            .eq("cognitive_key", cognitive_key)
            .limit(1)
            .execute()
        )
        return res.data[0] if res and res.data else None
    except Exception as e:
        logger.warning(f"[COG] cache read failed for {cognitive_key}: {e}")
        return None


def _pending_is_stale(updated_at_str: str | None) -> bool:
    """A 'pending' row this old means the generation task died without writing a
    terminal status (e.g. the instance was recycled mid-generation). Well past the
    generation cap, so it's safe to treat as failed rather than poll forever."""
    if not updated_at_str:
        return False
    try:
        ts = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age > COGNITIVE_STALE_PENDING_SECONDS
    except Exception:
        return False


async def cleanup_expired_cognitive():
    """Delete cognitive rows older than the TTL. Throttled so the DELETE runs at
    most once every few minutes regardless of request volume."""
    global _last_cog_cleanup
    now = time.time()
    if now - _last_cog_cleanup < _COG_CLEANUP_INTERVAL:
        return
    _last_cog_cleanup = now

    g_vars = get_g_vars()
    supabase = g_vars["supabase"]
    if not supabase:
        return
    cutoff = (datetime.utcnow() - timedelta(seconds=COGNITIVE_TTL_SECONDS)).isoformat()
    try:
        await asyncio.to_thread(
            lambda: supabase.table("cognitive_answers")
            .delete()
            .lt("updated_at", cutoff)
            .execute()
        )
    except Exception as e:
        logger.warning(f"[COG] cleanup failed: {e}")

async def _log_no_rag_question(question: str, endpoint: str, client_id: str | None):
    """Fire-and-forget: log a question with no RAG context into meeting_questions."""
    try:
        g_vars = get_g_vars()
        supabase = g_vars["supabase"]
        if not supabase:
            return
        now = datetime.utcnow().isoformat()
        await asyncio.to_thread(
            lambda: supabase.table("meeting_questions").insert({
                "question": question,
                "answer_transcript": "NO_INFO_FROM_RAG",
                "rag_verdict": "NO_INFO",
                "endpoint": endpoint,
                "client_id": client_id,
                "created_at": now,
                "updated_at": now,
            }).execute()
        )
    except Exception as e:
        logger.warning(f"Failed to log no-RAG question: {e}")
# Retrieval budget shared by the live (Groq) and cognitive (background) answers.
# Both read the same retrieved context; the cognitive answer no longer truncates
# it, so all RAG_TOP_K_LIVE chunks reach the reasoning model.
RAG_TOP_K_LIVE = 10

# The handsfree LIVE (voice) answer only needs the very top chunks — a smaller
# prompt means faster prefill / time-to-first-token. Retrieval still fetches the
# full RAG_TOP_K_LIVE set so the background cognitive answer keeps full depth.
RAG_TOP_K_HANDSFREE_LIVE = 5

# Short-lived cache of the *retrieval* result keyed by normalized question + scope.
# In a live meeting the same question is frequently re-detected from the transcript;
# a hit skips both the local query embedding (the main handsfree latency cost) and
# the DB round-trip. Keyed on exact normalized text so cached chunks are always
# correct for the query.
_RAG_CACHE = {}
RAG_CACHE_TTL_SECONDS = 120

# Narrower cache of just the *query embedding vector*, one layer below
# _RAG_CACHE. _RAG_CACHE's key includes top_k, so the live call
# (RAG_TOP_K_LIVE) and the cognitive wide-retrieve call (COGNITIVE_WIDE_TOP_K)
# for the identical question are guaranteed misses against each other there --
# each would otherwise pay its own e5-large-v2 CPU forward pass for the exact
# same text. TTL only needs to bridge that live->cognitive gap within one
# request cycle, not act as a long-lived cache.
_QUERY_EMBEDDING_CACHE = {}
QUERY_EMBEDDING_CACHE_TTL_SECONDS = 60


def _normalize_question(q: str) -> str:
    return " ".join((q or "").lower().split())


def _rag_cache_key(question, user_id, source_ids, client_id, top_k, query_augment="") -> str:
    sid = ",".join(sorted(source_ids)) if source_ids else ""
    raw = f"{user_id}|{client_id}|{sid}|{top_k}|{query_augment}|{_normalize_question(question)}"
    return hashlib.sha256(raw.encode()).hexdigest()


async def _get_query_embedding(user_id: str, question: str, query_augment: str, embed_query: str):
    """Cache just the query embedding vector so the live retrieval call and
    the cognitive wide-retrieve call for the same question (different top_k,
    so different _RAG_CACHE entries -- see _QUERY_EMBEDDING_CACHE above)
    don't each pay their own embedding model call. query_augment stays in
    the key since it changes the actual text being embedded (KYC/persona
    steering), so a differently-augmented embedding is never served across
    contexts."""
    key = (user_id, _normalize_question(question), query_augment)
    now = time.time()
    cached = _QUERY_EMBEDDING_CACHE.get(key)
    if cached and now - cached["ts"] < QUERY_EMBEDDING_CACHE_TTL_SECONDS:
        return cached["embedding"]
    embedding = await get_embeddings([embed_query])
    _QUERY_EMBEDDING_CACHE[key] = {"embedding": embedding, "ts": now}
    return embedding


def _build_query_augment(settings) -> str:
    """Buyer context appended to the *retrieval* query so results skew toward the
    buyer's priorities, not just literal question matches (KYC-steered retrieval).
    It never reaches the LLM prompt — only the embedding + BM25 query."""
    parts = []
    kw = getattr(settings, "strategic_keywords", None)
    if isinstance(kw, list) and kw:
        parts.append(", ".join(str(k) for k in kw[:8]))
    elif isinstance(kw, str) and kw.strip():
        parts.append(kw.strip())
    pd = getattr(settings, "product_domain", None)
    if pd and str(pd).strip():
        parts.append(str(pd).strip())
    return " ".join(parts).strip()


def _items_to_rag(items: list, cache_key: str, now: float) -> dict:
    """Shared formatting for warm-index hits (large-model and small-model
    demo path alike) -- builds the same rag dict shape build_rag_context
    returns from the RPC path, and populates the retrieval cache."""
    context_text = format_context(items)
    unique_sources = {}
    for item in items:
        sid = item["source_id"]
        if sid not in unique_sources:
            unique_sources[sid] = {
                "filename": item["filename"],
                "description": None,
                "source_id": sid,
            }
    sources_list = list(unique_sources.values())
    context_hash = hashlib.sha256(context_text.encode()).hexdigest()
    _CONTEXT_CACHE[context_hash] = {"context": context_text, "sources": sources_list}
    rag = {
        "context_text": context_text,
        "sources": sources_list,
        "context_hash": context_hash,
        "chunks": items,
    }
    _RAG_CACHE[cache_key] = {"ts": now, "rag": rag}
    return rag


def format_context(items) -> str:
    """Render retrieved chunks into the prompt context block."""
    parts = []
    for item in items:
        parts.append(
            f"Source File: {item['filename']}\n"
            f"Content: {item['content']}\n\n"
        )
    return "".join(parts)


async def build_rag_context(
    question: str,
    user_id: str,
    source_ids: list = None,
    client_id: str = None,
    top_k: int = TOP_K,
    query_augment: str = "",
):
    """
    Build RAG context, scoped to the active client's source_ids when provided.

    Hybrid retrieval: dense (e5 vector) + sparse (BM25 / full-text) fused with RRF
    via the hybrid_match_chunks_by_sources RPC, with a graceful fallback to the
    legacy vector-only RPC if the hybrid function isn't deployed yet.

    Fail-closed semantics:
    - client_id present + source_ids missing/empty → return None (client has no docs).
    - client_id present + source_ids non-empty → filter to those source_ids only.
    - client_id None + source_ids None → legacy single-tenant path (all user docs).
    """
    logger.warning(f"[RAG] ENTER build_rag_context q={question!r} user_id={user_id} client_id={client_id} source_ids_len={len(source_ids) if source_ids else 0}")
    g_vars = get_g_vars()
    supabase = g_vars["supabase"]

    # Fail closed: a selected client with no docs must not fall back to all-docs.
    if client_id is not None and not source_ids:
        logger.warning(f"[RAG] fail-closed: client_id={client_id} but no source_ids")
        return None

    # Retrieval cache — skips the query embedding + DB round-trip on repeats.
    cache_key = _rag_cache_key(question, user_id, source_ids, client_id, top_k, query_augment)
    now = time.time()
    cached = _RAG_CACHE.get(cache_key)
    if cached and now - cached["ts"] <= RAG_CACHE_TTL_SECONDS:
        logger.warning("[RAG] cache hit")
        return cached["rag"]

    # KYC-steered retrieval: buyer keywords/domain are folded into the retrieval
    # query (both dense and sparse arms) but not the LLM prompt.
    embed_query = f"query: {question}\n{query_augment}" if query_augment else f"query: {question}"
    bm25_query = f"{question} {query_augment}".strip() if query_augment else question

    # Locked-down small-model demo path (sai@spiked.ai only, see warm_index
    # module docstring): entirely separate vector space, in-memory only, no
    # Supabase writes. Checked first so this account never pays for the
    # large-model embed below at all. Falls through to the normal path if
    # the small index for this user hasn't finished its background load yet.
    if user_id in SMALL_MODEL_DEMO_USER_IDS:
        small_query_embedding = await warm_index.embed_query_small(embed_query)
        small_items = warm_index.search_small(
            user_id, small_query_embedding, source_ids=source_ids, top_k=top_k
        )
        if small_items is not None:
            logger.warning(f"[RAG] small-model demo warm hit, {len(small_items)} chunks")
            if not small_items:
                return None
            return _items_to_rag(small_items, cache_key, now)

    query_embedding = await _get_query_embedding(user_id, question, query_augment, embed_query)

    # Warm path: this user's chunk embeddings are hot in memory -> skip the
    # network+DB round trip entirely and do the cosine search in-process.
    # Dense-only (see warm_index docstring); falls through to the hybrid RPC
    # below if the user isn't warm yet or has no chunks.
    warm_items = warm_index.search(
        user_id, query_embedding[0], source_ids=source_ids, top_k=top_k
    )
    if warm_items is not None:
        logger.warning(f"[RAG] warm-index hit, {len(warm_items)} chunks")
        if not warm_items:
            return None
        return _items_to_rag(warm_items, cache_key, now)

    # Miss: this user isn't warm yet. Serve this one request via the RPC
    # below as usual, but kick off a background load so every request after
    # this one from the same user hits the warm path above instead -- no
    # fixed demo-account list, coverage is earned per-user on first contact.
    warm_index.kick_off_warm_load(user_id)

    hybrid_params = {
        "query_embedding": query_embedding[0].tolist(),
        "query_text": bm25_query,
        "match_count": top_k,
        "p_user_id": user_id,
    }
    if source_ids:
        hybrid_params["p_source_ids"] = source_ids

    try:
        db_response = await asyncio.to_thread(
            lambda: supabase.rpc("hybrid_match_chunks_by_sources", hybrid_params).execute()
        )
    except Exception as e:
        logger.warning(f"[RAG] hybrid RPC unavailable ({e}); falling back to vector-only")
        legacy_params = {
            "query_embedding": query_embedding[0].tolist(),
            "match_threshold": 0.3,
            "match_count": top_k,
            "p_user_id": user_id,
        }
        if source_ids:
            legacy_params["p_source_ids"] = source_ids
        db_response = await asyncio.to_thread(
            lambda: supabase.rpc("match_chunks_by_sources", legacy_params).execute()
        )

    logger.warning(f"[RAG] returned {len(db_response.data or [])} rows; first_source_id={(db_response.data[0].get('source_id') if db_response.data else None)}")

    if not db_response.data:
        return None

    items = db_response.data
    context_text = format_context(items)

    unique_sources = {}
    for item in items:
        sid = item["source_id"]
        if sid not in unique_sources:
            unique_sources[sid] = {
                "filename": item["filename"],
                "description": item.get("description"),
                "source_id": sid
            }

    sources_list = list(unique_sources.values())
    context_hash = hashlib.sha256(context_text.encode()).hexdigest()

    _CONTEXT_CACHE[context_hash] = {
        "context": context_text,
        "sources": sources_list
    }

    rag = {
        "context_text": context_text,
        "sources": sources_list,
        "context_hash": context_hash,
        "chunks": items,
    }
    _RAG_CACHE[cache_key] = {"ts": now, "rag": rag}
    return rag

def build_cognitive_prompt(settings):
    sales_context_lines = []

    if settings.seller_company:
        sales_context_lines.append(f"Seller Company: {settings.seller_company}")
    if settings.seller_name:
        sales_context_lines.append(f"Seller Representative: {settings.seller_name}")
    if settings.client_company:
        sales_context_lines.append(f"Buyer Company: {settings.client_company}")
    if settings.client_names:
        sales_context_lines.append(f"Buyer Stakeholders: {settings.client_names}")
    if settings.products_services:
        sales_context_lines.append(f"Products/Services Offered: {settings.products_services}")
    if settings.product_domain:
        sales_context_lines.append(f"Product Domain: {settings.product_domain}")

    sales_context_str = "\n".join(sales_context_lines) or "N/A"

    keywords_str = settings.strategic_keywords
    if isinstance(keywords_str, list):
        keywords_str = ", ".join(keywords_str)

    persona_str = render_persona(settings.selected_persona)
    styles_str = render_styles(settings.selected_answer_styles)

    return f"""
You are SpikedAI — the seller ({settings.seller_company or "the seller"}) speaking directly to the buyer ({settings.client_company or "the buyer"}) in a strategic sales conversation. Tone: natural, confident, persuasive, concise — never robotic or formal, and never open with "Hello"/"Hi".

GROUNDING RULE (overrides everything below):
Answer only from the provided documents and SALES CONTEXT. If they don't contain enough to answer, say so plainly in one or two sentences and stop. Never invent facts, features, customers, metrics, numbers, pricing, integrations, timelines, or next steps that aren't explicitly supported by the context. Do not pad with generic sales language to disguise missing information. The style/structure directives below apply ONLY when the context is sufficient.

SALES CONTEXT:
{sales_context_str}

CUSTOM INSTRUCTIONS:
{settings.custom_prompt or "N/A"}

EXECUTIVE SNAPSHOT (BUYER PRIORITIES):
{settings.executive_snapshot or "N/A"}

STRATEGIC KEYWORDS:
{keywords_str or "None"}

AUDIENCE PROFILE:
{persona_str or "N/A"}

ANSWER STYLE DIRECTIVES (apply all):
{styles_str or "None"}

When the context is sufficient, write a grounded, buyer-focused answer that:
- Opens by addressing the buyer's situation/priorities (reference the executive snapshot or strategic keywords naturally).
- Connects those goals/challenges to the seller's offering, showing business impact (ROI, efficiency, risk, differentiation) wherever the context supports it.
- Ends with one concrete next step that moves the deal forward.
- Matches the AUDIENCE PROFILE tone and every ANSWER STYLE DIRECTIVE, in clear Markdown.

Be substantive but tight — every sentence should carry grounded information, not filler. Never mention being an AI or sound like documentation.
"""

def build_reasoning_sales_prompt(settings):

    sales_context_lines = []

    if settings.seller_company:
        sales_context_lines.append(f"Seller Company: {settings.seller_company}")
    if settings.seller_name:
        sales_context_lines.append(f"Seller Representative: {settings.seller_name}")
    if settings.client_company:
        sales_context_lines.append(f"Buyer Company: {settings.client_company}")
    if settings.client_names:
        sales_context_lines.append(f"Buyer Stakeholders: {settings.client_names}")
    if settings.products_services:
        sales_context_lines.append(f"Products/Services Offered: {settings.products_services}")
    if settings.product_domain:
        sales_context_lines.append(f"Product Domain: {settings.product_domain}")

    sales_context_str = "\n".join(sales_context_lines) or "N/A"

    keywords_str = settings.strategic_keywords
    if isinstance(keywords_str, list):
        keywords_str = ", ".join(keywords_str)

    persona_str = render_persona(settings.selected_persona)
    styles_str = render_styles(settings.selected_answer_styles)

    return f"""
You are SpikedAI — operating as the seller, speaking directly to the buyer in a strategic sales conversation.

Your tone should be natural, confident, persuasive, and smooth — not robotic, not overly formal, and never beginning with greetings like "Hello" or "Hi".

This answer must feel like it is coming from the seller ({settings.seller_company or "the seller"}) pitching directly to the buyer ({settings.client_company or "the buyer"}).

--------------------------------
GROUNDING RULE (overrides every other instruction below)
--------------------------------
If the provided documents and SALES CONTEXT do not contain enough information to answer the question, say so plainly in one or two sentences and stop. Do not invent or assume facts, product capabilities, features, customers, metrics, numbers, pricing, integrations, timelines, or next steps that are not explicitly supported by the provided context. Do not pad with generic sales language to disguise missing information. Do not fabricate steps in the REASONING PROCESS below — if Step 2 cannot extract real facts from the context, stop at the honest short answer. The persuasive style, reasoning structure, and answer-style directives below apply ONLY when the context contains sufficient grounded information; when it does not, an honest short answer takes priority over all of them.

--------------------------------
SALES CONTEXT
--------------------------------
{sales_context_str}

--------------------------------
CUSTOM INSTRUCTIONS
--------------------------------
{settings.custom_prompt or "N/A"}

--------------------------------
EXECUTIVE SNAPSHOT (BUYER PRIORITIES)
--------------------------------
{settings.executive_snapshot or "N/A"}

--------------------------------
STRATEGIC KEYWORDS
--------------------------------
{keywords_str or "None"}

--------------------------------
AUDIENCE PROFILE
--------------------------------
{persona_str or "N/A"}

--------------------------------
ANSWER STYLE DIRECTIVES (apply all)
--------------------------------
{styles_str or "None"}

--------------------------------
REASONING PROCESS
--------------------------------

Step 1 — Identify the buyer's goal or challenge.

Step 2 — Extract relevant facts from the context.

Step 3 — Reason step-by-step about how the seller’s offering helps.

Step 4 — Determine the business value for the buyer.

--------------------------------
RESPONSE GUIDELINES
--------------------------------

0. Match the AUDIENCE PROFILE tone and follow every ANSWER STYLE DIRECTIVE above.
1. Start naturally — directly addressing the buyer’s situation or priorities.
2. Explicitly connect the buyer’s goals/challenges to the seller’s offering.
3. Be deeply personalized to the buyer’s context, reference the buyer’s priorities, executive snapshot, or strategic keywords naturally
4. Use examples or scenarios tailored to the buyer’s industry/domain.
5. Show strategic reasoning:
   - Why this matters now
   - What risk exists if nothing changes
   - How this creates competitive advantage
6. Show measurable or business-level impact wherever possible.
7. Provide tactical next steps (e.g., pilot, implementation phase, workshop, stakeholder alignment).
8. Close with forward momentum, not a generic conclusion.
9. Never mention that you are an AI.
10. Never sound like documentation or a generic report.

Make the answer persuasive, structured, and deeply buyer-focused.
"""

def store_context(context_hash: str, context_text: str, sources: list):
    _CONTEXT_CACHE[context_hash] = {
        "context": context_text,
        "sources": sources
    }

def get_context(context_hash: str):
    return _CONTEXT_CACHE.get(context_hash)


# ---------------------------------------------------------------------------
# Cognitive deep-dive pipeline (background / polled).
#
# Lean "Bundle B" without hosted infra or the agentic retrieval loop:
#   wide retrieve  →  flash rerank  →  grounded generation (persona tone, no
#   citations)  →  local embedding grounding check  →  conditional single revise.
#
# The whole thing is bounded by COGNITIVE_TIMEOUT_SECONDS and every stage
# degrades gracefully, so the pipeline nearly always returns an answer.
# ---------------------------------------------------------------------------
COGNITIVE_WIDE_TOP_K = 40          # candidates pulled for the deep answer
COGNITIVE_RERANK_TOP_K = 12        # kept after flash rerank
COGNITIVE_TIMEOUT_SECONDS = 35     # hard bound on generation only (network calls); verify runs separately
COGNITIVE_VERIFY_TIMEOUT = 10      # separate bound on the (best-effort) background grounding+revise step

# Grounding safety-net. A substantive answer sentence whose best cosine match to
# any retrieved chunk is below the threshold is "unsupported". We only trigger a
# (costly) revise when a large fraction is unsupported, so a mis-tuned threshold
# can't cause needless revises. Both values want calibration on real data.
GROUNDING_THRESHOLD = 0.72
GROUNDING_FLAG_FRACTION = 0.34
GROUNDING_MIN_SENTENCE_WORDS = 6

COGNITIVE_RERANK_SYSTEM_PROMPT = (
    "You are a relevance ranker. Given a QUESTION and a numbered list of PASSAGES, "
    "return ONLY a JSON array of passage numbers ordered from most to least relevant "
    "for answering the question. Include a passage only if it is genuinely relevant. "
    "Example: [3, 0, 7]"
)

COGNITIVE_VERIFY_SYSTEM_PROMPT = (
    "You are editing a sales answer for factual grounding. Given the CONTEXT and the "
    "DRAFT, remove or correct any statement that is not supported by the CONTEXT. "
    "Preserve the tone, persona, structure, and persuasiveness of the draft EXACTLY — "
    "change wording only where a claim is unsupported. Do not add citations, source "
    "references, or disclaimers. Return only the revised answer text."
)


async def _flash_rerank(question: str, chunks: list, top_n: int) -> list:
    """Rerank retrieved chunks with a single flash call. Returns the top_n chunks
    in relevance order; falls back to original order on any failure."""
    if len(chunks) <= top_n:
        return chunks

    numbered = "\n".join(
        f"[{i}] {(c.get('content') or '')[:400]}" for i, c in enumerate(chunks)
    )
    prompt = f"QUESTION:\n{question}\n\nPASSAGES:\n{numbered}"

    try:
        raw = await call_gemini_flash_llm(
            prompt=prompt, system_prompt=COGNITIVE_RERANK_SYSTEM_PROMPT
        )
        match = re.search(r"\[[\s\d,]*\]", raw or "")
        order = json.loads(match.group(0)) if match else []
        seen, selected = set(), []
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                seen.add(idx)
                selected.append(chunks[idx])
            if len(selected) >= top_n:
                break
        if selected:
            return selected
    except Exception as e:
        logger.warning(f"[COG] flash rerank failed ({e}); using retrieval order")

    return chunks[:top_n]


async def _resolve_chunk_embeddings(chunk_texts: list, chunk_embeddings: list | None, expected_dim: int) -> np.ndarray:
    """Build the chunk-side embedding matrix for _grounding_fraction, reusing
    each precomputed vector in chunk_embeddings where valid and recomputing
    only the rest (missing, or wrong dimensionality) in a single batched
    get_embeddings call -- not one call per chunk."""
    if not chunk_embeddings:
        return np.asarray(await get_embeddings(chunk_texts), dtype=np.float32)

    resolved: list = [None] * len(chunk_texts)
    missing_idx = []
    for i, emb in enumerate(chunk_embeddings):
        arr = np.asarray(emb, dtype=np.float32) if emb is not None else None
        if arr is not None and arr.ndim == 1 and arr.shape[0] == expected_dim:
            resolved[i] = arr
        else:
            missing_idx.append(i)

    if missing_idx:
        fresh = np.asarray(
            await get_embeddings([chunk_texts[i] for i in missing_idx]), dtype=np.float32
        )
        for pos, i in enumerate(missing_idx):
            resolved[i] = fresh[pos] if fresh.size else np.zeros(expected_dim, dtype=np.float32)

    return np.stack(resolved) if resolved else np.zeros((0, expected_dim), dtype=np.float32)


async def _grounding_fraction(answer: str, chunk_texts: list, chunk_embeddings: list | None = None) -> float:
    """Fraction of substantive answer sentences not supported by any chunk, via
    e5 embedding cosine. Returns 0.0 (treat as grounded) if it can't run.

    chunk_embeddings, if given, are precomputed vectors (one per chunk_texts
    entry, or None where unavailable -- e.g. the RPC-fallback retrieval path
    doesn't return an embedding column) threaded through from retrieval, so
    this skips re-embedding chunk content it already has a vector for.
    A precomputed vector is only reused if its dimensionality matches the
    sentence embeddings' -- the locked-down small-model demo path's chunks
    carry a different (384 vs 1024) embedding space, and reusing those here
    would silently corrupt the cosine comparison rather than just being
    slower, so a mismatch is treated the same as "missing" and recomputed."""
    if not answer or not chunk_texts:
        return 0.0

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", answer)
        if len(s.split()) >= GROUNDING_MIN_SENTENCE_WORDS
    ]
    if not sentences:
        return 0.0

    sent_emb = np.asarray(await get_embeddings(sentences), dtype=np.float32)
    if sent_emb.size == 0:
        return 0.0
    expected_dim = sent_emb.shape[1]

    chunk_emb = await _resolve_chunk_embeddings(chunk_texts, chunk_embeddings, expected_dim)
    if chunk_emb.size == 0:
        return 0.0

    # A degenerate (zero-norm) embedding normalizes to NaN/inf, which makes the
    # matmul emit "divide by zero / invalid value" warnings and poisons the score.
    # We can't assess such a sentence, so treat it as supported (never flag) to
    # avoid spurious revises, and sanitize both sides so the matmul stays clean.
    sent_finite = np.isfinite(sent_emb).all(axis=1)
    sims = (
        np.nan_to_num(sent_emb, posinf=0.0, neginf=0.0)
        @ np.nan_to_num(chunk_emb, posinf=0.0, neginf=0.0).T
    )
    # Embeddings are L2-normalized, so the dot product is cosine similarity.
    best = sims.max(axis=1)
    best = np.where(sent_finite, best, 1.0)
    unsupported = int((best < GROUNDING_THRESHOLD).sum())
    return unsupported / len(sentences)


async def _verify_and_revise(
    answer: str, context_text: str, reranked_texts: list, reranked_embeddings: list | None = None
) -> str:
    """Grounding check + one conditional revise. Returns the (possibly revised)
    answer, and always falls back to the original draft on any internal failure."""
    flagged = await _grounding_fraction(answer, reranked_texts, reranked_embeddings)
    if flagged <= GROUNDING_FLAG_FRACTION:
        return answer
    logger.warning(f"[COG] {flagged:.0%} of sentences unsupported → revising")
    try:
        revised = await call_gemini_flash_llm(
            prompt=f"CONTEXT:\n{context_text}\n\nDRAFT:\n{answer}",
            system_prompt=COGNITIVE_VERIFY_SYSTEM_PROMPT,
        )
        return revised or answer
    except Exception as e:
        logger.warning(f"[COG] revise failed ({e}); keeping draft")
        return answer


async def _run_cognitive_pipeline(
    question: str,
    live_context_text: str,
    settings,
    user_id: str | None,
    source_ids: list | None,
    client_id: str | None,
):
    """Returns (answer, context_used, reranked_texts, reranked_embeddings). Each
    stage degrades gracefully so a generation almost always happens even if
    retrieval/rerank fail. Verification is intentionally NOT done here (see
    generate_cognitive_answer)."""
    context_text = live_context_text
    reranked_texts = None
    reranked_embeddings = None

    # 1) Wide retrieve — falls back to the live context if unavailable.
    if user_id:
        try:
            wide = await build_rag_context(
                question, user_id, source_ids=source_ids,
                client_id=client_id, top_k=COGNITIVE_WIDE_TOP_K,
                query_augment=_build_query_augment(settings),
            )
            if wide and wide.get("chunks"):
                # 2) Flash rerank down to the strongest chunks.
                selected = await _flash_rerank(question, wide["chunks"], COGNITIVE_RERANK_TOP_K)
                context_text = format_context(selected)
                reranked_texts = [c.get("content") or "" for c in selected]
                # Carried through so the grounding check below can skip
                # re-embedding chunk content it already has a vector for
                # (warm-index hits carry one; RPC-fallback hits don't -- see
                # _resolve_chunk_embeddings' per-entry fallback).
                reranked_embeddings = [c.get("embedding") for c in selected]
        except Exception as e:
            logger.warning(f"[COG] wide retrieve/rerank failed ({e}); using live context")

    # 3) Grounded generation — reuse the existing persona/tone reasoning prompt
    #    (natural, persuasive, no citations). Verification is deliberately NOT run
    #    here: it's a CPU-bound (e5) step that can't be reliably cancelled under
    #    Cloud Run CPU throttling, so running it inline risks the outer timeout
    #    discarding an answer that already generated. Instead the caller persists
    #    this answer first, then refines it in the background.
    answer = await call_gemini_flash_llm(
        prompt=f"Context:\n{context_text}\n\nQuestion:\n{question}",
        system_prompt=build_reasoning_sales_prompt(settings),
    )
    if not answer:
        return None, context_text, None, None

    return answer, context_text, reranked_texts, reranked_embeddings


async def _verify_and_update(
    cognitive_key: str, answer: str, context_text: str, reranked_texts: list, reranked_embeddings: list | None = None
):
    """Best-effort grounding check + revise, run AFTER the answer is already
    persisted as 'done'. If it produces a revision, update the row; if it times
    out or errors (e.g. CPU-starved embeddings), the persisted draft simply
    stands. It can never make cognitive fail or hang the poll."""
    if not reranked_texts:
        return
    try:
        revised = await asyncio.wait_for(
            _verify_and_revise(answer, context_text, reranked_texts, reranked_embeddings),
            timeout=COGNITIVE_VERIFY_TIMEOUT,
        )
        if revised and revised != answer:
            await _cog_write(cognitive_key, "done", answer=revised)
    except Exception as e:
        logger.warning(f"[COG] background verify skipped ({type(e).__name__}); keeping draft")


async def generate_cognitive_answer(
    cognitive_key: str,
    question: str,
    context_text: str,
    settings,
    endpoint: str | None = None,
    client_id: str | None = None,
    meeting_log_id: str | None = None,
    user_id: str | None = None,
    source_ids: list | None = None,
):
    try:
        # Generation only (wide retrieve → flash rerank → grounded generation),
        # hard-bounded. Verification is intentionally excluded from this bound so a
        # slow/uncancellable grounding pass can't discard an answer that generated.
        response, used_context, reranked_texts, reranked_embeddings = await asyncio.wait_for(
            _run_cognitive_pipeline(
                question, context_text, settings, user_id, source_ids, client_id
            ),
            timeout=COGNITIVE_TIMEOUT_SECONDS,
        )

        if not response:
            raise ValueError("Cognitive pipeline returned an empty response")

        # Persist the answer the instant it's generated — it's now available to the
        # poll regardless of what the (best-effort) verify step does next.
        await _cog_write(cognitive_key, "done", answer=response)

        # Refine in the background; never blocks availability, never fails the answer.
        asyncio.create_task(
            _verify_and_update(cognitive_key, response, used_context, reranked_texts, reranked_embeddings)
        )

        # Fire-and-forget RAG classifier. Runs after the answer is cached
        # so it never delays the /cognitive polling response. Safe to skip
        # if we don't have the endpoint tag (legacy callers). Judged against the
        # context the pipeline actually used (the reranked wide set).
        if endpoint:
            asyncio.create_task(
                classify_and_log(
                    question=question,
                    context_text=used_context,
                    answer=response,
                    endpoint=endpoint,
                    client_id=client_id,
                    meeting_log_id=meeting_log_id,
                )
            )

    except asyncio.TimeoutError as e:
        logger.error(f"Cognitive generation timeout for key {cognitive_key}", exc_info=True)

        await _cog_write(cognitive_key, "failed", error="timeout")

    except RateLimitError as e:
        logger.error(f"OpenAI rate limit hit for cognitive key {cognitive_key}", exc_info=True)

        await _cog_write(cognitive_key, "failed", error="rate_limit")

    except APIConnectionError as e:
        logger.error(f"OpenAI connection error for cognitive key {cognitive_key}", exc_info=True)

        await _cog_write(cognitive_key, "failed", error="connection_error")

    except APIError as e:
        logger.error(f"OpenAI API error for cognitive key {cognitive_key}: {e}", exc_info=True)

        await _cog_write(cognitive_key, "failed", error="api_error")

    except Exception as e:
        logger.error(f"Unexpected cognitive generation error for key {cognitive_key}: {e}", exc_info=True)

        await _cog_write(cognitive_key, "failed", error="unknown_error")

    finally:
        _COGNITIVE_TASKS.pop(cognitive_key, None)

async def launch_cognitive_background(
    cognitive_key: str,
    question: str,
    context_text: str,
    settings,
    endpoint: str | None = None,
    client_id: str | None = None,
    meeting_log_id: str | None = None,
    user_id: str | None = None,
    source_ids: list | None = None,
):
    async with _COGNITIVE_LOCK:
        if cognitive_key in _COGNITIVE_TASKS:
            return

        # No pending-row write here: it used to be awaited before returning,
        # which blocked the live answer's StreamingResponse on a Supabase
        # round trip for every single request. get_cognitive() below now
        # treats "key not in DB but in _COGNITIVE_TASKS" as pending, so the
        # first real DB write is generate_cognitive_answer's own "done"/
        # "failed" -- also sidesteps a clobber race where a delayed pending
        # write could land after and overwrite an already-terminal row.
        task = asyncio.create_task(
            generate_cognitive_answer(
                cognitive_key,
                question,
                context_text,
                settings,
                endpoint=endpoint,
                client_id=client_id,
                meeting_log_id=meeting_log_id,
                user_id=user_id,
                source_ids=source_ids,
            )
        )

        _COGNITIVE_TASKS[cognitive_key] = task


_KYC_OVERLAY_FIELDS = (
    "seller_name", "seller_company", "client_company", "client_names",
    "company_url", "products_services", "product_domain", "sub_domains",
    "executive_snapshot", "strategic_keywords",
)


async def _resolve_effective_settings(
    user_id: str,
    base_settings: SettingsModel,
    client_id: str | None,
    kyc_id: str | None,
) -> SettingsModel:
    """
    Resolve the buyer context per client.

    The buyer (client_names/client_company) must never come from the shared
    global user_configs — that caused one client's buyer to leak into every
    other client's answers. When a client is in scope, the buyer is sourced
    only from the per-(client, kyc) overlay:
      - KYC selected      -> overlay row (client_id, kyc_id)
      - no KYC selected   -> per-client manual row (client_id, MANUAL_KYC_ID)
      - neither populated  -> empty buyer context -> generic, name-less answer

    With no client_id at all (legacy callers), behavior is unchanged.
    """
    if not client_id:
        return base_settings

    effective_kyc_id = kyc_id or MANUAL_KYC_ID

    # Start from base, but drop every buyer/KYC-derived field so the shared
    # global user_configs can't leak one client's identity into another's
    # answers. These are all sourced per-client from the overlay below; if the
    # overlay doesn't supply them, a name-less/generic answer is correct.
    # Seller fields are intentionally NOT blanked — the seller is the user's own
    # company and is global by design.
    # Type is preserved per field so SettingsModel validation never fails back to
    # the (leaky) base settings.
    data = base_settings.model_dump(by_alias=False)
    _BUYER_SCOPED_FIELDS = (
        "client_names", "client_company", "company_url", "sub_domains",
        "products_services", "product_domain", "executive_snapshot",
        "strategic_keywords",
    )
    for f in _BUYER_SCOPED_FIELDS:
        data[f] = [] if isinstance(data.get(f), list) else ""

    overlay = await get_client_kyc_config(user_id, client_id, effective_kyc_id)
    if overlay:
        for f in _KYC_OVERLAY_FIELDS:
            val = overlay.get(f)
            if val is None:
                continue
            # Skip empty strings so an unset overlay column doesn't clobber a real base value.
            if isinstance(val, str) and val.strip() == "":
                continue
            data[f] = val

    # strategic_keywords is List[str] in the model, but base-blanking and the KYC
    # overlay (a DB text column) can leave it as a string ("" or "a, b, c"). Coerce
    # to a list so a type mismatch never fails validation and silently drops the
    # ENTIRE per-client overlay (which then leaked base/generic personalization).
    sk = data.get("strategic_keywords")
    if isinstance(sk, str):
        data["strategic_keywords"] = [s.strip() for s in sk.split(",") if s.strip()]

    try:
        return SettingsModel(**data)
    except Exception as e:
        logger.warning(f"Failed to resolve client overlay (using base settings): {e}")
        return base_settings


def _persona_styles_fingerprint(settings: SettingsModel) -> str:
    persona = (settings.selected_persona or "").strip()
    styles = settings.selected_answer_styles or []
    if isinstance(styles, str):
        styles = [styles]
    return f"{persona}|{','.join(sorted(styles))}"


@router.post("/ask/regular")
async def ask_regular(
    request: AskRequest,
    user_id: str = Depends(get_user_id_from_token),
    settings: SettingsModel = Depends(get_current_user_settings)
):
    asyncio.create_task(cleanup_expired_cognitive())

    # Overlay per-(client, kyc) KYC fields on top of base settings when provided.
    # Falls back to base settings for clients with no KYC, preserving legacy behavior.
    settings = await _resolve_effective_settings(
        user_id, settings, request.client_id, request.kyc_id
    )

    question = request.question.strip()
    # KYC-steered retrieval: bias results toward the buyer's priorities. No added
    # latency (just a richer query) and no reranker on the live answer — the extra
    # flash call added visible time-to-first-token on the streamed response.
    query_augment = _build_query_augment(settings)
    rag = await build_rag_context(
        question, user_id,
        source_ids=request.source_ids,
        client_id=request.client_id,
        top_k=RAG_TOP_K_LIVE,
        query_augment=query_augment,
    )

    if not rag:
        asyncio.create_task(_log_no_rag_question(question, "regular", request.client_id))
        # Zero-context RAG failure — log directly (no cognitive answer will
        # be generated, so the classifier hook in launch_cognitive_background
        # never fires for this case).
        asyncio.create_task(
            classify_and_log(
                question=question,
                context_text="",
                answer="I could not find relevant documents.",
                endpoint="regular",
                client_id=request.client_id,
                meeting_log_id=None,
            )
        )
        async def no_docs():
            yield "I could not find relevant documents."
        return StreamingResponse(no_docs(), media_type="text/event-stream")

    context_text = rag["context_text"]
    context_hash = rag["context_hash"]
    sources_json = json.dumps(rag["sources"])

    # Include client/kyc scope and the resolved buyer in the cache key so a
    # cached cognitive answer never leaks across clients (or across a buyer-name
    # override). Without this, switching clients but asking a similar question
    # returned the previous client's cached answer.
    cognitive_key = hashlib.sha256(
        (
            f"{user_id}:{request.client_id}:{request.kyc_id}:"
            f"{settings.client_company}:{settings.client_names}:"
            f"{question}:{context_hash}:{_persona_styles_fingerprint(settings)}"
        ).encode()
    ).hexdigest()

    await launch_cognitive_background(
        cognitive_key,
        question,
        context_text,
        settings,
        endpoint="regular",
        client_id=request.client_id,
        meeting_log_id=None,
        user_id=user_id,
        source_ids=request.source_ids,
    )

    system_prompt = build_cognitive_prompt(settings)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
    ]

    return StreamingResponse(
        stream_groq_response(messages),
        media_type="text/event-stream",
        headers={
            "X-Sources": sources_json,
            "X-Context-Hash": context_hash,
            "X-Cognitive-Key": cognitive_key,
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )



@router.get("/cognitive")
async def get_cognitive(cognitive_key: str):
    # Strip accidental newline / whitespace
    cognitive_key = cognitive_key.strip()

    cached = await _cog_read(cognitive_key)

    if not cached:
        # No row yet (nothing's been written -- generation only writes on
        # done/failed now, see launch_cognitive_background). Same-instance
        # fallback: if the task is still running in this process, it's
        # genuinely pending, not missing. Cross-instance, this still 404s
        # until the task's own write lands.
        if cognitive_key in _COGNITIVE_TASKS:
            return {"status": "pending"}
        raise HTTPException(status_code=404, detail="Cognitive key not found.")

    if cached["status"] == "done":
        return {
            "status": "done",
            "answer": cached["answer"]
        }

    if cached["status"] == "failed":
        return {
            "status": "failed"
        }

    # Still pending — but if it's been pending far longer than a generation can
    # take, the task died (e.g. instance recycled mid-generation); report failed
    # so the client stops polling and can retry instead of looping forever.
    if _pending_is_stale(cached.get("updated_at")):
        return {
            "status": "failed"
        }

    return {
        "status": "pending"
    }