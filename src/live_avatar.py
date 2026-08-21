import os
import json
import base64
import asyncio
import logging
import re
import sys
import time
import uuid
import hashlib
import hmac
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Callable, Awaitable
from urllib.parse import urlencode, quote

from src.supabase_client import get_user_keywords_and_products
from src.agent_policy import (
    AgentState,
    EchoSuppressor,
    FinalUtteranceBuffer,
    FloorState,
    InterjectionJudgment,
    SpeechGovernor,
    SustainedSpeechDetector,
    apply_validated_corrections,
    build_entity_catalog,
    closest_entities,
    compose_reply,
    detect_mute_command,
    estimate_speech_seconds,
    is_directly_addressed,
    evaluate_turn,
    is_probably_incomplete,
    looks_like_followup,
    MAX_QUESTION_WORDS,
    needs_context_resolution,
    normalize_reply,
    requires_company_knowledge,
)

from src.call_intelligence import CallIntelligence

import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

# Setup logging: terminal keeps today's exact output, and every process start
# (a fresh `npm start` launch, or a --reload respawn on file save) also gets
# its own on-disk copy at logs/app.log -- opened in "w" (truncate), not
# append or size-rotation, so each run starts a clean file instead of
# yesterday's session still being at the top when you scroll for today's.
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_file_handler = logging.FileHandler(_LOG_DIR / "app.log", mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_file_handler.setLevel(logging.INFO)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
_console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
logger = logging.getLogger("LiveAvatar-Spiked")

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API") or os.getenv("DEEPGRAM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
SPIKED_BACKEND_URL = os.getenv("SPIKED_BACKEND_URL", "https://spikedai-production-application-409019309412.us-central1.run.app")
RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")
# Defaults to ap-northeast-1 for local dev, where RECALL_API_KEY has
# historically been region-bound to that workspace. This is NOT a safe
# assumption for every deploy -- Recall API keys are region-bound, and a
# 401 with "Invalid API token... might be for another Recall region" means
# whichever RECALL_API_KEY is actually in play doesn't match this default.
# The production Cloud Run deploy sets RECALL_BASE_URL explicitly in
# cloudbuild.yaml instead of relying on this fallback -- keep the two in
# sync if either the key or its region ever changes.
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://ap-northeast-1.recall.ai")
RECALL_WEBHOOK_URL = os.getenv(
    "RECALL_WEBHOOK_URL", 
    "https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript"
)
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
LIVEAVATAR_AVATAR_ID = "9650a758-1085-4d49-8bf3-f347565ec229"
LIVEAVATAR_SANDBOX = os.getenv("LIVEAVATAR_SANDBOX", "false").lower() == "true"
# LiveAvatar auto-closes a session that sees no join/interaction for a while
# (docs.liveavatar.com/reference/keep_session_alive_v1_sessions_keep_alive_post).
# Tom is silent unless addressed, so a long idle stretch in a meeting would
# otherwise trip that timeout — ping it well under the observed ~3 minute cliff.
LIVEAVATAR_KEEPALIVE_INTERVAL_S = float(os.getenv("LIVEAVATAR_KEEPALIVE_INTERVAL_S", "60"))
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", 
    "https://spiked-ai-liveavatar-409019309412.us-central1.run.app"
)
AGENT_BARGE_IN_MS = int(os.getenv("AGENT_BARGE_IN_MS", "700"))
AGENT_ENDPOINTING_MS = int(os.getenv("AGENT_ENDPOINTING_MS", "300"))
# Deepgram rejects the whole connection with HTTP 400 when utterance_end_ms is
# below 1000, which manifests as an agent that hears nothing at all rather than
# as a config error. Clamped rather than trusted: a tuning value must not be
# able to silently deafen the bot.
DEEPGRAM_MIN_UTTERANCE_END_MS = 1000
_requested_utterance_end_ms = int(os.getenv("AGENT_UTTERANCE_END_MS", "1000"))
AGENT_UTTERANCE_END_MS = max(_requested_utterance_end_ms, DEEPGRAM_MIN_UTTERANCE_END_MS)
if _requested_utterance_end_ms < DEEPGRAM_MIN_UTTERANCE_END_MS:
    logger.warning(
        "[Deepgram] AGENT_UTTERANCE_END_MS=%d is below the API minimum; using %d",
        _requested_utterance_end_ms,
        AGENT_UTTERANCE_END_MS,
    )
AGENT_MAX_REPLY_WORDS = int(os.getenv("AGENT_MAX_REPLY_WORDS", "45"))
DEFAULT_BOT_NAME = os.getenv("BOT_NAME", "Tom").strip() or "Tom"
# Retrieval budget on the live speak path. Nobody is waiting on a prefetch, so
# that path passes AGENT_RAG_PREFETCH_TIMEOUT_S instead.
AGENT_RAG_TIMEOUT_S = float(os.getenv("AGENT_RAG_TIMEOUT_S", "12"))
# Off-path budget for the Level 1 insight prefetch (see _maybe_prefetch_insight
# below): nobody is waiting on it, so it can afford to wait longer than the
# live speak path before giving up. Was referenced but never defined — every
# prefetch NameError'd and got silently swallowed by its own except block.
AGENT_RAG_PREFETCH_TIMEOUT_S = float(os.getenv("AGENT_RAG_PREFETCH_TIMEOUT_S", "12"))

# TEMPORARY (see TEMP_WIRING.md): when the backend's live Groq answer stream
# errors, fall back to polling its slower cognitive (background) answer, which
# is currently the only healthy generation path. Remove once Groq is restored.
# TEMP (see TEMP_WIRING.md): off by default. The live Groq path is healthy as
# of 2026-08-17 and ~4s; the cognitive fallback adds ~16s. Flip to "true" only
# if the live stream starts erroring platform-wide again.
AGENT_COGNITIVE_FALLBACK = os.getenv("AGENT_COGNITIVE_FALLBACK", "false").lower() == "true"
AGENT_COGNITIVE_FALLBACK_TIMEOUT_S = float(os.getenv("AGENT_COGNITIVE_FALLBACK_TIMEOUT_S", "25"))
AGENT_COGNITIVE_POLL_INTERVAL_S = float(os.getenv("AGENT_COGNITIVE_POLL_INTERVAL_S", "1.5"))

# The backend's short error yields from a failed live answer stream. Kept in one
# place: query_spiked_rag uses them to trigger the cognitive fallback, and the
# reply path uses them to refuse composing a reply out of a non-answer.
BACKEND_STREAM_ERROR_MARKERS = ("request timed out", "service unavailable", "failed to get response")

# Per-run cache of resolved source_ids. Resolving them costs two backend round
# trips (~1s+) on the speak path; the client's document set changes on the scale
# of minutes, not turns.
_SOURCE_IDS_CACHE: Dict[str, Any] = {}
SOURCE_IDS_CACHE_TTL_S = float(os.getenv("AGENT_SOURCE_IDS_CACHE_TTL_S", "300"))

# Shared connection pool for backend calls on the speak path. A per-call
# AsyncClient pays a fresh TLS handshake to Cloud Run every turn; keeping the
# connection warm shaves a few hundred ms off every RAG round trip. Timeouts
# are passed per-request, so one pool serves both the live and prefetch paths.
_backend_http: Optional[httpx.AsyncClient] = None

def _get_backend_http() -> httpx.AsyncClient:
    global _backend_http
    if _backend_http is None or _backend_http.is_closed:
        _backend_http = httpx.AsyncClient(
            timeout=AGENT_RAG_TIMEOUT_S,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=120),
        )
    return _backend_http

# Level 1: the agent noticed something it could speak to but was not invited.
# The cue goes to the rep and the answer is warmed silently. Rate-limited so a
# technical stretch of conversation does not fire retrieval on every sentence.
AGENT_INSIGHT_COOLDOWN_S = float(os.getenv("AGENT_INSIGHT_COOLDOWN_S", "25"))
AGENT_INSIGHT_TTL_S = float(os.getenv("AGENT_INSIGHT_TTL_S", "120"))

# Level 1.5: autonomous interjection. A stricter judgment on top of the Level 1
# heuristic above — "would a solution architect actually volunteer here" vs.
# "the KB happens to cover this" — gated per-run (autospeak_enabled) and
# capped hard, since this is the only behavior where the agent takes the floor
# with no human in the loop. Cooldown is deliberately longer than the Level 1
# cue cooldown: cueing is cheap, speaking isn't.
AGENT_AUTOSPEAK_MIN_CONFIDENCE = float(os.getenv("AGENT_AUTOSPEAK_MIN_CONFIDENCE", "0.75"))
AGENT_AUTOSPEAK_COOLDOWN_S = float(os.getenv("AGENT_AUTOSPEAK_COOLDOWN_S", "90"))
AGENT_AUTOSPEAK_MAX_PER_RUN = int(os.getenv("AGENT_AUTOSPEAK_MAX_PER_RUN", "3"))

# Turn detection. Fragments from one speaker are merged before the gate runs, so a
# single sentence split by a pause cannot produce two replies.
AGENT_TURN_MERGE_MS = int(os.getenv("AGENT_TURN_MERGE_MS", "250"))
AGENT_TURN_MERGE_INCOMPLETE_MS = int(os.getenv("AGENT_TURN_MERGE_INCOMPLETE_MS", "700"))
# Floor control. Opt-in: 0 means the wake name is required on every single turn,
# which is the behavior the product expects. Raise it only to allow nameless
# question-shaped continuations shortly after the agent stops speaking.
AGENT_FOLLOWUP_WINDOW_MS = int(os.getenv("AGENT_FOLLOWUP_WINDOW_MS", "8000"))
AGENT_MAX_FOLLOWUPS = int(os.getenv("AGENT_MAX_FOLLOWUPS", "1"))
# The follow-up window shrinks by this factor on each successive nameless
# follow-up, down to AGENT_MIN_FOLLOWUP_WINDOW_MS, instead of hard-cutting at a
# fixed count — a long on-topic exchange tapers rather than hitting a cliff.
AGENT_FOLLOWUP_DECAY_RATE = float(os.getenv("AGENT_FOLLOWUP_DECAY_RATE", "0.6"))
AGENT_MIN_FOLLOWUP_WINDOW_MS = int(os.getenv("AGENT_MIN_FOLLOWUP_WINDOW_MS", "3000"))
# The window's base also stretches with how long the agent's last reply took
# to say (last_reply_seconds * this factor), so a long answer buys the
# listener proportionally more think-time than a short one instead of racing
# the same flat clock regardless of what was just said. 0 keeps it flat.
AGENT_FOLLOWUP_WINDOW_REPLY_SCALE = float(os.getenv("AGENT_FOLLOWUP_WINDOW_REPLY_SCALE", "0.5"))
# Speech governor: hard ceiling on reply frequency and repetition.
AGENT_REPLY_COOLDOWN_MS = int(os.getenv("AGENT_REPLY_COOLDOWN_MS", "2000"))
AGENT_MAX_REPLIES_PER_WINDOW = int(os.getenv("AGENT_MAX_REPLIES_PER_WINDOW", "4"))
AGENT_REPLY_WINDOW_S = float(os.getenv("AGENT_REPLY_WINDOW_S", "30"))
# Echo suppression: how close a transcript must be to the agent's own words.
AGENT_ECHO_SIMILARITY = float(os.getenv("AGENT_ECHO_SIMILARITY", "0.72"))
AGENT_ECHO_TAIL_S = float(os.getenv("AGENT_ECHO_TAIL_S", "2.5"))
# Watchdog: force LISTENING if the avatar never reports back.
# Renders the "what the agent heard" panel into the meeting camera feed, which
# every participant can see. Off unless explicitly opted into for local debugging.
AGENT_DEBUG_OVERLAY = os.getenv("AGENT_DEBUG_OVERLAY", "false").lower() == "true"
AGENT_SPEAK_START_TIMEOUT_S = float(os.getenv("AGENT_SPEAK_START_TIMEOUT_S", "4"))
AGENT_SPEAK_MAX_OVERRUN_S = float(os.getenv("AGENT_SPEAK_MAX_OVERRUN_S", "6"))

# Configure Modern Google GenAI Client
# Using gemini-3.5-flash-lite for cost-effective, low-latency function calling
gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("[Google GenAI] Official modern SDK client initialized")
    except Exception as e:
        logger.warning(f"[Google GenAI] Could not initialize client: {e}")
else:
    logger.warning("GEMINI_API_KEY is not set in environment")

app = FastAPI(title="LiveAvatar Spiked AI Orchestrator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session and run registry
_ACTIVE_RUNS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Helper: Extract User ID from Supabase JWT
# ---------------------------------------------------------------------------

def extract_user_id_from_jwt(token: str) -> str:
    """Extracts the 'sub' claim (user_id) from Supabase JWT without needing secret."""
    try:
        parts = token.strip().replace("Bearer ", "").split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
            return payload.get("sub") or payload.get("user_id") or "unknown_user"
    except Exception as e:
        logger.warning(f"Could not parse user_id from JWT: {e}")
    return "unknown_user"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CreateLiveAvatarRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None, description="HeyGen/LiveAvatar avatar identifier")
    quality: str = Field(default="medium", description="Video quality: low, medium (720p), high (1080p)")
    is_sandbox: bool = Field(default=LIVEAVATAR_SANDBOX, description="Sandbox mode for testing")
    mode: str = Field(default="FULL", description="FULL ($0.20/min, includes TTS) or LITE ($0.10/min, BYOTTS)")
    user_id: Optional[str] = Field(default=None, description="User ID for fetching context")
    client_id: Optional[str] = Field(default=None, description="Client ID for fetching context")
    token: Optional[str] = Field(default=None, description="Auth token")
    bot_name: str = Field(default=DEFAULT_BOT_NAME, description="Meeting display and invocation name")

class CreateBotWithLiveAvatarRequest(BaseModel):
    meeting_url: str = Field(..., description="Zoom, Google Meet, or MS Teams URL")
    user_id: Optional[str] = Field(default=None, description="Supabase user ID")
    token: Optional[str] = Field(default=None, description="User's Supabase JWT access token for document RAG")
    client_id: Optional[str] = Field(default=None, description="Client/Company scope identifier")
    kyc_id: Optional[str] = Field(default=None, description="Active KYC overlay for buyer-aware answers; backend falls back to the manual overlay when absent")
    bot_name: str = Field(default=DEFAULT_BOT_NAME, description="Name of the bot in the meeting")
    avatar_id: Optional[str] = Field(default=None, description="Specific LiveAvatar avatar ID")
    autospeak_enabled: bool = Field(default=False, description="Level 1.5: let the agent take the floor unprompted for high-value moments, capped per run")


class TranscriptCorrection(BaseModel):
    raw: str = ""
    replacement: str = ""
    confidence: float = 0.0


class TurnAnalysis(BaseModel):
    response_action: Literal["respond", "acknowledge", "silent"] = "respond"
    intent: Literal["company_knowledge", "meeting_context", "social", "command", "coaching"]
    resolved_query: str
    corrections: List[TranscriptCorrection] = Field(default_factory=list)


class TurnAnalysisAndReply(TurnAnalysis):
    """TurnAnalysis plus an optional draft reply, for the single-shot classify
    path (agent_policy: coaching/meeting_context/social/command).

    company_knowledge needs retrieved facts before it can answer, so the
    reply fields are left empty by prompt instruction whenever intent is
    company_knowledge or response_action isn't "respond" — those turns still
    take the classification result down the existing RAG or short-circuit
    path. For every other addressed turn, this collapses what used to be two
    sequential Gemini calls (classify, then compose) into one.
    """

    answer: str = Field(
        default="",
        description="The direct answer, for spoken delivery. Empty if intent is company_knowledge or response_action isn't 'respond'.",
    )
    bridge: str = Field(default="", description="At most one short clause connecting the answer to what the speaker is deciding. Empty when it would only pad.")
    next_question: str = Field(default="", description="One question opening the next useful step. Empty for greetings, audio checks, commands, and short confirmations.")


class GroundedReply(BaseModel):
    """A spoken turn split into its three parts so each gets its own budget.

    Generated as separate fields rather than one string because the closing
    question is the agent's signature and is also the last thing produced, which
    makes it the first casualty of any length cap applied to the whole reply.
    """

    answer: str = Field(
        description="The direct answer, for spoken delivery. No markdown, lists, or filler."
    )
    bridge: str = Field(
        default="",
        description=(
            "At most one short clause connecting the answer to what this speaker "
            "is deciding. Empty when it would only pad."
        ),
    )
    next_question: str = Field(
        default="",
        description=(
            "One question opening the next useful step in the conversation. "
            "Empty for greetings, audio checks, commands, and short factual "
            "confirmations."
        ),
    )


class InterjectionJudgmentModel(BaseModel):
    """Pydantic mirror of agent_policy.InterjectionJudgment, for structured
    parsing of the Level 1.5 judgment call. Kept separate from the dataclass
    the rest of the codebase consumes, the same split GroundedReply/compose_
    reply already uses between the wire schema and the internal type."""

    worth_interjecting: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

# ---------------------------------------------------------------------------
# Static Webpage Hosting (Self-Hosted avatar.html for Recall Output Media)
# ---------------------------------------------------------------------------

public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
if os.path.exists(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")

@app.get("/avatar.html")
async def get_avatar_html():
    html_path = os.path.join(public_dir, "avatar.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="avatar.html not found")

@app.get("/avatar.js")
async def get_avatar_js():
    js_path = os.path.join(public_dir, "avatar.js")
    if os.path.exists(js_path):
        return FileResponse(js_path)
    raise HTTPException(status_code=404, detail="avatar.js not found")

# ---------------------------------------------------------------------------
# RAG Helper: Query SpikedAI-Backend-One
# ---------------------------------------------------------------------------

async def resolve_source_ids(token_to_use: str, target_client_id: str) -> List[str]:
    """Resolve a client's ingested (COMPLETED) document/website ids via the
    backend's own /documents and /websites endpoints, authenticated with the
    caller's JWT. Cached per user+client (SOURCE_IDS_CACHE_TTL_S) since this
    costs two Cloud Run round trips and the document set changes on the scale
    of minutes, not turns.

    Split out from query_spiked_rag so callers on the speak path (see
    process_transcript_with_gemini) can kick this off *before* the turn is
    classified and overlap it with that Gemini call instead of paying for it
    serially afterward.
    """
    if not target_client_id or not token_to_use:
        return []
    cache_key = f"{extract_user_id_from_jwt(token_to_use)}|{target_client_id}"
    cached = _SOURCE_IDS_CACHE.get(cache_key)
    if cached and time.monotonic() - cached["ts"] <= SOURCE_IDS_CACHE_TTL_S:
        return cached["ids"]
    _t0 = time.monotonic()
    try:
        base = SPIKED_BACKEND_URL.rstrip('/')
        auth_headers = {"Authorization": f"Bearer {token_to_use}"}
        params = {"client_id": target_client_id}
        client = _get_backend_http()
        doc_res, web_res = await asyncio.gather(
            client.get(f"{base}/documents", headers=auth_headers, params=params, timeout=8.0),
            client.get(f"{base}/websites", headers=auth_headers, params=params, timeout=8.0),
        )
        resolved: List[str] = []
        for res in (doc_res, web_res):
            if res.status_code != 200:
                logger.warning("[RAG] Source listing %s returned %s", res.request.url.path, res.status_code)
                continue
            for item in res.json():
                if item.get("id") and item.get("status") == "COMPLETED":
                    resolved.append(item["id"])
        if resolved:
            _SOURCE_IDS_CACHE[cache_key] = {"ts": time.monotonic(), "ids": resolved}
            logger.info("[RAG] Auto-resolved %d completed source_ids for client_id=%s", len(resolved), target_client_id)
        logger.info("[RAG][TIMING] resolve_source_ids=%.2fs (cache_miss)", time.monotonic() - _t0)
        return resolved
    except Exception as err:
        # With a client_id set and no source_ids, the backend will fail
        # closed — this request cannot succeed. Error, not warning.
        logger.error("[RAG] Failed to auto-resolve source_ids (request will fail closed): %s", err)
        return []


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


async def query_spiked_rag(
    question: str,
    auth_token: str,
    client_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    timeout_s: float = AGENT_RAG_TIMEOUT_S,
    kyc_id: Optional[str] = None,
    source_ids_task: "Optional[asyncio.Task[List[str]]]" = None,
    on_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    # /ask/regular, not /ask/handsfree: measured, not assumed. /ask/handsfree
    # was picked earlier for its smaller 5-chunk retrieval and terser prompt on
    # the theory that less work is faster — a real side-by-side showed the
    # opposite: /ask/regular (10 chunks, full prompt) at ~4s from the console vs
    # /ask/handsfree at ~9.7s from here for a comparable question. /ask/regular
    # also keeps KYC-steered query augmentation (search.py's ask_handsfree omits
    # _build_query_augment), which handsfree traded away for no measured benefit.
    url = f"{SPIKED_BACKEND_URL.rstrip('/')}/ask/regular"
    token_to_use = auth_token or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    target_client_id = client_id or os.getenv("DEFAULT_CLIENT_ID") or ""
    headers = {
        "Authorization": f"Bearer {token_to_use}" if token_to_use else "",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    payload: Dict[str, Any] = {
        "question": question,
    }
    if target_client_id:
        payload["client_id"] = target_client_id
    if kyc_id:
        payload["kyc_id"] = kyc_id
    # If source_ids is missing but client_id is present, resolve the client's
    # ingested documents. The backend fails CLOSED on client_id + empty
    # source_ids (a selected client with no docs must not fall back to
    # all-docs), so an empty resolution here guarantees "no relevant
    # documents". A caller that already started resolution in parallel with
    # classification (see process_transcript_with_gemini) passes the task in
    # via source_ids_task so this call just joins it instead of resolving
    # again — the whole ~1s round trip then costs nothing on the speak path.
    effective_source_ids = source_ids
    if not effective_source_ids and source_ids_task is not None:
        effective_source_ids = await source_ids_task
    elif not effective_source_ids and target_client_id and token_to_use:
        effective_source_ids = await resolve_source_ids(token_to_use, target_client_id)

    if effective_source_ids:
        payload["source_ids"] = effective_source_ids
    elif target_client_id:
        logger.error(
            "[RAG] client_id=%s with no source_ids — backend will answer 'no relevant documents'",
            target_client_id,
        )

    logger.info("[RAG] Querying SpikedAI-Backend-One client_id=%s sources=%s query_chars=%d has_auth=%s kyc=%s", target_client_id, len(effective_source_ids or []), len(question), bool(token_to_use), bool(kyc_id))

    t0 = time.monotonic()
    ttfb = None
    first_chunk_at = None
    try:
        # Stream so the read timeout applies per-chunk, not to the whole
        # generation — a long but healthy answer must not be killed mid-stream.
        client = _get_backend_http()
        async with client.stream("POST", url, headers=headers, json=payload, timeout=timeout_s) as response:
            ttfb = time.monotonic() - t0
            if response.status_code != 200:
                body = (await response.aread())[:200]
                logger.error("[RAG] Failed status=%s body=%s ttfb=%.2fs", response.status_code, body, ttfb)
                return "I could not retrieve the relevant company documents for this question."

            # /ask/regular streams plain text fragments (the backend already
            # unwraps the LLM SSE). Concatenate verbatim; a real newline from
            # the model is preserved as a space by the whitespace collapse
            # downstream, and no separator is invented between fragments.
            parts: List[str] = []
            # When on_sentence is given, sentence boundaries are detected as
            # fragments arrive and each complete sentence is handed off
            # immediately — the caller can start speaking it before the rest
            # of the answer has even finished generating. pending holds text
            # since the last boundary; it may never resolve to a full
            # sentence (e.g. punctuation-free text), so it's flushed as-is
            # after the loop.
            pending = ""
            # on_sentence, when given, awaits actual TTS playback of each
            # chunk before returning (see _speak_chunk) — that time is real,
            # but it's Tom talking, not the backend generating. Tracked
            # separately so the [RAG][TIMING] log below still means what it
            # always meant (backend response time), not backend time plus
            # however long the answer took to speak.
            playback_wait_s = 0.0
            async for line in response.aiter_lines():
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic() - t0
                text_piece: Optional[str] = None
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    if chunk.startswith('"') and chunk.endswith('"'):
                        try:
                            chunk = json.loads(chunk)
                        except Exception:
                            pass
                    text_piece = chunk if isinstance(chunk, str) else str(chunk)
                    parts.append(text_piece)
                    parts.append(" ")
                elif not line.startswith("event:") and not line.startswith("id:"):
                    text_piece = line
                    parts.append(text_piece)
                    parts.append("\n")

                if text_piece and on_sentence is not None:
                    pending += text_piece + " "
                    segments = _SENTENCE_BOUNDARY_RE.split(pending)
                    if len(segments) > 1:
                        for segment in segments[:-1]:
                            segment = segment.strip()
                            if segment:
                                _t_wait_start = time.monotonic()
                                await on_sentence(segment)
                                playback_wait_s += time.monotonic() - _t_wait_start
                        pending = segments[-1]

            if on_sentence is not None and pending.strip():
                _t_wait_start = time.monotonic()
                await on_sentence(pending.strip())
                playback_wait_s += time.monotonic() - _t_wait_start

            result = "".join(parts).strip()
            cognitive_key = (response.headers.get("x-cognitive-key") or "").strip()
            total = time.monotonic() - t0
            backend_total = total - playback_wait_s
            logger.info(
                "[RAG][TIMING] ttfb=%.2fs first_chunk=%.2fs backend_total=%.2fs playback_wait=%.2fs wall_total=%.2fs chars=%d",
                ttfb, first_chunk_at if first_chunk_at is not None else -1, backend_total, playback_wait_s, total, len(result),
            )

        # TEMPORARY (see TEMP_WIRING.md): the live Groq stream is currently
        # failing platform-wide. When it yields one of its short error strings,
        # fall back to the backend's cognitive (background) answer for the same
        # question — same retrieval context, healthier model, just slower.
        lowered = result.lower()
        live_failed = lowered.startswith("error:") or any(m in lowered for m in BACKEND_STREAM_ERROR_MARKERS)
        if live_failed and AGENT_COGNITIVE_FALLBACK and cognitive_key:
            cognitive = await _poll_cognitive_answer(cognitive_key, headers["Authorization"])
            if cognitive:
                return cognitive
        return result or "No specific documentation found for this query."

    except Exception as e:
        logger.error(f"[RAG] Error calling SpikedAI backend: {e}", exc_info=True)
        return "An error occurred while accessing the company knowledge base."


async def _poll_cognitive_answer(cognitive_key: str, authorization: str) -> Optional[str]:
    """TEMPORARY (see TEMP_WIRING.md): poll the backend's background cognitive
    answer until it lands or the budget runs out. Only called when the live
    answer stream has already failed, so the added wait is strictly better than
    the apology Tom would otherwise give."""
    url = f"{SPIKED_BACKEND_URL.rstrip('/')}/cognitive"
    deadline = time.monotonic() + AGENT_COGNITIVE_FALLBACK_TIMEOUT_S
    logger.warning("[RAG] Live answer errored; polling cognitive answer key=%s...", cognitive_key[:12])
    try:
        client = _get_backend_http()
        while time.monotonic() < deadline:
            res = await client.get(
                url,
                params={"cognitive_key": cognitive_key},
                headers={"Authorization": authorization},
                timeout=10.0,
            )
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "done" and data.get("answer"):
                    logger.info("[RAG] Cognitive fallback answered (%d chars)", len(data["answer"]))
                    return data["answer"]
                if data.get("status") == "failed":
                    logger.warning("[RAG] Cognitive fallback reported failed")
                    return None
            elif res.status_code != 404:
                # 404 just means the row hasn't been written yet; keep waiting.
                logger.warning("[RAG] Cognitive poll returned %s", res.status_code)
                return None
            await asyncio.sleep(AGENT_COGNITIVE_POLL_INTERVAL_S)
    except Exception as err:
        logger.warning("[RAG] Cognitive fallback poll failed: %s", err)
    logger.warning("[RAG] Cognitive fallback timed out after %.0fs", AGENT_COGNITIVE_FALLBACK_TIMEOUT_S)
    return None

# ---------------------------------------------------------------------------
# Gemini turn routing with deterministic RAG execution
# ---------------------------------------------------------------------------

async def process_transcript_with_gemini(
    transcript: str,
    speaker: str,
    conversation_history: List[Dict[str, str]],
    auth_token: str,
    client_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    intel: Optional[CallIntelligence] = None,
    source_ids: Optional[List[str]] = None,
    kyc_id: Optional[str] = None,
    run: Optional[Dict[str, Any]] = None,
    turn_id: Optional[int] = None,
) -> Optional[str]:
    """Route an already-addressed, complete turn and produce a short spoken reply.

    run+turn_id are optional, passed straight through to _generate_grounded_reply
    to enable sentence-by-sentence streaming dispatch — see its docstring for
    the contract callers must honor (check run["_streamed_turn_id"] afterward).
    """
    _t_turn_start = time.monotonic()
    if not gemini_client:
        logger.error("[Google GenAI] Client is not initialized")
        return None

    ctx = user_context or {}
    company_name = ctx.get("company_name", "SpikedAI")
    products_services = ctx.get("products_services", "")
    bot_name = ctx.get("bot_name") or "Tom"
    product_domain = ctx.get("product_domain", "Enterprise AI & Automated Sales Engineering")
    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    catalog = build_entity_catalog(ctx)
    candidate_entities = closest_entities(transcript, catalog)
    history_text = "\n".join(
        f"{turn.get('speaker', 'Participant')}: {turn.get('text', '')}"
        for turn in conversation_history[-12:]
    )

    # Speculative source-id resolution: most addressed, wake-name-matched
    # turns end up needing RAG, and this has no dependency on classification's
    # result — only on client_id + the JWT, both known already. Kicking it off
    # now overlaps its ~1s (cold) / ~0ms (cached) cost with the classification
    # call below instead of paying for it serially after. Cancelled wherever
    # the turn turns out not to need it.
    target_client_id_for_prefetch = client_id or os.getenv("DEFAULT_CLIENT_ID") or ""
    source_ids_task: "Optional[asyncio.Task[List[str]]]" = None
    if not source_ids and target_client_id_for_prefetch and auth_token:
        source_ids_task = asyncio.create_task(resolve_source_ids(auth_token, target_client_id_for_prefetch))

    try:
        if (
            is_directly_addressed(transcript, bot_name)
            and requires_company_knowledge(transcript, catalog)
            and not needs_context_resolution(transcript)
        ):
            logger.info("[Agent Route] Fast path: skipped classification round trip")
            analysis = TurnAnalysis(
                response_action="respond",
                intent="company_knowledge",
                resolved_query=transcript,
                corrections=[],
            )
            reply = await _generate_grounded_reply(
                analysis=analysis,
                transcript=transcript,
                speaker=speaker,
                bot_name=bot_name,
                company_name=company_name,
                history_text=history_text,
                catalog=catalog,
                auth_token=auth_token,
                client_id=client_id,
                preferred_model=preferred_model,
                intel=intel,
                source_ids=source_ids,
                kyc_id=kyc_id,
                source_ids_task=source_ids_task,
                run=run,
                turn_id=turn_id,
            )
            logger.info("[TIMING] process_transcript_with_gemini total=%.2fs (fast_path)", time.monotonic() - _t_turn_start)
            return reply

        call_state = intel.call_state_block() if intel else ""
        # Dossier/room, needed only if this turn ends up coaching/social/
        # meeting_context/command, are computed up front (cheap, local intel
        # reads, no I/O) so they can ride along in the single classify+reply
        # call below instead of requiring a second call once the intent is
        # known — company_knowledge is the one intent that still can't be
        # answered here, since it needs facts this call has no access to.
        dossier = intel.speaker_dossier(speaker) if intel else ""
        room = intel.room_block() if intel else ""
        detailed_request = any(
            phrase in transcript.casefold()
            for phrase in ("more detail", "in detail", "elaborate", "explain fully", "deep dive")
        )
        reply_word_limit = 90 if detailed_request else AGENT_MAX_REPLY_WORDS
        analysis_prompt = f"""Decide whether {bot_name} should respond, then classify and normalize this wake-name-matched meeting turn. If the turn also needs a spoken reply and isn't a company_knowledge question, draft that reply in the same response.
Company: {company_name}
Offerings: {products_services or product_domain}
Verified entity candidates: {candidate_entities}
Recent finalized conversation:
{history_text}
{f'What this call has established so far:{chr(10)}{call_state}' if call_state else ''}
{f'In the room:{chr(10)}{room}' if room else ''}
{f'Who is speaking: {dossier}' if dossier else ''}
Current speaker: {speaker}
Raw ASR: {transcript}

Set response_action to:
- respond: the speaker directly asks {bot_name} a question, requests an action/opinion, or gives {bot_name} a command.
- acknowledge: the speaker directly gives {bot_name} information or a simple instruction that only needs a brief confirmation.
- silent: {bot_name} is merely mentioned, quoted, discussed in third person, explicitly told not to answer, or the request is directed to somebody else.
The presence of the wake name alone is not sufficient. Prefer silent when the addressee is ambiguous.
Use company_knowledge for company/product/features/pricing/security/SLA/integration questions.
Use meeting_context for questions about what meeting participants said or discussed, and for questions about this specific call's live state — sentiment, mood, engagement, or how a participant seems — never company_knowledge for those, since the knowledge base has no data on this call or its participants.
Use coaching when the sales rep asks {bot_name} for help running the call itself rather than for an answer to relay: what to ask next, what is being missed, how to handle an objection, where the conversation should go — including a vague invitation to contribute ("can you hop in", "jump in here", "chime in", "anything to add") with no specific ask attached. Treat that as coaching, not social: synthesize one useful thing from Recent finalized conversation / What this call has established, the same as if the rep had asked "what am I missing?"
Use social only for actual greetings, audio checks, and questions about {bot_name}'s own identity/role ("who are you", "what are you") — never company_knowledge for those, since the knowledge base has no document about {bot_name} himself. Use command for stop/wait/repeat commands.
Resolve pronouns and omitted context only in resolved_query. Propose corrections only from the verified entity candidates.

Reply fields (answer/bridge/next_question): leave ALL THREE empty if intent is company_knowledge (a separate step with retrieved facts will answer it) or if response_action is not "respond". Otherwise fill them in for spoken delivery:
- answer: the direct answer, at most {reply_word_limit} words. No markdown, lists, or filler. If intent is coaching, this is one specific, actionable suggestion grounded in what this call has established — name the gap or risk plainly, do not summarize the call back, do not pitch — and aim it at what {f"{dossier}" if dossier else "the sales rep"} is accountable for if that framing is known.
- bridge: at most one short clause connecting the answer to what {speaker} is trying to decide. Leave empty rather than padding.
- next_question: one question, at most {MAX_QUESTION_WORDS} words, that opens the next useful step. For coaching, phrase it as the question {speaker} should ask the room next, verbatim. Ask a question by default — leave it empty only when one would be unwelcome (greetings, audio checks, a command being acknowledged, a short confirmation).
Neither answer nor next_question may restate or re-ask anything Recent finalized conversation or What this call has established already covers — check both before writing either field. If the only thing you have to offer already happened, say what's genuinely still open instead."""
        _t_classify_start = time.monotonic()
        try:
            analysis_response = await gemini_client.aio.models.generate_content(
                model=preferred_model,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=320,
                    response_mime_type="application/json",
                    response_schema=TurnAnalysisAndReply,
                ),
            )
            logger.info("[TIMING] classification_call=%.2fs", time.monotonic() - _t_classify_start)
            parsed = getattr(analysis_response, "parsed", None)
            if isinstance(parsed, TurnAnalysisAndReply):
                analysis = parsed
            elif parsed is not None:
                analysis = TurnAnalysisAndReply.model_validate(parsed)
            else:
                analysis = TurnAnalysisAndReply.model_validate_json(analysis_response.text)
        except Exception:
            logger.warning("[Agent Route] Structured routing failed; defaulting substantive turn to RAG", exc_info=True)
            analysis = TurnAnalysisAndReply(
                response_action="respond",
                intent="company_knowledge",
                resolved_query=transcript,
                corrections=[],
            )

        logger.info(
            "[LLM Response Gate] speaker=%s action=%s intent=%s",
            speaker,
            analysis.response_action,
            analysis.intent,
        )
        if analysis.response_action == "silent":
            if source_ids_task is not None:
                source_ids_task.cancel()
            return None
        if analysis.response_action == "acknowledge":
            if source_ids_task is not None:
                source_ids_task.cancel()
            return "Understood."

        if analysis.intent not in ("social", "command") and requires_company_knowledge(transcript, catalog):
            analysis.intent = "company_knowledge"

        if analysis.intent == "company_knowledge":
            # The one intent this single call can't finish — it needs RAG
            # facts this prompt never saw. Falls through to the existing
            # retrieval + reply path; the draft reply fields above (left
            # empty by instruction) are simply unused here.
            reply = await _generate_grounded_reply(
                analysis=analysis,
                transcript=transcript,
                speaker=speaker,
                bot_name=bot_name,
                company_name=company_name,
                history_text=history_text,
                catalog=catalog,
                auth_token=auth_token,
                client_id=client_id,
                preferred_model=preferred_model,
                intel=intel,
                source_ids=source_ids,
                kyc_id=kyc_id,
                source_ids_task=source_ids_task,
                run=run,
                turn_id=turn_id,
            )
            logger.info("[TIMING] process_transcript_with_gemini total=%.2fs (classified_path, rag)", time.monotonic() - _t_turn_start)
            return reply

        # coaching / meeting_context / social / command: the reply already
        # came back in this same call — compose it directly, no second
        # Gemini round trip.
        if source_ids_task is not None:
            source_ids_task.cancel()
        reply = compose_reply(
            answer=analysis.answer,
            bridge=analysis.bridge,
            next_question=analysis.next_question,
            max_answer_words=reply_word_limit,
        )
        logger.info(
            "[Agent Route] speaker=%s intent=%s rag_used=False corrections=%d reply_words=%d asked=%s (single_shot)",
            speaker,
            analysis.intent,
            len(analysis.corrections),
            len(reply.split()),
            reply.endswith("?"),
        )
        logger.info("[TIMING] process_transcript_with_gemini total=%.2fs (classified_path, single_shot)", time.monotonic() - _t_turn_start)
        return reply or None
    except Exception as e:
        if source_ids_task is not None:
            source_ids_task.cancel()
        logger.error(f"[Gemini Agent] Inference error with google-genai SDK: {e}", exc_info=True)
        return None


async def _generate_grounded_reply(
    analysis: TurnAnalysis,
    transcript: str,
    speaker: str,
    bot_name: str,
    company_name: str,
    history_text: str,
    catalog: List[str],
    auth_token: str,
    client_id: Optional[str],
    preferred_model: str,
    intel: Optional[CallIntelligence] = None,
    source_ids: Optional[List[str]] = None,
    rag_timeout_s: float = AGENT_RAG_TIMEOUT_S,
    kyc_id: Optional[str] = None,
    source_ids_task: "Optional[asyncio.Task[List[str]]]" = None,
    run: Optional[Dict[str, Any]] = None,
    turn_id: Optional[int] = None,
) -> Optional[str]:
    """Run RAG when the intent requires it, then produce the spoken reply.

    run+turn_id are optional and used only to enable sentence-by-sentence
    streaming dispatch on the company_knowledge path: when both are given,
    the answer is spoken chunk-by-chunk as it streams in from the backend
    instead of being spoken all at once after the full answer arrives, and
    dispatch happens as a side effect inside this call — the caller must
    check run["_streamed_turn_id"] == turn_id afterward and, if it matches,
    must NOT also call _dispatch_reply with the returned text (it was
    already spoken). Omit both (as the Level 1 insight prefetch does) to get
    the original behavior exactly: pure generation, no side effects, caller
    always dispatches.
    """

    corrections = [item.model_dump() for item in analysis.corrections]
    corrected_transcript = apply_validated_corrections(transcript, corrections, catalog)
    resolved_query = analysis.resolved_query.strip() or corrected_transcript
    detailed_request = any(
        phrase in transcript.casefold()
        for phrase in ("more detail", "in detail", "elaborate", "explain fully", "deep dive")
    )
    reply_word_limit = 90 if detailed_request else AGENT_MAX_REPLY_WORDS
    rag_result = ""
    rag_used = analysis.intent == "company_knowledge"
    if rag_used:
        # Degraded-answer markers. The first three are our own client-side
        # fallbacks; the rest are the backend's: its fail-closed empty-context
        # message and the short "Error: ..." strings its Groq stream yields on
        # timeout/rate-limit. None of these are facts, so none may reach the
        # compose prompt — a model told to "never invent" will paraphrase them
        # into a confident-sounding refusal.
        def _rag_degraded(text: str) -> bool:
            lowered = text.lower()
            return not text or lowered.startswith("error:") or any(marker in lowered for marker in (
                "could not retrieve", "no specific documentation", "an error occurred",
                "could not find relevant documents",
                # /ask/handsfree's empty-context fallback (worded differently
                # from /ask/regular's, same fail-closed meaning).
                "no relevant information found",
                "request timed out", "service unavailable", "failed to get response",
            ))

        # A short one-liner only, not a full instruction paragraph. /ask/regular's
        # `question` field is used verbatim for BOTH the retrieval embedding
        # (build_rag_context(question, ...)) and the final "Question: {question}"
        # line in the Groq prompt (search.py:893-952) — there is no separate
        # field for instructions, so whatever travels here also becomes the
        # retrieval query. An earlier version prepended a ~90-word persona/
        # format paragraph: it dominated the embedding and pulled visibly
        # different chunks/facts than the console's clean question against the
        # same knowledge base. This one-liner is short enough to identify Tom
        # without meaningfully diluting the question's own embedding. Format
        # shaping (markdown strip, word backstop) still happens entirely in
        # post-processing below, not in this line.
        # "Answer concisely" alone (no number) let real answers run 135-249
        # words in practice — a spoken reply that long takes 25-30s to
        # actually say, which is a bigger drag on the conversation than any
        # of the numbers this session has been chasing. A single explicit
        # word count is a few tokens, nowhere near the ~90-word paragraph
        # this replaced, so it shouldn't meaningfully repollute the query.
        # First person, explicitly: this string is appended to the question and
        # travels verbatim into the backend's own prompt as its final
        # "Question: ..." line (no separate persona/system field exists there —
        # see the note above). A third-person aside describing "{bot_name},
        # an ... avatar" here previously read as background about someone
        # else, not an instruction for who is answering, and produced replies
        # like "Tom would explain..." instead of Tom actually answering.
        persona_hint = (
            f" (You are {bot_name}, a Solution Architect avatar, not the pitch person — "
            f"answer as {bot_name}, first person, under {reply_word_limit} words, "
            f"end with a relevant question.)"
        )
        shaped_query = f"{resolved_query}{persona_hint}"

        stream_enabled = run is not None and turn_id is not None
        on_sentence, stream_state = (
            _make_streaming_sentence_handler(run, turn_id, reply_word_limit)
            if stream_enabled else (None, None)
        )
        rag_result = await query_spiked_rag(
            shaped_query, auth_token, client_id, source_ids=source_ids, timeout_s=rag_timeout_s,
            kyc_id=kyc_id, source_ids_task=source_ids_task, on_sentence=on_sentence,
        )
        source_ids_task = None  # consumed; a retry below must not double-await it

        if stream_enabled and stream_state["dispatched"]:
            # Already spoken chunk-by-chunk as it streamed in — the degraded/
            # retry/backstop logic below never runs for this attempt, since
            # the streaming handler already applied its own equivalents
            # (first-sentence degraded check, incremental word budget) before
            # ever speaking anything.
            spoken_text = " ".join(stream_state["spoken_parts"])
            run["_streamed_turn_id"] = turn_id
            _finish_streamed_reply(run, turn_id, spoken_text)
            return spoken_text

        if _rag_degraded(rag_result) and rag_result.lower().startswith("error:") and not AGENT_COGNITIVE_FALLBACK:
            # The backend reached retrieval but its LLM call failed — these are
            # typically transient (rate limit / upstream blip), so one retry is
            # cheap and often the difference between an answer and an apology.
            # Skipped when the cognitive fallback is active: that path already
            # spent its own budget waiting, and stacking a second full attempt
            # on the speak path would blow past anyone's patience.
            logger.warning("[Agent Route] Backend answer stream errored (%r); retrying once", rag_result)
            retry_on_sentence, retry_state = (
                _make_streaming_sentence_handler(run, turn_id, reply_word_limit)
                if stream_enabled else (None, None)
            )
            rag_result = await query_spiked_rag(
                shaped_query, auth_token, client_id, source_ids=source_ids, timeout_s=rag_timeout_s,
                kyc_id=kyc_id, on_sentence=retry_on_sentence,
            )
            if stream_enabled and retry_state["dispatched"]:
                spoken_text = " ".join(retry_state["spoken_parts"])
                run["_streamed_turn_id"] = turn_id
                _finish_streamed_reply(run, turn_id, spoken_text)
                return spoken_text
        if _rag_degraded(rag_result):
            logger.warning("[Agent Route] RAG unavailable for addressed knowledge turn (result=%r)", rag_result[:80])
            return "I don’t have verified information on that available right now."
        # Strip markdown, source tags [1], and list markers, then apply a
        # generous word-budget backstop. Not a strict word-count slice: the
        # backend has no word-budget instruction anymore (removed with the
        # persona paragraph — see query_spiked_rag), so its answers are full
        # document-length text, and a raw words[:N] cut lops off mid-sentence,
        # producing a spoken reply that trails off instead of ending on a
        # thought. Trim at the last complete sentence that still fits the
        # budget instead — the same total-length ceiling, without the
        # mid-clause cutoff.
        rag_result = re.sub(r"\[\d+\]", "", rag_result)
        rag_result = re.sub(r"[#*`_~]", "", rag_result)
        rag_result = re.sub(r"^\s*[-+•]\s+", "", rag_result, flags=re.MULTILINE)
        rag_result = re.sub(r"\s+", " ", rag_result).strip()
        words = rag_result.split()
        backstop_words = reply_word_limit + MAX_QUESTION_WORDS
        if len(words) > backstop_words:
            sentences = re.split(r"(?<=[.!?])\s+", rag_result)
            kept: List[str] = []
            kept_words = 0
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if kept and kept_words + sentence_words > backstop_words:
                    break
                kept.append(sentence)
                kept_words += sentence_words
                if kept_words >= backstop_words:
                    break
            rag_result = " ".join(kept)
            trimmed_words = rag_result.split()
            if len(trimmed_words) > backstop_words:
                # The only sentence that fit was itself oversized (a single
                # long run-on, or punctuation-free text) — fall back to a
                # hard word cut so the budget is still a real ceiling.
                rag_result = " ".join(trimmed_words[:backstop_words]).rstrip(",;:")
                if rag_result[-1:] not in ".!?":
                    rag_result += "."
            logger.warning(
                "[Agent Route] Single-shot answer exceeded backstop (%d words); trimmed to %d complete sentence(s)",
                len(words), len(kept) or 1,
            )
        return rag_result
    elif source_ids_task is not None:
        # Speculatively started before classification decided this turn does
        # not need RAG after all — don't leave it running unobserved.
        source_ids_task.cancel()

    dossier = intel.speaker_dossier(speaker) if intel else ""
    room = intel.room_block() if intel else ""
    call_state = intel.call_state_block() if intel else ""

    if analysis.intent == "coaching":
        # Coaching is addressed to the rep about how to run the call, so it is
        # aimed at the conversation rather than at the knowledge base. The reply
        # is still spoken into the room: there is no rep-private channel here.
        task_block = f"""{speaker} is the sales rep asking you for help running this call, not for a fact to relay.
Give one specific, actionable suggestion grounded in what this call has actually established.
Name the gap or the risk plainly. Do not summarize the call back to them, and do not pitch.
Your next_question should be the question you think {speaker} should ask the room next, phrased so they can say it verbatim."""
    else:
        task_block = f"""Answer {speaker}'s addressed turn."""

    answer_prompt = f"""You are {bot_name}, an American Solution Architect representing {company_name} in this meeting.
You are the technical authority supporting the sales rep. You are not the salesperson, and you never pitch.
You are concise, conversational, and technically credible. Never invent facts.
{f'{chr(10)}Who is speaking: {dossier}' if dossier else ''}{f'{chr(10)}Aim the answer at what this person is accountable for. A finance stakeholder and an engineering stakeholder need the same fact framed differently.' if dossier else ''}
{task_block}

Produce the reply for spoken delivery, in three parts:
- answer: the direct answer, at most {reply_word_limit} words. No markdown, lists, or filler.
- bridge: at most one short clause connecting the answer to what {speaker} is trying to decide. Leave empty rather than padding.
- next_question: one question, at most {MAX_QUESTION_WORDS} words, that opens the next useful step in the conversation.

Ask a question by default: it is how you hand the conversation back. Leave next_question empty only when one would be unwelcome — greetings and audio checks, a command you are acknowledging, or a short factual confirmation the speaker only needed verified.
Use the speaker's first name only if it improves a greeting or clarification.
Neither answer nor next_question may restate or re-ask anything Recent finalized conversation or What this call has established already covers — check both before writing either field.

Intent: {analysis.intent}
Corrected turn: {corrected_transcript}
Resolved meaning: {resolved_query}
{f'In the room:{chr(10)}{room}' if room else ''}
{f'What this call has established so far:{chr(10)}{call_state}' if call_state else ''}
Recent finalized conversation:
{history_text}
Verified RAG facts (the only source for company facts):
{rag_result if rag_used else '(not required for this intent)'}"""
    response = await gemini_client.aio.models.generate_content(
        model=preferred_model,
        contents=answer_prompt,
        config=types.GenerateContentConfig(
            # Three short fields plus JSON scaffolding. A cap tight enough to
            # truncate the payload yields unparseable JSON, not a shorter reply.
            max_output_tokens=360 if detailed_request else 220,
            response_mime_type="application/json",
            response_schema=GroundedReply,
        ),
    )
    parsed = getattr(response, "parsed", None)
    try:
        if isinstance(parsed, GroundedReply):
            structured = parsed
        elif parsed is not None:
            structured = GroundedReply.model_validate(parsed)
        else:
            structured = GroundedReply.model_validate_json(response.text or "")
    except Exception:
        # Falling back to the raw text keeps a usable turn when structured
        # decoding fails, at the cost of the closing question for that turn.
        logger.warning("[Agent Route] Structured reply decode failed; using raw text", exc_info=True)
        reply = normalize_reply(response.text or "", reply_word_limit, 3)
    else:
        reply = compose_reply(
            answer=structured.answer,
            bridge=structured.bridge,
            next_question=structured.next_question,
            max_answer_words=reply_word_limit,
        )
    logger.info(
        "[Agent Route] speaker=%s intent=%s rag_used=%s corrections=%d reply_words=%d asked=%s",
        speaker,
        analysis.intent,
        rag_used,
        len(corrections),
        len(reply.split()),
        reply.endswith("?"),
    )
    return reply or None


async def _judge_interjection(
    transcript: str,
    history_text: str,
    bot_name: str,
    preferred_model: str,
) -> InterjectionJudgment:
    """Level 1.5: is this warmed reply worth volunteering unprompted?

    Deliberately a second, stricter classification rather than a threshold on
    the Level 1 heuristic (looks_like_followup + requires_company_knowledge)
    that already ran to get here — that heuristic answers "is this even a
    candidate", this answers "would a solution architect actually speak up
    right now". Runs only for turns that already cleared the cheap heuristic,
    so the extra call is bounded, not per-sentence.
    """
    judgment_prompt = f"""You are judging whether {bot_name}, a Solution Architect silently sitting in on this sales call, should interrupt the conversation right now with unsolicited input — nobody asked him anything.

Recent finalized conversation:
{history_text}

The moment in question: {transcript}

{bot_name} has a warmed, accurate answer ready. Set worth_interjecting to true ONLY if staying silent would let something real go wrong: a wrong technical assumption is being stated as fact, a decision-blocking gap is being glossed over, or a genuine risk/compliance issue is going unmentioned. Do NOT set it true just because the knowledge base happens to cover the topic, or because the answer would be a nice-to-have addition — a real solution architect lets most things pass without comment. When in doubt, false.

confidence should reflect how clearly this crosses that bar, not how confident you are in the answer's factual accuracy.
reason: one short clause, for an internal audit log a sales rep will read."""
    try:
        response = await gemini_client.aio.models.generate_content(
            model=preferred_model,
            contents=judgment_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=120,
                response_mime_type="application/json",
                response_schema=InterjectionJudgmentModel,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, InterjectionJudgmentModel):
            structured = parsed
        elif parsed is not None:
            structured = InterjectionJudgmentModel.model_validate(parsed)
        else:
            structured = InterjectionJudgmentModel.model_validate_json(response.text or "")
        return InterjectionJudgment(
            worth_interjecting=structured.worth_interjecting,
            confidence=structured.confidence,
            reason=structured.reason,
        )
    except Exception:
        # Fail closed: an unparseable or failed judgment call must never be
        # treated as permission to take the floor unprompted.
        logger.warning("[Autospeak] Judgment call failed", exc_info=True)
        return InterjectionJudgment(False, 0.0, "judgment_failed")

# ---------------------------------------------------------------------------
# API Endpoints: Session & Bot Creation
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LiveAvatar-Spiked",
        "deepgram_configured": bool(DEEPGRAM_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "recall_configured": bool(RECALL_API_KEY),
        "spiked_backend_url": SPIKED_BACKEND_URL,
        "recall_webhook_url": RECALL_WEBHOOK_URL,
        "public_base_url": PUBLIC_BASE_URL
    }

class InvokeAgentRequest(BaseModel):
    question: Optional[str] = Field(
        default=None,
        description="Explicit question to answer. Omit to accept the agent's pending insight.",
    )
    coaching: bool = Field(
        default=False,
        description="Ask the agent how to run the call rather than for an answer to relay.",
    )


class FloorUnavailable(Exception):
    """The agent cannot take the floor right now (busy, or nothing to say).

    Callers translate this however fits their context: invoke_agent as an
    HTTP 409/400, the Level 1.5 autonomous path as a silent no-op that falls
    back to the existing cue-only behavior.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


async def _take_floor_and_speak(
    run: Dict[str, Any],
    question: str,
    speaker: str,
    *,
    coaching: bool = False,
    warm_reply: Optional[str] = None,
    source: str = "invoke",
) -> Dict[str, Any]:
    """Take the floor and speak, bypassing the wake name/addressee gate but
    never the floor, governor, or duplicate guard.

    This is the discipline Level 2 invitation established: the rep pressing
    "Ask Tom" bypasses the wake name, not the safety rails. Shared by
    invoke_agent (source="invoke") and the Level 1.5 autonomous interjection
    path (source="autonomous") so both go through identical turn bookkeeping
    and dispatch instead of two parallel implementations drifting apart.
    """
    if run.get("state") != AgentState.LISTENING:
        # Busy, not broken. The caller can retry once the current turn lands.
        raise FloorUnavailable(f"agent_is_{run.get('state')}")
    if not question:
        raise FloorUnavailable("no_question")

    run["turn_counter"] = int(run.get("turn_counter", 0)) + 1
    turn_id = run["turn_counter"]
    run["active_turn_id"] = turn_id
    run["state"] = AgentState.THINKING

    previous_task = run.get("active_response_task")
    if previous_task and not previous_task.done():
        previous_task.cancel()

    async def deliver() -> None:
        try:
            answer = warm_reply
            if not answer:
                ctx = run.get("user_context") or {}
                history = run.get("history") or []
                answer = await _generate_grounded_reply(
                    analysis=TurnAnalysis(
                        response_action="respond",
                        intent="coaching" if coaching else "company_knowledge",
                        resolved_query=question,
                        corrections=[],
                    ),
                    transcript=question,
                    speaker=speaker,
                    bot_name=run.get("bot_name") or "Tom",
                    company_name=ctx.get("company_name", "SpikedAI"),
                    history_text="\n".join(
                        f"{t.get('speaker', 'Participant')}: {t.get('text', '')}"
                        for t in history[-12:]
                    ),
                    catalog=build_entity_catalog(ctx),
                    auth_token=run.get("token") or "",
                    client_id=run.get("client_id"),
                    preferred_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                    intel=run.get("intel"),
                    kyc_id=run.get("kyc_id"),
                    run=run,
                    turn_id=turn_id,
                )
            if run.get("_streamed_turn_id") == turn_id:
                # Already spoken chunk-by-chunk — see process_transcript_with_
                # gemini's respond() for the same pattern and why.
                return
            if turn_id != run.get("active_turn_id"):
                return
            if not answer:
                _release_floor(run)
                return
            await _dispatch_reply(run, answer, turn_id, source=source)
        except asyncio.CancelledError:
            logger.info("[%s] Cancelled superseded turn_id=%s", source, turn_id)
        except Exception:
            _release_floor(run)
            logger.error("[%s] Failed turn_id=%s", source, turn_id, exc_info=True)

    run["active_response_task"] = asyncio.create_task(deliver())
    return {"accepted": True, "turn_id": turn_id, "warm": bool(warm_reply)}


@app.post("/api/runs/{run_id}/invoke")
async def invoke_agent(run_id: str, payload: Optional[InvokeAgentRequest] = None):
    """Level 2 by explicit invitation — the 'Ask Tom' / 'Bring Tom In' path.

    The rep pressing the button *is* the invocation, so this bypasses the wake
    name and the addressee gate. It does not bypass the floor or the governor:
    an invited agent still cannot talk over anybody, which is the one rule that
    holds regardless of who asked.
    """
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Unknown run")

    payload = payload or InvokeAgentRequest()
    _clear_mute(run, reason="invoked")

    if run.get("state") != AgentState.LISTENING:
        # Busy, not broken. The caller can retry once the current turn lands.
        raise HTTPException(status_code=409, detail=f"Agent is {run.get('state')}")

    now = time.monotonic()
    insight = run.get("pending_insight") or {}
    insight_fresh = bool(insight) and (now - insight.get("created_at", 0)) <= AGENT_INSIGHT_TTL_S

    question = (payload.question or "").strip()
    if not question:
        if not insight_fresh:
            raise HTTPException(status_code=400, detail="No question supplied and no fresh insight pending")
        question = insight.get("question") or ""

    # Accepting a pending insight verbatim is the fast path the prefetch exists
    # for: the answer is already composed, so this is a socket send.
    warm = insight.get("reply") if (insight_fresh and not payload.question and not payload.coaching) else None
    run["pending_insight"] = None

    try:
        result = await _take_floor_and_speak(
            run,
            question,
            insight.get("speaker") or "the rep",
            coaching=payload.coaching,
            warm_reply=warm,
            source="invoke",
        )
    except FloorUnavailable as exc:
        raise HTTPException(status_code=409, detail=exc.reason)

    logger.info(
        "[Invoke] run_id=%s turn_id=%s warm=%s coaching=%s",
        run_id, result["turn_id"], result["warm"], payload.coaching,
    )
    return result


@app.get("/api/runs/{run_id}/credentials")
async def get_run_credentials(run_id: str, token: Optional[str] = Query(None)):
    """Provides LiveKit room credentials to avatar.html when loaded by Recall."""
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or expired")
    return {
        "session_id": run.get("session_id"),
        "livekit_url": run.get("livekit_url"),
        "livekit_token": run.get("livekit_token"),
    }

@app.post("/create-live-avatar")
async def create_avatar(payload: CreateLiveAvatarRequest):
    """
    Creates a LiveAvatar / HeyGen session.
    FULL Mode ($0.20/min) = includes built-in STT/LLM/TTS; supports avatar.speak_text.
    LITE Mode ($0.10/min) = visual-only, requires BYOTTS (agent.speak with PCM audio).
    """
    if not LIVEAVATAR_API_KEY:
        raise HTTPException(status_code=500, detail="LIVEAVATAR_API_KEY is not configured")

    avatar_id = payload.avatar_id or LIVEAVATAR_AVATAR_ID or None
    la_headers = {
        "Content-Type": "application/json",
        "X-API-KEY": LIVEAVATAR_API_KEY
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        # The application owns knowledge retrieval and response generation. Keep
        # LiveAvatar output-only rather than creating a second persona/context that
        # can greet or answer independently of the meeting turn gate.
        context_id = None

        # Build session token payload
        token_payload = {
            "mode": payload.mode,
            "avatar_id": avatar_id,
            "video_settings": {
                "quality": payload.quality,
                "encoding": "H264"
            },
            "is_sandbox": payload.is_sandbox
        }

        # FULL mode requires interactivity_type + avatar_persona
        # Set interactivity_type to PUSH_TO_TALK to prevent HeyGen's built-in LLM from responding automatically
        if payload.mode == "FULL":
            token_payload["interactivity_type"] = "PUSH_TO_TALK"
            if context_id:
                token_payload["avatar_persona"] = {"context_id": context_id}
            else:
                # Fallback: empty persona so FULL mode still works
                token_payload["avatar_persona"] = {}

        token_res = await client.post(
            f"{LIVEAVATAR_BASE_URL}/v1/sessions/token",
            headers=la_headers,
            json=token_payload
        )
        
        if token_res.status_code not in (200, 201):
            logger.error(f"LiveAvatar token creation failed: {token_res.text}")
            raise HTTPException(status_code=token_res.status_code, detail=token_res.text)
        
        token_json = token_res.json()
        token_data = token_json.get("data", {}) if isinstance(token_json.get("data"), dict) else token_json
        session_token = token_data.get("session_token")
        
        start_res = await client.post(
            f"{LIVEAVATAR_BASE_URL}/v1/sessions/start",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session_token}"
            },
            json={}
        )
        
        if start_res.status_code not in (200, 201):
            logger.error(f"LiveAvatar start session failed: {start_res.text}")
            raise HTTPException(status_code=start_res.status_code, detail=start_res.text)
            
        start_json = start_res.json()
        start_data = start_json.get("data", {}) if isinstance(start_json.get("data"), dict) else start_json
        
        logger.info(f"[LiveAvatar] Session started in {payload.mode} mode: session_id={start_data.get('session_id')}")
        
        return {
            "mode": payload.mode,
            "rate": "$0.20/minute" if payload.mode == "FULL" else "$0.10/minute",
            "session_id": start_data.get("session_id"),
            "livekit_url": start_data.get("livekit_url"),
            "livekit_token": start_data.get("livekit_client_token"),
            "session_token": session_token
        }

async def _keep_avatar_session_alive(run_id: str, session_id: str) -> None:
    """Ping LiveAvatar's keep-alive endpoint for the life of the run.

    LiveAvatar auto-closes sessions that see no join/interaction for a while;
    Tom stays silent unless addressed, so without this a quiet stretch in a
    meeting kills the avatar session out from under an otherwise-healthy run.
    """
    la_headers = {"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                await asyncio.sleep(LIVEAVATAR_KEEPALIVE_INTERVAL_S)
                if run_id not in _ACTIVE_RUNS:
                    return
                try:
                    resp = await client.post(
                        f"{LIVEAVATAR_BASE_URL}/v1/sessions/keep-alive",
                        headers=la_headers,
                        json={"session_id": session_id},
                    )
                    if resp.status_code not in (200, 201):
                        logger.warning(
                            "[LiveAvatar] keep-alive for session %s -> %s: %s",
                            session_id, resp.status_code, resp.text,
                        )
                except Exception:
                    logger.warning("[LiveAvatar] keep-alive request failed for session %s", session_id, exc_info=True)
    except asyncio.CancelledError:
        raise


async def _deploy_live_avatar_bot(
    meeting_url: str,
    token: str,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    kyc_id: Optional[str] = None,
    bot_name: str = DEFAULT_BOT_NAME,
    avatar_id: Optional[str] = None,
    autospeak_enabled: bool = False,
    request: Optional[Request] = None
) -> Dict[str, Any]:
    """Internal core method to deploy the LiveAvatar Bot into a meeting."""
    try:
        if not RECALL_API_KEY:
            raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")

        if not user_id:
            user_id = extract_user_id_from_jwt(token)

        run_id = f"run_{uuid.uuid4().hex}"
        recall_ws_token = uuid.uuid4().hex

        # 1. Create LiveAvatar FULL Session ($0.20/min, includes TTS) with user context
        avatar_session = await create_avatar(CreateLiveAvatarRequest(
            avatar_id=avatar_id,
            mode="FULL",
            user_id=user_id,
            client_id=client_id,
            token=token,
            bot_name=bot_name
        ))

        # 2. Store session credentials for Recall's avatar.html
        _ACTIVE_RUNS[run_id] = {
            "run_id": run_id,
            "avatar_id": avatar_id,
            "user_id": user_id,
            "client_id": client_id,
            "kyc_id": kyc_id,
            "token": token,
            "session_id": avatar_session.get("session_id"),
            "livekit_url": avatar_session.get("livekit_url"),
            "livekit_token": avatar_session.get("livekit_token"),
            "bot_name": bot_name,
            "recall_ws_token": recall_ws_token,
            "state": AgentState.LISTENING,
            "history": [],
            "turn_counter": 0,
            "control_ws": None,
            # Separate from control_ws on purpose: that socket belongs to the
            # avatar page and is a single slot. A rep's browser connecting there
            # would displace the avatar and silence the bot entirely.
            # A set, not a slot: the console legitimately has several consumers
            # (insight panel, transcript feed) and may be open in more than one
            # tab. A single slot would let the newest connection silently
            # displace the others, which is the failure control_ws already has.
            "rep_sockets": set(),
            "active_response_task": None,
            "watchdog_task": None,
            "keepalive_task": None,
            # "Tom, stay quiet for 30 seconds" — muted_until is monotonic, used
            # for gating; muted_until_epoch_ms is wall-clock, sent to the
            # frontend so it can render a countdown against Date.now().
            "muted_until": None,
            "muted_until_epoch_ms": None,
            "mute_expiry_task": None,
            # Sentence-by-sentence dispatch: chunk_id -> asyncio.Event, set
            # when that chunk's avatar_speak_ended arrives. Lets the
            # streaming dispatch loop wait for exactly one chunk to finish
            # playing before sending the next, so audio never overlaps.
            "chunk_events": {},
            "pending_turns": {},
            "intel": None,
            # Level 1: a warmed answer the agent is holding but was not invited
            # to give. Consumed by the invoke endpoint, expired by TTL.
            "pending_insight": None,
            "last_insight_at": None,
            # Level 1.5: opt-in per run (see CreateBotWithLiveAvatarRequest).
            # Off by default — the agent taking the floor with no human in the
            # loop is the highest blast-radius behavior in the app so far.
            "autospeak_enabled": autospeak_enabled,
            "last_autospeak_at": None,
            "autospeak_count": 0,
            "floor": FloorState(),
            "echo": EchoSuppressor(
                similarity_threshold=AGENT_ECHO_SIMILARITY,
                tail_seconds=AGENT_ECHO_TAIL_S,
            ),
            "governor": SpeechGovernor(
                cooldown_seconds=AGENT_REPLY_COOLDOWN_MS / 1000,
                max_replies_per_window=AGENT_MAX_REPLIES_PER_WINDOW,
                window_seconds=AGENT_REPLY_WINDOW_S,
            ),
        }

        # Warm the conversation-intelligence snapshot in the background. It is
        # empty for the first cycle, which simply means the earliest turns get
        # the name-only behavior until the backend has something to say.
        intel = CallIntelligence(auth_token=token)
        intel.start()
        _ACTIVE_RUNS[run_id]["intel"] = intel

        session_id = avatar_session.get("session_id")
        if session_id:
            _ACTIVE_RUNS[run_id]["keepalive_task"] = asyncio.create_task(
                _keep_avatar_session_alive(run_id, session_id)
            )

        # Warm the source-id cache now instead of paying for it on the first
        # real question — resolve_source_ids costs ~0.7-1.9s the first time
        # it runs, and nothing else is happening yet to overlap it with (later
        # turns overlap it with classification, but the very first one never
        # gets that for free). Cache key matches exactly what process_
        # transcript_with_gemini uses later, so a hit here is a hit there too.
        # Cheap even if it goes to waste: if the first real question comes
        # after the 5-minute cache TTL, this was just one harmless early call.
        if client_id and token:
            asyncio.create_task(resolve_source_ids(token, client_id))

        # 3. Build Output Media URL
        base_url = PUBLIC_BASE_URL.rstrip('/')
        if request and "localhost" in base_url and not "localhost" in str(request.base_url):
            base_url = str(request.base_url).rstrip('/')

        # debug=1 renders the "what the agent heard" panel into the meeting camera
        # feed. Everyone in the call sees it, so turn AGENT_DEBUG_OVERLAY off for
        # anything customer-facing.
        avatar_page_url = (
            f"{base_url}/avatar.html"
            f"?run={run_id}"
            f"{'&debug=1' if AGENT_DEBUG_OVERLAY else ''}"
        )
        recall_ws_base = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        recall_audio_url = f"{recall_ws_base}/ws/recall/audio/{run_id}?token={recall_ws_token}"

        # 4. Recall Bot Payload with Output Media + Dual-Track Webhook
        recall_payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "variant": {
                "zoom": "web_4_core",
                "google_meet": "web_4_core",
                "microsoft_teams": "web_4_core"
            },
            "output_media": {
                "camera": {
                    "kind": "webpage",
                    "config": {
                        "url": avatar_page_url
                    }
                }
            },
            "metadata": {
                "user_id": user_id,
                "client_id": client_id or ""
            },
            "recording_config": {
                "audio_separate_raw": {},
                "retention": {
                    "type": "timed",
                    "hours": 168
                },
                "include_bot_in_recording": {
                    "audio": True
                },
                "transcript": {
                    "provider": {
                        "deepgram_streaming": {
                            "model": "nova-3"
                        }
                    },
                    "diarization": {
                        "use_separate_streams_when_available": True
                    }
                },
                "realtime_endpoints": [
                    {
                        "type": "websocket",
                        "url": recall_audio_url,
                        "events": [
                            "audio_separate_raw.data",
                            "participant_events.join",
                            "participant_events.leave",
                            "participant_events.speech_on",
                            "participant_events.speech_off"
                        ]
                    },
                    {
                        "type": "webhook",
                        "url": RECALL_WEBHOOK_URL,
                        "events": ["transcript.data", "transcript.partial_data"]
                    }
                ]
            }
        }

        headers = {
            "Authorization": f"Token {RECALL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{RECALL_BASE_URL.rstrip('/')}/api/v1/bot/",
                headers=headers,
                json=recall_payload
            )
            
            if resp.status_code not in (200, 201):
                logger.error(f"Recall Bot creation failed: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            
            bot_data = resp.json()
            bot_id = bot_data.get("id")
            _ACTIVE_RUNS[run_id]["bot_id"] = bot_id

            status_changes = bot_data.get("status_changes")
            latest_status = "created"
            if isinstance(status_changes, list) and len(status_changes) > 0 and isinstance(status_changes[-1], dict):
                latest_status = status_changes[-1].get("code", "created")

            return {
                "id": bot_id,
                "bot_id": bot_id,
                "run_id": run_id,
                "liveavatar_session_id": avatar_session.get("session_id"),
                "avatar_page_url": avatar_page_url,
                "status": latest_status
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying live avatar bot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start")
async def start_bot_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Unified Start Endpoint compatible with Frontend recallBotService.ts:
    Accepts both JSON and Multipart/FormData payloads.
    """
    try:
        token = ""
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "").strip()

        meeting_url = ""
        client_id = None
        kyc_id = None
        bot_name = DEFAULT_BOT_NAME
        avatar_id = None
        autospeak_enabled = False

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            meeting_url = body.get("meeting_url", "")
            client_id = body.get("client_id")
            kyc_id = body.get("kyc_id")
            bot_name = body.get("bot_name", DEFAULT_BOT_NAME)
            avatar_id = body.get("avatar_id")
            autospeak_enabled = bool(body.get("autospeak_enabled", False))
            if not token:
                token = body.get("token", "")
        else:
            form = await request.form()
            meeting_url = form.get("meeting_url", "")
            client_id = form.get("client_id")
            kyc_id = form.get("kyc_id")
            bot_name = form.get("bot_name", DEFAULT_BOT_NAME)
            avatar_id = form.get("avatar_id")
            autospeak_enabled = str(form.get("autospeak_enabled", "")).strip().lower() in ("true", "1", "on")
            if not token:
                token = form.get("token", "")

        if not meeting_url:
            raise HTTPException(status_code=400, detail="meeting_url is required")

        result = await _deploy_live_avatar_bot(
            meeting_url=meeting_url,
            token=token,
            client_id=client_id,
            kyc_id=kyc_id,
            bot_name=bot_name,
            avatar_id=avatar_id,
            autospeak_enabled=autospeak_enabled,
            request=request
        )
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"start_bot_endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-live-avatar-bot")
async def create_live_avatar_bot(
    payload: CreateBotWithLiveAvatarRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Explicit JSON endpoint to deploy a LiveAvatar bot."""
    token = payload.token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()

    return await _deploy_live_avatar_bot(
        meeting_url=payload.meeting_url,
        token=token or "",
        user_id=payload.user_id,
        client_id=payload.client_id,
        kyc_id=payload.kyc_id,
        bot_name=payload.bot_name,
        avatar_id=payload.avatar_id,
        autospeak_enabled=payload.autospeak_enabled,
        request=request
    )

async def _stop_avatar_session(session_id: Optional[str]) -> Optional[int]:
    """Release the LiveAvatar session so the concurrency slot is freed.

    LiveAvatar allows one concurrent session per key, so a session that outlives
    its meeting does not merely bill — it blocks every subsequent start with
    `4032 Session concurrency limit reached`. Stopping it is the single most
    important half of teardown.
    """
    if not session_id or not LIVEAVATAR_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{LIVEAVATAR_BASE_URL}/v1/sessions/stop",
                headers={"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
                json={"session_id": session_id},
            )
        logger.info("[Teardown] LiveAvatar session %s stop -> %s", session_id, resp.status_code)
        return resp.status_code
    except Exception:
        logger.error("[Teardown] Failed to stop LiveAvatar session %s", session_id, exc_info=True)
        return None


async def _leave_recall_call(bot_id: Optional[str]) -> Optional[int]:
    if not bot_id or not RECALL_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{RECALL_BASE_URL.rstrip('/')}/api/v1/bot/{bot_id}/leave_call/",
                headers={"Authorization": f"Token {RECALL_API_KEY}"},
            )
        logger.info("[Teardown] Recall bot %s leave_call -> %s", bot_id, resp.status_code)
        return resp.status_code
    except Exception:
        logger.error("[Teardown] Failed to evict Recall bot %s", bot_id, exc_info=True)
        return None


async def _teardown_run(run_id: str) -> Dict[str, Any]:
    """Release every resource a run holds, then evict it from the registry.

    Written to be idempotent and to never raise: disconnect is the path a user
    reaches for when something has already gone wrong, so a failure in one leg
    must not strand the others. Each leg is attempted independently and the
    run is evicted regardless of what the remote services report.
    """
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        return {"ok": True, "already_gone": True}

    for key in ("active_response_task", "watchdog_task", "keepalive_task", "mute_expiry_task"):
        task = run.get(key)
        if task and not task.done():
            task.cancel()
    for entry in (run.get("pending_turns") or {}).values():
        timer = entry.get("timer")
        if timer and not timer.done():
            timer.cancel()

    intel = run.get("intel")
    if intel:
        intel.stop()

    control_ws = run.get("control_ws")
    if control_ws:
        try:
            await control_ws.close()
        except Exception:
            pass
        run["control_ws"] = None

    for socket in list(run.get("rep_sockets") or ()):
        try:
            await socket.close()
        except Exception:
            pass
    run["rep_sockets"] = set()

    recall_status, avatar_status = await asyncio.gather(
        _leave_recall_call(run.get("bot_id")),
        _stop_avatar_session(run.get("session_id")),
    )

    _ACTIVE_RUNS.pop(run_id, None)
    logger.info("[Teardown] Run %s released (%d still active)", run_id, len(_ACTIVE_RUNS))
    return {
        "ok": True,
        "run_id": run_id,
        "recall_status": recall_status,
        "avatar_session_status": avatar_status,
    }


def _find_run_id_by_bot(bot_id: str) -> Optional[str]:
    for run_id, run in _ACTIVE_RUNS.items():
        if run.get("bot_id") == bot_id:
            return run_id
    return None


@app.get("/api/active-runs")
async def list_active_runs():
    """Every run this process is currently driving.

    The orchestrator is the only component that knows a self-started avatar bot
    exists — the recall backend never saw it created. Exposing the registry lets
    the mock backend (and, later, run recovery after a refresh) answer "is a bot
    live for this user?" truthfully instead of guessing.
    """
    return {
        "runs": [
            {
                "run_id": run_id,
                "bot_id": run.get("bot_id"),
                "user_id": run.get("user_id"),
                "client_id": run.get("client_id"),
                "bot_name": run.get("bot_name"),
                "state": str(run.get("state")),
                "muted_until_epoch_ms": (
                    run.get("muted_until_epoch_ms")
                    if run.get("muted_until") is not None and time.monotonic() < run["muted_until"]
                    else None
                ),
            }
            for run_id, run in _ACTIVE_RUNS.items()
            if run.get("bot_id")
        ]
    }


@app.post("/api/runs/{run_id}/stop")
async def stop_run_endpoint(run_id: str):
    """Full teardown by run id. The disconnect path the frontend should use."""
    return await _teardown_run(run_id)


@app.post("/remove-bot/{bot_id}")
@app.post("/leave-call/{bot_id}")
async def leave_call_endpoint(bot_id: str):
    """Evict a bot by Recall id, tearing down its run when one is known.

    Kept on the bot id because that is what callers hold, but it no longer stops
    at telling Recall to leave: without releasing the avatar session too, the
    next start fails on the concurrency limit.
    """
    run_id = _find_run_id_by_bot(bot_id)
    if run_id:
        return await _teardown_run(run_id)

    # No run in this process — a redeploy, another instance, or a plain
    # audio-mode bot. Evicting from the meeting is still the right thing.
    status = await _leave_recall_call(bot_id)
    if status is None and not RECALL_API_KEY:
        raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")
    return {"ok": True, "status": status, "run_matched": False}

@app.get("/bot/{bot_id}")
async def get_bot_endpoint(bot_id: str):
    """Retrieves live status of a bot from Recall API."""
    if not RECALL_API_KEY:
        raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")

    headers = {"Authorization": f"Token {RECALL_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{RECALL_BASE_URL.rstrip('/')}/api/v1/bot/{bot_id}/",
            headers=headers
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

# ---------------------------------------------------------------------------
# Participant-separated Audio & Control WebSockets
# ---------------------------------------------------------------------------


def verify_recall_websocket(headers: Any, fallback_token: str, supplied_token: str) -> bool:
    """Verify Recall's upgrade signature, with a per-run token for legacy workspaces."""
    if not RECALL_WEBHOOK_SECRET:
        return bool(fallback_token) and hmac.compare_digest(fallback_token, supplied_token or "")
    try:
        secret = RECALL_WEBHOOK_SECRET
        if not secret.startswith("whsec_"):
            return False
        message_id = headers.get("webhook-id") or headers.get("svix-id")
        timestamp = headers.get("webhook-timestamp") or headers.get("svix-timestamp")
        signatures = headers.get("webhook-signature") or headers.get("svix-signature")
        if not message_id or not timestamp or not signatures:
            return False
        key = base64.b64decode(secret.removeprefix("whsec_"))
        expected = hmac.new(key, f"{message_id}.{timestamp}.".encode(), hashlib.sha256).digest()
        for versioned in signatures.split(" "):
            version, _, signature = versioned.partition(",")
            if version == "v1" and hmac.compare_digest(expected, base64.b64decode(signature)):
                return True
    except Exception:
        logger.warning("[Recall WS] Signature verification failed", exc_info=True)
    return False


def _push_heard(
    run: Dict[str, Any],
    speaker: str,
    text: str,
    reply: bool,
    reason: str,
) -> None:
    """Mirror the gate's verdict onto the avatar overlay and the rep console.

    In avatar mode this is the *only* path a transcript can reach the frontend:
    the recall backend never saw this bot created, so its transcript stream is
    empty by construction. These are the same finalized, speaker-attributed
    turns the agent itself reasons over.
    """
    _push_rep(run, {
        "type": "heard",
        "speaker": speaker,
        "text": text,
        "reply": reply,
        "reason": reason,
    })

    control_ws = run.get("control_ws")
    if not control_ws:
        return

    async def send() -> None:
        try:
            await control_ws.send_json({
                "type": "heard",
                "speaker": speaker,
                "text": text,
                "reply": reply,
                "reason": reason,
            })
        except Exception:
            pass

    asyncio.create_task(send())


def _push_control(run: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Fire-and-forget a message to the avatar page itself (/ws/control).

    Distinct from _push_rep: that reaches the rep's console, this reaches the
    actual meeting video feed avatar.js renders — used for the mute countdown
    overlay, which every meeting participant sees, not just the rep.
    """
    control_ws = run.get("control_ws")
    if not control_ws:
        return

    async def send() -> None:
        try:
            await control_ws.send_json(message)
        except Exception:
            pass

    try:
        asyncio.create_task(send())
    except RuntimeError:
        send().close()


def _push_rep(run: Dict[str, Any], message: Dict[str, Any]) -> None:
    """Fire-and-forget a message to the rep's console, if one is attached.

    Never awaited by the turn pipeline: a rep with a wedged socket must not be
    able to slow down or block what the agent says in the room.
    """
    sockets = list(run.get("rep_sockets") or ())
    if not sockets:
        return

    async def send() -> None:
        for socket in sockets:
            try:
                await socket.send_json(message)
            except Exception:
                # A dead console is dropped on its own disconnect path; failing
                # to reach one must not stop the others from being told.
                pass

    try:
        asyncio.create_task(send())
    except RuntimeError:
        # No running loop. Notifying the console is never worth raising into a
        # caller — _release_floor is on the speak path, and an exception there
        # would strand the agent out of LISTENING permanently.
        send().close()


def _set_mute(run: Dict[str, Any], seconds: int) -> None:
    """Mute the agent for `seconds`. A voice command reissued while already
    muted simply replaces the previous timer, matching how the wake name
    always grants the floor regardless of current state."""
    now_mono = time.monotonic()
    until_mono = now_mono + seconds
    until_epoch_ms = int((time.time() + seconds) * 1000)
    run["muted_until"] = until_mono
    run["muted_until_epoch_ms"] = until_epoch_ms

    previous = run.get("mute_expiry_task")
    if previous and not previous.done():
        previous.cancel()

    logger.info("[Mute] Agent muted for %ds run_id=%s", seconds, run.get("run_id"))
    mute_message = {"type": "agent_muted", "muted_until_epoch_ms": until_epoch_ms, "seconds": seconds}
    _push_rep(run, mute_message)
    _push_control(run, mute_message)

    async def expire() -> None:
        try:
            await asyncio.sleep(seconds)
            # Only clear if nothing re-muted (or cleared) it in the meantime.
            if run.get("muted_until") == until_mono:
                run["muted_until"] = None
                run["muted_until_epoch_ms"] = None
                logger.info("[Mute] Expired run_id=%s", run.get("run_id"))
                _push_rep(run, {"type": "agent_unmuted", "reason": "expired"})
                _push_control(run, {"type": "agent_unmuted", "reason": "expired"})
        except asyncio.CancelledError:
            pass

    run["mute_expiry_task"] = asyncio.create_task(expire())


def _clear_mute(run: Dict[str, Any], reason: str = "invoked") -> None:
    """Ask Tom / explicit invocation always overrides mute — the button press
    is itself the invitation, so it bypasses mute the same way it bypasses
    the wake name and the addressee gate."""
    if run.get("muted_until") is None:
        return
    run["muted_until"] = None
    run["muted_until_epoch_ms"] = None
    task = run.get("mute_expiry_task")
    if task and not task.done():
        task.cancel()
    logger.info("[Mute] Cleared (%s) run_id=%s", reason, run.get("run_id"))
    _push_rep(run, {"type": "agent_unmuted", "reason": reason})
    _push_control(run, {"type": "agent_unmuted", "reason": reason})


def _push_insight(run: Dict[str, Any], speaker: str, topic: str) -> None:
    """Signal that the agent has something, without it taking the floor.

    This is the whole of Level 1 on the wire: the rep's console learns the agent
    could contribute, and nothing is said in the room unless somebody accepts.
    """
    _push_rep(run, {
        "type": "insight_available",
        "speaker": speaker,
        "topic": topic,
    })


async def _dispatch_reply(run: Dict[str, Any], answer: str, turn_id: int, source: str = "addressed") -> bool:
    """Send a finished reply to the avatar and arm the speak watchdog.

    Shared by the addressed-turn path, the invoke endpoint, and the Level 1.5
    autonomous path so every turn passes the same duplicate guard and echo
    suppression — invitation bypasses the wake name, not the safety rails.
    `source` ("addressed" | "invoke" | "autonomous") rides along on the rep
    socket's agent_spoke event only, purely for console display/audit — it
    has no effect on gating.
    """
    governor: SpeechGovernor = run["governor"]
    bot_name = run.get("bot_name") or "Tom"
    spoken_at = time.monotonic()

    if governor.is_duplicate(answer, spoken_at):
        logger.info("[Governor] Suppressed duplicate reply turn_id=%s", turn_id)
        _release_floor(run, spoken_at)
        return False

    control_ws = run.get("control_ws")
    if not control_ws:
        logger.warning("[Agent] Avatar control socket unavailable for turn_id=%s", turn_id)
        _release_floor(run, spoken_at)
        return False

    history = run.setdefault("history", [])
    history.append({"speaker": bot_name, "participant_id": "bot", "text": answer})
    del history[:-40]
    # Register before dispatch: the echo can return before the socket ack.
    run["echo"].note_bot_speech(answer, spoken_at)
    governor.note_reply(answer, spoken_at)
    _push_rep(run, {"type": "agent_spoke", "text": answer, "turn_id": turn_id, "source": source})

    timing = run.setdefault("turn_timing", {}).setdefault(turn_id, {})
    timing["dispatched_at"] = spoken_at
    finalized_at = timing.get("finalized_at")
    if finalized_at is not None:
        logger.info(
            "[TIMING] turn_id=%s finalize->dispatch=%.2fs (turn gate + classification + RAG)",
            turn_id, spoken_at - finalized_at,
        )

    await control_ws.send_json({"type": "avatar_speak", "text": answer, "turn_id": turn_id})
    run["watchdog_task"] = asyncio.create_task(_speak_watchdog(run, turn_id, answer))
    return True


async def _speak_chunk(run: Dict[str, Any], turn_id: int, chunk_id: str, text: str) -> bool:
    """Send one sentence and wait for it to actually finish playing before
    returning, so sequential chunks of one answer never overlap in the
    avatar's audio. Returns False once the turn is no longer live (barge-in,
    superseded turn, or the control socket dropped) — callers use that to
    stop dispatching further chunks rather than talking over a turn that's
    already over.
    """
    control_ws = run.get("control_ws")
    # active_turn_id alone isn't enough: an in-place barge-in during SPEAKING
    # releases the floor (state -> LISTENING) without changing active_turn_id
    # at all, so a state check is the only thing that actually catches "this
    # exact turn was just interrupted" between one chunk and the next.
    if (
        not control_ws
        or run.get("active_turn_id") != turn_id
        or run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING)
    ):
        return False
    event = asyncio.Event()
    run.setdefault("chunk_events", {})[chunk_id] = event
    # Stamp once, on this turn's first chunk only: the same field name
    # _dispatch_reply writes for the legacy single-shot path, so the
    # avatar_speak_started handler's dispatch->speaking log works for both
    # without duplicating that logic. Streamed replies never went through
    # _dispatch_reply at all, so without this, that log line's gating field
    # was silently never set and the line never printed — see live_avatar.py
    # avatar_control_endpoint's avatar_speak_started handling.
    timing = run.setdefault("turn_timing", {}).setdefault(turn_id, {})
    timing.setdefault("dispatched_at", time.monotonic())
    try:
        await control_ws.send_json({"type": "avatar_speak", "text": text, "turn_id": turn_id, "chunk_id": chunk_id})
        timeout = estimate_speech_seconds(text) + 5.0
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[Stream] turn_id=%s chunk_id=%s timed out waiting for speak_ended; continuing", turn_id, chunk_id)
    finally:
        (run.get("chunk_events") or {}).pop(chunk_id, None)
    return run.get("active_turn_id") == turn_id and run.get("state") in (AgentState.THINKING, AgentState.SPEAKING)


def _make_streaming_sentence_handler(
    run: Dict[str, Any], turn_id: int, reply_word_limit: int,
) -> "tuple[Callable[[str], Awaitable[None]], Dict[str, Any]]":
    """Build the on_sentence callback query_spiked_rag calls per sentence.

    Speaks each complete sentence as soon as it's ready instead of waiting
    for the whole answer — the entire point of streaming. Two safety checks
    apply before anything is actually spoken:
    - The first sentence is checked against the same degraded/error markers
      _generate_grounded_reply already checks the full answer against. If it
      looks like a failure, nothing is spoken here at all — the caller falls
      through to the existing full-buffer degraded-check-and-retry path
      exactly as if streaming had never been attempted.
    - A running word count enforces the same backstop as the non-streaming
      path (reply_word_limit + MAX_QUESTION_WORDS), stopping at a sentence
      boundary instead of a raw word-count slice — so a long answer still
      ends on a complete thought, just decided incrementally instead of
      after the fact.

    Returns (callback, state) — state accumulates what was actually spoken
    (state["spoken_parts"]) and whether anything was dispatched
    (state["dispatched"]), which the caller uses to decide whether this
    attempt already committed to speaking or is still free to retry.
    """
    backstop_words = reply_word_limit + MAX_QUESTION_WORDS
    state: Dict[str, Any] = {
        "dispatched": False,
        "aborted": False,
        "seen_first": False,
        "cumulative_words": 0,
        "chunk_n": 0,
        "spoken_parts": [],
    }

    def _degraded_marker(text: str) -> bool:
        lowered = text.lower()
        return not text or lowered.startswith("error:") or any(marker in lowered for marker in (
            "could not retrieve", "no specific documentation", "an error occurred",
            "could not find relevant documents", "no relevant information found",
            "request timed out", "service unavailable", "failed to get response",
        ))

    async def on_sentence(raw_sentence: str) -> None:
        if state["aborted"]:
            return
        cleaned = re.sub(r"\[\d+\]", "", raw_sentence)
        cleaned = re.sub(r"[#*`_~]", "", cleaned)
        cleaned = re.sub(r"^\s*[-+•]\s+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return
        if not state["seen_first"]:
            state["seen_first"] = True
            if _degraded_marker(cleaned):
                # Don't speak a raw failure string — bail out entirely and
                # let the caller's existing full-buffer path (degraded
                # check, one retry) handle it exactly as before.
                state["aborted"] = True
                return
        if state["dispatched"] and (
            run.get("active_turn_id") != turn_id
            or run.get("state") not in (AgentState.THINKING, AgentState.SPEAKING)
        ):
            # Superseded, or barge-in released the floor without changing
            # active_turn_id (see _speak_chunk) — stop speaking further
            # chunks. The backend stream is still left to finish reading
            # quietly in the background so the connection isn't torn down
            # mid-response, but nothing more will be dispatched.
            state["aborted"] = True
            return
        sentence_words = len(cleaned.split())
        if state["cumulative_words"] and state["cumulative_words"] + sentence_words > backstop_words:
            state["aborted"] = True  # budget hit; stop, same as the non-streaming backstop
            return
        state["chunk_n"] += 1
        chunk_id = f"{turn_id}-{state['chunk_n']}"
        state["dispatched"] = True
        state["cumulative_words"] += sentence_words
        state["spoken_parts"].append(cleaned)
        still_live = await _speak_chunk(run, turn_id, chunk_id, cleaned)
        if not still_live:
            state["aborted"] = True
        if state["cumulative_words"] >= backstop_words:
            state["aborted"] = True

    return on_sentence, state


def _finish_streamed_reply(run: Dict[str, Any], turn_id: int, spoken_text: str) -> None:
    """Bookkeeping for a reply already spoken chunk-by-chunk via _speak_chunk.

    Mirrors what _dispatch_reply does after a single-shot send — history,
    echo registration, governor, timing log — without re-sending audio.
    Unlike _dispatch_reply, this can't check governor.is_duplicate() before
    speaking (there's no full text to check until streaming is already
    underway), so within-window duplicate suppression doesn't apply to a
    streamed reply — a real, accepted gap, not an oversight.
    """
    bot_name = run.get("bot_name") or "Tom"
    spoken_at = time.monotonic()
    governor: SpeechGovernor = run["governor"]

    history = run.setdefault("history", [])
    history.append({"speaker": bot_name, "participant_id": "bot", "text": spoken_text})
    del history[:-40]
    run["echo"].note_bot_speech(spoken_text, spoken_at)
    governor.note_reply(spoken_text, spoken_at)
    _push_rep(run, {"type": "agent_spoke", "text": spoken_text, "turn_id": turn_id})

    timing = (run.get("turn_timing") or {}).pop(turn_id, None)
    if timing and timing.get("finalized_at") is not None:
        logger.info(
            "[TIMING] turn_id=%s streamed reply complete, total=%.2fs (turn_finalize->last_chunk_spoken)",
            turn_id, spoken_at - timing["finalized_at"],
        )
    _release_floor(run, spoken_at, reply_text=spoken_text)


def _release_floor(run: Dict[str, Any], now: Optional[float] = None, reply_text: str = "") -> None:
    """Return the agent to LISTENING and open the follow-up window.

    reply_text, when given, sizes the follow-up window to how long that reply
    actually took to say (see AGENT_FOLLOWUP_WINDOW_REPLY_SCALE) — omitted on
    release paths where nothing was actually spoken (failures, duplicates,
    watchdog timeouts), which correctly leaves the window at its flat floor.
    """
    run["state"] = AgentState.LISTENING
    floor: FloorState = run["floor"]
    floor.last_bot_finished_at = now if now is not None else time.monotonic()
    if reply_text:
        floor.last_reply_seconds = estimate_speech_seconds(reply_text)
    _push_rep(run, {"type": "agent_state", "state": AgentState.LISTENING.value})


async def _interrupt_avatar(run: Dict[str, Any], participant_id: str) -> None:
    """Stop an in-flight turn. Playback and pending inference are both cancelled."""
    state = run.get("state")
    if state not in (AgentState.SPEAKING, AgentState.THINKING):
        return

    if state == AgentState.THINKING:
        # The answer is not spoken yet and is already stale; drop it silently
        # rather than delivering a reply to a question the room moved past.
        pending = run.get("active_response_task")
        if pending and not pending.done():
            pending.cancel()
        logger.info("[Barge-In] Cancelled pending turn participant_id=%s", participant_id)
        _release_floor(run)
        return

    control_ws = run.get("control_ws")
    if not control_ws:
        return
    run["state"] = AgentState.INTERRUPTING
    logger.info("[Barge-In] participant_id=%s sustained_ms=%d", participant_id, AGENT_BARGE_IN_MS)
    await control_ws.send_json({
        "type": "avatar_interrupt",
        "turn_id": run.get("active_turn_id"),
    })


async def _speak_watchdog(run: Dict[str, Any], turn_id: int, text: str) -> None:
    """Force LISTENING if the avatar never reports speak_started/speak_ended.

    Without this a single dropped LiveKit data event wedges the run in SPEAKING
    forever, after which every reply is suppressed and only barge-in ever fires.
    """
    try:
        await asyncio.sleep(AGENT_SPEAK_START_TIMEOUT_S)
        if run.get("active_turn_id") != turn_id:
            return
        if run.get("state") == AgentState.THINKING:
            logger.warning("[Watchdog] No speak_started for turn_id=%s; releasing floor", turn_id)
            _release_floor(run)
            return

        remaining = estimate_speech_seconds(text) + AGENT_SPEAK_MAX_OVERRUN_S
        await asyncio.sleep(remaining)
        if run.get("active_turn_id") != turn_id:
            return
        if run.get("state") != AgentState.LISTENING:
            logger.warning(
                "[Watchdog] turn_id=%s stuck in %s after %.1fs; releasing floor",
                turn_id,
                run.get("state"),
                AGENT_SPEAK_START_TIMEOUT_S + remaining,
            )
            _release_floor(run)
    except asyncio.CancelledError:
        pass


def _ingest_utterance(
    run: Dict[str, Any],
    participant_id: str,
    participant_name: str,
    utterance: Dict[str, Any],
) -> None:
    """Drop self-audio, then merge fragments from one speaker into a single turn."""
    transcript = (utterance.get("text") or "").strip()
    if not transcript:
        return

    now = time.monotonic()
    echo: EchoSuppressor = run["echo"]
    if echo.is_echo(transcript, now):
        # The agent hearing itself, either from Recall's bot audio or from a
        # participant's speakers. Never reaches history or the gate.
        logger.info(
            "[Echo] Suppressed self-audio participant_id=%s text=%r",
            participant_id,
            transcript[:80],
        )
        _push_heard(run, participant_name, transcript, False, "echo_suppressed")
        return

    pending: Dict[str, Any] = run.setdefault("pending_turns", {})
    entry = pending.get(participant_id)
    # started_at: first fragment of this turn (Deepgram's earliest is_final
    # segment) — the closest thing to "when the participant started talking."
    # last_fragment_at: this fragment's arrival — updated every merge, so the
    # gap to the eventual flush is purely the deliberate merge-pause wait, not
    # speech itself.
    started_at = entry.get("started_at", now) if entry else now
    if entry:
        timer: Optional[asyncio.Task] = entry.get("timer")
        if timer and not timer.done():
            timer.cancel()
        transcript = f"{entry['text']} {transcript}".strip()

    delay_ms = (
        AGENT_TURN_MERGE_INCOMPLETE_MS
        if is_probably_incomplete(transcript)
        else AGENT_TURN_MERGE_MS
    )

    async def flush_after_pause() -> None:
        try:
            await asyncio.sleep(delay_ms / 1000)
            current = run.get("pending_turns", {}).pop(participant_id, None)
            if current:
                _finalize_turn(
                    run, participant_id, participant_name, current["text"],
                    speech_started_at=current.get("started_at"),
                    speech_ended_at=current.get("last_fragment_at"),
                )
        except asyncio.CancelledError:
            pass

    pending[participant_id] = {
        "text": transcript,
        "timer": asyncio.create_task(flush_after_pause()),
        "started_at": started_at,
        "last_fragment_at": now,
    }


def _consider_insight(
    run: Dict[str, Any],
    speaker: str,
    transcript: str,
    history_snapshot: List[Dict[str, str]],
) -> None:
    """Level 1: notice a moment worth speaking into, without speaking into it.

    The agent was not addressed, so it stays silent — that rule is not relaxed
    here. What it does instead is tell the rep it has something and warm the
    answer in the background, so accepting the offer costs a socket round trip
    instead of a retrieval one. An agent that pauses eight seconds after being
    waved in is not perceived as the smartest participant in the room, whatever
    the answer eventually says.
    """
    if run.get("state") != AgentState.LISTENING:
        return

    ctx = run.get("user_context") or {}
    catalog = build_entity_catalog(ctx)
    # Deliberately narrow: a question-shaped turn about something the knowledge
    # base actually covers. Everything else is conversation the agent has no
    # standing to volunteer into.
    if not looks_like_followup(transcript) or not requires_company_knowledge(transcript, catalog):
        return

    now = time.monotonic()
    last = run.get("last_insight_at")
    if last is not None and now - last < AGENT_INSIGHT_COOLDOWN_S:
        return
    run["last_insight_at"] = now

    logger.info("[Insight] Level 1 moment speaker=%s text=%r", speaker, transcript[:80])
    _push_insight(run, speaker, transcript)

    bot_name = run.get("bot_name") or "Tom"
    history_text = "\n".join(
        f"{turn.get('speaker', 'Participant')}: {turn.get('text', '')}"
        for turn in history_snapshot[-12:]
    )

    async def prefetch() -> None:
        try:
            reply = await _generate_grounded_reply(
                analysis=TurnAnalysis(
                    response_action="respond",
                    intent="company_knowledge",
                    resolved_query=transcript,
                    corrections=[],
                ),
                transcript=transcript,
                speaker=speaker,
                bot_name=bot_name,
                company_name=ctx.get("company_name", "SpikedAI"),
                history_text=history_text,
                catalog=catalog,
                auth_token=run.get("token") or "",
                client_id=run.get("client_id"),
                preferred_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                intel=run.get("intel"),
                rag_timeout_s=AGENT_RAG_PREFETCH_TIMEOUT_S,
                kyc_id=run.get("kyc_id"),
            )
            if reply:
                run["pending_insight"] = {
                    "speaker": speaker,
                    "question": transcript,
                    "reply": reply,
                    "created_at": time.monotonic(),
                }
                logger.info("[Insight] Warmed reply words=%d", len(reply.split()))
                if run.get("autospeak_enabled"):
                    await _consider_autospeak(run, speaker, transcript, reply, history_text)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A failed prefetch costs nothing: the cue already reached the rep,
            # and invoking regenerates from scratch.
            logger.warning("[Insight] Prefetch failed", exc_info=True)

    asyncio.create_task(prefetch())


async def _consider_autospeak(
    run: Dict[str, Any],
    speaker: str,
    transcript: str,
    warmed_reply: str,
    history_text: str,
) -> None:
    """Level 1.5: decide whether to auto-accept the Level 1 insight just
    warmed above, i.e. take the floor unprompted instead of only cueing.

    Only ever called for a run that opted in (autospeak_enabled) and only
    after the Level 1 heuristic + a successful RAG warm already passed —
    this is an additional, stricter filter on top of that path, not a
    replacement for it. Every check here fails closed: any doubt, and Tom
    falls back to the existing cue-only behavior with `pending_insight`
    left in place for the rep to accept manually.
    """
    bot_name = run.get("bot_name") or "Tom"
    run_id = run.get("run_id")

    if run.get("autospeak_count", 0) >= AGENT_AUTOSPEAK_MAX_PER_RUN:
        logger.info("[Autospeak] Skipped run_id=%s reason=cap_reached", run_id)
        return
    last = run.get("last_autospeak_at")
    now = time.monotonic()
    if last is not None and now - last < AGENT_AUTOSPEAK_COOLDOWN_S:
        logger.info("[Autospeak] Skipped run_id=%s reason=cooldown", run_id)
        return
    # Re-check floor/governor here, not just in _consider_insight's caller:
    # the RAG warm above was an await, so state may have moved on since.
    if run.get("state") != AgentState.LISTENING:
        logger.info("[Autospeak] Skipped run_id=%s reason=agent_not_idle state=%s", run_id, run.get("state"))
        return
    governor: SpeechGovernor = run["governor"]
    allowed, governor_reason = governor.allows_reply(time.monotonic())
    if not allowed:
        logger.info("[Autospeak] Skipped run_id=%s reason=governor:%s", run_id, governor_reason)
        return

    judgment = await _judge_interjection(
        transcript=transcript,
        history_text=history_text,
        bot_name=bot_name,
        preferred_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )
    logger.info(
        "[Autospeak] Judged run_id=%s worth_interjecting=%s confidence=%.2f reason=%r",
        run_id, judgment.worth_interjecting, judgment.confidence, judgment.reason,
    )
    _push_rep(run, {
        "type": "autospeak_reasoning",
        "worth_interjecting": judgment.worth_interjecting,
        "confidence": judgment.confidence,
        "reason": judgment.reason,
    })
    if not judgment.worth_interjecting or judgment.confidence < AGENT_AUTOSPEAK_MIN_CONFIDENCE:
        logger.info("[Autospeak] Skipped run_id=%s reason=below_bar", run_id)
        return
    # Recheck floor/governor once more: the judgment call was itself an
    # await, and this is the last moment before actually taking the floor.
    if run.get("state") != AgentState.LISTENING:
        logger.info("[Autospeak] Skipped run_id=%s reason=agent_not_idle_after_judgment state=%s", run_id, run.get("state"))
        return
    allowed, governor_reason = governor.allows_reply(time.monotonic())
    if not allowed:
        logger.info("[Autospeak] Skipped run_id=%s reason=governor_after_judgment:%s", run_id, governor_reason)
        return

    run["autospeak_count"] = run.get("autospeak_count", 0) + 1
    run["last_autospeak_at"] = time.monotonic()
    run["pending_insight"] = None  # being spoken now, not offered

    logger.info(
        "[Autospeak] Taking floor speaker=%s confidence=%.2f reason=%r count=%d",
        speaker, judgment.confidence, judgment.reason, run["autospeak_count"],
    )
    try:
        await _take_floor_and_speak(
            run, transcript, speaker, coaching=False, warm_reply=warmed_reply, source="autonomous",
        )
    except FloorUnavailable:
        # Lost the floor between the recheck above and here (no await in
        # between today, but this keeps the function honest if that changes).
        logger.info("[Autospeak] Floor unavailable at the last moment; staying silent")


def _consider_autospeak_candidate(
    run: Dict[str, Any],
    speaker: str,
    transcript: str,
    history_snapshot: List[Dict[str, str]],
) -> None:
    """Level 1.5's own trigger — independent of Level 1's cue gate.

    _consider_insight's heuristic (looks_like_followup + requires_company_
    knowledge) is deliberately narrow: it decides what's worth showing the
    rep a card for, and a keyword-list topic gate is the right amount of
    conservative for that UX surface. But that gate also has real recall
    gaps — a genuine technical question phrased in domain language that
    doesn't happen to contain one of the hardcoded factual_terms (e.g. "how
    does state transfer work when orchestrated using Kubernetes") never
    reaches the KB at all, silently, whether or not it's question-shaped.

    Level 1.5 never shows anything to the rep unless _judge_interjection
    approves it, so it doesn't inherit that conservatism: this evaluates
    every complete, non-addressed turn — question or statement, on a
    recognized topic or not — and lets RAG's own "no relevant documents"
    response be the real relevance filter. That's cheap on a miss (retrieval
    only, no generation, no cognitive pipeline — see build_rag_context's
    early return) and it means the only things standing between a good
    moment and Tom staying silent are the cooldown, the cap, and the
    judgment call — not a static keyword list. Shares _consider_insight's
    cooldown clock, and may run alongside it for the same on-topic
    question-shaped turn (two prefetches, not one) — an accepted duplication
    now that a miss is cheap, in exchange for never structurally missing a
    real moment the way the combined old gate did.
    """
    if not run.get("autospeak_enabled"):
        return
    if run.get("state") != AgentState.LISTENING:
        return
    if is_probably_incomplete(transcript):
        return

    ctx = run.get("user_context") or {}
    catalog = build_entity_catalog(ctx)

    now = time.monotonic()
    last = run.get("last_insight_at")
    if last is not None and now - last < AGENT_INSIGHT_COOLDOWN_S:
        return
    run["last_insight_at"] = now

    logger.info("[Autospeak] Candidate turn speaker=%s text=%r", speaker, transcript[:80])

    bot_name = run.get("bot_name") or "Tom"
    history_text = "\n".join(
        f"{turn.get('speaker', 'Participant')}: {turn.get('text', '')}"
        for turn in history_snapshot[-12:]
    )

    async def prefetch_and_judge() -> None:
        try:
            reply = await _generate_grounded_reply(
                analysis=TurnAnalysis(
                    response_action="respond",
                    intent="company_knowledge",
                    resolved_query=transcript,
                    corrections=[],
                ),
                transcript=transcript,
                speaker=speaker,
                bot_name=bot_name,
                company_name=ctx.get("company_name", "SpikedAI"),
                history_text=history_text,
                catalog=catalog,
                auth_token=run.get("token") or "",
                client_id=run.get("client_id"),
                preferred_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                intel=run.get("intel"),
                rag_timeout_s=AGENT_RAG_PREFETCH_TIMEOUT_S,
                kyc_id=run.get("kyc_id"),
            )
            if not reply:
                return
            await _consider_autospeak(run, speaker, transcript, reply, history_text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[Autospeak] Declarative prefetch failed", exc_info=True)

    asyncio.create_task(prefetch_and_judge())


def _finalize_turn(
    run: Dict[str, Any],
    participant_id: str,
    participant_name: str,
    transcript: str,
    speech_started_at: Optional[float] = None,
    speech_ended_at: Optional[float] = None,
) -> None:
    now = time.monotonic()
    bot_name = run.get("bot_name") or "Tom"
    floor: FloorState = run["floor"]
    governor: SpeechGovernor = run["governor"]

    if speech_started_at is not None and speech_ended_at is not None:
        logger.info(
            "[TIMING] speech_duration=%.2fs merge_pause=%.2fs (last_fragment->turn_finalize) participant_id=%s",
            speech_ended_at - speech_started_at,
            now - speech_ended_at,
            participant_id,
        )

    # A mute command is checked before anything else, including while already
    # muted — re-issuing it (or a different duration) always takes effect, the
    # same way the wake name always grants the floor regardless of state.
    mute_seconds = detect_mute_command(transcript, bot_name)
    if mute_seconds is not None:
        logger.info("[Turn Gate] Detected mute command text=%r seconds=%d", transcript[:200], mute_seconds)
        _set_mute(run, mute_seconds)
        _push_heard(run, participant_name, transcript, False, "mute_command")
        return

    if run.get("muted_until") is not None and now < run["muted_until"]:
        # Muted: only the "Ask Tom" button (_clear_mute, called from invoke)
        # or the timer expiring can end this — not the wake name.
        logger.info("[Turn Gate] Dropped (muted) text=%r", transcript[:200])
        _push_heard(run, participant_name, transcript, False, "muted")
        return

    decision = evaluate_turn(
        transcript,
        participant_id,
        bot_name,
        floor,
        now,
        followup_window_seconds=AGENT_FOLLOWUP_WINDOW_MS / 1000,
        max_consecutive_followups=AGENT_MAX_FOLLOWUPS,
        followup_decay_rate=AGENT_FOLLOWUP_DECAY_RATE,
        min_followup_window_seconds=AGENT_MIN_FOLLOWUP_WINDOW_MS / 1000,
        agent_is_idle=run.get("state") == AgentState.LISTENING,
        reply_length_scale=AGENT_FOLLOWUP_WINDOW_REPLY_SCALE,
    )

    history = run.setdefault("history", [])
    history_snapshot = list(history[-20:])
    history.append({"speaker": participant_name, "participant_id": participant_id, "text": transcript})
    del history[:-40]

    logger.info(
        "[Turn Gate] participant_id=%s participant_name=%s bot_name=%s reply=%s reason=%s matched_name=%s text=%r",
        participant_id,
        participant_name,
        bot_name,
        decision.should_reply,
        decision.reason,
        decision.matched_name,
        transcript[:200],
    )
    _push_heard(run, participant_name, transcript, decision.should_reply, decision.reason)
    if not decision.should_reply:
        _consider_insight(run, participant_name, transcript, history_snapshot)
        _consider_autospeak_candidate(run, participant_name, transcript, history_snapshot)
        return

    allowed, governor_reason = governor.allows_reply(now)
    if not allowed:
        logger.info("[Governor] Suppressed reply participant_id=%s reason=%s", participant_id, governor_reason)
        _push_heard(run, participant_name, transcript, False, governor_reason)
        return

    if decision.reason == "followup_window":
        floor.consecutive_followups += 1
    else:
        floor.consecutive_followups = 0
    floor.last_addressed_participant = participant_id

    previous_task = run.get("active_response_task")
    if previous_task and not previous_task.done():
        previous_task.cancel()
    previous_watchdog = run.get("watchdog_task")
    if previous_watchdog and not previous_watchdog.done():
        previous_watchdog.cancel()

    run["turn_counter"] = int(run.get("turn_counter", 0)) + 1
    turn_id = run["turn_counter"]
    run["active_turn_id"] = turn_id
    run["state"] = AgentState.THINKING
    run.setdefault("turn_timing", {})[turn_id] = {"finalized_at": now}

    async def respond() -> None:
        try:
            answer = await process_transcript_with_gemini(
                transcript=transcript,
                speaker=participant_name,
                conversation_history=history_snapshot,
                auth_token=run.get("token") or "",
                client_id=run.get("client_id") or (run.get("user_context") or {}).get("client_id"),
                user_context=run.get("user_context") or {},
                intel=run.get("intel"),
                source_ids=run.get("source_ids") or (run.get("user_context") or {}).get("source_ids"),
                kyc_id=run.get("kyc_id"),
                run=run,
                turn_id=turn_id,
            )
            if run.get("_streamed_turn_id") == turn_id:
                # Already spoken chunk-by-chunk inside _generate_grounded_reply
                # (see its docstring) — floor already released, history/echo/
                # governor already updated. Nothing left to do here.
                return
            if turn_id != run.get("active_turn_id"):
                logger.info("[Agent] Discarded stale response turn_id=%s", turn_id)
                (run.get("turn_timing") or {}).pop(turn_id, None)
                return
            if not answer:
                _release_floor(run)
                (run.get("turn_timing") or {}).pop(turn_id, None)
                return

            await _dispatch_reply(run, answer, turn_id)
        except asyncio.CancelledError:
            logger.info("[Agent] Cancelled superseded turn_id=%s", turn_id)
            (run.get("turn_timing") or {}).pop(turn_id, None)
        except Exception:
            _release_floor(run)
            logger.error("[Agent] Failed addressed turn_id=%s", turn_id, exc_info=True)
            (run.get("turn_timing") or {}).pop(turn_id, None)

    run["active_response_task"] = asyncio.create_task(respond())


class ParticipantTranscriber:
    def __init__(
        self,
        participant_id: str,
        participant_name: str,
        keywords: List[str],
        on_utterance: Any,
    ):
        self.participant_id = participant_id
        self.participant_name = participant_name
        self.keywords = keywords
        self.on_utterance = on_utterance
        self.ws: Any = None
        self.receiver_task: Optional[asyncio.Task] = None
        self.buffer = FinalUtteranceBuffer()
        self.start_lock = asyncio.Lock()
        # Wall-clock time the last Results message carrying actual words arrived
        # (interim or final) — the closest proxy we have to "when this person
        # stopped talking," since we have no independent timestamp for that.
        # Diffing this against the flush-triggering message's own arrival time
        # isolates Deepgram's own endpointing/utterance_end wait from our own
        # merge_pause (already logged separately in _ingest_utterance).
        self._last_words_at: Optional[float] = None

    async def ensure_started(self) -> None:
        if self.ws:
            return
        async with self.start_lock:
            if self.ws:
                return
            params = {
                "model": "nova-3",
                "encoding": "linear16",
                "sample_rate": "16000",
                "channels": "1",
                "smart_format": "true",
                "interim_results": "true",
                "endpointing": str(AGENT_ENDPOINTING_MS),
                "utterance_end_ms": str(AGENT_UTTERANCE_END_MS),
                "vad_events": "true",
                "punctuate": "true",
            }
            url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
            keyterms: List[str] = []
            keyterm_tokens = 0
            for item in self.keywords[:100]:
                if not item or not item.strip():
                    continue
                estimated_tokens = max(1, len(item.split()))
                if keyterm_tokens + estimated_tokens > 450:
                    break
                keyterms.append(item.strip())
                keyterm_tokens += estimated_tokens
            if keyterms:
                url += "&" + "&".join(f"keyterm={quote(item.strip())}" for item in keyterms)
            headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            try:
                try:
                    self.ws = await websockets.connect(url, additional_headers=headers)
                except TypeError:
                    self.ws = await websockets.connect(url, extra_headers=headers)
            except websockets.exceptions.InvalidStatus as exc:
                # Deepgram puts the actual reason in the response body. Without
                # logging it, a rejected query parameter looks identical to an
                # agent that simply never hears anybody.
                detail = ""
                try:
                    detail = exc.response.body.decode()[:300]
                except Exception:
                    pass
                logger.error(
                    "[Deepgram] Handshake rejected status=%s detail=%s params=%s",
                    exc.response.status_code, detail, params,
                )
                raise
            self.receiver_task = asyncio.create_task(self._receive())
            logger.info(
                "[Deepgram] Connected participant_id=%s participant_name=%s",
                self.participant_id,
                self.participant_name,
            )

    async def send(self, pcm: bytes) -> None:
        await self.ensure_started()
        await self.ws.send(pcm)

    async def _receive(self) -> None:
        try:
            while True:
                now = time.monotonic()
                data = json.loads(await self.ws.recv())
                msg_type = data.get("type")

                has_words = False
                is_speech_final = False
                if msg_type == "Results":
                    alternatives = data.get("channel", {}).get("alternatives", [])
                    has_words = bool(alternatives and (alternatives[0].get("transcript") or "").strip())
                    is_speech_final = bool(data.get("speech_final"))

                utterance = self.buffer.add_result(data)
                if utterance:
                    # Flush triggered by this message (speech_final Results, or an
                    # UtteranceEnd with no words of its own). If speech_final fired
                    # on a content-bearing message, this gap is ~0 — the fast path
                    # working as intended. A large gap means it fell back to
                    # utterance_end_ms (floored at Deepgram's 1000ms API minimum).
                    if self._last_words_at is not None:
                        logger.info(
                            "[TIMING] deepgram_finalize_wait=%.2fs trigger=%s participant_id=%s "
                            "(last words heard -> Deepgram signaled utterance done)",
                            now - self._last_words_at, msg_type, self.participant_id,
                        )
                    self._last_words_at = None
                    self.on_utterance(self.participant_id, self.participant_name, utterance)
                elif has_words:
                    self._last_words_at = now

                if msg_type == "Error":
                    logger.error("[Deepgram] participant_id=%s error=%s", self.participant_id, data)
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception:
            logger.error("[Deepgram] Receiver failed participant_id=%s", self.participant_id, exc_info=True)

    async def close(self) -> None:
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass
        if self.receiver_task:
            self.receiver_task.cancel()


@app.websocket("/ws/control/{run_id}")
async def avatar_control_endpoint(websocket: WebSocket, run_id: str):
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    run["control_ws"] = websocket
    logger.info("[Control WS] Avatar connected run_id=%s", run_id)
    try:
        while True:
            event = await websocket.receive_json()
            event_type = event.get("type")
            turn_id = event.get("turn_id")
            if turn_id is not None and turn_id != run.get("active_turn_id"):
                continue
            if event_type == "avatar_speak_started":
                run["state"] = AgentState.SPEAKING
                timing = (run.get("turn_timing") or {}).pop(turn_id, None) if turn_id is not None else None
                if timing:
                    speak_started_at = time.monotonic()
                    dispatched_at = timing.get("dispatched_at")
                    finalized_at = timing.get("finalized_at")
                    if dispatched_at is not None:
                        parts = [f"dispatch->speaking={speak_started_at - dispatched_at:.2f}s (network + LiveKit + HeyGen TTS start)"]
                        if finalized_at is not None:
                            parts.append(f"total={speak_started_at - finalized_at:.2f}s (turn_finalize->speaking)")
                        logger.info("[TIMING] turn_id=%s %s", turn_id, " ".join(parts))
            elif event_type == "avatar_speak_ended":
                chunk_id = event.get("chunk_id")
                if chunk_id is not None:
                    # Sentence-by-sentence dispatch: this is one chunk of a
                    # longer answer, not necessarily the last. Unblock the
                    # streaming dispatch loop waiting on it; that loop — not
                    # this handler — decides when the whole turn is done and
                    # releases the floor.
                    chunk_event = (run.get("chunk_events") or {}).get(chunk_id)
                    if chunk_event is not None:
                        chunk_event.set()
                    continue
                # Legacy single-shot dispatch: this is the whole answer, so
                # it opens the follow-up window immediately, same as before.
                # _dispatch_reply appended it to history right before sending,
                # so the last bot turn there is what was just spoken.
                history = run.get("history") or []
                last_reply = history[-1]["text"] if history and history[-1].get("participant_id") == "bot" else ""
                _release_floor(run, reply_text=last_reply)
                watchdog = run.get("watchdog_task")
                if watchdog and not watchdog.done():
                    watchdog.cancel()
            elif event_type == "avatar_speak_interrupted":
                # Barge-in: unblock a streaming dispatch loop that may be
                # mid-answer waiting on a chunk that will now never arrive,
                # then release the floor same as the legacy path always did.
                for chunk_event in (run.get("chunk_events") or {}).values():
                    chunk_event.set()
                run["chunk_events"] = {}
                _release_floor(run)
                watchdog = run.get("watchdog_task")
                if watchdog and not watchdog.done():
                    watchdog.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if run.get("control_ws") is websocket:
            run["control_ws"] = None
        logger.info("[Control WS] Avatar disconnected run_id=%s", run_id)


@app.websocket("/ws/rep/{run_id}")
async def rep_console_endpoint(websocket: WebSocket, run_id: str):
    """The sales rep's read-mostly view of the agent.

    Deliberately a different socket from /ws/control: that one is the avatar's
    speech transport and holds a single slot. This one carries Level 1 cues and
    agent state to the rep's console and can drop at any time without affecting
    what happens in the meeting.
    """
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    run.setdefault("rep_sockets", set()).add(websocket)
    logger.info("[Rep WS] Console connected run_id=%s", run_id)

    # Replay a still-fresh insight so a console that connects late, or
    # reconnects after a refresh, does not miss the offer.
    insight = run.get("pending_insight") or {}
    if insight and (time.monotonic() - insight.get("created_at", 0)) <= AGENT_INSIGHT_TTL_S:
        try:
            await websocket.send_json({
                "type": "insight_available",
                "speaker": insight.get("speaker"),
                "topic": insight.get("question"),
            })
        except Exception:
            pass

    try:
        while True:
            # The console is not a control surface; speaking goes through the
            # invoke endpoint so it passes the same gates as any other turn.
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        (run.get("rep_sockets") or set()).discard(websocket)
        logger.info("[Rep WS] Console disconnected run_id=%s", run_id)


@app.websocket("/ws/recall/audio/{run_id}")
async def recall_separate_audio_endpoint(
    websocket: WebSocket,
    run_id: str,
    token: Optional[str] = Query(None),
):
    run = _ACTIVE_RUNS.get(run_id)
    if not run or not verify_recall_websocket(
        websocket.headers,
        run.get("recall_ws_token") or "",
        token or "",
    ):
        await websocket.close(code=4401)
        return
    if not DEEPGRAM_API_KEY:
        await websocket.close(code=1011)
        return

    await websocket.accept()
    user_context = await get_user_keywords_and_products(
        user_id=run.get("user_id"),
        client_id=run.get("client_id"),
        auth_token=run.get("token"),
    )
    user_context["bot_name"] = run.get("bot_name") or "Tom"
    run["user_context"] = user_context
    keywords = build_entity_catalog(user_context)
    transcribers: Dict[str, ParticipantTranscriber] = {}
    detectors: Dict[str, SustainedSpeechDetector] = {}
    logger.info("[Recall WS] Separate audio connected run_id=%s", run_id)

    try:
        while True:
            message = await websocket.receive_json()
            event_type = message.get("event")
            envelope = message.get("data") or {}
            payload = envelope.get("data") or envelope
            participant = payload.get("participant") or {}
            participant_id = str(participant.get("id") or "unknown")
            participant_name = (participant.get("name") or f"Participant {participant_id}").strip()

            if event_type == "participant_events.leave":
                transcriber = transcribers.pop(participant_id, None)
                detectors.pop(participant_id, None)
                stale = run.get("pending_turns", {}).pop(participant_id, None)
                if stale and stale.get("timer") and not stale["timer"].done():
                    stale["timer"].cancel()
                if transcriber:
                    await transcriber.close()
                continue
            if event_type != "audio_separate_raw.data":
                continue

            encoded = payload.get("buffer") or ""
            if not encoded:
                continue
            try:
                pcm = base64.b64decode(encoded)
            except Exception:
                logger.warning("[Recall WS] Invalid audio buffer participant_id=%s", participant_id)
                continue
            if not pcm:
                continue

            # First line of defense against self-audio. EchoSuppressor is the
            # second, and catches what name matching cannot: platform-decorated
            # names, and the avatar returning through a participant's speakers.
            if participant.get("is_bot") or participant_id == str(run.get("bot_participant_id") or ""):
                continue
            if participant_name.casefold() == (run.get("bot_name") or DEFAULT_BOT_NAME).casefold():
                continue

            detector = detectors.setdefault(
                participant_id,
                SustainedSpeechDetector(threshold_ms=AGENT_BARGE_IN_MS),
            )
            if run.get("state") in (AgentState.SPEAKING, AgentState.THINKING):
                if detector.feed(pcm):
                    await _interrupt_avatar(run, participant_id)
            else:
                detector.reset()

            transcriber = transcribers.get(participant_id)
            if not transcriber:
                transcriber = ParticipantTranscriber(
                    participant_id,
                    participant_name,
                    keywords,
                    lambda pid, name, utterance: _ingest_utterance(run, pid, name, utterance),
                )
                transcribers[participant_id] = transcriber
            await transcriber.send(pcm)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.error("[Recall WS] Separate audio failed run_id=%s", run_id, exc_info=True)
    finally:
        for entry in run.get("pending_turns", {}).values():
            timer = entry.get("timer")
            if timer and not timer.done():
                timer.cancel()
        run["pending_turns"] = {}
        await asyncio.gather(*(item.close() for item in transcribers.values()), return_exceptions=True)
        logger.info("[Recall WS] Separate audio disconnected run_id=%s", run_id)
