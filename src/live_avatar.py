import os
import io
import json
import struct
import base64
import asyncio
import logging
import re
import time
import uuid
import hashlib
import hmac
import wave
from typing import Optional, List, Dict, Any, Literal, AsyncGenerator
from urllib.parse import urlencode, quote

from src.supabase_client import get_user_keywords_and_products
from src.agent_policy import (
    AgentState,
    EchoSuppressor,
    FloorState,
    SpeechGovernor,
    SustainedSpeechDetector,
    apply_validated_corrections,
    build_entity_catalog,
    closest_entities,
    estimate_speech_seconds,
    is_directly_addressed,
    evaluate_turn,
    is_probably_incomplete,
    needs_context_resolution,
    normalize_reply,
    requires_company_knowledge,
)

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

from src import rag_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LiveAvatar-Spiked")

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API") or os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
SPIKED_BACKEND_URL = os.getenv("SPIKED_BACKEND_URL", "https://spikedai-production-application-409019309412.us-central1.run.app")
RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_WEBHOOK_SECRET = os.getenv("RECALL_WEBHOOK_SECRET", "")
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://us-west-2.recall.ai")
RECALL_WEBHOOK_URL = os.getenv(
    "RECALL_WEBHOOK_URL", 
    "https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript"
)
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID", "69cf601f-b35b-4d1b-a701-c854d223b5a5")
LIVEAVATAR_SANDBOX = os.getenv("LIVEAVATAR_SANDBOX", "false").lower() == "true"
# Simli AI avatar renderer (free plan: $10 signup credit + 50 min/month top-up).
# Simli only renders + lip-syncs; the audio is synthesized by Cartesia below.
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "")
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "")
SIMLI_BASE_URL = os.getenv("SIMLI_API_URL", "https://api.simli.ai")
SIMLI_MAX_SESSION_SECONDS = int(os.getenv("SIMLI_MAX_SESSION_SECONDS", "1800"))
SIMLI_MAX_IDLE_SECONDS = int(os.getenv("SIMLI_MAX_IDLE_SECONDS", "300"))
# Audio frames forwarded to the avatar page. 6000 bytes = 3000 samples = 187.5ms
# at 16kHz, which is Simli's preferred chunk size.
SIMLI_AUDIO_CHUNK_BYTES = int(os.getenv("SIMLI_AUDIO_CHUNK_BYTES", "6000"))
# Cartesia TTS (free plan). Output is pinned to raw PCM16 @ 16kHz mono, the
# exact format Simli's WebRTC endpoint consumes, so the page resamples nothing.
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_BASE_URL = os.getenv("CARTESIA_BASE_URL", "https://api.cartesia.ai")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-3.5")
CARTESIA_SAMPLE_RATE = int(os.getenv("CARTESIA_SAMPLE_RATE", "16000"))
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", 
    "https://spiked-ai-liveavatar-409019309412.us-central1.run.app"
)
AGENT_BARGE_IN_MS = int(os.getenv("AGENT_BARGE_IN_MS", "500"))
AGENT_ENDPOINTING_MS = int(os.getenv("AGENT_ENDPOINTING_MS", "250"))
AGENT_UTTERANCE_END_MS = int(os.getenv("AGENT_UTTERANCE_END_MS", "1000"))
AGENT_MAX_REPLY_WORDS = int(os.getenv("AGENT_MAX_REPLY_WORDS", "45"))
DEFAULT_BOT_NAME = os.getenv("BOT_NAME", "Tom").strip() or "Tom"

# Turn detection. Fragments from one speaker are merged before the gate runs, so a
# single sentence split by a pause cannot produce two replies.
AGENT_TURN_MERGE_MS = int(os.getenv("AGENT_TURN_MERGE_MS", "250"))
AGENT_TURN_MERGE_INCOMPLETE_MS = int(os.getenv("AGENT_TURN_MERGE_INCOMPLETE_MS", "700"))
# Floor control. 1 allows a single nameless follow-up shortly after the agent
# finishes, so natural conversation does not require chanting the wake name.
AGENT_FOLLOWUP_WINDOW_MS = int(os.getenv("AGENT_FOLLOWUP_WINDOW_MS", "10000"))
AGENT_MAX_FOLLOWUPS = int(os.getenv("AGENT_MAX_FOLLOWUPS", "1"))
# Speech governor: hard ceiling on reply frequency and repetition.
AGENT_REPLY_COOLDOWN_MS = int(os.getenv("AGENT_REPLY_COOLDOWN_MS", "1500"))
AGENT_MAX_REPLIES_PER_WINDOW = int(os.getenv("AGENT_MAX_REPLIES_PER_WINDOW", "5"))
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
    bot_name: str = Field(default=DEFAULT_BOT_NAME, description="Name of the bot in the meeting")
    avatar_id: Optional[str] = Field(default=None, description="Specific LiveAvatar avatar ID")


class TranscriptCorrection(BaseModel):
    raw: str = ""
    replacement: str = ""
    confidence: float = 0.0


class TurnAnalysis(BaseModel):
    response_action: Literal["respond", "acknowledge", "silent"] = "respond"
    intent: Literal["company_knowledge", "meeting_context", "social", "command"]
    resolved_query: str
    corrections: List[TranscriptCorrection] = Field(default_factory=list)

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

async def query_spiked_rag(question: str, auth_token: str, client_id: Optional[str] = None) -> str:
    """
    Executes Document RAG by querying SpikedAI's knowledge base.
    Uses rag_client for authenticated queries with source_ids.
    """
    logger.info("[RAG] Querying knowledge base client_id=%s query_chars=%d", client_id, len(question))
    
    try:
        result = await rag_client.query_rag(question, client_id=client_id)
        if result:
            logger.info("[RAG] Received grounded answer (%d chars)", len(result))
            return result
        return "No specific documentation found for this query."
    except Exception as e:
        logger.error(f"[RAG] Error querying knowledge base: {e}", exc_info=True)
        return "An error occurred while accessing the company knowledge base."

# ---------------------------------------------------------------------------
# Streaming pipeline: route turn → stream Gemini → stream Cartesia → avatar
# ---------------------------------------------------------------------------

async def _route_and_stream(
    transcript: str,
    speaker: str,
    conversation_history: List[Dict[str, str]],
    auth_token: str,
    client_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    run: Optional[Dict[str, Any]] = None,
    turn_id: Optional[int] = None,
) -> Optional[str]:
    """Route the turn then stream the answer directly into Cartesia TTS.

    This replaces the two-step ``process_transcript_with_gemini`` →
    ``_stream_tts_for_reply`` path with a single pipelined flow so the
    avatar starts speaking as soon as the first sentence is ready, while
    Gemini is still generating the rest of the answer.
    """
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

    analysis: Optional[TurnAnalysis] = None
    rag_task: Optional[asyncio.Task] = None

    # --- Routing ---
    try:
        if is_directly_addressed(transcript, bot_name):
            if requires_company_knowledge(transcript, catalog):
                logger.info("[Agent Route] Fast path: company knowledge (no routing LLM)")
                analysis = TurnAnalysis(
                    response_action="respond",
                    intent="company_knowledge",
                    resolved_query=transcript,
                    corrections=[],
                )
                rag_task = asyncio.create_task(
                    query_spiked_rag(transcript, auth_token, client_id)
                )
            else:
                remainder = _strip_wake_name(transcript, bot_name).strip().lower()
                _social_words = {"hello", "hi", "hey", "good", "morning", "afternoon", "evening", "how", "what's", "whats"}
                if any(w in remainder.split() for w in _social_words) or remainder.rstrip("?!") in ("hello", "hi", "hey"):
                    logger.info("[Agent Route] Fast path: social greeting (no routing LLM)")
                    greeting = normalize_reply(f"Hey {speaker}! How can I help you today?")
                    if run and turn_id is not None:
                        await _stream_tts_for_reply(run, turn_id, greeting)
                    return greeting

        if analysis is None:
            analysis_prompt = f"""Decide whether {bot_name} should respond, then classify and normalize this wake-name-matched meeting turn.
Company: {company_name}
Offerings: {products_services or product_domain}
Verified entity candidates: {candidate_entities}
Recent finalized conversation:
{history_text}
Current speaker: {speaker}
Raw ASR: {transcript}

Set response_action to:
- respond: the speaker directly asks {bot_name} a question, requests an action/opinion, or gives {bot_name} a command.
- acknowledge: the speaker directly gives {bot_name} information or a simple instruction that only needs a brief confirmation.
- silent: {bot_name} is merely mentioned, quoted, discussed in third person, explicitly told not to answer, or the request is directed to somebody else.
The presence of the wake name alone is not sufficient. Prefer silent when the addressee is ambiguous.
Use company_knowledge for company/product/features/pricing/security/SLA/integration questions.
Use meeting_context only for questions about what meeting participants said or discussed.
Use social for greetings and audio checks. Use command for stop/wait/repeat commands.
Resolve pronouns and omitted context only in resolved_query. Propose corrections only from the verified entity candidates."""
            try:
                analysis_response = await gemini_client.aio.models.generate_content(
                    model=preferred_model,
                    contents=analysis_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=220,
                        response_mime_type="application/json",
                        response_schema=TurnAnalysis,
                    ),
                )
                parsed = getattr(analysis_response, "parsed", None)
                if isinstance(parsed, TurnAnalysis):
                    analysis = parsed
                elif parsed is not None:
                    analysis = TurnAnalysis.model_validate(parsed)
                else:
                    analysis = TurnAnalysis.model_validate_json(analysis_response.text)
            except Exception:
                logger.warning("[Agent Route] Structured routing failed; defaulting substantive turn to RAG", exc_info=True)
                analysis = TurnAnalysis(
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
            return None
        if analysis.response_action == "acknowledge":
            if run and turn_id is not None:
                await _stream_tts_for_reply(run, turn_id, "Understood.")
            return "Understood."
        if analysis.intent not in ("social", "command") and requires_company_knowledge(transcript, catalog):
            analysis.intent = "company_knowledge"
    except Exception as e:
        logger.error(f"[Gemini Agent] Inference error with google-genai SDK: {e}", exc_info=True)
        return None

    # --- Streaming generate + TTS ---
    if run and turn_id is not None:
        return await _generate_and_stream_reply(
            run=run,
            turn_id=turn_id,
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
            rag_precomputed=rag_task,
        )
    else:
        return await _generate_grounded_reply(
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
            rag_precomputed=rag_task,
        )


# ---------------------------------------------------------------------------
# Gemini turn routing with deterministic RAG execution
# ---------------------------------------------------------------------------

async def process_transcript_with_gemini(
    transcript: str,
    speaker: str,
    conversation_history: List[Dict[str, str]],
    auth_token: str,
    client_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Route an already-addressed, complete turn and produce a short spoken reply."""
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
    try:
        # Fast path: skip the classification round trip when the turn is
        # unambiguously addressed to the agent. The answer model already
        # receives conversation history and can resolve pronouns on its own,
        # so we no longer require needs_context_resolution to be False.
        # Also handles greetings directly, skipping the routing LLM entirely.
        if is_directly_addressed(transcript, bot_name):
            if requires_company_knowledge(transcript, catalog):
                logger.info("[Agent Route] Fast path: company knowledge (no routing LLM)")
                analysis = TurnAnalysis(
                    response_action="respond",
                    intent="company_knowledge",
                    resolved_query=transcript,
                    corrections=[],
                )
                rag_task = asyncio.create_task(
                    query_spiked_rag(transcript, auth_token, client_id)
                )
                return await _generate_grounded_reply(
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
                    rag_precomputed=rag_task,
                )
            # Greeting / social fast path: "Tom, hello", "Tom, how are you"
            remainder = _strip_wake_name(transcript, bot_name).strip().lower()
            _social_words = {"hello", "hi", "hey", "good", "morning", "afternoon", "evening", "how", "what's", "whats"}
            if any(w in remainder.split() for w in _social_words) or remainder.rstrip("?!") in ("hello", "hi", "hey"):
                logger.info("[Agent Route] Fast path: social greeting (no routing LLM)")
                return normalize_reply(f"Hey {speaker}! How can I help you today?")

        analysis_prompt = f"""Decide whether {bot_name} should respond, then classify and normalize this wake-name-matched meeting turn.
Company: {company_name}
Offerings: {products_services or product_domain}
Verified entity candidates: {candidate_entities}
Recent finalized conversation:
{history_text}
Current speaker: {speaker}
Raw ASR: {transcript}

Set response_action to:
- respond: the speaker directly asks {bot_name} a question, requests an action/opinion, or gives {bot_name} a command.
- acknowledge: the speaker directly gives {bot_name} information or a simple instruction that only needs a brief confirmation.
- silent: {bot_name} is merely mentioned, quoted, discussed in third person, explicitly told not to answer, or the request is directed to somebody else.
The presence of the wake name alone is not sufficient. Prefer silent when the addressee is ambiguous.
Use company_knowledge for company/product/features/pricing/security/SLA/integration questions.
Use meeting_context only for questions about what meeting participants said or discussed.
Use social for greetings and audio checks. Use command for stop/wait/repeat commands.
Resolve pronouns and omitted context only in resolved_query. Propose corrections only from the verified entity candidates."""
        try:
            analysis_response = await gemini_client.aio.models.generate_content(
                model=preferred_model,
                contents=analysis_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=220,
                    response_mime_type="application/json",
                    response_schema=TurnAnalysis,
                ),
            )
            parsed = getattr(analysis_response, "parsed", None)
            if isinstance(parsed, TurnAnalysis):
                analysis = parsed
            elif parsed is not None:
                analysis = TurnAnalysis.model_validate(parsed)
            else:
                analysis = TurnAnalysis.model_validate_json(analysis_response.text)
        except Exception:
            logger.warning("[Agent Route] Structured routing failed; defaulting substantive turn to RAG", exc_info=True)
            analysis = TurnAnalysis(
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
            return None
        if analysis.response_action == "acknowledge":
            return "Understood."

        if analysis.intent not in ("social", "command") and requires_company_knowledge(transcript, catalog):
            analysis.intent = "company_knowledge"

        return await _generate_grounded_reply(
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
        )
    except Exception as e:
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
    rag_precomputed: Optional[asyncio.Task] = None,
) -> Optional[str]:
    """Run RAG when the intent requires it, then produce the spoken reply.

    When *rag_precomputed* is an asyncio.Task (fired in parallel with the caller),
    it is awaited instead of launching a new query — this shaves the full RAG
    round-trip off the hot path.
    """

    corrections = [item.model_dump() for item in analysis.corrections]
    corrected_transcript = apply_validated_corrections(transcript, corrections, catalog)
    resolved_query = analysis.resolved_query.strip() or corrected_transcript
    detailed_request = any(
        phrase in transcript.casefold()
        for phrase in ("more detail", "in detail", "elaborate", "explain fully", "deep dive")
    )
    reply_word_limit = 90 if detailed_request else AGENT_MAX_REPLY_WORDS
    reply_sentence_limit = 4 if detailed_request else 2
    rag_result = ""
    rag_used = analysis.intent == "company_knowledge"
    rag_fallback = False
    if rag_used:
        if rag_precomputed is not None:
            rag_result = await rag_precomputed
        else:
            rag_result = await query_spiked_rag(resolved_query, auth_token, client_id)
        if not rag_result or any(marker in rag_result.lower() for marker in (
            "could not retrieve", "no specific documentation", "an error occurred",
            "no relevant information",
        )):
            logger.info("[Agent Route] RAG empty; falling back to general knowledge")
            rag_result = ""
            rag_fallback = True

    answer_prompt = f"""You are {bot_name}, a natural and conversational meeting participant representing {company_name}.
Answer {speaker}'s addressed turn as a real person would in a live conversation — warm, clear, and concise.
Use at most {reply_sentence_limit} sentences and {reply_word_limit} words. No markdown, lists, or filler.
Use the speaker's first name only if it improves a greeting or clarification.
Never invent facts about {company_name}'s products or services.

CRITICAL — end every reply with a short natural question that keeps the conversation flowing. The question must be:
- Contextually relevant to what was just discussed
- Something the other person would naturally want to answer
- Short (under 12 words)
Good examples: "Does that align with what you had in mind?", "Want me to go deeper on any of that?", "Shall I walk you through the next steps?"
Bad examples: generic "How can I help?" or "Any other questions?" — these feel robotic.

Intent: {analysis.intent}
Corrected turn: {corrected_transcript}
Resolved meaning: {resolved_query}
Recent finalized conversation:
{history_text}
Verified RAG facts (the only source for company-specific facts):
{rag_result if rag_used and not rag_fallback else '(not available — use your general knowledge)'}
{'Note: no company documentation was found for this question. Answer using your general knowledge, but be clear you are speaking generally rather than on behalf of ' + company_name + '.' if rag_fallback else ''}"""
    response = await gemini_client.aio.models.generate_content(
        model=preferred_model,
        contents=answer_prompt,
        config=types.GenerateContentConfig(max_output_tokens=120),
    )
    reply = normalize_reply(response.text or "", reply_word_limit, reply_sentence_limit)
    logger.info(
        "[Agent Route] speaker=%s intent=%s rag_used=%s corrections=%d reply_words=%d",
        speaker,
        analysis.intent,
        rag_used,
        len(corrections),
        len(reply.split()),
    )
    return reply or None

# ---------------------------------------------------------------------------
# API Endpoints: Session & Bot Creation
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "LiveAvatar-Spiked",
        "stt_provider": "groq-whisper-large-v3-turbo" if GROQ_API_KEY else "none",
        "groq_configured": bool(GROQ_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "recall_configured": bool(RECALL_API_KEY),
        "simli_configured": bool(SIMLI_API_KEY and SIMLI_FACE_ID),
        "cartesia_configured": bool(CARTESIA_API_KEY),
        "spiked_backend_url": SPIKED_BACKEND_URL,
        "recall_webhook_url": RECALL_WEBHOOK_URL,
        "public_base_url": PUBLIC_BASE_URL
    }

@app.get("/api/runs/{run_id}/credentials")
async def get_run_credentials(run_id: str, token: Optional[str] = Query(None)):
    """Provides Simli session credentials to avatar.html when loaded by Recall."""
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or expired")
    return {
        "session_id": run.get("session_id"),
        "livekit_url": run.get("livekit_url"),
        "livekit_token": run.get("livekit_token"),
        "simli_session_token": run.get("simli_session_token"),
        "simli_base_url": run.get("simli_base_url") or SIMLI_BASE_URL,
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

async def create_simli_session() -> str:
    """Create a Simli compose session token (free-plan friendly).

    Simli only renders the avatar and lip-syncs; the audio driving it is
    synthesized by Cartesia and streamed to the page over the control WS.
    """
    if not SIMLI_API_KEY:
        raise HTTPException(status_code=500, detail="SIMLI_API_KEY is not configured")
    if not SIMLI_FACE_ID:
        raise HTTPException(status_code=500, detail="SIMLI_FACE_ID is not configured")

    url = f"{SIMLI_BASE_URL.rstrip('/')}/compose/token"
    headers = {"Content-Type": "application/json", "x-simli-api-key": SIMLI_API_KEY}
    payload = {
        "faceId": SIMLI_FACE_ID,
        "maxSessionLength": SIMLI_MAX_SESSION_SECONDS,
        "maxIdleTime": SIMLI_MAX_IDLE_SECONDS,
        "handleSilence": True,
        "audioInputFormat": "pcm16",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code not in (200, 201):
        logger.error("[Simli] token creation failed: %s", resp.text)
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])

    data = resp.json()
    token = data.get("session_token") if isinstance(data, dict) else None
    if not token or not isinstance(token, str) or token in ("FAIL TOKEN", "FAIL"):
        logger.error("[Simli] token response invalid: %s", data)
        raise HTTPException(status_code=500, detail="Simli session token creation failed")

    logger.info("[Simli] Session token created face_id=%s", SIMLI_FACE_ID)
    return token


async def stream_cartesia_tts(text: str) -> AsyncGenerator[bytes, None]:
    """Synthesize `text` to raw 16kHz mono PCM16 and yield it in chunks.

    Uses the Cartesia /tts/bytes endpoint: one POST per utterance, with the raw
    PCM streamed back as the response body. output_format is pinned to what
    Simli's WebRTC endpoint consumes (pcm_s16le @ 16kHz) so nothing is
    resampled before the avatar page forwards it.
    """
    if not CARTESIA_API_KEY:
        raise RuntimeError("CARTESIA_API_KEY is not configured")

    url = f"{CARTESIA_BASE_URL.rstrip('/')}/tts/bytes"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CARTESIA_API_KEY}",
        "Cartesia-Version": "2026-03-01",
    }
    payload = {
        "model_id": CARTESIA_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": CARTESIA_SAMPLE_RATE,
        },
        "language": "en",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                logger.error("[Cartesia] TTS failed: %s %s", resp.status_code, body[:300])
                raise RuntimeError(f"Cartesia TTS failed ({resp.status_code})")
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


async def _ws_send(run: Dict[str, Any], payload: dict) -> bool:
    """Serialize all writes to the avatar control socket through a per-run lock."""
    ws = run.get("control_ws")
    if not ws:
        return False
    lock: asyncio.Lock = run.setdefault("ws_send_lock", asyncio.Lock())
    async with lock:
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            logger.warning("[WS] Control socket send failed: %s", payload.get("type"))
            return False


async def _speak_reply(run: Dict[str, Any], turn_id: int, text: str) -> None:
    """Turn `text` into spoken audio and stream it to the avatar page.

    Flow: Cartesia TTS (raw PCM16 @ 16kHz) -> base64 frames over the control WS
    -> avatar.js -> Simli WebRTC for lip-sync + playback. The page echoes back
    avatar_speak_started / avatar_speak_ended so the floor state stays accurate.
    """
    if not run.get("control_ws"):
        logger.warning("[TTS] Control socket unavailable turn_id=%s", turn_id)
        return

    async def emit(payload: dict) -> None:
        await _ws_send(run, payload)

    try:
        await emit({"type": "avatar_speak", "text": text, "turn_id": turn_id})
        frame = bytearray()
        async for chunk in stream_cartesia_tts(text):
            frame.extend(chunk)
            while len(frame) >= SIMLI_AUDIO_CHUNK_BYTES:
                piece = bytes(frame[:SIMLI_AUDIO_CHUNK_BYTES])
                del frame[:SIMLI_AUDIO_CHUNK_BYTES]
                await emit({
                    "type": "avatar_audio",
                    "data": base64.b64encode(piece).decode("ascii"),
                    "turn_id": turn_id,
                })
        if frame:
            await emit({
                "type": "avatar_audio",
                "data": base64.b64encode(bytes(frame)).decode("ascii"),
                "turn_id": turn_id,
            })
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})
        logger.info("[TTS] Streamed reply turn_id=%s chars=%d", turn_id, len(text))
    except asyncio.CancelledError:
        logger.info("[TTS] Cancelled reply turn_id=%s", turn_id)
        raise
    except Exception:
        logger.error("[TTS] Failed to stream reply turn_id=%s", turn_id, exc_info=True)
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})


async def _generate_and_stream_reply(
    run: Dict[str, Any],
    turn_id: int,
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
    rag_precomputed: Optional[asyncio.Task] = None,
) -> Optional[str]:
    """Stream Gemini tokens directly into Cartesia TTS for minimal latency.

    Instead of waiting for the full answer then sending it to TTS, this function
    buffers streaming Gemini output into sentences and sends each sentence to
    Cartesia as soon as it's complete. The avatar starts speaking while Gemini
    is still generating the rest of the answer.
    """
    corrections = [item.model_dump() for item in analysis.corrections]
    corrected_transcript = apply_validated_corrections(transcript, corrections, catalog)
    resolved_query = analysis.resolved_query.strip() or corrected_transcript
    detailed_request = any(
        phrase in transcript.casefold()
        for phrase in ("more detail", "in detail", "elaborate", "explain fully", "deep dive")
    )
    reply_word_limit = 90 if detailed_request else AGENT_MAX_REPLY_WORDS
    reply_sentence_limit = 4 if detailed_request else 2
    rag_result = ""
    rag_used = analysis.intent == "company_knowledge"
    rag_fallback = False
    if rag_used:
        if rag_precomputed is not None:
            rag_result = await rag_precomputed
        else:
            rag_result = await query_spiked_rag(resolved_query, auth_token, client_id)
        if not rag_result or any(marker in rag_result.lower() for marker in (
            "could not retrieve", "no specific documentation", "an error occurred",
            "no relevant information",
        )):
            logger.info("[Agent Route] RAG empty; falling back to general knowledge")
            rag_result = ""
            rag_fallback = True

    answer_prompt = f"""You are {bot_name}, a natural and conversational meeting participant representing {company_name}.
Answer {speaker}'s addressed turn as a real person would in a live conversation — warm, clear, and concise.
Use at most {reply_sentence_limit} sentences and {reply_word_limit} words. No markdown, lists, or filler.
Use the speaker's first name only if it improves a greeting or clarification.
Never invent facts about {company_name}'s products or services.

CRITICAL — end every reply with a short natural question that keeps the conversation flowing. The question must be:
- Contextually relevant to what was just discussed
- Something the other person would naturally want to answer
- Short (under 12 words)
Good examples: "Does that align with what you had in mind?", "Want me to go deeper on any of that?", "Shall I walk you through the next steps?"
Bad examples: generic "How can I help?" or "Any other questions?" — these feel robotic.

Intent: {analysis.intent}
Corrected turn: {corrected_transcript}
Resolved meaning: {resolved_query}
Recent finalized conversation:
{history_text}
Verified RAG facts (the only source for company-specific facts):
{rag_result if rag_used and not rag_fallback else '(not available — use your general knowledge)'}
{'Note: no company documentation was found for this question. Answer using your general knowledge, but be clear you are speaking generally rather than on behalf of ' + company_name + '.' if rag_fallback else ''}"""

    if not run.get("control_ws"):
        logger.warning("[Agent] Control socket unavailable for streaming turn_id=%s", turn_id)
        return None

    async def emit(payload: dict) -> None:
        await _ws_send(run, payload)

    full_text = ""
    try:
        sentence_buf = ""
        word_count = 0
        await emit({"type": "avatar_speak", "text": "", "turn_id": turn_id})
        stream = await gemini_client.aio.models.generate_content_stream(
            model=preferred_model,
            contents=answer_prompt,
            config=types.GenerateContentConfig(max_output_tokens=120),
        )
        async for chunk in stream:
            token = getattr(chunk, "text", None) or ""
            if not token:
                continue
            full_text += token
            sentence_buf += token
            word_count += len(token.split())
            while _SENTENCE_BREAKS.search(sentence_buf):
                match = _SENTENCE_BREAKS.search(sentence_buf)
                sentence, sentence_buf = sentence_buf[:match.end()].strip(), sentence_buf[match.end():]
                if sentence and word_count <= reply_word_limit:
                    await _send_sentence_tts(emit, sentence, turn_id)
            if word_count >= reply_word_limit:
                break
        remaining = sentence_buf.strip()
        if remaining and word_count <= reply_word_limit + 10:
            await _send_sentence_tts(emit, remaining, turn_id)
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})
        reply = normalize_reply(full_text, reply_word_limit, reply_sentence_limit)
        logger.info(
            "[Agent Stream] speaker=%s intent=%s rag_used=%s words=%d",
            speaker, analysis.intent, rag_used, len(reply.split()),
        )
        return reply or None
    except asyncio.CancelledError:
        logger.info("[Agent Stream] Cancelled turn_id=%s", turn_id)
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})
        raise
    except Exception:
        logger.error("[Agent Stream] Failed turn_id=%s", turn_id, exc_info=True)
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})
        return None


async def _send_sentence_tts(emit, sentence: str, turn_id: int) -> None:
    """Send one sentence through Cartesia TTS and stream PCM to avatar."""
    if not sentence.strip():
        return
    frame = bytearray()
    try:
        async for tts_chunk in stream_cartesia_tts(sentence):
            frame.extend(tts_chunk)
            while len(frame) >= SIMLI_AUDIO_CHUNK_BYTES:
                piece = bytes(frame[:SIMLI_AUDIO_CHUNK_BYTES])
                del frame[:SIMLI_AUDIO_CHUNK_BYTES]
                await emit({
                    "type": "avatar_audio",
                    "data": base64.b64encode(piece).decode("ascii"),
                    "turn_id": turn_id,
                })
        if frame:
            await emit({
                "type": "avatar_audio",
                "data": base64.b64encode(bytes(frame)).decode("ascii"),
                "turn_id": turn_id,
            })
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("[TTS] Sentence TTS failed: %s", sentence[:40], exc_info=True)


# Sentence-boundary characters for splitting streaming Gemini output into
# Cartesia-friendly chunks. Each chunk is small enough to keep latency low
# while producing natural prosody.
_SENTENCE_BREAKS = re.compile(r'(?<=[.!?])\s+')


async def _stream_tts_for_reply(
    run: Dict[str, Any],
    turn_id: int,
    text: str,
) -> None:
    """Stream TTS for a complete answer, breaking it into sentence-sized
    Cartesia calls so the first sentence starts speaking immediately."""
    if not run.get("control_ws"):
        return

    async def emit(payload: dict) -> None:
        await _ws_send(run, payload)

    try:
        await emit({"type": "avatar_speak", "text": text, "turn_id": turn_id})
        sentences = _SENTENCE_BREAKS.split(text)
        # Merge very short fragments to avoid excessive Cartesia round-trips.
        merged: List[str] = []
        buf = ""
        for s in sentences:
            buf = f"{buf} {s}".strip() if buf else s
            if len(buf) >= 60 or s.rstrip().endswith((".", "!", "?")):
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)

        for sentence in merged:
            if not sentence.strip():
                continue
            frame = bytearray()
            try:
                async for chunk in stream_cartesia_tts(sentence):
                    frame.extend(chunk)
                    while len(frame) >= SIMLI_AUDIO_CHUNK_BYTES:
                        piece = bytes(frame[:SIMLI_AUDIO_CHUNK_BYTES])
                        del frame[:SIMLI_AUDIO_CHUNK_BYTES]
                        await emit({
                            "type": "avatar_audio",
                            "data": base64.b64encode(piece).decode("ascii"),
                            "turn_id": turn_id,
                        })
                if frame:
                    await emit({
                        "type": "avatar_audio",
                        "data": base64.b64encode(bytes(frame)).decode("ascii"),
                        "turn_id": turn_id,
                    })
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("[TTS] Sentence TTS failed, continuing: %s", sentence[:40], exc_info=True)
                continue
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})
        logger.info("[TTS] Streamed reply turn_id=%s sentences=%d chars=%d", turn_id, len(merged), len(text))
    except asyncio.CancelledError:
        logger.info("[TTS] Cancelled reply turn_id=%s", turn_id)
        raise
    except Exception:
        logger.error("[TTS] Failed to stream reply turn_id=%s", turn_id, exc_info=True)
        await emit({"type": "avatar_speak_end", "turn_id": turn_id})


async def _deploy_live_avatar_bot(
    meeting_url: str,
    token: str,
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    bot_name: str = DEFAULT_BOT_NAME,
    avatar_id: Optional[str] = None,
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

        # 1. Create a Simli session (free plan). Simli renders + lip-syncs only;
        #    Groq Whisper STT, Gemini LLM and Cartesia TTS all run in this process.
        simli_session_token = await create_simli_session()

        # 2. Store session credentials for Recall's avatar.html
        _ACTIVE_RUNS[run_id] = {
            "run_id": run_id,
            "user_id": user_id,
            "client_id": client_id,
            "token": token,
            "session_id": None,
            "livekit_url": None,
            "livekit_token": None,
            "simli_session_token": simli_session_token,
            "simli_base_url": SIMLI_BASE_URL,
            "bot_name": bot_name,
            "recall_ws_token": recall_ws_token,
            "state": AgentState.LISTENING,
            "history": [],
            "turn_counter": 0,
            "control_ws": None,
            "active_response_task": None,
            "watchdog_task": None,
            "pending_turns": {},
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
                "simli_session_token": simli_session_token,
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
        bot_name = DEFAULT_BOT_NAME
        avatar_id = None

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            meeting_url = body.get("meeting_url", "")
            client_id = body.get("client_id")
            bot_name = body.get("bot_name", DEFAULT_BOT_NAME)
            avatar_id = body.get("avatar_id")
            if not token:
                token = body.get("token", "")
        else:
            form = await request.form()
            meeting_url = form.get("meeting_url", "")
            client_id = form.get("client_id")
            bot_name = form.get("bot_name", DEFAULT_BOT_NAME)
            avatar_id = form.get("avatar_id")
            if not token:
                token = form.get("token", "")

        if not meeting_url:
            raise HTTPException(status_code=400, detail="meeting_url is required")

        result = await _deploy_live_avatar_bot(
            meeting_url=meeting_url,
            token=token,
            client_id=client_id,
            bot_name=bot_name,
            avatar_id=avatar_id,
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
        bot_name=payload.bot_name,
        avatar_id=payload.avatar_id,
        request=request
    )

@app.post("/remove-bot/{bot_id}")
@app.post("/leave-call/{bot_id}")
async def leave_call_endpoint(bot_id: str):
    """Instructs Recall bot to leave meeting call immediately."""
    if not RECALL_API_KEY:
        raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")

    headers = {"Authorization": f"Token {RECALL_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{RECALL_BASE_URL.rstrip('/')}/api/v1/bot/{bot_id}/leave_call/",
            headers=headers
        )
        return {"ok": True, "status": resp.status_code}

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
    """Mirror the gate's verdict onto the avatar page's debug overlay."""
    if not AGENT_DEBUG_OVERLAY or not run.get("control_ws"):
        return

    async def send() -> None:
        await _ws_send(run, {
            "type": "heard",
            "speaker": speaker,
            "text": text,
            "reply": reply,
            "reason": reason,
        })

    asyncio.create_task(send())


def _release_floor(run: Dict[str, Any], now: Optional[float] = None) -> None:
    """Return the agent to LISTENING and open the follow-up window."""
    run["state"] = AgentState.LISTENING
    floor: FloorState = run["floor"]
    floor.last_bot_finished_at = now if now is not None else time.monotonic()


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

    if not run.get("control_ws"):
        return
    run["state"] = AgentState.INTERRUPTING
    logger.info("[Barge-In] participant_id=%s sustained_ms=%d", participant_id, AGENT_BARGE_IN_MS)
    await _ws_send(run, {
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
                _finalize_turn(run, participant_id, participant_name, current["text"])
        except asyncio.CancelledError:
            pass

    pending[participant_id] = {
        "text": transcript,
        "timer": asyncio.create_task(flush_after_pause()),
    }


def _finalize_turn(
    run: Dict[str, Any],
    participant_id: str,
    participant_name: str,
    transcript: str,
) -> None:
    now = time.monotonic()
    bot_name = run.get("bot_name") or "Tom"
    floor: FloorState = run["floor"]
    governor: SpeechGovernor = run["governor"]

    decision = evaluate_turn(
        transcript,
        participant_id,
        bot_name,
        floor,
        now,
        followup_window_seconds=AGENT_FOLLOWUP_WINDOW_MS / 1000,
        max_consecutive_followups=AGENT_MAX_FOLLOWUPS,
        agent_is_idle=run.get("state") == AgentState.LISTENING,
    )

    history = run.setdefault("history", [])
    history_snapshot = list(history[-20:])
    history.append({"speaker": participant_name, "participant_id": participant_id, "text": transcript})
    del history[:-40]

    logger.info(
        "[Turn Gate] participant_id=%s participant_name=%s bot_name=%s reply=%s reason=%s matched_name=%s",
        participant_id,
        participant_name,
        bot_name,
        decision.should_reply,
        decision.reason,
        decision.matched_name,
    )
    _push_heard(run, participant_name, transcript, decision.should_reply, decision.reason)
    if not decision.should_reply:
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

    async def respond() -> None:
        try:
            # Streaming path: route + generate + TTS all pipelined so
            # the avatar starts speaking before Gemini finishes generating.
            # _route_and_stream handles TTS internally when run+turn_id are given.
            answer = await _route_and_stream(
                transcript=transcript,
                speaker=participant_name,
                conversation_history=history_snapshot,
                auth_token=run.get("token") or "",
                client_id=run.get("client_id"),
                user_context=run.get("user_context") or {},
                run=run,
                turn_id=turn_id,
            )
            if turn_id != run.get("active_turn_id"):
                logger.info("[Agent] Discarded stale response turn_id=%s", turn_id)
                return
            if not answer:
                _release_floor(run)
                return

            spoken_at = time.monotonic()
            # Note: TTS was already streamed by _route_and_stream, so we
            # skip _stream_tts_for_reply here and just record bookkeeping.
            control_ws = run.get("control_ws")
            if not control_ws:
                logger.warning("[Agent] Avatar control socket unavailable for turn_id=%s", turn_id)
                _release_floor(run, spoken_at)
                return

            history.append({"speaker": bot_name, "participant_id": "bot", "text": answer})
            del history[:-40]
            run["echo"].note_bot_speech(answer, spoken_at)
            governor.note_reply(answer, spoken_at)
            run["watchdog_task"] = asyncio.create_task(_speak_watchdog(run, turn_id, answer))
        except asyncio.CancelledError:
            logger.info("[Agent] Cancelled superseded turn_id=%s", turn_id)
        except Exception:
            _release_floor(run)
            logger.error("[Agent] Failed addressed turn_id=%s", turn_id, exc_info=True)

    run["active_response_task"] = asyncio.create_task(respond())


class ParticipantTranscriber:
    """Streaming STT via Groq Whisper with energy-based VAD."""

    _SILENCE_THRESHOLD = 350
    _SILENCE_TIMEOUT_MS = 400
    _MIN_SPEECH_MS = 200
    _MAX_SEGMENT_MS = 12_000
    _SAMPLE_RATE = 16000
    _BYTES_PER_SAMPLE = 2

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
        self._pcm_buf = bytearray()
        self._has_speech = False
        self._last_speech_time: float = 0.0
        self._started_at: float = 0.0
        self._lock = asyncio.Lock()
        self._closed = False
        self._flush_task: Optional[asyncio.Task] = None

    async def ensure_started(self) -> None:
        pass

    async def send(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        async with self._lock:
            self._pcm_buf.extend(pcm)
            now = time.monotonic()
            rms = self._pcm_rms(pcm)
            if rms >= self._SILENCE_THRESHOLD:
                if not self._has_speech:
                    self._started_at = now
                self._has_speech = True
                self._last_speech_time = now
            elif self._has_speech:
                elapsed_ms = (now - self._last_speech_time) * 1000
                if elapsed_ms >= self._SILENCE_TIMEOUT_MS:
                    await self._flush()
                    return
                total_ms = (now - self._started_at) * 1000
                if total_ms >= self._MAX_SEGMENT_MS:
                    await self._flush()
                    return

    async def close(self) -> None:
        self._closed = True
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        async with self._lock:
            if self._pcm_buf and self._has_speech:
                await self._flush_nolock()

    @staticmethod
    def _pcm_rms(pcm: bytes) -> float:
        if len(pcm) < 2:
            return 0.0
        count = len(pcm) // 2
        samples = struct.unpack(f"<{count}h", pcm[: count * 2])
        return (sum(s * s for s in samples) / count) ** 0.5

    async def _flush(self) -> None:
        await self._flush_nolock()

    async def _flush_nolock(self) -> None:
        pcm = bytes(self._pcm_buf)
        self._pcm_buf.clear()
        self._has_speech = False
        if not pcm:
            return
        duration_ms = len(pcm) / (self._SAMPLE_RATE * self._BYTES_PER_SAMPLE / 1000)
        if duration_ms < self._MIN_SPEECH_MS:
            return
        self._flush_task = asyncio.create_task(self._transcribe(pcm))

    async def _transcribe(self, pcm: bytes) -> None:
        if self._closed or not GROQ_API_KEY:
            return
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._SAMPLE_RATE)
            wf.writeframes(pcm)
        wav_bytes = wav_buf.getvalue()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data={
                        "model": "whisper-large-v3-turbo",
                        "language": "en",
                        "temperature": "0",
                        "response_format": "json",
                    },
                )
                if resp.status_code == 429:
                    logger.warning("[GroqWhisper] Rate-limited, dropping segment")
                    return
                if resp.status_code != 200:
                    logger.error("[GroqWhisper] Transcription failed %s: %s", resp.status_code, resp.text[:200])
                    return
                text = (resp.json().get("text") or "").strip()
                if text:
                    logger.info("[GroqWhisper] participant=%s text=%r", self.participant_name, text)
                    self.on_utterance(self.participant_id, self.participant_name, {"text": text, "words": []})
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.error("[GroqWhisper] Transcription error", exc_info=True)


@app.websocket("/ws/control/{run_id}")
async def avatar_control_endpoint(websocket: WebSocket, run_id: str):
    run = _ACTIVE_RUNS.get(run_id)
    if not run:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    run["control_ws"] = websocket
    logger.info("[Control WS] Avatar connected run_id=%s", run_id)
    
    # Send initial greeting after avatar connects
    async def send_initial_greeting():
        await asyncio.sleep(2)  # Wait for avatar to be ready
        bot_name = run.get("bot_name", "Tom")
        greeting = f"Hello! I'm {bot_name}, an AI agent from SpikedAI. I'm your digital twin and I'm here to help you with any questions or tasks you might have. How can I assist you today?"
        turn_id = run.get("turn_counter", 0) + 1
        run["turn_counter"] = turn_id
        run["active_turn_id"] = turn_id
        run["state"] = AgentState.SPEAKING
        
        # Stream the greeting via TTS
        try:
            await _stream_tts_for_reply(run, turn_id, greeting)
        except Exception as e:
            logger.error("[Greeting] Failed to send initial greeting: %s", e)
    
    asyncio.create_task(send_initial_greeting())
    
    try:
        while True:
            event = await websocket.receive_json()
            event_type = event.get("type")
            turn_id = event.get("turn_id")
            if turn_id is not None and turn_id != run.get("active_turn_id"):
                continue
            if event_type == "avatar_speak_started":
                run["state"] = AgentState.SPEAKING
            elif event_type in ("avatar_speak_ended", "avatar_speak_interrupted"):
                # Opens the follow-up window so the same speaker can continue
                # without repeating the wake name.
                _release_floor(run)
                watchdog = run.get("watchdog_task")
                if watchdog and not watchdog.done():
                    watchdog.cancel()
                if event_type == "avatar_speak_interrupted":
                    # Stop the backend from streaming further TTS frames: the
                    # page already flushed Simli's buffer with "SKIP".
                    pending = run.get("active_response_task")
                    if pending and not pending.done():
                        pending.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if run.get("control_ws") is websocket:
            run["control_ws"] = None
        logger.info("[Control WS] Avatar disconnected run_id=%s", run_id)


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
    if not GROQ_API_KEY:
        logger.error("[Recall WS] GROQ_API_KEY not set — cannot transcribe")
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
