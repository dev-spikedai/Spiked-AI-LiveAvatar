import os
import json
import base64
import asyncio
import logging
import re
import uuid
import hashlib
import hmac
from typing import Optional, List, Dict, Any, Literal
from urllib.parse import urlencode, quote

from src.supabase_client import get_user_keywords_and_products
from src.agent_policy import (
    AgentState,
    FinalUtteranceBuffer,
    SustainedSpeechDetector,
    apply_validated_corrections,
    build_entity_catalog,
    closest_entities,
    detect_invocation,
    normalize_reply,
    requires_company_knowledge,
)

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LiveAvatar-Spiked")

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API") or os.getenv("DEEPGRAM_API_KEY", "")
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
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", 
    "https://spiked-ai-liveavatar-409019309412.us-central1.run.app"
)
AGENT_BARGE_IN_MS = int(os.getenv("AGENT_BARGE_IN_MS", "700"))
AGENT_ENDPOINTING_MS = int(os.getenv("AGENT_ENDPOINTING_MS", "500"))
AGENT_UTTERANCE_END_MS = int(os.getenv("AGENT_UTTERANCE_END_MS", "1000"))
AGENT_MAX_REPLY_WORDS = int(os.getenv("AGENT_MAX_REPLY_WORDS", "45"))
DEFAULT_BOT_NAME = os.getenv("BOT_NAME", "Lara").strip() or "Lara"

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
    Executes Document RAG by calling SpikedAI-Backend-One /ask/handsfree endpoint.
    Passes the customer's Supabase JWT in the Authorization header.
    """
    url = f"{SPIKED_BACKEND_URL.rstrip('/')}/ask/handsfree"
    headers = {
        "Authorization": f"Bearer {auth_token}" if auth_token else "",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    payload = {
        "question": question,
        "client_id": client_id
    }
    
    logger.info("[RAG] Querying SpikedAI-Backend-One client_id=%s query_chars=%d", client_id, len(question))
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error("[RAG] Failed with status %s", response.status_code)
                return "I could not retrieve the relevant company documents for this question."
            
            full_text = ""
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunk = line.replace("data:", "").strip()
                    full_text += chunk + " "
                elif line:
                    full_text += line + " "
            
            result = full_text.strip()
            logger.info("[RAG] Received grounded answer (%d chars)", len(result))
            return result or "No specific documentation found for this query."
            
    except Exception as e:
        logger.error(f"[RAG] Error calling SpikedAI backend: {e}", exc_info=True)
        return "An error occurred while accessing the company knowledge base."

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
    bot_name = ctx.get("bot_name") or "Lara"
    product_domain = ctx.get("product_domain", "Enterprise AI & Automated Sales Engineering")
    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    catalog = build_entity_catalog(ctx)
    candidate_entities = closest_entities(transcript, catalog)
    history_text = "\n".join(
        f"{turn.get('speaker', 'Participant')}: {turn.get('text', '')}"
        for turn in conversation_history[-12:]
    )
    try:
        analysis_prompt = f"""Classify and normalize this addressed meeting turn.
Company: {company_name}
Offerings: {products_services or product_domain}
Verified entity candidates: {candidate_entities}
Recent finalized conversation:
{history_text}
Current speaker: {speaker}
Raw ASR: {transcript}

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
                intent="company_knowledge",
                resolved_query=transcript,
                corrections=[],
            )

        if analysis.intent not in ("social", "command") and requires_company_knowledge(transcript, catalog):
            analysis.intent = "company_knowledge"

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
        if rag_used:
            rag_result = await query_spiked_rag(resolved_query, auth_token, client_id)
            if not rag_result or any(marker in rag_result.lower() for marker in (
                "could not retrieve", "no specific documentation", "an error occurred"
            )):
                logger.warning("[Agent Route] RAG unavailable for addressed knowledge turn")
                return "I don’t have verified information on that available right now."

        answer_prompt = f"""You are {bot_name}, a concise meeting participant representing {company_name}.
Answer {speaker}'s addressed turn naturally for spoken delivery.
Use at most {reply_sentence_limit} sentences and {reply_word_limit} words. No markdown, lists, filler, sales pitch, or automatic follow-up question.
Use the speaker's first name only if it improves a greeting or clarification.
Never invent facts.

Intent: {analysis.intent}
Corrected turn: {corrected_transcript}
Resolved meaning: {resolved_query}
Recent finalized conversation:
{history_text}
Verified RAG facts (the only source for company facts):
{rag_result if rag_used else '(not required for this intent)'}"""
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
    except Exception as e:
        logger.error(f"[Gemini Agent] Inference error with google-genai SDK: {e}", exc_info=True)
        return None

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
            "user_id": user_id,
            "client_id": client_id,
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
            "active_response_task": None,
        }

        # 3. Build Output Media URL
        base_url = PUBLIC_BASE_URL.rstrip('/')
        if request and "localhost" in base_url and not "localhost" in str(request.base_url):
            base_url = str(request.base_url).rstrip('/')

        avatar_page_url = (
            f"{base_url}/avatar.html"
            f"?run={run_id}"
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


async def _interrupt_avatar(run: Dict[str, Any], participant_id: str) -> None:
    if run.get("state") != AgentState.SPEAKING:
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


def _schedule_completed_turn(
    run: Dict[str, Any],
    participant_id: str,
    participant_name: str,
    utterance: Dict[str, Any],
) -> None:
    transcript = (utterance.get("text") or "").strip()
    if not transcript:
        return
    bot_name = run.get("bot_name") or "Lara"
    invocation = detect_invocation(transcript, bot_name)
    history = run.setdefault("history", [])
    history_snapshot = list(history[-20:])
    history.append({"speaker": participant_name, "participant_id": participant_id, "text": transcript})
    del history[:-40]
    logger.info(
        "[Turn Gate] participant_id=%s participant_name=%s bot_name=%s addressed=%s matched_name=%s reason=%s transcript=%r",
        participant_id,
        participant_name,
        bot_name,
        invocation.addressed,
        invocation.matched_name,
        invocation.reason,
        re.sub(r"\s+", " ", transcript)[:160],
    )
    if not invocation.addressed:
        return

    previous_task = run.get("active_response_task")
    if previous_task and not previous_task.done():
        previous_task.cancel()

    run["turn_counter"] = int(run.get("turn_counter", 0)) + 1
    turn_id = run["turn_counter"]
    run["active_turn_id"] = turn_id
    run["state"] = AgentState.THINKING

    async def respond() -> None:
        try:
            answer = await process_transcript_with_gemini(
                transcript=transcript,
                speaker=participant_name,
                conversation_history=history_snapshot,
                auth_token=run.get("token") or "",
                client_id=run.get("client_id"),
                user_context=run.get("user_context") or {},
            )
            if turn_id != run.get("active_turn_id"):
                logger.info("[Agent] Discarded stale response turn_id=%s", turn_id)
                return
            if not answer:
                run["state"] = AgentState.LISTENING
                return
            control_ws = run.get("control_ws")
            if not control_ws:
                logger.warning("[Agent] Avatar control socket unavailable for turn_id=%s", turn_id)
                run["state"] = AgentState.LISTENING
                return
            history.append({"speaker": bot_name, "participant_id": "bot", "text": answer})
            del history[:-40]
            await control_ws.send_json({"type": "avatar_speak", "text": answer, "turn_id": turn_id})
        except asyncio.CancelledError:
            logger.info("[Agent] Cancelled superseded turn_id=%s", turn_id)
        except Exception:
            run["state"] = AgentState.LISTENING
            logger.error("[Agent] Failed addressed turn_id=%s", turn_id, exc_info=True)

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
                self.ws = await websockets.connect(url, additional_headers=headers)
            except TypeError:
                self.ws = await websockets.connect(url, extra_headers=headers)
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
                data = json.loads(await self.ws.recv())
                utterance = self.buffer.add_result(data)
                if utterance:
                    self.on_utterance(self.participant_id, self.participant_name, utterance)
                if data.get("type") == "Error":
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
            elif event_type in ("avatar_speak_ended", "avatar_speak_interrupted"):
                run["state"] = AgentState.LISTENING
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
    if not DEEPGRAM_API_KEY:
        await websocket.close(code=1011)
        return

    await websocket.accept()
    user_context = await get_user_keywords_and_products(
        user_id=run.get("user_id"),
        client_id=run.get("client_id"),
        auth_token=run.get("token"),
    )
    user_context["bot_name"] = run.get("bot_name") or "Lara"
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

            if participant_name.casefold() == (run.get("bot_name") or "Lara").casefold():
                continue

            detector = detectors.setdefault(
                participant_id,
                SustainedSpeechDetector(threshold_ms=AGENT_BARGE_IN_MS),
            )
            if run.get("state") == AgentState.SPEAKING:
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
                    lambda pid, name, utterance: _schedule_completed_turn(run, pid, name, utterance),
                )
                transcribers[participant_id] = transcriber
            await transcriber.send(pcm)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.error("[Recall WS] Separate audio failed run_id=%s", run_id, exc_info=True)
    finally:
        await asyncio.gather(*(item.close() for item in transcribers.values()), return_exceptions=True)
        logger.info("[Recall WS] Separate audio disconnected run_id=%s", run_id)
