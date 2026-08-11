import os
import json
import base64
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import google.generativeai as genai

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
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://us-west-2.recall.ai")
RECALL_WEBHOOK_URL = os.getenv(
    "RECALL_WEBHOOK_URL", 
    "https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript"
)
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID", "dd73ea75-1218-4ef3-92ce-606d5f7fbc0a")
LIVEAVATAR_SANDBOX = os.getenv("LIVEAVATAR_SANDBOX", "false").lower() == "true"
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL", 
    "https://spiked-ai-liveavatar-409019309412.us-central1.run.app"
)

# Configure Gemini Client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini API configured successfully")
else:
    logger.warning("GEMINI_API is not set in environment")

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
    mode: str = Field(default="LITE", description="LITE ($0.10/min) or FULL ($0.20/min)")

class CreateBotWithLiveAvatarRequest(BaseModel):
    meeting_url: str = Field(..., description="Zoom, Google Meet, or MS Teams URL")
    user_id: Optional[str] = Field(default=None, description="Supabase user ID")
    token: Optional[str] = Field(default=None, description="User's Supabase JWT access token for document RAG")
    client_id: Optional[str] = Field(default=None, description="Client/Company scope identifier")
    bot_name: str = Field(default="SpikedAI", description="Name of the bot in the meeting")
    avatar_id: Optional[str] = Field(default=None, description="Specific LiveAvatar avatar ID")

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
    
    logger.info(f"[RAG] Querying SpikedAI-Backend-One for: '{question}' (client_id: {client_id})")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"[RAG] Failed with status {response.status_code}: {response.text}")
                return "I could not retrieve the relevant company documents for this question."
            
            full_text = ""
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunk = line.replace("data:", "").strip()
                    full_text += chunk + " "
                elif line:
                    full_text += line + " "
            
            result = full_text.strip()
            logger.info(f"[RAG] Received grounded answer ({len(result)} chars)")
            return result or "No specific documentation found for this query."
            
    except Exception as e:
        logger.error(f"[RAG] Error calling SpikedAI backend: {e}", exc_info=True)
        return "An error occurred while accessing the company knowledge base."

# ---------------------------------------------------------------------------
# Gemini Agent with Dynamic RAG Tool Calling
# ---------------------------------------------------------------------------

async def process_transcript_with_gemini(
    transcript: str,
    speaker: str,
    conversation_history: List[Dict[str, str]],
    auth_token: str,
    client_id: Optional[str] = None
) -> Optional[str]:
    """
    Evaluates transcript turn with Gemini 3.5 Flash Lite equipped with the RAG tool.
    Returns the final answer text to speak, or None if the bot should stay silent.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API key is missing")
        return None

    # Define tool declaration for Gemini
    rag_tool_def = {
        "name": "generate_system_answer",
        "description": "Retrieves verified company knowledge, pricing, SLAs, technical specs, and seller persona guidelines from the user's document RAG database.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "The specific question or topic to look up in the company knowledge base."
                }
            },
            "required": ["query"]
        }
    }

    system_instruction = """
You are SpikedAI, a live interactive AI meeting assistant and sales representative avatar.
You are a participant in this meeting. You can hear what everyone says via real-time speaker diarization.

RULES OF ENGAGEMENT:
1. WHEN TO SPEAK:
   - ALWAYS respond to greetings, hellos, questions directed at you, or any product/business question.
   - If someone says "hello", "hey", "hi", "can you hear me", or addresses you in any way -> RESPOND warmly.
   - ONLY output "[SILENT]" if the speakers are clearly having a private side-conversation not involving you at all.
   - When in doubt, RESPOND rather than staying silent. You are a helpful meeting participant.
2. USING THE RAG TOOL:
   - For questions about company offerings, pricing, architecture, SLAs, product features, or onboarding -> call `generate_system_answer`.
   - If the tool returns "No relevant information found" or similar, respond honestly: "I don't have specific information on that topic in my knowledge base, but I'd be happy to help with anything else."
   - Do NOT call the tool again if it already returned no results for the same query.
   - For casual greetings or social pleasantries, respond conversationally without calling tools.
3. OUTPUT STYLE:
   - Keep answers natural, confident, concise, and direct (max 2 to 4 sentences).
   - Never output markdown formatting or bullet points (your response will be spoken aloud by a video avatar).
   - Always produce a plain text response. Never produce a function call without accompanying text.
"""

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    candidate_models = [
        preferred_model,
        "gemini-3.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    seen = set()
    candidate_models = [m for m in candidate_models if not (m in seen or seen.add(m))]

    model = None
    for m_name in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=m_name,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 200
                },
                system_instruction=system_instruction,
                tools=[{"function_declarations": [rag_tool_def]}]
            )
            break
        except Exception as e:
            logger.warning(f"Could not initialize model {m_name}: {e}")

    if not model:
        logger.error("Failed to initialize any Gemini model candidate")
        return None

    # Build conversation context
    formatted_context = ""
    for turn in conversation_history[-6:]:
        formatted_context += f"Speaker {turn.get('speaker', 'Participant')}: {turn.get('text', '')}\n"
    formatted_context += f"Speaker {speaker}: {transcript}\n"

    logger.info(f"[Gemini Agent] Evaluating transcript turn: Speaker {speaker} -> '{transcript}'")

    try:
        chat = model.start_chat(enable_automatic_function_calling=False)
        response = await asyncio.to_thread(chat.send_message, formatted_context)
        
        # Handle up to 3 rounds of tool calls (Gemini may chain tool calls)
        max_tool_rounds = 3
        for tool_round in range(max_tool_rounds):
            # Extract text from response, handling function_call parts safely
            has_function_call = False
            text_parts = []
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    fn_call = getattr(part, "function_call", None)
                    if fn_call and fn_call.name == "generate_system_answer":
                        has_function_call = True
                        query_arg = fn_call.args.get("query", transcript)
                        logger.info(f"[Gemini Agent] >>> TOOL CALL (round {tool_round+1}): generate_system_answer(query='{query_arg}')")
                        
                        # Execute RAG query
                        rag_result = await query_spiked_rag(query_arg, auth_token, client_id)
                        logger.info(f"[Gemini Agent] RAG Result preview: '{rag_result[:120]}...'")
                        
                        tool_response_part = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name="generate_system_answer",
                                response={"result": rag_result}
                            )
                        )
                        response = await asyncio.to_thread(chat.send_message, tool_response_part)
                        break  # Re-evaluate the new response in next loop iteration
                    elif hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
            
            if not has_function_call:
                # No more tool calls - extract final text
                reply_text = " ".join(text_parts).strip()
                logger.info(f"[Gemini Agent] Final reply (round {tool_round+1}): '{reply_text}'")
                if reply_text == "[SILENT]" or not reply_text:
                    logger.info("[Gemini Agent] Model chose to remain silent")
                    return None
                logger.info(f"[Gemini Agent] >>> SPEAKING RESPONSE: '{reply_text}' <<<")
                return reply_text
        
        # Exhausted tool rounds - try to extract any text from the last response
        logger.warning(f"[Gemini Agent] Exhausted {max_tool_rounds} tool rounds, extracting fallback text")
        try:
            reply_text = response.text.strip()
        except ValueError:
            # Last response was still a function_call with no text
            reply_text = "I wasn't able to find specific information on that, but I'm happy to help with anything else."
        
        if reply_text == "[SILENT]" or not reply_text:
            return None
        logger.info(f"[Gemini Agent] >>> FALLBACK RESPONSE: '{reply_text}' <<<")
        return reply_text

    except Exception as e:
        logger.error(f"[Gemini Agent] Inference error: {e}", exc_info=True)
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
    Creates a LiveAvatar / HeyGen session (LITE Mode = $0.10/min).
    """
    if not LIVEAVATAR_API_KEY:
        raise HTTPException(status_code=500, detail="LIVEAVATAR_API_KEY is not configured")

    avatar_id = payload.avatar_id or LIVEAVATAR_AVATAR_ID or None

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_payload = {
            "mode": payload.mode,
            "avatar_id": avatar_id,
            "video_settings": {
                "quality": payload.quality,
                "encoding": "H264"
            },
            "is_sandbox": payload.is_sandbox
        }
        
        token_res = await client.post(
            f"{LIVEAVATAR_BASE_URL}/v1/sessions/token",
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": LIVEAVATAR_API_KEY
            },
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
        
        return {
            "mode": payload.mode,
            "rate": "$0.10/minute" if payload.mode == "LITE" else "$0.20/minute",
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
    bot_name: str = "SpikedAI",
    avatar_id: Optional[str] = None,
    request: Optional[Request] = None
) -> Dict[str, Any]:
    """Internal core method to deploy the LiveAvatar Bot into a meeting."""
    try:
        if not RECALL_API_KEY:
            raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")

        if not user_id:
            user_id = extract_user_id_from_jwt(token)

        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # 1. Create LiveAvatar LITE Session ($0.10/min)
        avatar_session = await create_avatar(CreateLiveAvatarRequest(
            avatar_id=avatar_id,
            mode="LITE"
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
        }

        # 3. Build Output Media URL
        base_url = PUBLIC_BASE_URL.rstrip('/')
        if request and "localhost" in base_url and not "localhost" in str(request.base_url):
            base_url = str(request.base_url).rstrip('/')

        avatar_page_url = (
            f"{base_url}/avatar.html"
            f"?run={run_id}"
            f"&token={token}"
            f"&client_id={client_id or ''}"
        )

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
                    }
                },
                "realtime_endpoints": [
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
        bot_name = "SpikedAI"
        avatar_id = None

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            meeting_url = body.get("meeting_url", "")
            client_id = body.get("client_id")
            bot_name = body.get("bot_name", "SpikedAI")
            avatar_id = body.get("avatar_id")
            if not token:
                token = body.get("token", "")
        else:
            form = await request.form()
            meeting_url = form.get("meeting_url", "")
            client_id = form.get("client_id")
            bot_name = form.get("bot_name", "SpikedAI")
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
# WebSocket Audio Ingress & Real-Time Orchestration Pipeline
# ---------------------------------------------------------------------------

@app.websocket("/ws/audio/{session_id}")
async def websocket_audio_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
    client_id: Optional[str] = Query(None),
    bot_id: Optional[str] = Query(None)
):
    """
    Receives raw meeting audio stream from avatar.html (inside Recall headless browser).
    Pipes audio to Deepgram Streaming STT -> Detects Speech Turns -> Evaluates with Gemini 3.5 Flash Lite
    -> Executes SpikedAI Document RAG -> Sends Avatar Speak instructions back to avatar.html.
    """
    await websocket.accept()
    logger.info(f"[WS] Client connected for session {session_id} (bot_id: {bot_id}, client_id: {client_id})")

    if not DEEPGRAM_API_KEY:
        logger.error("[WS] DEEPGRAM_API_KEY is not configured")
        await websocket.send_json({"type": "error", "message": "Deepgram API key missing on server"})
        await websocket.close()
        return

    # Deepgram WebSocket Live Transcription URL
    # MediaRecorder in avatar.js sends WebM containerized Opus audio.
    # Omit encoding & sample_rate so Deepgram automatically detects the container!
    deepgram_params = {
        "model": "nova-2",
        "diarize": "true",
        "smart_format": "true",
        "interim_results": "false",
        "endpointing": "300",
        "punctuate": "true"
    }
    deepgram_ws_url = f"wss://api.deepgram.com/v1/listen?{urlencode(deepgram_params)}"
    deepgram_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    conversation_history: List[Dict[str, str]] = []
    chunk_count = 0
    total_bytes = 0

    try:
        # websockets >= 13.0 uses 'additional_headers', older versions use 'extra_headers'
        try:
            ws_client_cm = websockets.connect(deepgram_ws_url, additional_headers=deepgram_headers)
        except TypeError:
            ws_client_cm = websockets.connect(deepgram_ws_url, extra_headers=deepgram_headers)

        async with ws_client_cm as dg_ws:
            logger.info(f"[WS] Successfully connected to Deepgram Live Streaming STT for session {session_id}")

            async def forward_audio():
                """Forwards incoming binary audio frames from Recall browser to Deepgram."""
                nonlocal chunk_count, total_bytes
                try:
                    while True:
                        try:
                            message = await websocket.receive()
                        except (WebSocketDisconnect, RuntimeError):
                            break

                        if message.get("type") == "websocket.disconnect":
                            break

                        if "bytes" in message and message["bytes"]:
                            chunk_count += 1
                            total_bytes += len(message["bytes"])
                            if chunk_count % 30 == 0:
                                logger.info(f"[WS Ingress] Forwarded {chunk_count} chunks ({total_bytes} bytes) to Deepgram")
                            await dg_ws.send(message["bytes"])
                        elif "text" in message:
                            try:
                                cmd = json.loads(message["text"])
                                if cmd.get("type") == "auth_update":
                                    nonlocal token
                                    token = cmd.get("token", token)
                            except Exception:
                                pass
                except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError):
                    pass
                finally:
                    logger.info(f"[WS Ingress] Audio stream closed after {chunk_count} chunks ({total_bytes} bytes)")

            async def receive_transcripts():
                """Receives transcribed turns from Deepgram, routes to Gemini + RAG, and triggers avatar speak."""
                try:
                    while True:
                        raw_msg = await dg_ws.recv()
                        data = json.loads(raw_msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                transcript_text = alternatives[0].get("transcript", "").strip()
                                if transcript_text:
                                    words = alternatives[0].get("words", [])
                                    speaker_id = str(words[0].get("speaker", "0")) if words else "0"
                                    
                                    logger.info(f"[Deepgram STT] Speaker {speaker_id} Turn: '{transcript_text}'")
                                    
                                    answer_to_speak = await process_transcript_with_gemini(
                                        transcript=transcript_text,
                                        speaker=speaker_id,
                                        conversation_history=conversation_history,
                                        auth_token=token or "",
                                        client_id=client_id
                                    )
                                    
                                    conversation_history.append({
                                        "speaker": speaker_id,
                                        "text": transcript_text
                                    })
                                    
                                    if answer_to_speak:
                                        logger.info(f"[Agent] >>> Sending avatar_speak instruction: '{answer_to_speak}'")
                                        conversation_history.append({
                                            "speaker": "SpikedAI Avatar",
                                            "text": answer_to_speak
                                        })
                                        
                                        await websocket.send_json({
                                            "type": "avatar_speak",
                                            "text": answer_to_speak,
                                            "session_id": session_id
                                        })
                        elif msg_type == "Metadata":
                            logger.info(f"[Deepgram] Session metadata received: request_id={data.get('request_id')}")
                        elif msg_type == "Error":
                            logger.error(f"[Deepgram Error] {data}")

                except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                    logger.info("[WS] Deepgram receiver stream closed")

            forward_task = asyncio.create_task(forward_audio())
            receive_task = asyncio.create_task(receive_transcripts())
            done, pending = await asyncio.wait(
                [forward_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

    except (WebSocketDisconnect, RuntimeError):
        logger.info(f"[WS] Client disconnected cleanly for session {session_id}")
    except Exception as e:
        logger.error(f"[WS] WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"[WS] Cleaned up connection for session {session_id}")
