import os
import json
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
SPIKED_BACKEND_URL = os.getenv("SPIKED_BACKEND_URL", "https://spikedai-production-application-409019309412.us-central1.run.app")
RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
RECALL_BASE_URL = os.getenv("RECALL_BASE_URL", "https://us-west-2.recall.ai")
RECALL_WEBHOOK_URL = os.getenv(
    "RECALL_WEBHOOK_URL", 
    "https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript"
)
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_BASE_URL = os.getenv("LIVEAVATAR_API_URL", "https://api.liveavatar.com")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID", "")
LIVEAVATAR_SANDBOX = os.getenv("LIVEAVATAR_SANDBOX", "false").lower() == "true"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

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
# Models
# ---------------------------------------------------------------------------

class CreateLiveAvatarRequest(BaseModel):
    avatar_id: Optional[str] = Field(default=None, description="HeyGen/LiveAvatar avatar identifier")
    quality: str = Field(default="medium", description="Video quality: low, medium (720p), high (1080p)")
    is_sandbox: bool = Field(default=LIVEAVATAR_SANDBOX, description="Sandbox mode for testing")
    mode: str = Field(default="LITE", description="LITE ($0.10/min) or FULL ($0.20/min)")

class CreateBotWithLiveAvatarRequest(BaseModel):
    meeting_url: str = Field(..., description="Zoom, Google Meet, or MS Teams URL")
    user_id: str = Field(..., description="Supabase user ID")
    token: str = Field(..., description="User's Supabase JWT access token for document RAG")
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
            
            # Read streaming response or text
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
# Gemini 2.0 Flash Agent with Dynamic RAG Tool Calling
# ---------------------------------------------------------------------------

async def process_transcript_with_gemini(
    transcript: str,
    speaker: str,
    conversation_history: List[Dict[str, str]],
    auth_token: str,
    client_id: Optional[str] = None
) -> Optional[str]:
    """
    Evaluates transcript turn with Gemini 2.0 Flash equipped with the RAG tool.
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
You are listening to an ongoing meeting with real-time speaker diarization.

RULES OF ENGAGEMENT:
1. DECIDE WHETHER TO SPEAK:
   - If a participant directly asks you a question, greets you, or asks a product/business question requiring your input -> RESPOND.
   - If participants are talking amongst themselves and do not need your input -> Respond with the exact string: "[SILENT]".
2. USING THE RAG TOOL:
   - For ANY questions regarding company offerings, pricing, architecture, SLAs, product features, or onboarding -> You MUST call the `generate_system_answer` tool to retrieve verified facts.
   - For casual greetings or quick social pleasantries (e.g. "Hello Spiked", "Can you hear me?"), respond conversationally without calling tools.
3. OUTPUT STYLE:
   - Keep answers natural, confident, concise, and direct (max 2 to 4 sentences).
   - Never output markdown formatting or bullet points (your response will be spoken aloud by a video avatar).
"""

    # Prefer Gemini 2.0 Flash for sub-200ms TTFT and state-of-the-art agentic tool calling
    model_name = "gemini-2.0-flash"
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 200
            },
            system_instruction=system_instruction,
            tools=[{"function_declarations": [rag_tool_def]}]
        )
    except Exception:
        # Fallback to gemini-1.5-flash if 2.0 is not yet enabled on the key
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 200
            },
            system_instruction=system_instruction,
            tools=[{"function_declarations": [rag_tool_def]}]
        )

    # Build conversation context
    formatted_context = ""
    for turn in conversation_history[-6:]:
        formatted_context += f"Speaker {turn.get('speaker', 'Participant')}: {turn.get('text', '')}\n"
    formatted_context += f"Speaker {speaker}: {transcript}\n"

    try:
        chat = model.start_chat(enable_automatic_function_calling=False)
        response = await asyncio.to_thread(chat.send_message, formatted_context)
        
        # Check if model requested a tool call
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call and fn_call.name == "generate_system_answer":
                    query_arg = fn_call.args.get("query", transcript)
                    logger.info(f"[Gemini 2.0] Tool call triggered: generate_system_answer(query='{query_arg}')")
                    
                    # Execute RAG query against SpikedAI-Backend-One
                    rag_result = await query_spiked_rag(query_arg, auth_token, client_id)
                    
                    # Feed tool output back to Gemini for final spoken formulation
                    tool_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name="generate_system_answer",
                            response={"result": rag_result}
                        )
                    )
                    final_res = await asyncio.to_thread(chat.send_message, tool_response_part)
                    reply_text = final_res.text.strip()
                    
                    if reply_text == "[SILENT]" or not reply_text:
                        return None
                    return reply_text

        # If direct text was returned without tool calling
        reply_text = response.text.strip()
        if reply_text == "[SILENT]" or not reply_text:
            return None
        return reply_text

    except Exception as e:
        logger.error(f"[Gemini 2.0] Inference error: {e}", exc_info=True)
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
        "recall_configured": bool(RECALL_API_KEY),
        "spiked_backend_url": SPIKED_BACKEND_URL,
        "recall_webhook_url": RECALL_WEBHOOK_URL
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
        
        if token_res.status_code != 200:
            logger.error(f"LiveAvatar token creation failed: {token_res.text}")
            raise HTTPException(status_code=token_res.status_code, detail=token_res.text)
        
        token_data = token_res.json().get("data", {})
        session_token = token_data.get("session_token")
        
        start_res = await client.post(
            f"{LIVEAVATAR_BASE_URL}/v1/sessions/start",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {session_token}"
            },
            json={}
        )
        
        if start_res.status_code != 200:
            logger.error(f"LiveAvatar start session failed: {start_res.text}")
            raise HTTPException(status_code=start_res.status_code, detail=start_res.text)
            
        start_data = start_res.json().get("data", {})
        
        return {
            "mode": payload.mode,
            "rate": "$0.10/minute" if payload.mode == "LITE" else "$0.20/minute",
            "session_id": start_data.get("session_id"),
            "livekit_url": start_data.get("livekit_url"),
            "livekit_token": start_data.get("livekit_client_token"),
            "session_token": session_token
        }

@app.post("/create-live-avatar-bot")
async def create_live_avatar_bot(payload: CreateBotWithLiveAvatarRequest):
    """
    Creates a full LiveAvatar Bot in Recall.ai:
    1. Initializes LiveAvatar LITE session ($0.10/min).
    2. Registers the run with local registry.
    3. Deploys Recall bot with Output Media camera pointing to self-hosted avatar.html.
    4. Sets realtime_endpoints webhook to recall_backend to preserve all dashboard features.
    5. Injects metadata: { user_id, client_id } for fail-safe webhook user resolution.
    """
    if not RECALL_API_KEY:
        raise HTTPException(status_code=500, detail="RECALL_API_KEY is not configured")

    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # 1. Create LiveAvatar session
    avatar_session = await create_avatar(CreateLiveAvatarRequest(
        avatar_id=payload.avatar_id,
        mode="LITE"
    ))

    # 2. Store session credentials for Recall's avatar.html
    _ACTIVE_RUNS[run_id] = {
        "run_id": run_id,
        "user_id": payload.user_id,
        "client_id": payload.client_id,
        "token": payload.token,
        "session_id": avatar_session.get("session_id"),
        "livekit_url": avatar_session.get("livekit_url"),
        "livekit_token": avatar_session.get("livekit_token"),
    }

    # 3. Build Output Media URL
    avatar_page_url = (
        f"{PUBLIC_BASE_URL.rstrip('/')}/avatar.html"
        f"?run={run_id}"
        f"&token={payload.token}"
        f"&client_id={payload.client_id or ''}"
    )

    # 4. Recall Bot Payload with Dual-Track Webhook and Metadata
    recall_payload = {
        "meeting_url": payload.meeting_url,
        "bot_name": payload.bot_name,
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
            "user_id": payload.user_id,
            "client_id": payload.client_id or ""
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

        return {
            "success": True,
            "bot_id": bot_id,
            "run_id": run_id,
            "liveavatar_session_id": avatar_session.get("session_id"),
            "avatar_page_url": avatar_page_url,
            "status": bot_data.get("status_changes", [{}])[-1].get("code", "created")
        }

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
    Pipes audio to Deepgram Streaming STT -> Detects Speech Turns -> Evaluates with Gemini 2.0 Flash
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
    deepgram_params = {
        "model": "nova-2",
        "diarize": "true",
        "smart_format": "true",
        "interim_results": "false",
        "endpointing": "300",
        "punctuate": "true",
        "encoding": "linear16",
        "sample_rate": "16000"
    }
    deepgram_ws_url = f"wss://api.deepgram.com/v1/listen?{urlencode(deepgram_params)}"
    deepgram_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    conversation_history: List[Dict[str, str]] = []

    try:
        async with websockets.connect(deepgram_ws_url, extra_headers=deepgram_headers) as dg_ws:
            logger.info("[WS] Successfully connected to Deepgram Live Streaming STT")

            async def forward_audio():
                """Forwards incoming binary audio frames from Recall browser to Deepgram."""
                try:
                    while True:
                        message = await websocket.receive()
                        if "bytes" in message and message["bytes"]:
                            await dg_ws.send(message["bytes"])
                        elif "text" in message:
                            try:
                                cmd = json.loads(message["text"])
                                if cmd.get("type") == "auth_update":
                                    nonlocal token
                                    token = cmd.get("token", token)
                            except Exception:
                                pass
                except (WebSocketDisconnect, asyncio.CancelledError):
                    logger.info("[WS] Audio forwarding stream ended")

            async def receive_transcripts():
                """Receives transcribed turns from Deepgram, routes to Gemini + RAG, and triggers avatar speak."""
                try:
                    while True:
                        raw_msg = await dg_ws.recv()
                        data = json.loads(raw_msg)
                        
                        if data.get("type") == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                transcript_text = alternatives[0].get("transcript", "").strip()
                                if transcript_text:
                                    words = alternatives[0].get("words", [])
                                    speaker_id = str(words[0].get("speaker", "0")) if words else "0"
                                    
                                    logger.info(f"[Deepgram] Speaker {speaker_id}: '{transcript_text}'")
                                    
                                    # Process turn with Gemini 2.0 Flash & RAG Tool
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
                                    
                                    # If Gemini formulated an answer to speak
                                    if answer_to_speak:
                                        logger.info(f"[Agent] Speaking response: '{answer_to_speak}'")
                                        conversation_history.append({
                                            "speaker": "SpikedAI Avatar",
                                            "text": answer_to_speak
                                        })
                                        
                                        # Send speak instruction to avatar.html
                                        await websocket.send_json({
                                            "type": "avatar_speak",
                                            "text": answer_to_speak,
                                            "session_id": session_id
                                        })

                except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                    logger.info("[WS] Deepgram receiver stream closed")

            # Run audio forwarding and transcript processing concurrently
            await asyncio.gather(forward_audio(), receive_transcripts())

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"[WS] WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"[WS] Cleaned up connection for session {session_id}")
