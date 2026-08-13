import os
import json
import base64
import time
import asyncio
import logging
import uuid
from typing import Optional, List, Dict, Any
from urllib.parse import urlencode, quote

from src.supabase_client import get_user_keywords_and_products, correct_stt_text

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Header, Form
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

# Configure Modern Google GenAI Client
# Using gemini-3.5-flash-lite for cost-effective, low-latency function calling
gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"[Google GenAI] Official modern SDK client initialized with key: {GEMINI_API_KEY[:8]}...{GEMINI_API_KEY[-6:]}")
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

class CreateBotWithLiveAvatarRequest(BaseModel):
    meeting_url: str = Field(..., description="Zoom, Google Meet, or MS Teams URL")
    user_id: Optional[str] = Field(default=None, description="Supabase user ID")
    token: Optional[str] = Field(default=None, description="User's Supabase JWT access token for document RAG")
    client_id: Optional[str] = Field(default=None, description="Client/Company scope identifier")
    bot_name: str = Field(default="Tom", description="Name of the bot in the meeting")
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
    client_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Evaluates transcript turn with Gemini 3.5 Flash Lite equipped with the RAG tool.
    Uses SOTA conversational meeting bot dialogue policy with intelligent intent classification.
    Returns the final answer text to speak, or None if the bot should stay silent.
    """
    if not gemini_client:
        logger.error("[Google GenAI] Client is not initialized")
        return None

    # Extract dynamic company context
    ctx = user_context or {}
    company_name = ctx.get("company_name", "SpikedAI")
    products_services = ctx.get("products_services", "")
    bot_name = ctx.get("bot_name") or "Tom"
    product_domain = ctx.get("product_domain", "Enterprise AI & Automated Sales Engineering")
    keywords_list = ctx.get("keywords") or []
    formatted_keywords = ", ".join(keywords_list[:40]) if keywords_list else "SpikedAI, s3cura AI, 3CAI, CRM, RAG"

    # Define Tool using modern Google GenAI SDK
    rag_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="generate_system_answer",
                description=f"REQUIRED tool to retrieve verified facts, documentation, products, pricing, and architecture for {company_name} from the company document RAG database.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description=f"The specific, high-density question or topic to search in {company_name}'s knowledge base."
                        )
                    },
                    required=["query"]
                )
            )
        ]
    )

    system_instruction = f"""
You are {bot_name}, a live interactive AI meeting participant and sales engineer representing {company_name}.
You are actively listening to this live meeting with real-time speaker diarization.

COMPANY BACKGROUND & OFFERINGS:
- Company: {company_name}
- Domain & Offerings: {products_services or product_domain or "Enterprise AI Solutions & Meeting Intelligence"}
- Verified Company Keywords: {formatted_keywords}

YOUR CAPABILITIES & TOOLSET:
- You are equipped with real-time Document RAG connected to {company_name}'s verified knowledge base via the `generate_system_answer` tool.
- Capabilities:
  * Answering product, technical, architectural, pricing, and SLA questions accurately using verified documents.
  * Assisting live in meetings with contextual notes, insights, and objections handling.
  * Interfacing with integrated enterprise platforms (e.g. {products_services or "CRM, Cloud, and Knowledge Bases"}).
- If asked "What can you do?", "What tools do you have?", or "How can you help?", describe these capabilities directly, proudly, and conversationally in 1-2 punchy sentences.

CRITICAL GATING RULE - ONLY SPEAK WHEN EXPLICITLY ADDRESSED AS {bot_name.upper()}:
- You must ONLY speak when an attendee in the meeting explicitly addresses you by name (e.g., "{bot_name}", "Hey {bot_name}", "{bot_name}, can you...", "What do you think, {bot_name}?", "{bot_name}, what are your products?").
- If the attendees are talking to each other, having internal discussions, discussing slides, asking questions to the room, or not explicitly calling out {bot_name} by name -> You MUST output the exact string: "[SILENT]".
- NEVER butt in, interject, or answer unaddressed general chatter. Silence is mandatory unless {bot_name} is explicitly invoked.

TRANSCRIPT REPAIR & ASR NOISE RECONSTRUCTION (CRITICAL):
- The incoming transcript is produced in real-time by speech-to-text (Deepgram Nova-3) and is noisy, fragmented, and frequently contains acoustic/phonetic errors, split words, and phonetic drift.
- You MUST recognize that words in the transcript may be broken or misspelled.
- When you are addressed, ALWAYS repair the broken transcript internally using the VERIFIED COMPANY KEYWORDS and COMPANY OFFERINGS before reasoning:
  * "secure a" / "secure ai" / "secura" -> {company_name}
  * "three c" / "three cai" / "3c ai" -> 3CAI / 3CAI Architecture
  * "spider" / "spike the eye" / "spike ai" -> Spiked / SpikedAI
  * "comic gp" / "karma gp" -> Karmic GP
  * "context graf" / "contact graph" -> Context Graph
  * "call sim" / "sales sim" -> Hyper-Realistic Sales Simulator / Call Simulator
- Always map any phonetic approximations or misheard words back to the closest matching offering from COMPANY OFFERINGS.

IMPLICIT & PRONOUN RESOLUTION:
- Use conversation context heavily to resolve pronouns ("it", "that", "this thing", "you guys", "your platform", "your tool") into concrete company offerings.
- When attendees ask "{bot_name}, what do you offer?", "{bot_name}, what does it cost?", or "{bot_name}, how do you handle security?", recognize that "you" / "it" refers to {company_name} and its offerings.

RULES OF ENGAGEMENT & DIALOGUE POLICY (WHEN ADDRESSED AS {bot_name.upper()}):
1. MANDATORY RAG TOOL CALL FOR ALL QUESTIONS (CRITICAL):
   - You do NOT have raw internal knowledge stored in memory.
   - For ANY question asked to {bot_name} regarding company offerings, products, features, pricing, architecture, roadmap, security, SLAs, integrations, or technical capabilities:
     YOU MUST ALWAYS EXECUTE THE TOOL `generate_system_answer(query=...)`.
   - Do NOT generate direct text answers for knowledge questions without calling `generate_system_answer`.
   - When calling `generate_system_answer`:
     1. Repair any broken or misheard words into the verified company keywords.
     2. Resolve pronouns into explicit product/company names.
     3. Formulate a self-contained, high-density semantic retrieval search query (e.g., "{company_name} core products, 3CAI architecture, and AI assistance features overview" or "{company_name} enterprise pricing tiers and SLA terms").
   - If the address is purely a casual greeting or social pleasantry (e.g., "Hey {bot_name}, how are you?", "Can you hear me {bot_name}?"), respond directly, warmly, and conversationally in 1 sentence without calling RAG.

2. HONESTY & ACCURACY:
   - If the RAG tool returns no matching facts, be honest and transparent: "I don't have specific details on that in our knowledge base yet, but I'd be glad to check with the team and get back to you."
   - Never invent false metrics or hallucinate non-existent features.

3. ALWAYS END WITH A FOLLOW-UP QUESTION (MANDATORY):
   - Every answer you provide MUST conclude with a relevant, natural conversational follow-up question to keep the conversation engaging and proactive.
   - Examples of ending questions:
     * "Would you like me to dive deeper into how that works?"
     * "Does that align with what your team is looking for?"
     * "Should I pull up more technical details on that for you?"
     * "Would you like to explore our pricing or integrations for that?"

4. SPOKEN DELIVERY & CADENCE:
   - Your response will be spoken aloud by a live avatar in a video meeting.
   - Maximum 2 to 4 punchy, natural sentences (ending with your follow-up question).
   - NO markdown, NO bullet points, NO asterisks, NO numbered lists.
   - Tone: Confident, helpful, concise, and professional.
"""

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    logger.info(f"[Prompt Debug] Dynamic Context: Company='{company_name}', Products='{products_services[:100]}...', Domain='{product_domain}', Bot='{bot_name}'")
    logger.info(f"[Prompt Debug] Evaluating transcript turn: Speaker {speaker} -> '{transcript}'")

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=250,
        system_instruction=system_instruction,
        tools=[rag_tool]
    )

    # Build conversation context with previous meeting turns
    formatted_context = ""
    for turn in conversation_history[-12:]:
        formatted_context += f"Speaker {turn.get('speaker', 'Participant')}: {turn.get('text', '')}\n"
    formatted_context += f"Speaker {speaker}: {transcript}\n"

    try:
        chat = gemini_client.aio.chats.create(model=preferred_model, config=config)
        response = await chat.send_message(formatted_context)
        
        # Multi-round tool execution (up to 3 rounds)
        max_tool_rounds = 3
        for tool_round in range(max_tool_rounds):
            if response.function_calls:
                fn_call = response.function_calls[0]
                if fn_call.name == "generate_system_answer":
                    query_arg = fn_call.args.get("query", transcript) if isinstance(fn_call.args, dict) else transcript
                    logger.info(f"[Gemini Agent] >>> TOOL CALL (round {tool_round+1}): generate_system_answer(query='{query_arg}')")
                    
                    # Execute RAG query against SpikedAI-Backend-One
                    rag_result = await query_spiked_rag(query_arg, auth_token, client_id)
                    
                    # If RAG returned no matching docs, fallback to injected company product catalog
                    if not rag_result or "No relevant information found" in rag_result or "could not retrieve" in rag_result:
                        fallback_facts = f"Verified company knowledge: {company_name} offers {products_services}. Domain: {product_domain}."
                        logger.info(f"[Gemini Agent] RAG empty -> Enhancing with user_configs: '{fallback_facts[:120]}...'")
                        rag_result = f"{rag_result}\n\n{fallback_facts}"
                    else:
                        logger.info(f"[Gemini Agent] RAG Result preview: '{rag_result[:120]}...'")
                    
                    tool_resp_part = types.Part.from_function_response(
                        name="generate_system_answer",
                        response={"result": rag_result}
                    )
                    response = await chat.send_message(tool_resp_part)
                    continue
            
            # No function call - extract final spoken text
            reply_text = response.text.strip() if response.text else ""
            logger.info(f"[Gemini Agent] Final reply (round {tool_round+1}): '{reply_text}'")
            if reply_text == "[SILENT]" or not reply_text:
                logger.info("[Gemini Agent] Model chose to remain silent")
                return None
            logger.info(f"[Gemini Agent] >>> SPEAKING RESPONSE: '{reply_text}' <<<")
            return reply_text
            
        reply_text = response.text.strip() if response.text else ""
        if reply_text == "[SILENT]" or not reply_text:
            return None
        logger.info(f"[Gemini Agent] >>> FALLBACK RESPONSE: '{reply_text}' <<<")
        return reply_text

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
        # For FULL mode, load user company context and create a Context (persona)
        context_id = None
        if payload.mode == "FULL":
            user_ctx = await get_user_keywords_and_products(
                user_id=payload.user_id,
                client_id=payload.client_id,
                auth_token=payload.token
            )
            c_name = user_ctx.get("company_name") or "SpikedAI"
            p_desc = user_ctx.get("products_services") or user_ctx.get("product_domain") or "Enterprise AI Solutions"
            b_name = "Tom"  # Explicitly named Tom as requested
            
            logger.info(f"[LiveAvatar Context] Initializing with Company='{c_name}', Products='{p_desc[:80]}...', Bot='{b_name}'")
            
            ctx_payload = {
                "name": f"tom-bot-{uuid.uuid4().hex[:8]}",
                "prompt": f"You are Tom, a helpful and concise sales engineer and meeting participant representing {c_name}. Products & offerings: {p_desc}. Only speak when explicitly addressed as Tom. Keep answers to 2-4 sentences spoken naturally.",
                "opening_text": f"Hi everyone, I'm Tom from {c_name}. I'm here and ready to help."
            }
            ctx_res = await client.post(
                f"{LIVEAVATAR_BASE_URL}/v1/contexts",
                headers=la_headers,
                json=ctx_payload
            )
            if ctx_res.status_code in (200, 201):
                ctx_data = ctx_res.json()
                context_id = ctx_data.get("data", {}).get("id") if isinstance(ctx_data.get("data"), dict) else None
                logger.info(f"[LiveAvatar] Created context: {context_id}")
            else:
                logger.warning(f"[LiveAvatar] Context creation failed: {ctx_res.text}, proceeding without")

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
        if payload.mode == "FULL":
            token_payload["interactivity_type"] = "CONVERSATIONAL"
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
    bot_name: str = "Tom",
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

        # 1. Create LiveAvatar FULL Session ($0.20/min, includes TTS) with user context
        avatar_session = await create_avatar(CreateLiveAvatarRequest(
            avatar_id=avatar_id,
            mode="FULL",
            user_id=user_id,
            client_id=client_id,
            token=token
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

    user_id = extract_user_id_from_jwt(token)
    user_context = await get_user_keywords_and_products(
        user_id=user_id,
        client_id=client_id,
        auth_token=token
    )
    user_keywords = user_context.get("keywords", [])

    deepgram_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    async def connect_to_deepgram():
        models_to_try = ["nova-3", "nova-2"]
        last_err = None
        for m in models_to_try:
            params = {
                "model": m,
                "diarize": "true",
                "smart_format": "true",
                "interim_results": "false",
                "endpointing": "300",
                "punctuate": "true"
            }
            url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
            if user_keywords:
                if m.startswith("nova-3"):
                    # Nova-3 uses keyterm parameter without weights
                    kw_params = [f"keyterm={quote(kw.strip())}" for kw in user_keywords[:40] if kw and kw.strip()]
                else:
                    # Nova-2 uses legacy keywords with weights
                    kw_params = [f"keywords={quote(kw.strip())}:5" for kw in user_keywords[:40] if kw and kw.strip()]
                if kw_params:
                    url = f"{url}&{'&'.join(kw_params)}"
            try:
                try:
                    conn = await websockets.connect(url, additional_headers=deepgram_headers)
                except TypeError:
                    conn = await websockets.connect(url, extra_headers=deepgram_headers)
                logger.info(f"[WS] Successfully connected to Deepgram Live STT using model '{m}' (session: {session_id})")
                return conn
            except Exception as e:
                logger.warning(f"[WS] Deepgram connection with model '{m}' failed: {e}")
                last_err = e
        raise last_err

    conversation_history: List[Dict[str, str]] = []
    chunk_count = 0
    total_bytes = 0
    last_speak_timestamp = 0.0

    try:
        dg_ws = await connect_to_deepgram()
        async with dg_ws:

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
                nonlocal last_speak_timestamp
                try:
                    while True:
                        raw_msg = await dg_ws.recv()
                        data = json.loads(raw_msg)
                        msg_type = data.get("type")
                        
                        if msg_type == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                raw_transcript = alternatives[0].get("transcript", "").strip()
                                if raw_transcript:
                                    words = alternatives[0].get("words", [])
                                    speaker_id = str(words[0].get("speaker", "0")) if words else "0"
                                    
                                    # Apply phonetic & keyword correction
                                    transcript_text = correct_stt_text(raw_transcript, user_keywords)
                                    logger.info(f"[Deepgram STT] Speaker {speaker_id} Turn: '{raw_transcript}' -> Corrected: '{transcript_text}'")
                                    
                                    # Barge-in: If user speaks within 6s of avatar speaking, notify avatar to interrupt
                                    if time.time() - last_speak_timestamp < 6.0:
                                        logger.info("[Barge-In] User spoke while avatar was active — sending interrupt")
                                        await websocket.send_json({
                                            "type": "avatar_interrupt",
                                            "session_id": session_id
                                        })

                                    answer_to_speak = await process_transcript_with_gemini(
                                        transcript=transcript_text,
                                        speaker=speaker_id,
                                        conversation_history=conversation_history,
                                        auth_token=token or "",
                                        client_id=client_id,
                                        user_context=user_context
                                    )
                                    
                                    conversation_history.append({
                                        "speaker": speaker_id,
                                        "text": transcript_text
                                    })
                                    
                                    if answer_to_speak:
                                        logger.info(f"[Agent] >>> Sending avatar_speak instruction: '{answer_to_speak}'")
                                        conversation_history.append({
                                            "speaker": user_context.get("bot_name", "SpikedAI"),
                                            "text": answer_to_speak
                                        })
                                        
                                        last_speak_timestamp = time.time()
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
