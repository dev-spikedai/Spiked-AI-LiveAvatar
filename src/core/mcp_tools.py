"""Controlled, read-only MCP enrichment for the meeting brain.

The router is deliberately optional. Company knowledge continues to use the
warm RAG path; MCP is for platform intelligence that is not naturally part of
the document index (meeting goals, live questions, settings, and live-note
state). No write-capable tool is accepted here.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("SpikedMeetingAgent")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "").strip()
MCP_TIMEOUT_S = float(os.getenv("MCP_TIMEOUT_S", "1.5"))

READ_ONLY_TOOLS = frozenset({
    "get_active_session",
    "get_livenote",
    "get_live_questions",
    "get_meeting_sentiment",
    "list_knowledge_base",
    "get_knowledge_base_document",
    "get_meeting_goals",
    "get_settings",
})


def choose_tool(transcript: str, intent: str, has_bot_id: bool) -> Optional[str]:
    """Choose at most one read-only tool for a completed turn."""
    lowered = (transcript or "").casefold()
    if intent == "meeting_context" and has_bot_id:
        if any(term in lowered for term in ("question", "asked", "ask", "wondering")):
            return "get_live_questions"
        if any(term in lowered for term in ("said", "say", "discussed", "catch that", "summarize", "summary")):
            return "get_livenote"
        if any(term in lowered for term in ("sentiment", "mood", "feeling", "engaged", "engagement")):
            return "get_meeting_sentiment"
    if any(term in lowered for term in ("meeting goal", "goal", "objective", "topic coverage")):
        return "get_meeting_goals"
    if any(term in lowered for term in ("my settings", "my preferences", "configured", "personalization")):
        return "get_settings"
    return None


def _jsonrpc_payload(response: httpx.Response) -> Dict[str, Any]:
    """Extract JSON from JSON or simple SSE-wrapped MCP responses."""
    content_type = (response.headers.get("content-type") or "").casefold()
    if "text/event-stream" not in content_type:
        value = response.json()
        return value if isinstance(value, dict) else {}
    latest: Dict[str, Any] = {}
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            latest = value
    return latest


def _content_text(result: Dict[str, Any]) -> str:
    error = result.get("error")
    if error:
        raise RuntimeError(str(error))
    payload = result.get("result") or {}
    content = payload.get("content") or []
    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
    return "\n".join(part for part in text_parts if part).strip()


async def call_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    auth_token: str,
) -> Optional[str]:
    """Call one allowlisted MCP tool, failing closed on every error."""
    if not MCP_SERVER_URL or tool_name not in READ_ONLY_TOOLS:
        return None
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        async with httpx.AsyncClient(timeout=MCP_TIMEOUT_S) as client:
            response = await client.post(MCP_SERVER_URL, headers=headers, json=body)
        response.raise_for_status()
        text = _content_text(_jsonrpc_payload(response))
        return text or None
    except Exception as exc:
        logger.info("[MCP] read-only tool %s unavailable: %s", tool_name, exc)
        return None


async def enrich_turn(
    transcript: str,
    intent: str,
    run: Dict[str, Any],
    auth_token: str,
) -> Optional[str]:
    """Fetch one scoped platform fact for the current turn."""
    tool_name = choose_tool(transcript, intent, bool(run.get("bot_id")))
    if not tool_name:
        return None
    arguments: Dict[str, Any] = {}
    if tool_name in {"get_livenote", "get_live_questions", "get_meeting_sentiment"}:
        arguments["bot_id"] = run.get("bot_id")
    if tool_name == "get_meeting_sentiment":
        arguments["surface"] = "participants"
    if tool_name == "list_knowledge_base" and run.get("client_id"):
        arguments["client_id"] = run.get("client_id")
    result = await call_tool(tool_name, arguments, auth_token)
    if result:
        logger.info("[MCP] Enriched turn with %s", tool_name)
        return f"MCP read-only context ({tool_name}):\n{result}"
    return None
