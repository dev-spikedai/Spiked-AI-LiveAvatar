"""Speech dispatch across provider shapes. See docs/PROVIDER_REFACTOR_PLAN.md §4."""

import asyncio
import base64
import logging
from typing import Any, Dict, Optional

from src.core import protocol

logger = logging.getLogger("SpikedMeetingAgent")

# ~187ms of PCM16@16k: small enough that barge-in cuts promptly, large enough
# not to flood the socket.
AUDIO_FRAME_BYTES = 6000


async def emit_speech(run: Dict[str, Any], turn_id: int, chunk_id: Optional[str], text: str) -> bool:
    """Send one utterance to whichever provider this run uses. False if it could not be sent."""
    control_ws = run.get("control_ws")
    if not control_ws:
        return False

    providers = run.get("providers")
    if providers is None or providers.video.accepts == "text":
        await control_ws.send_json(protocol.avatar_speak(text, turn_id, chunk_id))
        return True

    return await _emit_synthesized(run, control_ws, providers, turn_id, chunk_id, text)


async def _emit_synthesized(run, control_ws, providers, turn_id, chunk_id, text) -> bool:
    """Lip-sync-only provider: synthesize, then stream PCM frames."""
    if providers.tts is None:
        logger.error("[Speech] %s needs a TTS provider and has none", providers.video.name)
        return False

    # Marks the start of the utterance; the frames that follow are the speech.
    await control_ws.send_json(protocol.avatar_speak(text, turn_id, chunk_id))

    frame_bytes = getattr(providers.video, "chunk_bytes", AUDIO_FRAME_BYTES)
    buffer = bytearray()
    try:
        async for chunk in providers.tts.stream(text):
            buffer.extend(chunk)
            while len(buffer) >= frame_bytes:
                await _send_frame(control_ws, bytes(buffer[:frame_bytes]), turn_id)
                del buffer[:frame_bytes]
        if buffer:
            await _send_frame(control_ws, bytes(buffer), turn_id)
    except asyncio.CancelledError:
        # Still close the utterance, or the page waits forever for frames.
        await control_ws.send_json(protocol.avatar_speak_end(turn_id, chunk_id))
        raise
    except Exception:
        logger.error("[Speech] TTS failed turn_id=%s chunk_id=%s", turn_id, chunk_id, exc_info=True)
        await control_ws.send_json(protocol.avatar_speak_end(turn_id, chunk_id))
        return False

    await control_ws.send_json(protocol.avatar_speak_end(turn_id, chunk_id))
    return True


async def _send_frame(control_ws, pcm: bytes, turn_id: int) -> None:
    await control_ws.send_json(protocol.avatar_audio(base64.b64encode(pcm).decode("ascii"), turn_id))


async def delegate_turn(run: Dict[str, Any], turn_id: int, question: str, timeout: float) -> Optional[str]:
    """Hand the turn to the vendor's own brain; return what it said, if anything."""
    control_ws = run.get("control_ws")
    if not control_ws:
        return None

    waiter: "asyncio.Future[str]" = asyncio.get_running_loop().create_future()
    run.setdefault("vendor_reply_waiters", {})[turn_id] = waiter
    try:
        await control_ws.send_json(protocol.avatar_user_message(question, turn_id))
        return await asyncio.wait_for(waiter, timeout)
    except asyncio.TimeoutError:
        # Floor must be released either way or the agent stays SPEAKING forever.
        logger.warning("[Delegated] vendor never replied turn_id=%s", turn_id)
        return None
    finally:
        (run.get("vendor_reply_waiters") or {}).pop(turn_id, None)


def resolve_vendor_reply(run: Dict[str, Any], turn_id: int, text: str) -> None:
    """Called from the control socket when the vendor reports what it said."""
    waiter = (run.get("vendor_reply_waiters") or {}).get(turn_id)
    if waiter is not None and not waiter.done():
        waiter.set_result(text)
