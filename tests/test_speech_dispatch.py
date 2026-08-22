"""Speech dispatch must look identical to the agent loop whether the provider
takes text or PCM. These tests pin the wire output of each shape."""

import asyncio
from types import SimpleNamespace

import pytest

from src.core import speech


class FakeWs:
    def __init__(self):
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)


class FakeTts:
    """Yields `total` bytes in fixed-size pieces, ignoring the text."""

    def __init__(self, total=14000, piece=5000, fail=False):
        self.total, self.piece, self.fail = total, piece, fail
        self.calls = []

    async def stream(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("tts exploded")
        sent = 0
        while sent < self.total:
            n = min(self.piece, self.total - sent)
            sent += n
            yield b"\x01" * n


def _run(accepts="text", tts=None, chunk_bytes=6000):
    ws = FakeWs()
    video = SimpleNamespace(accepts=accepts, name=accepts, chunk_bytes=chunk_bytes)
    return {"control_ws": ws, "providers": SimpleNamespace(video=video, tts=tts)}, ws


def test_text_provider_gets_one_message():
    run, ws = _run("text")
    assert asyncio.run(speech.emit_speech(run, 1, "1-1", "Hello."))
    assert ws.sent == [{"type": "avatar_speak", "text": "Hello.", "turn_id": 1, "chunk_id": "1-1"}]


def test_run_without_providers_falls_back_to_text():
    """Runs created before the provider layer must keep working mid-deploy."""
    ws = FakeWs()
    assert asyncio.run(speech.emit_speech({"control_ws": ws}, 1, None, "Hello."))
    assert ws.sent == [{"type": "avatar_speak", "text": "Hello.", "turn_id": 1}]


def test_audio_provider_gets_framed_pcm_bracketed_by_speak_and_end():
    tts = FakeTts(total=14000, piece=5000)
    run, ws = _run("audio", tts=tts, chunk_bytes=6000)
    assert asyncio.run(speech.emit_speech(run, 7, "7-1", "Hello."))

    kinds = [m["type"] for m in ws.sent]
    assert kinds[0] == "avatar_speak"
    assert kinds[-1] == "avatar_speak_end"
    assert set(kinds[1:-1]) == {"avatar_audio"}
    assert tts.calls == ["Hello."]

    # 14000 bytes at 6000/frame -> two full frames and a 2000-byte remainder.
    import base64
    sizes = [len(base64.b64decode(m["data"])) for m in ws.sent if m["type"] == "avatar_audio"]
    assert sizes == [6000, 6000, 2000]
    assert sum(sizes) == 14000, "every synthesized byte must reach the avatar"


def test_audio_path_round_trips_chunk_id_on_the_end_marker():
    """Without chunk_id the ack reads as a single-shot reply and the backend
    releases the floor while the rest of the answer is still queued."""
    run, ws = _run("audio", tts=FakeTts(total=100, piece=100))
    asyncio.run(speech.emit_speech(run, 7, "7-2", "Hi."))
    end = [m for m in ws.sent if m["type"] == "avatar_speak_end"]
    assert end == [{"type": "avatar_speak_end", "turn_id": 7, "chunk_id": "7-2"}]


def test_audio_provider_without_a_voice_sends_nothing():
    run, ws = _run("audio", tts=None)
    assert asyncio.run(speech.emit_speech(run, 1, "1-1", "Hello.")) is False
    assert ws.sent == []


def test_tts_failure_still_closes_the_utterance():
    """A page left waiting for frames that never come never acks, and the turn
    hangs until the watchdog fires."""
    run, ws = _run("audio", tts=FakeTts(fail=True))
    assert asyncio.run(speech.emit_speech(run, 3, "3-1", "Hello.")) is False
    assert ws.sent[-1]["type"] == "avatar_speak_end"


def test_no_control_socket_is_not_an_error():
    assert asyncio.run(speech.emit_speech({"control_ws": None}, 1, None, "Hi.")) is False


def test_delegated_turn_returns_what_the_vendor_said():
    run, ws = _run("text")

    async def scenario():
        task = asyncio.create_task(speech.delegate_turn(run, 5, "What is SOC 2?", timeout=2))
        await asyncio.sleep(0)  # let it register its waiter
        speech.resolve_vendor_reply(run, 5, "We hold SOC 2 Type II.")
        return await task

    assert asyncio.run(scenario()) == "We hold SOC 2 Type II."
    assert ws.sent == [{"type": "avatar_user_message", "text": "What is SOC 2?", "turn_id": 5}]


def test_delegated_turn_gives_up_rather_than_holding_the_floor():
    run, _ = _run("text")
    assert asyncio.run(speech.delegate_turn(run, 5, "q", timeout=0.05)) is None
    assert run["vendor_reply_waiters"] == {}, "waiter must not leak"


def test_stale_vendor_reply_is_ignored():
    """A reply for a superseded turn must not resolve the live one."""
    run, _ = _run("text")

    async def scenario():
        task = asyncio.create_task(speech.delegate_turn(run, 9, "q", timeout=0.15))
        await asyncio.sleep(0)
        speech.resolve_vendor_reply(run, 8, "answer to an older turn")
        return await task

    assert asyncio.run(scenario()) is None
