import asyncio

from src.core.asr import ParticipantTranscriber


class FakeSocket:
    async def send(self, _message):
        return None

    async def close(self):
        return None


class ReconnectingTranscriber(ParticipantTranscriber):
    def __init__(self):
        super().__init__("p1", "Client", [], lambda *_args: None)
        self.sockets = []

    async def ensure_started(self):
        if not self.ws:
            socket = FakeSocket()
            if not self.sockets:
                async def fail_once(_message):
                    raise OSError("simulated dropped stream")
                socket.send = fail_once
            self.sockets.append(socket)
            self.ws = socket


def test_participant_send_reconnects_without_failing_the_shared_audio_loop():
    async def scenario():
        transcriber = ReconnectingTranscriber()
        await transcriber.send(b"pcm")
        return transcriber.sockets

    sockets = asyncio.run(scenario())

    assert len(sockets) == 2


def test_transcriber_close_flushes_finalized_words_without_promoting_interim():
    received = []

    async def scenario():
        transcriber = ParticipantTranscriber(
            "p1", "Client", [],
            lambda pid, name, utterance: received.append((pid, name, utterance)),
        )
        transcriber.ws = FakeSocket()
        transcriber.receiver_task = asyncio.create_task(asyncio.sleep(0))
        transcriber.buffer.add_result({
            "type": "Results",
            "is_final": True,
            "speech_final": False,
            "channel": {"alternatives": [{"transcript": "Tom, did you catch that?"}]},
        })
        await transcriber.close()

    asyncio.run(scenario())

    assert received == [
        ("p1", "Client", {"text": "Tom, did you catch that?", "words": []})
    ]
