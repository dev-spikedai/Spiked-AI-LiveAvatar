# Transcript replay testing

The replay harness behaves like a small scripted meeting stream. It feeds
multi-participant turns through the same production turn path used after
Deepgram finalization:

```text
JSON steps -> interim observation -> _ingest_utterance
            -> merge/finalize -> turn gate -> Gemini -> RAG/autospeak
```

Only the final speech/video transport is replaced. The harness records Tom's
filler, answer chunks, autonomous interjections, and interruption events instead
of opening Recall or LiveAvatar sessions. Speech is deliberately slowed down so
the next participant event can barge in while Tom is speaking.

## Run it

From `Spiked-AI-LiveAvatar`:

```powershell
.\.venv\Scripts\python.exe scripts\replay_meeting.py examples\meeting_scenario.json
```

For machine-readable output:

```powershell
.\.venv\Scripts\python.exe scripts\replay_meeting.py examples\meeting_scenario.json --json
```

For a safe local mechanics test that never calls Gemini, Supabase, or RAG:

```powershell
.\.venv\Scripts\python.exe scripts\replay_meeting.py examples\meeting_scenario.json --offline
```

The offline brain emits deterministic filler and answer chunks, so it verifies
the simulator's waits, simulated speech duration, and barge-in behavior. Omit
`--offline` to use the real Gemini/RAG path with the normal local environment.

The replay uses live Gemini and RAG when the normal environment variables are
available. Set `settings.token` in a private local copy, or set
`SUPABASE_ACCESS_TOKEN`, when the RAG backend requires a user bearer token.
Never commit a real token to a fixture.

## Scenario format

The scenario uses ordered `steps`. A participant turn looks like:

```json
{
  "turn": {
    "participant_id": "client",
    "speaker": "Client",
    "say": "Tom, what is the migration approach?"
  },
  "wait_for": [
    { "kind": "tom_speech", "phase": "filler", "contains": "check", "timeout_s": 30 },
    { "kind": "tom_answer", "contains": "migration", "timeout_s": 45 }
  ]
}
```

`wait_for` expectations are consumed in order, so the second expectation only
matches speech emitted after the first one. Supported expectation fields are:

- `kind`: `tom_speech`, `tom_answer`, `tom_interjection`, or `tom_interrupted`;
- `phase`: `filler` or `answer` for `tom_speech`;
- `contains`: a string or list of strings that must appear in Tom's text;
- `not_contains`: text that must not appear;
- `source`: usually `addressed` or `autonomous`;
- `timeout_s`: how long to wait before failing the scenario.

Use a separate assertion-only step when needed:

```json
{
  "expect": [
    { "kind": "tom_answer", "contains": "AWS", "timeout_s": 30 }
  ]
}
```

To simulate an interruption while Tom is speaking:

```json
{
  "barge_in": {
    "participant_id": "client",
    "speaker": "Client",
    "say": "Stop there, I have a question.",
    "after_s": 0.2
  },
  "expect": [
    { "kind": "tom_interrupted", "timeout_s": 5 }
  ]
}
```

Normal `turn` steps also automatically interrupt Tom when
`settings.auto_barge_in` is true. Set it to false if you want to test a turn
arriving while Tom is speaking without simulating the audio floor detector.

The original flat `turns` format remains supported for backward compatibility.

## Useful scenarios

- An unprompted implementation-cost concern that should trigger autospeak.
- A third-person "I told Tom..." mention that must stay silent.
- "Tom, did you catch that?" after a multi-participant discussion.
- A barge-in during a filler or answer chunk.
- A repeated question blocked by cooldown or duplicate handling.

This is a decision/reasoning replay, not an audio-quality test. Recall,
Deepgram endpointing, LiveAvatar latency, and real barge-in PCM behavior still
require a live smoke test.
