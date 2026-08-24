# Cognitive Twin Rollout Runbook

This runbook covers the order in which the warm RAG service, orchestrator, and
memory schema should be promoted. The orchestrator is deliberately backward
compatible: a missing warm endpoint, MCP endpoint, or memory table must not
prevent a meeting from starting.

## 1. Deploy the warm RAG service first

Deploy `feat-fastembeddings` and verify:

```text
GET /health             -> status=ok
POST /warm              -> authenticated user, chunks_loaded > 0 when docs exist
POST /ask/regular       -> streamed answer with the user's bearer token
```

`POST /warm` derives the user id from the bearer token. Never expose a route
that accepts an arbitrary user id for production warming; the existing
`/debug/warm/{user_id}` route is for local/test use only.

## 2. Configure the orchestrator

Set:

```text
SPIKED_BACKEND_URL=<legacy/general backend, if cognitive fallback is used>
RAG_BACKEND_URL=<warm feat-fastembeddings Cloud Run URL>
MCP_SERVER_URL=<optional read-only MCP endpoint>
```

Do not point `RAG_BACKEND_URL` at a test service until that service has the
correct Supabase project, authentication configuration, and client-scoped
documents.

The Cloud Build manifest carries these as explicit substitutions. Promote the
warm and MCP services with:

```text
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_RAG_BACKEND_URL=<warm-service-url>,_MCP_SERVER_URL=<mcp-url>
```

The orchestrator build checks that the selected warm service responds with
`{"status":"ok"}` before it builds or deploys. This catches an incorrect or
unreachable RAG URL during rollout instead of after a meeting starts.

`/health` reports whether warm RAG, MCP, FULL-mode LiveAvatar, and persistent
memory credentials are configured. It never reports secret values.

### Deployment secret contract

The orchestrator Cloud Run revision expects these Secret Manager bindings:

| Secret Manager name | Runtime variable | Required for |
|---|---|---|
| `DEEPGRAM_API` | `DEEPGRAM_API` | Nova-3 participant transcription |
| `LIVEAVATAR_API_KEY` | `LIVEAVATAR_API_KEY` | FULL-mode avatar sessions |
| `RECALL_API_KEY` | `RECALL_API_KEY` | Recall bot join/leave |
| `GEMINI_API_KEY` | `GEMINI_API_KEY` | Turn classification and reasoning |
| `SUPABASE_KEY_TRANSCRIPT` | `SUPABASE_KEY` | User config and persistent memory |
| `SUPABASE_URL_TRANSCRIPT` | `SUPABASE_URL` | Supabase project endpoint |

`SUPABASE_KEY_TRANSCRIPT` must be service-role-capable, or explicitly have
access to `sources`, `chunks`, `user_configs`, `client_provider_configs`, and
`agent_memory`. The warm RAG service has its own Supabase/LLM environment and
is not populated by the orchestrator's `--set-secrets` list.

The current rollout deliberately leaves `MCP_SERVER_URL` empty. If MCP is
enabled later, the MCP adapter holds no credentials; it forwards the caller's
Supabase bearer token and only exposes read-only tools. `RECALL_WEBHOOK_SECRET`
is also not required by this deployment: the orchestrator's audio socket uses
the per-run token, while transcript webhooks target the separate Recall
backend.

Before promoting, confirm the Cloud Run runtime service account can access all
six secrets, the `agent_memory` schema is applied, and the Recall key belongs
to the configured `us-west-2` region. The build then checks the warm service's
public `/health` endpoint before deploying.

## 3. Deploy the memory schema

Apply [`PERSISTENT_MEMORY_SCHEMA.sql`](./PERSISTENT_MEMORY_SCHEMA.sql) to the
application's Supabase project. Confirm that service-role access is available
to the orchestrator and that browser clients do not receive a policy granting
arbitrary access to `agent_memory`.

Before enabling persistent writes, test:

- A global user preference uses `client_id=''`.
- A client-specific preference is visible only for that client.
- A repeated explicit fact upserts instead of duplicating.
- A missing table leaves the meeting functional.

## 4. Start a FULL-mode smoke meeting

Use the normal frontend start path with `video_provider=liveavatar` and no TTS
provider. Confirm the run reports:

```text
Recall bot joined
control WebSocket connected
separate audio WebSocket connected
LiveAvatar reports FULL mode
Deepgram participant stream connected
```

The LiveAvatar provider must publish `avatar.stop_listening` immediately after
connecting. The orchestrator is the only component allowed to decide speech.

## 5. Behavioral smoke script

Run these in a two-or-more participant meeting:

1. `Tom, did you catch that?` — answer from the immediately preceding context.
2. `Tom, what is our implementation SLA?` — warm RAG answer.
3. Say a question without Tom — remain silent by default.
4. Say `I prefer you to speak up when you can help.`
5. Say `We are worried about implementation cost.` — proactive candidate.
6. Say `Time, what does it cost?` — tolerate the likely ASR name variant.
7. Say `What time can you meet?` — do not invoke Tom.
8. Interrupt Tom for at least 700 ms — playback stops and the floor returns.
9. Say `Tom, stay quiet for 30 seconds.` — mute overlay/countdown appears.
10. Disconnect — Recall leaves, LiveAvatar stops, sockets close, and the run is
    removed from `/api/active-runs`.

## 6. Latency evidence to collect

While the bot is live, `GET /api/runs/{run_id}/diagnostics` exposes sanitized
turn timing, interim wake candidates, proactive prefetch state, warm-task
completion, and MCP enrichment state. It does not return tokens, meeting URLs,
or raw conversation history.

For each addressed and proactive turn, record the correlation timestamps in
Cloud Run logs:

```text
first interim transcript
wake candidate
Deepgram final
turn finalize
RAG warm/cache hit or miss
RAG time-to-first-byte
first answer sentence
avatar dispatch
avatar_speak_started
avatar_speak_ended
```

The most important first comparison is `Deepgram final -> avatar_speak_started`.
If that is already low but the user still perceives delay, the remaining cost
is vendor rendering/playback and should be masked with sentence streaming or a
short truthful filler, not another LLM call.

## 7. Rollback

Rollback is configuration-first:

1. Set `RAG_BACKEND_URL` back to the previous known-good backend.
2. Set `MCP_SERVER_URL` empty if MCP enrichment is unhealthy.
3. Leave the memory table in place; failed reads/writes are fail-safe.
4. Redeploy the previous orchestrator revision if the control protocol itself
   is broken.
5. Verify Recall eviction and LiveAvatar session stop before trying another
   meeting, because a leaked FULL-mode session consumes the concurrency slot.
