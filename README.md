# Spiked-AI-LiveAvatar.

Real-time Live Avatar Orchestrator for SpikedAI..

This service coordinates:
- **Recall.ai Output Media**: Injects live video and audio directly into Zoom, Google Meet, and Microsoft Teams meetings via a self-hosted `avatar.html` webpage (`web_4_core` variant).
- **Recall Separate Participant Audio**: 16 kHz PCM streams with stable participant IDs and display names for multi-person meetings.
- **Deepgram Live Streaming STT**: One Nova-3 stream per active participant, with complete-utterance buffering and keyterm prompting.
- **Google Gemini 3.5 Flash-Lite**: Structured intent routing, contextual query repair, and concise spoken-answer generation.
- **SpikedAI Document RAG (`/ask/handsfree`)**: Retrieves grounded company facts, SLAs, pricing, and persona guidelines from Supabase vector embeddings with Groq inference acceleration.
- **LiveAvatar / HeyGen Lip-Sync WebRTC**: Output-only FULL-mode rendering and TTS; meeting audio is never published into LiveAvatar.
- **Dual-Track Webhook Preservation**: Forwards transcripts to `recall_backend` (`/webhook/recall/transcript`) to maintain all user dashboard notes, sentiment analysis, and meeting logs.

## Setup & Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```bash
DEEPGRAM_API=your_deepgram_key
GEMINI_API=your_gemini_key
RECALL_API_KEY=your_recall_key
RECALL_WEBHOOK_SECRET=whsec_your_recall_workspace_secret
LIVEAVATAR_API_KEY=your_liveavatar_key
LIVEAVATAR_AVATAR_ID=your_avatar_id
SPIKED_BACKEND_URL=https://spikedai-production-application-409019309412.us-central1.run.app
RECALL_WEBHOOK_URL=https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript
PUBLIC_BASE_URL=https://your-domain.ngrok-free.app
AGENT_BARGE_IN_MS=700
AGENT_ENDPOINTING_MS=500
AGENT_UTTERANCE_END_MS=1000
AGENT_MAX_REPLY_WORDS=45
```

`RECALL_WEBHOOK_SECRET` is recommended. Workspaces without one use a cryptographically random per-run token embedded only in Recall’s realtime endpoint URL.

## Meeting behavior

- Tom speaks only when a finalized participant utterance explicitly contains `Tom` or `Thom`. Every turn requires the name.
- Participant names come from Recall and are preserved in conversation history.
- Company, product, pricing, security, SLA, and integration questions always call document RAG before an answer is generated.
- Normal answers are capped at two sentences and 45 spoken words; no automatic follow-up question is added.
- While Tom is speaking, 700 ms of sustained speech interrupts playback but does not trigger a reply unless the new turn also invokes Tom.

3. Run the development server:
```bash
uvicorn src.live_avatar:app --host 0.0.0.0 --port 8080 --reload
```

## Deployment

CI/CD is automated via Google Cloud Build on commits to `main`:
```bash
gcloud builds submit --config cloudbuild.yaml
```

The deployment is currently pinned to one Cloud Run instance because live run/control state is in memory. Move that state and control fan-out to a shared store before raising `--max-instances`.
