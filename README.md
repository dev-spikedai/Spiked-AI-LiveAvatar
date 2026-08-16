# Spiked-AI-LiveAvatar.

Real-time Live Avatar Orchestrator for SpikedAI..

This service coordinates:
- **Recall.ai Output Media**: Injects live video and audio directly into Zoom, Google Meet, and Microsoft Teams meetings via a self-hosted `avatar.html` webpage (`web_4_core` variant).
- **Recall Separate Participant Audio**: 16 kHz PCM streams with stable participant IDs and display names for multi-person meetings.
- **Deepgram Live Streaming STT**: One Nova-3 stream per active participant, with complete-utterance buffering and keyterm prompting.
- **Google Gemini 3.5 Flash-Lite**: Structured intent routing, contextual query repair, and concise spoken-answer generation.
- **SpikedAI Document RAG (`/ask/handsfree`)**: Retrieves grounded company facts, SLAs, pricing, and persona guidelines from Supabase vector embeddings with Groq inference acceleration.
- **Cartesia TTS**: Synthesizes the avatar's spoken replies (raw PCM16 @ 16kHz) and streams them to the page.
- **Simli AI Avatar**: Renders + lip-syncs the avatar (WebRTC p2p) in the meeting camera feed. Simli is output-only; all STT/LLM/TTS run here, and meeting audio never reaches Simli.
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
SIMLI_API_KEY=your_simli_key
SIMLI_FACE_ID=your_simli_face_id
CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=your_cartesia_voice_id
RECALL_API_KEY=your_recall_key
RECALL_WEBHOOK_SECRET=whsec_your_recall_workspace_secret
SPIKED_BACKEND_URL=https://spikedai-production-application-409019309412.us-central1.run.app
RECALL_WEBHOOK_URL=https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript
PUBLIC_BASE_URL=https://your-domain.ngrok-free.app
AGENT_BARGE_IN_MS=700
AGENT_ENDPOINTING_MS=500
AGENT_UTTERANCE_END_MS=1000
AGENT_MAX_REPLY_WORDS=45
```

`RECALL_WEBHOOK_SECRET` is recommended. Workspaces without one use a cryptographically random per-run token embedded only in Recall’s realtime endpoint URL.

Keys (all have free tiers):
- **Deepgram** (STT): https://console.deepgram.com
- **Google AI Studio** (Gemini LLM): https://aistudio.google.com
- **Simli** (avatar): https://app.simli.com — free $10 credit + 50 min/month top-up
- **Cartesia** (TTS): https://play.cartesia.ai — free tier with monthly credits

## Meeting behavior

- Tiya speaks only when a finalized participant utterance explicitly contains `Tiya`. Every turn requires the name.
- After wake-name detection, a constrained LLM gate chooses `respond`, `acknowledge`, or `silent`, so third-person mentions and explicit requests not to answer do not make Tiya speak.
- Participant names come from Recall and are preserved in conversation history.
- Company, product, pricing, security, SLA, and integration questions always call document RAG before an answer is generated.
- Normal answers are capped at two sentences and 45 spoken words; no automatic follow-up question is added.
- While Tiya is speaking, 700 ms of sustained speech interrupts playback but does not trigger a reply unless the new turn also invokes Tiya.

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
