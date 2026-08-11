# Spiked-AI-LiveAvatar

Real-time Live Avatar Orchestrator for SpikedAI.

This service coordinates:
- **Recall.ai Output Media**: Injects live video and audio directly into Zoom, Google Meet, and Microsoft Teams meetings via a self-hosted `avatar.html` webpage (`web_4_core` variant).
- **Deepgram Live Streaming STT**: Low-latency speech-to-text with speaker diarization and voice activity detection (VAD).
- **Google Gemini 2.0 Flash**: Intelligent conversational agent with native tool-calling (`generate_system_answer`).
- **SpikedAI Document RAG (`/ask/handsfree`)**: Retrieves grounded company facts, SLAs, pricing, and persona guidelines from Supabase vector embeddings with Groq inference acceleration.
- **LiveAvatar / HeyGen Lip-Sync WebRTC**: Real-time video lip-sync rendering in **LITE Mode ($0.10/min)**.
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
LIVEAVATAR_API_KEY=your_liveavatar_key
LIVEAVATAR_AVATAR_ID=your_avatar_id
SPIKED_BACKEND_URL=https://spikedai-production-application-409019309412.us-central1.run.app
RECALL_WEBHOOK_URL=https://recall-backend-production-409019309412.us-central1.run.app/webhook/recall/transcript
PUBLIC_BASE_URL=https://your-domain.ngrok-free.app
```

3. Run the development server:
```bash
uvicorn src.live_avatar:app --host 0.0.0.0 --port 8080 --reload
```

## Deployment

CI/CD is automated via Google Cloud Build on commits to `main`:
```bash
gcloud builds submit --config cloudbuild.yaml
```
