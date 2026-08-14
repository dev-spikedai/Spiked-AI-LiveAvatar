# SpikedAI LiveAvatar Integration Architecture

This document outlines the real-time orchestrator architecture for **SpikedAI's LiveAvatar meeting assistant (Tom)**. It details how the different backend services, third-party APIs, and WebRTC streaming channels integrate to deliver low-latency, grounded conversational meeting participation.

---

## 1. System Architecture

Below is the visual integration flow of the system. It follows the **Direct Custom RAG Pipeline via WebRTC Data Channel** design:

```mermaid
sequenceDiagram
    autonumber
    participant Participant as Meeting Participant
    participant Bot as Recall Bot
    participant LASDK as LiveAvatar SDK
    participant Backend as Orchestrator
    participant DG as Deepgram Nova-3
    participant Gemini as Gemini 3.5
    participant SpikedRAG as SpikedAI RAG
    participant Analytics as recall_backend

    %% Session Initialization
    Note over Bot, Backend: 1. Meeting Start & Credentials
    Bot->>Backend: GET credentials
    Backend-->>Bot: Return WebRTC Token
    Bot->>LASDK: Initialize WebRTC
    Bot->>Backend: Open output-control WebSocket

    %% Audio Streaming & STT
    Note over Participant, Backend: 2. Real-Time Audio & Transcription
    Participant->>Bot: Speak in Meeting
    Bot->>Backend: Recall realtime endpoint sends participant-separated PCM + participant identity
    Backend->>DG: Forward each participant to a dedicated Nova-3 stream
    DG-->>Backend: Return interim/final segments and end-of-turn events

    %% Orchestrator Evaluation
    Note over Backend, Gemini: 3. Addressing & Gating Check
    Backend->>Backend: Assemble complete turn and apply deterministic name gate
    alt Explicitly addressed to Tom on this turn
        Backend->>Gemini: Structured intent + contextual query repair
        Gemini-->>Backend: Intent, resolved query, validated correction candidates
        
        %% RAG Execution
        Backend->>SpikedRAG: Deterministic POST ask handsfree for company knowledge
        SpikedRAG-->>Backend: Return grounded documents
        
        Backend->>Gemini: Generate short answer from RAG results
        Gemini-->>Backend: Final response with spoken text
        
        %% Speaking Command
        Backend->>Bot: Send avatar_speak
        Bot->>LASDK: Publish speak_text
        LASDK->>Participant: Render video and play Audio
    else General Room Talk
        Backend->>Backend: Remain Silent
    end

    %% Webhook Sync
    Note over Backend, Analytics: 4. Analytics & Note Logging
    Backend->>Analytics: POST webhook transcript
```

---

## 2. Core Components & Backend Services

The integration is divided across the following key services:

### A. `LiveAvatar-Spiked` (This Service)
* **Role**: The main real-time orchestrator. It runs on Google Cloud Run.
* **Responsibilities**:
  * Manages active WebSocket connections from headless bots.
  * Forwards real-time raw PCM/Opus audio chunks to Deepgram STT.
  * Applies ASR transcript repair using seller company strategic keywords.
  * Executes the conversational dialogue policy, gating rules, and name-invocation logic.
  * Coordinates Gemini LLM inference and executes the RAG retrieval.
  * Controls the avatar's speech, barge-in interruptions, and synchronization.

### B. `Recall.ai` (Meeting Connector)
* **Role**: Headless virtual meeting bot.
* **Responsibilities**:
  * Joins meetings (Zoom, MS Teams, Google Meet) as a visual participant.
  * Loads our customized containerized web page ([`avatar.html`](file:///Users/diptanshukumar/Callendar/LiveAvatar-Spiked/public/avatar.html)) inside an isolated headless Chrome browser.
  * Automatically handles `getUserMedia` permissions, piping in-meeting speaker audio to our WebSocket and rendering the incoming LiveKit avatar WebRTC video track.

### C. `LiveAvatar (HeyGen) Streaming API`
* **Role**: Lip-Sync Video Generation.
* **Responsibilities**:
  * Renders the photorealistic 3D/2D avatar (Tom) with sub-second lipsync.
  * Streams raw audio and H264 video back to the headless browser via a LiveKit-hosted WebRTC room.
* **Interactivity Configuration**: Configured with `interactivity_type: "PUSH_TO_TALK"`, kept in `avatar.stop_listening`, and used only through `avatar.speak_text`. No meeting microphone is published into LiveAvatar, so its internal ASR/LLM cannot race the orchestrator.

### D. `SpikedAI-Backend-One`
* **Endpoint**: `/ask/handsfree`
* **Role**: Dynamic Document RAG Service.
* **Responsibilities**:
  * Performs vector similarity search against the user's uploaded company knowledge base.
  * Returns factual information, pricing, SLAs, and technical specifications mapped to the user's specific `client_id` and Supabase JWT context.

### E. `recall_backend`
* **Endpoint**: `/webhook/recall/transcript`
* **Role**: Analytics Sync Service.
* **Responsibilities**:
  * Logs all transcribed meeting turns, sentiment analysis, CRM integrations, and summary notes to the main SpikedAI user dashboard.

---

## 3. Key Orchestration Logic

### 1. Conversational Dialogue & Gating Policy
Tom stays silent unless each finalized utterance explicitly contains `Tom` or the safe spelling variant `Thom`. Common words such as `time` and `tone`, other names such as `Dom`, and automatic post-answer continuation are deliberately rejected.

### 2. ASR Noise Repair
Nova-3 receives up to 100 verified company/product keyterms. Gemini may propose entity corrections after an addressed, complete turn, but the backend accepts only high-confidence replacements found in the verified entity catalog. Pronouns and omitted context are expanded in a separate retrieval query without rewriting the stored transcript.

### 3. Graceful Barge-In & Interruption
* Participant-separated PCM is evaluated in 20 ms VAD frames. At least 700 ms of sustained non-bot speech sends one `avatar_interrupt` command.
* An interruption stops playback only. The completed participant turn still needs to invoke Tom before it can receive an answer.

### 4. Sliding Context Window
* A bounded named-speaker history is retained per run. The last 12 turns are supplied for routing, contextual repair, pronoun resolution, and answer generation.

---

## 4. Local Run & Deployment Command Sheet

### Running Locally
1. Start the server (ensure `.env` has active keys):
   ```bash
   uvicorn src.live_avatar:app --host 0.0.0.0 --port 8080 --reload
   ```
2. Proxy your local server to expose the public endpoint for Recall:
   ```bash
   ngrok http 8080
   ```

### Deploying to Cloud Run
To trigger the automated container rebuild and deploy the latest changes to Cloud Run:
```bash
gcloud builds submit --config cloudbuild.yaml
```
*Note: Make sure your `gcloud` context is authenticated as `dev@spiked.ai` in the `spikedai-production` project.*
