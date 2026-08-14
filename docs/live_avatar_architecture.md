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
    Bot->>Backend: Open Audio WebSocket

    %% Audio Streaming & STT
    Note over Participant, Backend: 2. Real-Time Audio & Transcription
    Participant->>Bot: Speak in Meeting
    Bot->>Backend: Stream raw audio chunks
    Backend->>DG: Forward audio stream
    DG-->>Backend: Return transcripts

    %% Orchestrator Evaluation
    Note over Backend, Gemini: 3. Addressing & Gating Check
    Backend->>Backend: Repair STT noise
    alt Addressed to Tom OR Conversational Continuation
        Backend->>Gemini: POST generateContent
        Note right of Gemini: System Prompt instructs RAG tool calling
        Gemini-->>Backend: Tool Call generate_system_answer
        
        %% RAG Execution
        Backend->>SpikedRAG: POST ask handsfree
        SpikedRAG-->>Backend: Return grounded documents
        
        Backend->>Gemini: Send RAG results
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
  * **Interactivity Configuration**: Configured with `interactivity_type: "PUSH_TO_TALK"`. This completely deactivates HeyGen's built-in, server-side LLM conversational engine, preventing race conditions or generic responses while Gemini processes custom RAG answers. The avatar is controlled strictly by our orchestrator via the `avatar.speak_text` command.

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
Tom stays silent (`[SILENT]`) unless explicitly addressed. Gating is calculated deterministically by checking:
1. **Explicit Name Invocation**: If the transcript contains the name `"Tom"` or any of its misheard phonetic approximations (e.g. `Thom`, `Dom`, `Tong`, `Tone`, `Toom`, `Time`).
2. **Dialog Continuation**: If Tom spoke on the *immediate previous turn* in the history, any fast user response is treated as a conversational continuation, allowing back-and-forth dialogue without needing the user to repeat "Tom" every sentence.

### 2. ASR Noise Repair
Since real-time meeting audio transcribes with errors, transcripts are normalized before hitting Gemini using strategic keywords:
* `"secure a" / "secura" / "secure ai" -> "s3cura AI"`
* `"three c" / "three cai" / "3c ai" -> "3CAI"`
* `"spider" / "spike the eye" / "spike ai" -> "SpikedAI"`
* `"comic gp" / "karma gp" -> "Karmic GP"`
* `"context graf" / "contact graph" -> "Context Graph"`

### 3. Graceful Barge-In & Interruption
* If a participant interrupts and speaks while the avatar is talking, the orchestrator sends an `avatar_interrupt` command to halt the avatar's video speech instantly.
* **STT Grace Window**: A `1.5s` guard window is enforced. Barge-ins cannot trigger within the first 1.5 seconds of starting speech to prevent back-to-back transcripts in the STT queue from canceling the output before the user has actually heard it.

### 4. Sliding Context Window
* A sliding history window of the last **12 turns** is injected into each Gemini request to preserve conversational context, pronoun resolution ("it", "that platform", "your service"), and dialogue coherence.

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
