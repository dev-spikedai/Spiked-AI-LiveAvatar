// LiveAvatar-Spiked: public/avatar.js
// Handles WebRTC video playback, room audio capture, and low-latency WebSocket communication.

(async function () {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const videoEl = document.getElementById("avatar-video");

  function updateStatus(text, state = "pending") {
    statusText.textContent = text;
    statusDot.className = `status-dot ${state === "active" ? "active" : state === "error" ? "error" : ""}`;
  }

  try {
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("run") || "default";
    const token = params.get("token") || "";
    const clientId = params.get("client_id") || "";
    let sessionId = params.get("session_id") || "";
    let livekitUrl = params.get("livekit_url") || "";
    let livekitToken = params.get("livekit_token") || "";

    updateStatus("Fetching session credentials...");

    // 1. If credentials are not in URL query params, fetch them from backend
    if (!livekitUrl || !livekitToken) {
      const res = await fetch(`/api/runs/${runId}/credentials?token=${encodeURIComponent(token)}`);
      if (!res.ok) throw new Error(`Failed to load credentials: ${res.status}`);
      const data = await res.json();
      sessionId = data.session_id;
      livekitUrl = data.livekit_url;
      livekitToken = data.livekit_token;
    }

    updateStatus("Connecting to LiveKit WebRTC...");

    // 2. Connect to LiveAvatar LiveKit Room
    const room = new LivekitClient.Room({
      adaptiveStream: true,
      dynacast: true,
    });

    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
      console.log(`[LiveKit] Track subscribed: ${track.kind}`);
      if (track.kind === "video") {
        track.attach(videoEl);
        videoEl.play().catch(console.warn);
        updateStatus("Avatar Video Rendering", "active");
      }
      if (track.kind === "audio") {
        const audioEl = track.attach();
        audioEl.autoplay = true;
        document.body.appendChild(audioEl);
        console.log("[LiveKit] Avatar audio track attached to DOM");
      }
    });

    const decoder = new TextDecoder();
    room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind, topic) => {
      try {
        const decoded = JSON.parse(decoder.decode(payload));
        console.log(`[LiveKit DataReceived] topic=${topic}:`, decoded);
      } catch {
        console.log(`[LiveKit DataReceived] topic=${topic} (raw)`);
      }
    });

    await room.connect(livekitUrl, livekitToken);
    console.log("[LiveKit] Connected to room successfully");
    updateStatus("Live Avatar Connected", "active");

    // Allow autoplay for audio
    room.startAudio().catch(() => {
      document.addEventListener("click", () => room.startAudio(), { once: true });
    });

    // Publish page microphone into LiveKit room (Recall pipes meeting audio into it)
    try {
      await room.localParticipant.setMicrophoneEnabled(true, {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      });
      console.log("[LiveKit] Microphone published to LiveKit room");
    } catch (micErr) {
      console.warn("[LiveKit] Could not publish microphone:", micErr);
    }

    // Tell LiveAvatar agent to start listening (REQUIRED per LiveAvatar SDK)
    const encoder = new TextEncoder();
    function sendLiveAvatarCommand(eventType, payload = {}) {
      const event = {
        event_id: crypto.randomUUID(),
        event_type: eventType,
        session_id: sessionId,
        source_event_id: null,
        ...payload,
        payload,
      };
      console.log(`[LiveKit] Publishing command: ${eventType}`, event);
      room.localParticipant.publishData(encoder.encode(JSON.stringify(event)), {
        reliable: true,
        topic: "agent-control",
      });
    }

    sendLiveAvatarCommand("avatar.start_listening");
    console.log("[avatar.js] Sent avatar.start_listening command");

    // 3. Capture In-Meeting Audio (Recall auto-grants getUserMedia)
    updateStatus("Initializing In-Room Audio Capture...");
    const audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: 16000,
        channelCount: 1,
      },
    });

    // 4. Open Real-Time WebSocket to LiveAvatar-Spiked Backend
    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${location.host}/ws/audio/${sessionId}?token=${encodeURIComponent(token)}&client_id=${encodeURIComponent(clientId)}&bot_id=${encodeURIComponent(runId)}`;
    console.log(`[WS] Connecting to backend: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    let sentChunks = 0;
    ws.onopen = () => {
      console.log("[WS] WebSocket connection established");
      updateStatus("Audio Stream Live", "active");
      
      // Start recording raw audio frames and sending to WebSocket
      const mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          sentChunks++;
          if (sentChunks % 30 === 0) {
            console.log(`[MediaRecorder] Sent ${sentChunks} audio chunks (${event.data.size} bytes/chunk)`);
          }
          ws.send(event.data);
        }
      };

      mediaRecorder.start(100); // 100ms chunks for minimum latency
      console.log("[MediaRecorder] Started recording at 100ms interval");
    };

    // 5. Receive "avatar_speak" commands from Gemini/RAG and speak in meeting
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[WS] Received message from backend:", data);
        
        if (data.type === "avatar_speak" && data.text) {
          console.log(`[avatar.js] >>> Publishing avatar.speak_text: "${data.text}"`);
          sendLiveAvatarCommand("avatar.speak_text", { text: data.text });

          updateStatus(`Avatar Speaking: "${data.text.slice(0, 30)}..."`, "active");
          setTimeout(() => updateStatus("Listening...", "active"), 4000);
        }
      } catch (err) {
        console.error("Error processing WS message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("[WS] WebSocket error:", err);
      updateStatus("Audio WS Error", "error");
    };

    ws.onclose = () => {
      console.log("[WS] WebSocket connection closed");
      updateStatus("Audio Stream Closed", "pending");
    };

  } catch (err) {
    console.error("Initialization error:", err);
    updateStatus(`Error: ${err.message}`, "error");
  }
})();
