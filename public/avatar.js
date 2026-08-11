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
      if (track.kind === "video") {
        track.attach(videoEl);
        videoEl.play().catch(console.warn);
      }
      if (track.kind === "audio") {
        const audioEl = track.attach();
        audioEl.autoplay = true;
        document.body.appendChild(audioEl);
      }
    });

    await room.connect(livekitUrl, livekitToken);
    updateStatus("Live Avatar Connected", "active");

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
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      updateStatus("Audio Stream Live", "active");
      
      // Start recording raw audio frames and sending to WebSocket
      const mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          ws.send(event.data);
        }
      };

      mediaRecorder.start(100); // 100ms chunks for minimum latency
    };

    // 5. Receive "avatar_speak" commands from Gemini/RAG and speak in meeting
    const encoder = new TextEncoder();
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === "avatar_speak" && data.text) {
          const command = {
            event_id: crypto.randomUUID(),
            event_type: "avatar.speak_text",
            session_id: sessionId,
            text: data.text,
            payload: {
              text: data.text,
            },
          };

          // Publish command to LiveKit agent-control topic
          room.localParticipant.publishData(encoder.encode(JSON.stringify(command)), {
            reliable: true,
            topic: "agent-control",
          });

          updateStatus("Avatar Speaking...", "active");
          setTimeout(() => updateStatus("Listening...", "active"), 3000);
        }
      } catch (err) {
        console.error("Error processing WS message:", err);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      updateStatus("Audio WS Error", "error");
    };

  } catch (err) {
    console.error("Initialization error:", err);
    updateStatus(`Error: ${err.message}`, "error");
  }
})();
