// LiveAvatar-Spiked: public/avatar.js
// Handles LiveAvatar playback and backend control. Meeting audio goes directly from Recall to the backend.

(async function () {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const videoEl = document.getElementById("avatar-video");

  function updateStatus(text, state = "pending") {
    statusText.textContent = text;
    statusDot.className = `status-dot ${state === "active" ? "active" : state === "error" ? "error" : ""}`;
  }

  // Debug overlay: shows every finalized turn the backend evaluated, with the
  // gate's verdict. Clears on each speaking -> listening transition so the panel
  // always reflects the current turn rather than the whole meeting.
  const heardOverlay = document.getElementById("heard-overlay");
  const heardLines = document.getElementById("heard-lines");
  const debugEnabled = new URLSearchParams(window.location.search).get("debug") === "1";

  function renderHeard(entry) {
    if (!debugEnabled) return;
    const row = document.createElement("div");
    row.className = "heard-line";

    const speaker = document.createElement("span");
    speaker.className = "heard-speaker";
    speaker.textContent = `${entry.speaker || "?"}:`;

    const text = document.createElement("span");
    text.className = "heard-text";
    text.textContent = entry.text || "";

    const verdict = document.createElement("span");
    verdict.className = `heard-verdict ${entry.reply ? "reply" : "blocked"}`;
    verdict.textContent = entry.reason || (entry.reply ? "reply" : "ignored");

    row.append(speaker, text, verdict);
    heardLines.appendChild(row);
    while (heardLines.childElementCount > 6) heardLines.removeChild(heardLines.firstChild);
    heardOverlay.classList.add("visible");
  }

  function clearHeard() {
    if (!debugEnabled) return;
    heardLines.replaceChildren();
    heardOverlay.classList.remove("visible");
  }

  try {
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("run") || "default";
    let sessionId = params.get("session_id") || "";
    let livekitUrl = params.get("livekit_url") || "";
    let livekitToken = params.get("livekit_token") || "";

    updateStatus("Fetching session credentials...");

    // 1. If credentials are not in URL query params, fetch them from backend
    if (!livekitUrl || !livekitToken) {
      const res = await fetch(`/api/runs/${runId}/credentials`);
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

    let currentTurnId = null;
    const speakEventTurns = new Map();
    let controlWs = null;
    function sendControlState(type, turnId = currentTurnId) {
      if (turnId === null || turnId === undefined) return;
      if (controlWs && controlWs.readyState === WebSocket.OPEN) {
        controlWs.send(JSON.stringify({ type, turn_id: turnId }));
      }
    }
    const decoder = new TextDecoder();
    room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind, topic) => {
      try {
        if (topic && topic !== "agent-response") return;
        const decoded = JSON.parse(decoder.decode(payload));
        const eventType = decoded.event_type || decoded.type;
        const eventTurnId = speakEventTurns.get(decoded.source_event_id) ?? currentTurnId;
        console.log(`[LiveKit DataReceived] topic=${topic}:`, decoded);
        
        if (eventType === "avatar.speak_started") {
          sendControlState("avatar_speak_started", eventTurnId);
          updateStatus("Avatar Speaking", "active");
        } else if (eventType === "avatar.speak_ended") {
          sendControlState("avatar_speak_ended", eventTurnId);
          if (decoded.source_event_id) speakEventTurns.delete(decoded.source_event_id);
          updateStatus("Listening...", "active");
          clearHeard();
        }
      } catch {
        console.log(`[LiveKit DataReceived] topic=${topic} (raw)`);
      }
    });

    let reconnecting = false;
    room.on(LivekitClient.RoomEvent.Disconnected, async (reason) => {
      console.warn(`[LiveKit] Room disconnected: ${reason}`);
      if (reconnecting) return;
      reconnecting = true;
      updateStatus("Disconnected — reconnecting...", "error");
      // The LiveAvatar session itself may have closed server-side (idle
      // timeout, crash, etc.), so re-fetch fresh credentials rather than
      // just re-dialing the stale ones.
      for (let attempt = 1; attempt <= 5; attempt++) {
        await new Promise((r) => setTimeout(r, Math.min(2000 * attempt, 10000)));
        try {
          const res = await fetch(`/api/runs/${runId}/credentials`);
          if (!res.ok) throw new Error(`credentials ${res.status}`);
          const data = await res.json();
          sessionId = data.session_id;
          await room.connect(data.livekit_url, data.livekit_token);
          console.log("[LiveKit] Reconnected to room successfully");
          updateStatus("Live Avatar Connected", "active");
          reconnecting = false;
          return;
        } catch (err) {
          console.warn(`[LiveKit] Reconnect attempt ${attempt} failed:`, err);
        }
      }
      updateStatus("Connection lost — refresh to rejoin", "error");
      reconnecting = false;
    });

    await room.connect(livekitUrl, livekitToken);
    console.log("[LiveKit] Connected to room successfully");
    updateStatus("Live Avatar Connected", "active");

    // Allow autoplay for audio
    room.startAudio().catch(() => {
      document.addEventListener("click", () => room.startAudio(), { once: true });
    });

    // LiveAvatar is output-only. Recall and the backend own listening and turn-taking.
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
      return event.event_id;
    }

    sendLiveAvatarCommand("avatar.stop_listening");
    console.log("[avatar.js] Sent avatar.stop_listening command");

    // 3. Open control-only WebSocket to LiveAvatar-Spiked Backend.
    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${location.host}/ws/control/${encodeURIComponent(runId)}`;
    console.log(`[WS] Connecting to backend: ${wsUrl}`);
    controlWs = new WebSocket(wsUrl);

    controlWs.onopen = () => {
      console.log("[WS] Control connection established");
      updateStatus("Listening...", "active");
    };

    // 4. Receive avatar commands from the backend.
    controlWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[WS] Received message from backend:", data);
        
        if (data.type === "heard") {
          renderHeard(data);
          return;
        }

        if (data.type === "avatar_speak" && data.text) {
          console.log(`[avatar.js] >>> Publishing avatar.speak_text: "${data.text}"`);
          currentTurnId = data.turn_id;
          const eventId = sendLiveAvatarCommand("avatar.speak_text", { text: data.text });
          speakEventTurns.set(eventId, currentTurnId);
          updateStatus(`Avatar Speaking: "${data.text.slice(0, 30)}..."`, "active");
        } else if (data.type === "avatar_interrupt") {
          console.log("[avatar.js] >>> Received avatar_interrupt command from backend");
          sendLiveAvatarCommand("avatar.interrupt");
          sendControlState("avatar_speak_interrupted");
          updateStatus("Listening...", "active");
        }
      } catch (err) {
        console.error("Error processing WS message:", err);
      }
    };

    controlWs.onerror = (err) => {
      console.error("[WS] WebSocket error:", err);
      updateStatus("Control WS Error", "error");
    };

    controlWs.onclose = () => {
      console.log("[WS] Control connection closed");
      updateStatus("Control Stream Closed", "pending");
    };

  } catch (err) {
    console.error("Initialization error:", err);
    updateStatus(`Error: ${err.message}`, "error");
  }
})();
