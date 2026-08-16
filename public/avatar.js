// SpikedAI-Simli: public/avatar.js
// Renders the Simli AI avatar (WebRTC p2p) into the meeting camera feed.
//
// The backend owns the full brain: Deepgram STT, Gemini LLM, and Cartesia TTS.
// Cartesia PCM16@16kHz audio is streamed here as base64 frames over the control
// WebSocket; each frame is forwarded to Simli, which lip-syncs and plays it.
// Meeting audio itself never touches this page or Simli — Recall feeds it
// straight to the backend.

(async function () {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const videoEl = document.getElementById("avatar-video");
  const audioEl = document.getElementById("avatar-audio");

  function updateStatus(text, state = "pending") {
    statusText.textContent = text;
    statusDot.className = `status-dot ${state === "active" ? "active" : state === "error" ? "error" : ""}`;
  }

  // Debug overlay: shows every finalized turn the backend evaluated, with the
  // gate's verdict. Renders into the meeting camera feed, so it is opt-in via
  // ?debug=1.
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

    updateStatus("Fetching session credentials...");
    const res = await fetch(`/api/runs/${runId}/credentials`);
    if (!res.ok) throw new Error(`Failed to load credentials: ${res.status}`);
    const creds = await res.json();
    const sessionToken = creds.simli_session_token;
    if (!sessionToken) throw new Error("No Simli session token in credentials");

    const simliBase = (creds.simli_base_url || "https://api.simli.ai").trim().replace(/\/+$/, "");
    const simliWsUrl = `${simliBase.replace(/^http/, "ws")}/compose/webrtc/p2p?session_token=${encodeURIComponent(sessionToken)}`;

    updateStatus("Connecting to Simli WebRTC...");

    // 1. Simli WebRTC (p2p): offer as first message, answer back, then feed the
    //    avatar raw PCM16@16kHz audio as binary frames over the same socket.
    const simliWs = new WebSocket(simliWsUrl);
    simliWs.binaryType = "arraybuffer";

    let pc = null;
    let simliStarted = false;
    const preStartAudioQueue = [];

    // Simli's p2p protocol requires a priming "zero-audio" frame shortly after
    // START before it will ingest spoken audio. 64000 bytes == 2s of PCM16@16kHz.
    // Without it the avatar's video renders but the audio pipeline stays silent.
    function primeSimliAudio() {
      setTimeout(() => {
        if (simliWs.readyState !== WebSocket.OPEN) return;
        simliWs.send(new Uint8Array(64000));
        while (preStartAudioQueue.length) {
          if (simliWs.readyState !== WebSocket.OPEN) break;
          simliWs.send(preStartAudioQueue.shift());
        }
      }, 100);
    }

    simliWs.addEventListener("open", async () => {
      try {
        pc = new RTCPeerConnection({
          iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        });
        pc.addTransceiver("audio", { direction: "recvonly" });
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.ontrack = (evt) => {
          const stream = evt.streams?.[0] || new MediaStream([evt.track]);
          if (evt.track.kind === "video") {
            videoEl.srcObject = stream;
            videoEl.play().catch(console.warn);
            updateStatus("Avatar Video Rendering", "active");
          } else if (evt.track.kind === "audio") {
            audioEl.srcObject = stream;
            audioEl.play().catch(() => {
              const resume = () => audioEl.play().catch(() => {});
              document.addEventListener("click", resume, { once: true });
              document.addEventListener("keydown", resume, { once: true });
              const retry = setInterval(() => {
                audioEl.play().then(() => clearInterval(retry)).catch(() => {});
              }, 1000);
            });
          }
        };
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        simliWs.send(JSON.stringify(offer));
      } catch (err) {
        console.error("[Simli] offer setup failed:", err);
        updateStatus(`Simli offer error: ${err.message}`, "error");
      }
    });

    simliWs.addEventListener("message", (evt) => {
      const data = evt.data;
      if (typeof data !== "string") return;
      if (data === "START") {
        simliStarted = true;
        primeSimliAudio();
        updateStatus("Listening...", "active");
      } else if (data === "SPEAK") {
        updateStatus("Avatar Speaking", "active");
      } else if (data === "SILENT") {
        updateStatus("Listening...", "active");
      } else if (data === "STOP") {
        updateStatus("Session ended", "pending");
      } else if (data === "ACK") {
        // Server acknowledges an audio segment; nothing to do.
      } else if (data.startsWith("ERROR") || data.startsWith("RATE") || data.startsWith("CLOSING")) {
        console.error("[Simli] server message:", data);
        updateStatus(data, "error");
      } else {
        try {
          const msg = JSON.parse(data);
          if (msg.type === "answer") {
            pc?.setRemoteDescription(msg).catch(console.warn);
          }
        } catch {
          /* ignore non-JSON messages */
        }
      }
    });

    simliWs.addEventListener("close", () => {
      console.log("[Simli] WebSocket closed");
      updateStatus("Simli connection closed", "pending");
    });
    simliWs.addEventListener("error", () => {
      console.error("[Simli] WebSocket error");
      updateStatus("Simli connection error", "error");
    });

    function sendSimliAudio(pcm) {
      if (simliWs.readyState !== WebSocket.OPEN) return;
      if (!simliStarted) {
        preStartAudioQueue.push(pcm);
        return;
      }
      simliWs.send(pcm);
    }

    function sendSimliSignal(signal) {
      if (simliWs.readyState === WebSocket.OPEN) {
        simliWs.send(signal);
      }
    }

    // 2. Control-only WebSocket to the backend: receives TTS audio + commands,
    //    and reports speaking lifecycle events so the floor state stays accurate.
    let currentTurnId = null;
    let controlWs = null;
    function sendControlState(type, turnId = currentTurnId) {
      if (turnId === null || turnId === undefined) return;
      if (controlWs && controlWs.readyState === WebSocket.OPEN) {
        controlWs.send(JSON.stringify({ type, turn_id: turnId }));
      }
    }

    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const controlUrl = `${wsProtocol}//${location.host}/ws/control/${encodeURIComponent(runId)}`;
    controlWs = new WebSocket(controlUrl);

    controlWs.onopen = () => {
      console.log("[WS] Control connection established");
      updateStatus("Listening...", "active");
    };

    controlWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log("[WS] Received message from backend:", data);

        if (data.type === "heard") {
          renderHeard(data);
          return;
        }

        if (data.type === "avatar_speak" && data.text) {
          currentTurnId = data.turn_id;
          sendControlState("avatar_speak_started");
          updateStatus(`Avatar Speaking: "${data.text.slice(0, 30)}..."`, "active");
        } else if (data.type === "avatar_audio" && data.data) {
          const bin = atob(data.data);
          const bytes = new Uint8Array(bin.length);
          for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
          sendSimliAudio(bytes.buffer);
        } else if (data.type === "avatar_speak_end") {
          sendControlState("avatar_speak_ended");
          updateStatus("Listening...", "active");
          clearHeard();
        } else if (data.type === "avatar_interrupt") {
          console.log("[WS] >>> avatar_interrupt");
          sendSimliSignal("SKIP");
          sendControlState("avatar_speak_interrupted");
          updateStatus("Listening...", "active");
        }
      } catch (err) {
        console.error("Error processing WS message:", err);
      }
    };

    controlWs.onerror = (err) => {
      console.error("[WS] Control WebSocket error:", err);
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
