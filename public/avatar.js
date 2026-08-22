// SpikedMeetingAgent: public/avatar.js
//
// The provider-agnostic half of the avatar page. It owns everything that is
// the same no matter which vendor is rendering the face: credentials, the
// control socket, the overlays, and the protocol.
//
// It owns nothing vendor-specific. The backend tells it which ES module to
// import (`browser_module` from /credentials) and that module implements the
// transport. Adding a provider means adding a file under /providers, not
// editing this one.
//
// Meeting audio never reaches this page in any configuration -- Recall streams
// it to the backend, which owns listening and turn-taking. Every provider
// module is therefore output-only.

const PROTOCOL_VERSION = 1;

(async function () {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const videoEl = document.getElementById("avatar-video");

  function updateStatus(text, state = "pending") {
    statusText.textContent = text;
    statusDot.className = `status-dot ${state === "active" ? "active" : state === "error" ? "error" : ""}`;
  }

  // "Tom, stay quiet for 30 seconds" — bottom-center countdown ring on the
  // actual meeting video feed, visible to everyone, not just the rep's console.
  const muteOverlay = document.getElementById("mute-overlay");
  const muteCount = document.getElementById("mute-count");
  const muteRingProgress = document.getElementById("mute-ring-progress");
  const MUTE_RING_RADIUS = 28;
  const MUTE_RING_CIRCUMFERENCE = 2 * Math.PI * MUTE_RING_RADIUS;
  muteRingProgress.style.strokeDasharray = `${MUTE_RING_CIRCUMFERENCE}`;
  let muteIntervalId = null;

  function stopMuteOverlay() {
    if (muteIntervalId !== null) {
      clearInterval(muteIntervalId);
      muteIntervalId = null;
    }
    muteOverlay.classList.remove("visible");
  }

  function startMuteOverlay(mutedUntilEpochMs, totalSeconds) {
    if (muteIntervalId !== null) clearInterval(muteIntervalId);
    muteOverlay.classList.add("visible");

    function tick() {
      const remainingMs = mutedUntilEpochMs - Date.now();
      if (remainingMs <= 0) {
        stopMuteOverlay();
        return;
      }
      const remainingSeconds = Math.ceil(remainingMs / 1000);
      muteCount.textContent = String(remainingSeconds);
      const fraction = totalSeconds > 0 ? remainingSeconds / totalSeconds : 0;
      muteRingProgress.style.strokeDashoffset = String(MUTE_RING_CIRCUMFERENCE * (1 - fraction));
    }

    tick();
    muteIntervalId = setInterval(tick, 250);
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

  try {
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("run") || "default";

    updateStatus("Fetching session credentials...");
    const res = await fetch(`/api/runs/${runId}/credentials`);
    if (!res.ok) throw new Error(`Failed to load credentials: ${res.status}`);
    const credentials = await res.json();

    if (credentials.protocol_version && credentials.protocol_version !== PROTOCOL_VERSION) {
      // A stale cached page otherwise just never speaks.
      console.warn(
        `[avatar.js] Protocol mismatch: page speaks v${PROTOCOL_VERSION}, ` +
        `backend speaks v${credentials.protocol_version}`
      );
    }

    // Default keeps pre-refactor pages working mid-deploy.
    const moduleUrl = credentials.browser_module || "/providers/liveavatar.js";
    console.log(`[avatar.js] Loading provider module: ${moduleUrl}`);
    const provider = await import(moduleUrl);

    // --- control socket -----------------------------------------------------
    const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProtocol}//${location.host}/ws/control/${encodeURIComponent(runId)}`;
    const controlWs = new WebSocket(wsUrl);

    let currentTurnId = null;

    // Copied explicitly, not spread: an undefined turn_id would clobber the
    // fallback and a null chunk_id would fake a streamed chunk.
    function sendControl(type, extra = {}) {
      if (controlWs.readyState !== WebSocket.OPEN) return;
      const turnId = extra.turn_id ?? currentTurnId;
      if (turnId === null || turnId === undefined) return;
      const message = { type, turn_id: turnId };
      if (extra.chunk_id !== null && extra.chunk_id !== undefined) message.chunk_id = extra.chunk_id;
      if (extra.text !== undefined) message.text = extra.text;
      if (extra.interrupted !== undefined) message.interrupted = extra.interrupted;
      controlWs.send(JSON.stringify(message));
    }

    // Provider reports speech lifecycle; the shell renders it.
    function onSpeakStarted() {
      updateStatus("Avatar Speaking", "active");
    }
    function onSpeakEnded() {
      updateStatus("Listening...", "active");
      if (!debugEnabled) return;
      heardLines.replaceChildren();
      heardOverlay.classList.remove("visible");
    }

    const handle = await provider.connect({
      credentials,
      runId,
      videoEl,
      setStatus: updateStatus,
      sendControl,
      onSpeakStarted,
      onSpeakEnded,
    });

    controlWs.onopen = () => {
      console.log("[WS] Control connection established");
      updateStatus("Listening...", "active");
    };

    controlWs.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.error("[WS] Unparseable message:", err);
        return;
      }

      try {
        switch (data.type) {
          case "heard":
            renderHeard(data);
            break;

          case "avatar_speak":
            currentTurnId = data.turn_id;
            handle.speak({ text: data.text, turnId: data.turn_id, chunkId: data.chunk_id ?? null });
            if (data.text) {
              updateStatus(`Avatar Speaking: "${data.text.slice(0, 30)}..."`, "active");
            }
            break;

          case "avatar_audio":
            handle.audio?.({ data: data.data, turnId: data.turn_id });
            break;

          case "avatar_speak_end":
            handle.speakEnd?.({ turnId: data.turn_id, chunkId: data.chunk_id ?? null });
            break;

          case "avatar_user_message":
            // Delegated mode: the vendor's own brain answers this.
            currentTurnId = data.turn_id;
            handle.userMessage?.({ text: data.text, turnId: data.turn_id });
            break;

          case "avatar_interrupt":
            handle.interrupt?.();
            sendControl("avatar_speak_interrupted");
            updateStatus("Listening...", "active");
            break;

          case "agent_muted":
            if (data.muted_until_epoch_ms) startMuteOverlay(data.muted_until_epoch_ms, data.seconds || 0);
            break;

          case "agent_unmuted":
            stopMuteOverlay();
            break;

          default:
            console.log("[WS] Unhandled message type:", data.type);
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
