// Simli — browser half. Ported from the `simli` branch.
//
// Transport: Simli's p2p WebRTC endpoint. The same socket carries signalling
// (JSON offer/answer, then bare string state words) and the audio feed (binary
// PCM16@16kHz frames). No SDK; plain RTCPeerConnection.
//
// This provider only lip-syncs. It has no voice of its own, so the backend
// pairs it with a TtsProvider and sends PCM rather than text — the
// `accepts: "audio"` half of the contract.
//
//   connect(ctx) -> { speak, audio, speakEnd, interrupt }

export const name = "simli";
export const accepts = "audio";

export async function connect({
  credentials, videoEl, setStatus, sendControl, onSpeakStarted, onSpeakEnded,
}) {
  const sessionToken = credentials.simli_session_token;
  if (!sessionToken) throw new Error("No Simli session token in credentials");

  const base = (credentials.simli_base_url || "https://api.simli.ai").trim().replace(/\/+$/, "");
  const wsUrl = `${base.replace(/^http/, "ws")}/compose/webrtc/p2p?session_token=${encodeURIComponent(sessionToken)}`;

  // Simli publishes audio on its own track rather than muxing it into the
  // video element, so the page needs a second sink for it.
  let audioEl = document.getElementById("avatar-audio");
  if (!audioEl) {
    audioEl = document.createElement("audio");
    audioEl.id = "avatar-audio";
    audioEl.autoplay = true;
    document.body.appendChild(audioEl);
  }

  setStatus("Connecting to Simli WebRTC...");

  const simliWs = new WebSocket(wsUrl);
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

  const ready = new Promise((resolve, reject) => {
    simliWs.addEventListener("open", async () => {
      try {
        pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
        pc.addTransceiver("audio", { direction: "recvonly" });
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.ontrack = (evt) => {
          const stream = evt.streams?.[0] || new MediaStream([evt.track]);
          if (evt.track.kind === "video") {
            videoEl.srcObject = stream;
            videoEl.play().catch(console.warn);
            setStatus("Avatar Video Rendering", "active");
          } else if (evt.track.kind === "audio") {
            audioEl.srcObject = stream;
            // Autoplay is routinely blocked until the tab sees a gesture. In a
            // Recall browser there is no user to click, hence the retry loop as
            // well as the listeners.
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
        resolve();
      } catch (err) {
        console.error("[Simli] offer setup failed:", err);
        setStatus(`Simli offer error: ${err.message}`, "error");
        reject(err);
      }
    });
    simliWs.addEventListener("error", () => {
      console.error("[Simli] WebSocket error");
      setStatus("Simli connection error", "error");
      reject(new Error("Simli WebSocket error"));
    });
  });

  simliWs.addEventListener("message", (evt) => {
    const data = evt.data;
    if (typeof data !== "string") return;
    if (data === "START") {
      simliStarted = true;
      primeSimliAudio();
      setStatus("Listening...", "active");
    } else if (data === "SPEAK") {
      setStatus("Avatar Speaking", "active");
    } else if (data === "SILENT") {
      setStatus("Listening...", "active");
    } else if (data === "STOP") {
      setStatus("Session ended", "pending");
    } else if (data === "ACK") {
      // Server acknowledges an audio segment; nothing to do.
    } else if (data.startsWith("ERROR") || data.startsWith("RATE") || data.startsWith("CLOSING")) {
      console.error("[Simli] server message:", data);
      setStatus(data, "error");
    } else {
      try {
        const msg = JSON.parse(data);
        if (msg.type === "answer") pc?.setRemoteDescription(msg).catch(console.warn);
      } catch {
        /* ignore non-JSON messages */
      }
    }
  });

  simliWs.addEventListener("close", () => {
    console.log("[Simli] WebSocket closed");
    setStatus("Simli connection closed", "pending");
  });

  await ready;

  function sendAudio(pcm) {
    if (simliWs.readyState !== WebSocket.OPEN) return;
    // Frames that arrive before START are held, not dropped: the backend starts
    // synthesizing as soon as it has an answer and does not wait for the
    // WebRTC handshake.
    if (!simliStarted) {
      preStartAudioQueue.push(pcm);
      return;
    }
    simliWs.send(pcm);
  }

  return {
    // Text mode's "an utterance begins". Simli speaks nothing from this; the
    // audio frames that follow are the actual speech.
    speak({ turnId }) {
      sendControl("avatar_speak_started", { turn_id: turnId });
      onSpeakStarted();
    },
    audio({ data }) {
      if (!data) return;
      const bin = atob(data);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      sendAudio(bytes.buffer);
    },
    speakEnd({ turnId }) {
      sendControl("avatar_speak_ended", { turn_id: turnId });
      onSpeakEnded();
    },
    interrupt() {
      // SKIP discards whatever Simli still has buffered; without it the avatar
      // keeps talking through a barge-in for as long as the queue is deep.
      if (simliWs.readyState === WebSocket.OPEN) simliWs.send("SKIP");
      preStartAudioQueue.length = 0;
    },
  };
}
