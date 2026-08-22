// LiveAvatar (HeyGen) — browser half.
//
// Transport: a LiveKit room. Commands go out as data-channel messages on the
// "agent-control" topic; HeyGen echoes speak start/end back on the same channel
// with the event_id we sent, which is the only identity the backend gets to
// correlate a finished chunk with the sentence it dispatched.
//
// Implements the provider contract consumed by avatar.js:
//   connect(ctx) -> { speak, interrupt }
// No audio()/speakEnd(): this provider accepts text and does its own TTS.

const LIVEKIT_CDN = "https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js";

export const name = "liveavatar";
export const accepts = "text";

export async function connect({
  credentials, runId, videoEl, setStatus, sendControl, loadScript,
  onSpeakStarted, onSpeakEnded,
}) {
  await loadScript(LIVEKIT_CDN);
  const LivekitClient = window.LivekitClient;
  if (!LivekitClient) throw new Error("LiveKit SDK failed to load");

  let sessionId = credentials.session_id;
  const livekitUrl = credentials.livekit_url;
  const livekitToken = credentials.livekit_token;
  if (!livekitUrl || !livekitToken) throw new Error("No LiveKit credentials in /credentials");

  setStatus("Connecting to LiveKit WebRTC...");

  const room = new LivekitClient.Room({ adaptiveStream: true, dynacast: true });

  room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
    console.log(`[LiveKit] Track subscribed: ${track.kind}`);
    if (track.kind === "video") {
      track.attach(videoEl);
      videoEl.play().catch(console.warn);
      setStatus("Avatar Video Rendering", "active");
    }
    if (track.kind === "audio") {
      const audioEl = track.attach();
      audioEl.autoplay = true;
      document.body.appendChild(audioEl);
      console.log("[LiveKit] Avatar audio track attached to DOM");
    }
  });

  // Sentence-by-sentence dispatch: the backend sends one avatar_speak per
  // sentence, each carrying a chunk_id, and waits for that exact chunk's
  // speak_ended before sending the next — so audio never overlaps or races.
  // speakEventTurns maps the event_id HeyGen echoes back to {turnId, chunkId}
  // so the round trip carries enough identity for the backend to know which
  // chunk just finished.
  const speakEventTurns = new Map();

  const decoder = new TextDecoder();
  room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant, kind, topic) => {
    try {
      if (topic && topic !== "agent-response") return;
      const decoded = JSON.parse(decoder.decode(payload));
      const eventType = decoded.event_type || decoded.type;
      // Fall back to the current turn with no chunk: HeyGen does not always
      // echo a source_event_id, and dropping those events would strand the
      // dispatch loop waiting on a chunk that already finished playing.
      const eventInfo =
        speakEventTurns.get(decoded.source_event_id) ?? { turnId: undefined, chunkId: null };
      console.log(`[LiveKit DataReceived] topic=${topic}:`, decoded);

      if (eventType === "avatar.speak_started") {
        sendControl("avatar_speak_started", { turn_id: eventInfo.turnId, chunk_id: eventInfo.chunkId });
        onSpeakStarted();
      } else if (eventType === "avatar.speak_ended") {
        sendControl("avatar_speak_ended", { turn_id: eventInfo.turnId, chunk_id: eventInfo.chunkId });
        if (decoded.source_event_id) speakEventTurns.delete(decoded.source_event_id);
        onSpeakEnded();
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
    setStatus("Disconnected — reconnecting...", "error");
    // The LiveAvatar session itself may have closed server-side (idle timeout,
    // crash, etc.), so re-fetch fresh credentials rather than re-dialing stale
    // ones — a retry loop against a dead session never succeeds.
    for (let attempt = 1; attempt <= 5; attempt++) {
      await new Promise((r) => setTimeout(r, Math.min(2000 * attempt, 10000)));
      try {
        const res = await fetch(`/api/runs/${runId}/credentials`);
        if (!res.ok) throw new Error(`credentials ${res.status}`);
        const data = await res.json();
        sessionId = data.session_id;
        await room.connect(data.livekit_url, data.livekit_token);
        console.log("[LiveKit] Reconnected to room successfully");
        setStatus("Live Avatar Connected", "active");
        reconnecting = false;
        return;
      } catch (err) {
        console.warn(`[LiveKit] Reconnect attempt ${attempt} failed:`, err);
      }
    }
    setStatus("Connection lost — refresh to rejoin", "error");
    reconnecting = false;
  });

  await room.connect(livekitUrl, livekitToken);
  console.log("[LiveKit] Connected to room successfully");
  setStatus("Live Avatar Connected", "active");

  room.startAudio().catch(() => {
    document.addEventListener("click", () => room.startAudio(), { once: true });
  });

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

  // Output-only. Recall and the backend own listening and turn-taking; letting
  // HeyGen listen would give the meeting a second, ungated agent.
  sendLiveAvatarCommand("avatar.stop_listening");

  return {
    speak({ text, turnId, chunkId }) {
      if (!text) return;
      const eventId = sendLiveAvatarCommand("avatar.speak_text", { text });
      speakEventTurns.set(eventId, { turnId, chunkId });
    },
    interrupt() {
      sendLiveAvatarCommand("avatar.interrupt");
      speakEventTurns.clear();
    },
  };
}
