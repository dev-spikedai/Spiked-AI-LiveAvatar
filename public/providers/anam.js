// Anam browser half. Driven mode streams our text; native mode hands Anam the
// transcript and lets its own LLM answer. See docs/PROVIDER_REFACTOR_PLAN.md §5.

const ANAM_SDK = "https://esm.sh/@anam-ai/js-sdk";

export const name = "anam";
export const accepts = "text";

export async function connect({
  credentials, videoEl, setStatus, sendControl, onSpeakStarted, onSpeakEnded,
}) {
  const sessionToken = credentials.anam_session_token;
  if (!sessionToken) throw new Error("No Anam session token in credentials");
  const native = credentials.anam_mode === "native";

  setStatus("Loading Anam SDK...");
  const { createClient, AnamEvent } = await import(ANAM_SDK);

  setStatus("Connecting to Anam...");
  const client = createClient(sessionToken, {
    disableInputAudio: credentials.anam_disable_input_audio !== false,
  });

  let talkStream = null;
  let currentTurnId = null;

  client.addListener(AnamEvent.TALK_STREAM_INTERRUPTED, (correlationId) => {
    console.log("[Anam] talk stream interrupted:", correlationId);
    talkStream = null;
    sendControl("avatar_speak_interrupted", { turn_id: currentTurnId });
    onSpeakEnded();
  });

  if (native) {
    // Only channel through which the vendor's words reach us; the backend needs
    // them for echo suppression. Role is "persona", not "assistant".
    client.addListener(AnamEvent.MESSAGE_HISTORY_UPDATED, (messages) => {
      const last = Array.isArray(messages) ? messages[messages.length - 1] : null;
      if (!last || last.role !== "persona" || !last.content) return;
      sendControl("avatar_vendor_reply", { turn_id: currentTurnId, text: last.content });
    });
  }

  // One element carries both video and audio.
  await client.streamToVideoElement(videoEl.id);
  setStatus("Anam Connected", "active");

  return {
    speak({ text, turnId, chunkId }) {
      if (!text || native) return;
      if (turnId !== currentTurnId) {
        try { talkStream?.endMessage(); } catch { /* already closed */ }
        talkStream = null;
        currentTurnId = turnId;
      }
      talkStream = client.createTalkMessageStream(String(chunkId ?? turnId));
      onSpeakStarted();
      sendControl("avatar_speak_started", { turn_id: turnId, chunk_id: chunkId });
      try {
        talkStream.streamMessageChunk(text, false);
      } catch (err) {
        // Documented signal that the stream was interrupted, not a failure.
        console.warn("[Anam] chunk rejected (stream inactive):", err);
        talkStream = null;
        return;
      }
      // Anam has no per-chunk "finished playing" event, so a chunk is acked on
      // acceptance; pacing is Anam's from here.
      talkStream.endMessage();
      talkStream = null;
      sendControl("avatar_speak_ended", { turn_id: turnId, chunk_id: chunkId });
      onSpeakEnded();
    },

    userMessage({ text, turnId }) {
      if (!native) return;
      currentTurnId = turnId;
      onSpeakStarted();
      client.sendUserMessage(text);
    },

    interrupt() {
      // No documented interrupt API; ending the stream is the only lever.
      try { talkStream?.endMessage(); } catch { /* already closed */ }
      talkStream = null;
      onSpeakEnded();
    },
  };
}
