// Anam — browser half. Handles both configurations the backend can hand it.
//
//   driven  (llmId = CUSTOMER_CLIENT_V1): Anam's brain is off. Each avatar_speak
//           becomes a talk-message stream; one stream per turn, chunks appended
//           as sentences arrive. This is the drop-in for LiveAvatar FULL.
//
//   native  (a real llmId): Anam composes the reply itself. The backend sends
//           avatar_user_message with the gated transcript, and we report back
//           what Anam actually said — the core needs that text for the echo
//           suppressor, which is the only reason this path reports at all.
//
// Input audio is disabled in both. The page never listens; Recall does.
//
//   connect(ctx) -> { speak, interrupt, userMessage }

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

  let audioEl = document.getElementById("avatar-audio");
  if (!audioEl) {
    audioEl = document.createElement("audio");
    audioEl.id = "avatar-audio";
    audioEl.autoplay = true;
    document.body.appendChild(audioEl);
  }

  setStatus("Connecting to Anam...");
  const client = createClient(sessionToken, {
    // Anam would otherwise open a mic and answer the room directly, giving the
    // meeting a second agent that never passes the turn gate.
    disableInputAudio: credentials.anam_disable_input_audio !== false,
  });

  // One turn = one talk stream. Held here so chunks of the same answer append
  // to the same stream rather than racing as separate utterances.
  let talkStream = null;
  let currentTurnId = null;
  let spokenSoFar = "";

  client.addListener(AnamEvent.TALK_STREAM_INTERRUPTED, (correlationId) => {
    // Anam's correlationId is our chunk/turn identity — that mapping is what
    // lets a vendor-side barge-in land on the right turn.
    console.log("[Anam] talk stream interrupted:", correlationId);
    talkStream = null;
    sendControl("avatar_speak_interrupted", { turn_id: currentTurnId });
    onSpeakEnded();
  });

  if (native) {
    // The only channel through which the vendor's own words reach us. If this
    // event is late or missing, echo suppression is blind for that turn and the
    // agent can hear itself come back through Deepgram.
    client.addListener(AnamEvent.MESSAGE_HISTORY_UPDATED, (messages) => {
      const last = Array.isArray(messages) ? messages[messages.length - 1] : null;
      if (!last || last.role !== "assistant" || !last.content) return;
      spokenSoFar = last.content;
      sendControl("avatar_vendor_reply", { turn_id: currentTurnId, text: spokenSoFar });
    });
  }

  await client.streamToVideoAndAudioElements(videoEl.id, audioEl.id);
  setStatus("Anam Connected", "active");

  return {
    speak({ text, turnId, chunkId }) {
      if (!text) return;
      if (native) {
        // Nothing to do: in native mode the backend does not send us words to
        // say. Reaching here means a delegated run was wired to a streaming
        // engine, which the registry is supposed to have refused.
        console.warn("[Anam] avatar_speak received in native mode; ignoring");
        return;
      }
      if (turnId !== currentTurnId) {
        // New turn: close the previous stream rather than appending this
        // answer onto the last one.
        try { talkStream?.endMessage(); } catch { /* already closed */ }
        talkStream = null;
        currentTurnId = turnId;
      }
      if (!talkStream || !talkStream.isActive()) {
        talkStream = client.createTalkMessageStream(String(chunkId ?? turnId));
        onSpeakStarted();
        sendControl("avatar_speak_started", { turn_id: turnId, chunk_id: chunkId });
      }
      try {
        talkStream.streamMessageChunk(text, false);
      } catch (err) {
        // streamMessageChunk throws on an interrupted stream — that is the
        // documented signal, not an unexpected failure.
        console.warn("[Anam] chunk rejected (stream inactive):", err);
        talkStream = null;
        return;
      }
      // The backend waits for this before releasing the next sentence. Anam
      // has no per-chunk "finished playing" event, so the chunk is
      // acknowledged once it is accepted into the stream; pacing is Anam's
      // job from here.
      talkStream.endMessage();
      talkStream = null;
      sendControl("avatar_speak_ended", { turn_id: turnId, chunk_id: chunkId });
      onSpeakEnded();
    },

    userMessage({ text, turnId }) {
      if (!native) {
        console.warn("[Anam] avatar_user_message received in driven mode; ignoring");
        return;
      }
      currentTurnId = turnId;
      spokenSoFar = "";
      onSpeakStarted();
      // Anam receives this as if the user had said it, and its persona answers.
      client.sendUserMessage(text);
    },

    interrupt() {
      try { talkStream?.endMessage(); } catch { /* already closed */ }
      talkStream = null;
      client.interruptPersona?.();
      onSpeakEnded();
    },
  };
}
