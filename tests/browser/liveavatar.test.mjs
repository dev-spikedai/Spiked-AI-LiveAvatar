// public/providers/liveavatar.js against stubs. The Python suite cannot see
// this half, and it is where a broken speak/echo contract silences the bot.

import assert from "node:assert/strict";
import test, { before, describe } from "node:test";

import { encode, installDom, installFakeLiveKit } from "./stubs.mjs";

const MODULE = new URL("../../public/providers/liveavatar.js", import.meta.url);

let mod, handle, published, handlers, control, ui;

before(async () => {
  let liveKit;
  installDom({ onScriptSrc: () => { liveKit = installFakeLiveKit(); } });
  mod = await import(MODULE);

  control = [];
  ui = [];
  handle = await mod.connect({
    credentials: { session_id: "s1", livekit_url: "wss://fake", livekit_token: "tok" },
    runId: "r1",
    videoEl: { id: "avatar-video", play: async () => {}, srcObject: null },
    setStatus: () => {},
    sendControl: (type, extra) => control.push({ type, ...extra }),
    onSpeakStarted: () => ui.push("started"),
    onSpeakEnded: () => ui.push("ended"),
  });
  ({ published, handlers } = liveKit);
});

describe("liveavatar provider module", () => {
  test("declares the contract the registry expects", () => {
    assert.equal(mod.name, "liveavatar");
    assert.equal(mod.accepts, "text");
    assert.equal(typeof mod.connect, "function");
  });

  test("stops HeyGen listening on connect", () => {
    // Without this the meeting has a second agent that never passes our gate.
    const stop = published.find((p) => p.body.event_type === "avatar.stop_listening");
    assert.ok(stop, "avatar.stop_listening was never published");
  });

  test("speak() publishes the text on the agent-control topic", () => {
    handle.speak({ text: "Hello there.", turnId: 1, chunkId: "1-1" });
    const speak = published.find((p) => p.body.event_type === "avatar.speak_text");
    assert.ok(speak, "no avatar.speak_text published");
    assert.equal(speak.body.text, "Hello there.");
    assert.equal(speak.topic, "agent-control");
  });

  test("echoes round-trip turn_id and chunk_id", () => {
    // Losing chunk_id makes the ack read as single-shot, which releases the
    // floor while the rest of the answer is still queued.
    const speak = published.find((p) => p.body.event_type === "avatar.speak_text");
    for (const event_type of ["avatar.speak_started", "avatar.speak_ended"]) {
      handlers.DataReceived(
        encode({ event_type, source_event_id: speak.body.event_id }),
        null, null, "agent-response",
      );
    }
    const ended = control.find((c) => c.type === "avatar_speak_ended");
    assert.ok(control.find((c) => c.type === "avatar_speak_started"), "speak_started not reported");
    assert.ok(ended, "speak_ended not reported");
    assert.equal(ended.turn_id, 1);
    assert.equal(ended.chunk_id, "1-1");
    assert.deepEqual(ui, ["started", "ended"]);
  });

  test("ignores traffic on other topics", () => {
    const before = control.length;
    handlers.DataReceived(encode({ event_type: "avatar.speak_ended" }), null, null, "other-topic");
    assert.equal(control.length, before);
  });

  test("interrupt() publishes avatar.interrupt", () => {
    handle.interrupt();
    assert.ok(published.find((p) => p.body.event_type === "avatar.interrupt"));
  });
});
