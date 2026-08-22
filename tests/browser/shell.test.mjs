// public/avatar.js: the provider-agnostic shell. Asserts each backend message
// reaches the right provider method with its identity intact.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { before, describe } from "node:test";

import { installDom, installFakeWebSocket } from "./stubs.mjs";

const SHELL = new URL("../../public/avatar.js", import.meta.url);
const MOCK = new URL("./fixtures/mock_provider.mjs", import.meta.url).href;

let calls, ws;

const push = (msg) => ws.socket.onmessage({ data: JSON.stringify(msg) });

before(async () => {
  installDom();
  ws = installFakeWebSocket();
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ protocol_version: 1, browser_module: MOCK, provider: "mock" }),
  });

  // The shell is an IIFE, not a module, so it is evaluated rather than imported.
  await import("data:text/javascript," + encodeURIComponent(readFileSync(SHELL, "utf8")));
  await new Promise((r) => setTimeout(r, 50));
  ({ calls } = await import(MOCK));
});

describe("avatar.js shell", () => {
  test("loads the provider module the backend named", () => {
    const connect = calls.find((c) => c.fn === "connect");
    assert.ok(connect, "provider module was never connected");
    for (const key of ["credentials", "runId", "videoEl", "setStatus", "sendControl",
                       "onSpeakStarted", "onSpeakEnded"]) {
      assert.ok(key in connect.ctx, `connect ctx missing ${key}`);
    }
  });

  test("opens the control socket for the run in the URL", () => {
    assert.ok(ws.socket, "control socket never opened");
    assert.match(ws.socket.url, /\/ws\/control\/r1$/);
  });

  test("routes avatar_speak with turn and chunk identity", () => {
    push({ type: "avatar_speak", text: "Hi.", turn_id: 7, chunk_id: "7-1" });
    const speak = calls.find((c) => c.fn === "speak");
    assert.deepEqual(
      { text: speak.text, turnId: speak.turnId, chunkId: speak.chunkId },
      { text: "Hi.", turnId: 7, chunkId: "7-1" },
    );
  });

  test("routes the audio path, preserving chunk_id on the end marker", () => {
    push({ type: "avatar_audio", data: "QQ==", turn_id: 7 });
    push({ type: "avatar_speak_end", turn_id: 7, chunk_id: "7-1" });
    assert.equal(calls.find((c) => c.fn === "audio").data, "QQ==");
    assert.equal(calls.find((c) => c.fn === "speakEnd").chunkId, "7-1");
  });

  test("routes delegated turns to userMessage", () => {
    push({ type: "avatar_user_message", text: "q?", turn_id: 8 });
    assert.equal(calls.find((c) => c.fn === "userMessage").text, "q?");
  });

  test("interrupt reaches the provider and is reported back", () => {
    push({ type: "avatar_interrupt" });
    assert.ok(calls.find((c) => c.fn === "interrupt"));
    const reported = ws.socket.sent.find((m) => m.type === "avatar_speak_interrupted");
    assert.ok(reported, "backend was never told the turn was cut");
    assert.equal(reported.turn_id, 8, "interrupt must land on the current turn");
  });

  test("overlay messages do not reach the provider", () => {
    const before = calls.length;
    push({ type: "heard", speaker: "Ann", text: "hey", reply: true, reason: "addressed" });
    push({ type: "agent_muted", muted_until_epoch_ms: Date.now() + 5000, seconds: 5 });
    push({ type: "agent_unmuted", reason: "expired" });
    assert.equal(calls.length, before);
  });

  test("an unknown message type is ignored, not thrown", () => {
    assert.doesNotThrow(() => push({ type: "something_new" }));
  });
});
