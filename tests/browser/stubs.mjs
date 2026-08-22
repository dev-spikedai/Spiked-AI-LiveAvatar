// Minimal DOM/network stubs so the real browser modules can run under Node.
// Deliberately thin: anything these fake is something the test is not covering.

export function fakeElement(id = "avatar-video") {
  return {
    id, className: "", style: {}, textContent: "", srcObject: null,
    childElementCount: 0, firstChild: null,
    classList: { add() {}, remove() {} },
    append() {}, appendChild() {}, replaceChildren() {}, removeChild() {},
    play: async () => {}, addEventListener() {},
  };
}

export function installDom({ onScriptSrc } = {}) {
  globalThis.document = {
    getElementById: () => fakeElement(),
    createElement(tag) {
      const node = fakeElement();
      if (tag === "script" && onScriptSrc) {
        Object.defineProperty(node, "src", {
          set() { onScriptSrc(); setTimeout(() => node.onload && node.onload(), 0); },
        });
      }
      return node;
    },
    head: { appendChild() {} },
    body: { appendChild() {} },
    addEventListener() {},
  };
  globalThis.location = { protocol: "http:", host: "127.0.0.1:8765", search: "?run=r1&debug=1" };
  globalThis.window = { LivekitClient: null, location: globalThis.location };
}

/** LiveKit stand-in. Returns the handler map so tests can fire room events. */
export function installFakeLiveKit() {
  const handlers = {};
  const published = [];
  globalThis.window.LivekitClient = {
    RoomEvent: {
      TrackSubscribed: "TrackSubscribed",
      DataReceived: "DataReceived",
      Disconnected: "Disconnected",
    },
    Room: class {
      on(evt, fn) { handlers[evt] = fn; }
      async connect() {}
      async startAudio() {}
      localParticipant = {
        publishData(bytes, opts) {
          published.push({ topic: opts.topic, body: JSON.parse(new TextDecoder().decode(bytes)) });
        },
      };
    },
  };
  return { handlers, published };
}

export function installFakeWebSocket() {
  const state = { socket: null };
  globalThis.WebSocket = class {
    constructor(url) { this.url = url; this.readyState = 1; this.sent = []; state.socket = this; }
    send(data) { this.sent.push(JSON.parse(data)); }
  };
  globalThis.WebSocket.OPEN = 1;
  return state;
}

export const encode = (obj) => new TextEncoder().encode(JSON.stringify(obj));
