// Stand-in provider so the shell can be tested without any vendor SDK.
export const calls = [];
export const name = "mock";
export const accepts = "text";

export async function connect(ctx) {
  calls.push({ fn: "connect", ctx });
  return {
    speak: (a) => calls.push({ fn: "speak", ...a }),
    audio: (a) => calls.push({ fn: "audio", ...a }),
    speakEnd: (a) => calls.push({ fn: "speakEnd", ...a }),
    userMessage: (a) => calls.push({ fn: "userMessage", ...a }),
    interrupt: () => calls.push({ fn: "interrupt" }),
  };
}
