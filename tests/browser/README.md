# Browser tests

The Python suite stops at the control socket. These cover the other half —
`public/avatar.js` and `public/providers/*.js` — by executing the real files
under Node against DOM stubs.

```bash
node --test tests/browser/*.test.mjs
```

No dependencies and no browser: `node:test` and hand-written stubs only. A
directory argument does not work (`node --test tests/browser` resolves it as a
module); pass the files or the glob.

## What they catch

Contract breaks between the backend and a provider module — the class of bug
that produces an avatar which connects, looks healthy, and never speaks. Both
real regressions found during the refactor were of exactly this kind: a
`chunk_id` dropped from `avatar_speak_end`, and Anam's message role being
`persona` rather than `assistant`.

## What they do not catch

Anything past the vendor boundary: real LiveKit or WebRTC handshakes, HeyGen's
actual echo timing, autoplay policy in Recall's headless Chrome. Those need a
live meeting.

`anam.js` and `simli.js` have no tests here yet — neither has run against its
vendor, so a passing stub test would imply more confidence than exists.
