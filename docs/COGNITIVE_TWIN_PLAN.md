# Cognitive Digital Twin Runtime Plan

This document defines the intended behavior of Tom as a cognitive digital twin
in a live meeting. The avatar is currently kept in FULL mode; the orchestrator,
not the vendor's built-in brain, remains responsible for deciding what Tom
should say.

## Core behavior

Tom is not merely a wake-word chatbot. The wake name is one strong signal that
Tom owns the next turn, but the conversation controller also evaluates meeting
context, participant roles, client/user knowledge, Tom's configured role, and
the user's speaking preferences.

For example, "Tom, did you catch that?" means: reconstruct the immediately
preceding discussion, determine what the speaker is asking Tom to confirm or
answer, and respond with the most useful grounded answer. It is not classified
as a generic acknowledgement by default.

Likewise, if the user has explicitly taught Tom, "always speak up when you can
help," then a client statement such as "we are worried about implementation
cost" becomes a candidate for an immediate, role-appropriate interjection. In
the absence of that preference, Tom may prepare an insight silently or wait for
an explicit invitation.

## Runtime layers

```text
Recall participant audio
        |
        v
Deepgram live interim/final transcript streams
        |
        v
Conversation controller
  - live turn assembly
  - wake/name candidate detection
  - speaker and meeting context
  - role and preference policy
  - speak / prepare / suggest / stay silent decision
        |
        +--> warm per-call knowledge index (company + client context)
        +--> ephemeral meeting memory
        +--> persistent user/client memory
        +--> controlled MCP tool router
        |
        v
Gemini structured reasoning and grounded response generation
        |
        v
FULL-mode LiveAvatar control socket
        |
        v
Recall headless browser output inside the meeting
```

Recall's transcript/webhook path remains useful for analytics, notes, and
reconciliation. The real-time controller uses participant-separated audio and
Deepgram so it can react to interim speech, detect barge-in, and apply the same
speaker-aware policy consistently.

## Conversation states

The runtime should expose these states:

- `LISTENING`: collecting conversation and updating context.
- `CANDIDATE`: a live turn may involve Tom or may be useful to Tom's role.
- `THINKING`: the controller owns the turn and is retrieving/reasoning.
- `SPEAKING`: FULL-mode avatar playback is active.
- `INTERRUPTING`: participant speech is stopping current playback.
- `MUTED`: Tom continues learning but does not take the floor unless explicitly
  overridden.

The controller separates two decisions:

1. Is this relevant to Tom?
2. Should Tom speak now?

An addressed turn normally answers the second question with yes, subject to
conversation safety and confidence. A nameless turn can result in `prepare`,
`insight`, `autonomous_speak`, or `silent` depending on role, preferences,
confidence, interruption cost, and recent floor ownership.

## Memory model

### Working memory

The recent finalized turns, current incomplete turn, unresolved questions, and
active participants. This is optimized for fast routing and answer generation.

### Ephemeral meeting memory

Facts learned or established during the current call, including explicit user
instructions, decisions, concerns, terminology, and commitments. Every item
stores its source speaker, timestamp, confidence, and expiration behavior.

### Persistent memory

Only deliberate, useful facts survive the meeting: user preferences, client
systems, recurring objections, approved terminology, decision-makers, and past
commitments. Persistent memory is never created by blindly saving transcripts;
it is extracted, scoped, confidence-rated, and auditable.

An instruction such as "always speak up when you can help" is stored as a
preference with a scope and confidence, not as an unstructured prompt sentence.

## Warm knowledge retrieval

At call start, the service loads the user's/company's permitted knowledge into a
per-call in-memory index. The index is keyed by user/client/knowledge-version,
has an explicit memory budget and TTL, and is warmed before the first expected
question whenever possible.

Retrieval during the call should be local and fast. Supabase remains the source
of truth and is used for cache misses, refreshes, and version changes. A warm
index must never cross client or authorization boundaries.

Interim transcript signals may speculatively start retrieval. Final routing can
cancel or reuse that work once the turn is complete.

## Role templates and tools

Roles such as solution architect are structured runtime profiles. A role
controls the dimensions Tom should cover, tradeoffs to surface, tone, and
forbidden claims; it does not replace company grounding.

MCP tools are selected through an explicit policy layer. Read-only knowledge and
client-context tools may be used automatically when confidence is sufficient.
External side effects require an authorization path and are never inferred from
ordinary conversation alone.

## Delivery and latency targets

Every turn records timestamps for:

`audio -> interim -> wake candidate -> final -> route -> retrieval -> first
answer token -> avatar dispatch -> avatar speaking`.

The target is to overlap work rather than serialize it:

```text
interim transcript -> candidate detection + speculative retrieval
final transcript   -> route using already-warm context
first safe answer sentence -> FULL-mode avatar
remaining answer sentences -> stream sequentially
```

FULL mode's provider delay is accepted initially. Short, truthful thinking
fillers may mask retrieval latency when the answer is sufficiently likely, but
must never overlap the grounded answer or conceal uncertainty.

## Implementation goals

1. Establish the conversation-state and memory contracts.
2. Instrument every latency stage with a turn correlation ID.
3. Assemble live interim/final turns and detect tolerant Tom candidates.
4. Route addressed turns using the full meeting context, not wake-word type.
5. Add structured role templates and active meeting-state summaries.
6. Warm and serve a scoped per-call in-memory knowledge index.
7. Add proactive interjection scoring with user preference controls.
8. Add scoped persistent memory and controlled MCP tool routing.
9. Optimize FULL-mode progressive speech and latency masking.
10. Verify failure handling, teardown, multi-participant behavior, and rollout.
