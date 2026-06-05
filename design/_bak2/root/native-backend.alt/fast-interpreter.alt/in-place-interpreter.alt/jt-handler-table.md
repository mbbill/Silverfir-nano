---
status: explored, rejected same-day
---

- A rival in-place interpreter dispatching through a 256-entry table of
  handler function pointers (one `fn(&mut Ctx) -> Step` per opcode byte)
  instead of a match statement; otherwise identical (same jump table, same
  stack).
- Runtime-selectable against the match interpreter via env var for A/B
  measurement.
