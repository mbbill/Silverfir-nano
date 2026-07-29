- Exception instances are allocated directly into the shared cross-store
  reference registry as registry-owned, shared-ownership payloads; the handle
  a throw produces is the pooled registry index (`alloc_exn_in`).

- Both engines allocate exceptions through this one path; an exception's
  payload lifetime is independent of the instance that threw it.

## Moves

- 2026-07-28 (fec5adb5) replaced [[per-store-exn-arena]]: exceptions were
  already registry-owned through shared payloads, so the per-store arena was a
  second owner buying nothing — the registry-only path serves both engines and
  removed the arena plus its dead code in single-engine builds (code).
