- Exception instances live in a separate per-store arena, indexed by a `Copy`
  index handle (`ExnRef`), with the exception's field payload stored out of
  line from the registry entry.

- Allocating a throwable exception is two steps with two owners: the arena
  retains the shared payload, and registering the reference retains the same
  payload again in the cross-store reference registry.

## Moves

- 2026-07-28 (fec5adb5) replaced by [[exception-storage]]: exceptions were
  already registry-owned through shared payloads, so the per-store arena was a
  second owner buying nothing — the registry-only path serves both engines and
  removed the arena plus its dead code in single-engine builds (code).
