- At a direct local call the caller zeroes the entire callee frame prefix beyond
  the argument span before transferring control; the callee always sees a
  fully zero-initialized local-prefix window.

- The C entry path and the emulator pre-zero the callee frame prefix beyond the
  passed arguments; zero-init is unconditional and not driven by per-local
  liveness.

## Moves

- 2026-04-06 (94946b38) replaced by [[zero-init]]: the caller blindly zeroed the
  callee's whole local prefix at every call site (and the C/emulator entry
  pre-zeroed it), initializing locals the callee provably writes before reading;
  moving zero-init into the callee at function entry and gating it on a
  read-before-write liveness analysis elides every store for a provably-written
  local (code).
