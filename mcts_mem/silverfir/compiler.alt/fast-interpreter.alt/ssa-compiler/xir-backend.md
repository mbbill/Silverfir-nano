- The executable form (XIR) is a flat array of fixed-layout instructions, each
  a handler function pointer plus three 64-bit immediate fields; LIR physical
  registers map one-to-one onto the eight XIR register slots with no further
  allocation.

- Execution is threaded, not a decode loop: each handler does its work and
  tail-calls the next instruction's handler through a C trampoline using the
  `preserve_none` convention, keeping the abstract register set in CPU
  registers across the whole chain.

- Operations are specialized per (operation, type, register permutation): the
  handler pointer baked into each instruction already encodes which register
  slots are inputs and output, leaving no operand decoding at run time.

- Spill loads, spill stores, and register-to-register copies are explicit XIR
  instructions whose register and slot indices ride in the immediate fields.

- The full set of permutation handlers is generated at build time from a
  declarative specification rather than hand-written; handler bodies are split
  between C and safe Rust on a hot-path-versus-cold-path line, with the hot
  paths (including the common runtime-touching call/return forms and slot
  copies) in C behind a generated FFI shim ([[xir-backend/handler-language]]).

## Facts

- 2025-09-17 (aed2ff42) rationale: compute is register-based, not stack-based —
  every SSA value gets a home in a fixed per-function register file, so there is
  no implicit operand stack during compute; the shared value stack survives only
  for calls, returns, and rare fallbacks. This is the architectural inversion
  from the stack-window fast backend, where the operand stack is the working
  store and only the top few slots are register-resident (code).

- 2026-02-08 (89f91c70) rationale: the per-(operation, type, register
  permutation) handler scheme is bounded by a 2-address constraint baked into
  every instruction signature — the output register must equal the first input
  register. Without it the permutation count explodes (a 2->1 signature has 64
  arrangements, a 3->1 has 512); the 2-address constraint is what keeps the
  generated handler set finite (code).

- 2026-02-08 (da03882b) rationale: keeping the hot handlers in C avoids needing
  cross-language LTO while still getting C-only LTO to perform the guard-check
  (bounds/zero-check) elimination the handlers depend on (code).

- 2026-02-08 (da03882b) rationale: the hot handlers are written in C rather than
  Rust because cross-language LTO (Rust->C inlining) was fragile and at the time
  worked only on Windows with a special toolchain — the same rationale behind the
  fast backend's `handlers_c/` (sourced).
