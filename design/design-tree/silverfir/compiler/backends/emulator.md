- The emulator backend interprets finalized MachineIR directly against the
  NativeContext, runtime contract, and helper entrypoints, living under `arch/`
  alongside the real ISA backends; it exercises the same machine module a
  native backend would (`arch/emulator`).

- Emu64 and Emu32 are first-class variants of the unified compile-time
  native-backend enum, selected at build time by mutually-exclusive cfgs
  (`sf_backend_emu64` / `sf_backend_emu32`) that replace the host-native
  backend; there is no runtime or process-global mode and no Disabled
  fallback — when no backend is compiled, backend selection returns an error
  rather than defaulting to an emulator (`NativeBackend`).

- Emu32 uses its own 4-byte GP-unit budget preset to model a 32-bit-target
  MachineIR shape, while Emu64 uses an 8-byte GP-unit preset.

- The emulator backend is selected via the backend-emu64 / backend-emu32
  features as a build-time override: its module compiles only when its feature
  is on, and the emu backend is mutually exclusive with a native backend (an
  arm64-backend build does not compile the emulator).

## Facts

- 2026-03-11 (c4102007) rationale: the emulator exists only to validate
  machine-IR semantics above the ISA layer — a debugging oracle, never a
  fallback production engine, mixed-mode executor, or a target native code may
  jump into; a compiled native function must be entirely native with no
  per-function fallback stub handing execution to the emulator, because such a
  fallback destroys the backend boundary, hides correctness bugs, and makes
  performance numbers meaningless (author).

- 2026-03-18 (752b7eb4) rationale: emu32 is a semantic oracle that executes the
  optimized-and-legalized 32-bit-target MachineIR directly so legalization
  correctness (i64 arithmetic, f64 transport, memory access, control flow with
  block params and select) is validated before ARMv7A physical-register
  encoding is involved; passing emu32 proves the legalized MachineIR is
  correct but is explicitly not proof the ARMv7A register mapping is solved
  (author).

- 2026-03-17 (72f21214) rationale: on guard-page configurations the MachineIR
  omits explicit bounds-check TrapIf instructions and relies on a signal
  handler, but the emulator installs no signal handler, so every emulated
  load/store software-checks the access itself — a pointer is treated as a
  wasm linear-memory access when it falls inside the guard-page virtual
  reservation window and trapped if it runs past mem0_size, while frame/stack
  pointers (in a heap Vec outside that window) pass through unchecked (diff).

- 2026-04-08 (60498ab7) pitfall: the wasm-vs-host pointer classification test —
  whether a pointer falls inside the 8 GB GUARD_WINDOW above mem_base — only
  makes sense when guard-page targets reserve that virtual range; gating it on
  `target_pointer_width=64` alone misclassified host frame/stack pointers near
  the committed allocation as wasm-memory addresses on a 64-bit build without
  guard pages, so the check is now gated on `sf_has_guard_pages` and falls back
  to the committed-range check (`ptr < mem_base + mem_size`) when guard pages
  are absent (diff).

- 2026-04-07 (09ea65d5) pitfall: block-edge argument transfer read every arg
  through read_value, which errored outright on a `MachineValue::ReservedReg`
  (a cached-local value), so any edge carrying a reserved cache-edge move could
  not be processed; the fix special-cases reserved args as identity-only (the
  target register already holds the cached-local value, no move happens) and
  asserts they are non-moving, matching the native backends (diff).
