- The emulator/reference backend is compiled into the core engine on every
  target and build profile (gated only by the no-op `cfg_has_reference!` macro);
  --emu64 / --emu32 always work, including in release builds.

- On a target with no native backend, `active_native_backend()` always falls
  back to `NativeBackend::Reference` rather than erroring.

## Moves

- 2026-03-16 (25c9daac) replaced [[host-availability-selection]]: with three
  real ISA backends the NativeBackend enum, its config/dispatch arms, and the
  per-arch NativeCode entry slots are each cfg-gated to the host target_arch so
  a build compiles only its own backend; the reference/emulator backend is
  retained on debug builds and on hosts with no native backend (e.g. arm32) to
  keep --emu working (diff).

- 2026-04-08 (76e9c612) replaced by [[runtime-reference-backend-overlay]]: compiling the
  reference emulator unconditionally bloated every build, including the minimal
  no_std target that has a real native backend and never needs it; making it an
  opt-in feature keeps the emulator available where it is wanted (spectest, the
  WASI CLI, --emu64/--emu32) while letting production builds drop it entirely
  (diff).
