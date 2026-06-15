- The NativeBackend enum carries Arm64 and (debug-only) Reference;
  `host_native_backend` returns Arm64 on aarch64 and otherwise None, with
  Reference selected as the debug fallback.

- arch/mod.rs unconditionally declares `pub mod arm64;` and dispatches
  config/as_str over Arm64 plus the debug-gated Reference arm.

## Moves

- 2026-03-16 (25c9daac) replaced by [[emulator-always-compiled]]: with three
  real ISA backends the NativeBackend enum, its config/dispatch arms, and the
  per-arch NativeCode entry slots are each cfg-gated to the host target_arch so
  a build compiles only its own backend; the reference/emulator backend is
  retained on debug builds and on hosts with no native backend (e.g. arm32) to
  keep --emu working (diff).
