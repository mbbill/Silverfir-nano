- A single generic compile_module<A: ArchBackend> in common/pipeline.rs
  assembles a 64-bit backend's per-function artifacts into a module: it lays out
  each function's text with page alignment, builds the function-info table,
  resolves internal-entry addresses, and projects the backend's CompiledEntry
  into the uniform CompiledArchEntry shape.

- Both arm64 and x86_64 dispatch through this one arch-agnostic module linker
  (armv7a already uses its own separate 32-bit compile_module).

## Moves

- 2026-04-09 (f49c5712) replaced by [[module-linker]]: the 64-bit module-link
  pass needs 64-bit-only state — the NativeLocalCallInfo64 function-info table
  layout and the guard-page signal handler's body_local_error_offset capture — so
  factoring it into a 64-bit-specific shared linker lets arm64 and x86_64 share
  that 64-bit module-layout code instead of routing it through the arch-agnostic
  generic linker (diff).
