- The 64-bit backends (arm64, x86_64, RISC-V 64) assemble per-function artifacts
  into a module through a dedicated 64-bit shared module linker that owns the
  64-bit-only module-layout state, while the 32-bit backends use their own
  separate 32-bit linker (`compile_module_64`, `ModuleLinkBackend64`).

## Moves

- 2026-04-09 (f49c5712) replaced [[generic-module-linker]]: the 64-bit
  module-link pass needs 64-bit-only state — the NativeLocalCallInfo64
  function-info table layout and the guard-page signal handler's
  body_local_error_offset capture — so factoring it into a 64-bit-specific shared
  linker lets arm64 and x86_64 share that 64-bit module-layout code instead of
  routing it through the arch-agnostic generic linker (diff).
