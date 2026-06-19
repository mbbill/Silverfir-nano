- Guard-page memory backing is a build-derived capability (`sf_has_guard_pages`):
  build.rs enables it only on targets that can support the reservation, and every
  other target falls back to plain Vec-backed memory with explicit bounds checks
  (`emit_guard_pages_cfg`).

## Facts

- 2026-03-18 (dc2d5d66) statement: guard pages require a virtual address space
  large enough for the 8 GB + 64 KB reservation, so the build script only enables
  has_guard_pages on 64-bit pointer-width macos/linux targets running x86_64 or
  aarch64; on every other target the cfg is absent and instantiation falls back to
  plain Vec-backed memory with explicit bounds checks, so the same source compiles
  on 32-bit and bare-metal targets without the guard mechanism (code).

## Moves

- 2026-03-18 (dc2d5d66) replaced [[hand-set-guard-page-feature]]: a hand-set
  feature could be enabled on a target that cannot support guard pages (32-bit
  address spaces, non-POSIX), and it dragged in an unrelated micro-jit dependency;
  deriving has_guard_pages in build.rs from target arch/os/pointer-width (64-bit
  macos/linux on x86_64/aarch64 only) refuses unsupported targets and decouples
  the capability from that dependency (code).
