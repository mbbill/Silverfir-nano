- Guard-page memory backing and the signal-handler trap path are gated behind a
  default-on Cargo `guard-pages` feature that also pulls in the `micro-jit`
  feature, and the feature can be enabled on any target regardless of whether it
  can support guard pages.

## Moves

- 2026-03-18 (dc2d5d66) replaced by [[guard-page-availability]]: a hand-set
  feature could be enabled on a target that cannot support guard pages (32-bit
  address spaces, non-POSIX), and it dragged in an unrelated micro-jit dependency;
  deriving has_guard_pages in build.rs from target arch/os/pointer-width (64-bit
  macos/linux on x86_64/aarch64 only) refuses unsupported targets and decouples
  the capability from that dependency (code).
