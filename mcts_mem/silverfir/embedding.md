- Everything that varies by target — the ISA backend, the OS abstraction
  (hosted vs bare-metal `sf_os_none`), guard-page availability, and SIMD
  availability — is resolved from the target triple and feature set at build
  time by the core crate's `build.rs` (emitted as `sf_backend_*`, `sf_os_*`,
  `sf_has_guard_pages`, `sf_has_simd` cfgs), never detected at runtime.

- A bare-metal embedding receives only an executable-memory and OS-shim
  contract from the embedder; the engine itself stays `no_std` and pulls in no
  runtime dependency beyond `alloc`.

- The product ships binary crates over the core engine rather than folding a
  front end into it, keeping the embeddable core and the host tooling in
  distinct crates.

- What a build contains is chosen by feature configuration on those crates,
  not by maintaining a separate stripped-down front end: the size floor is a
  feature selection over the same CLI.

- SIMD is a build-time feature (`sf_has_simd`): when it is off, a v128 value type
  is rejected at each parse site where it can appear (signature, field, local,
  block type) and FD-prefix opcodes are rejected at decode, cleanly refusing
  SIMD-using modules on a non-SIMD build rather than mis-executing them
  (`ensure_enabled_in_build`).

## Facts

- 2026-02-14 (414d9557) statement: a real no_std/no_main embedder
  (sf-nano-cli-minimal) runs the core with only a GlobalAlloc (libc malloc/free)
  and a panic handler, reading the .wasm via raw libc syscalls — the first
  independent consumer proving the zero-runtime-dependency no_std API is
  embeddable standalone (code).

- 2026-04-07 (91d9a53c) rationale: build.rs is the single authority mapping
  user-facing Cargo features onto internal sf_* cfgs — source uses only sf_*
  cfgs, every sf_* cfg is declared via rustc-check-cfg so a typo'd cfg is a
  compile error, and invalid feature/target combinations are rejected in build.rs
  rather than miscompiling silently (code).

- 2026-04-07 (c1ada379) rationale: there is no user-facing `std` feature — libstd
  availability (sf_has_std) is derived in build.rs from whichever std-requiring
  feature the user selected (wasi, call-trace, or guard-pages), so the embedder
  cannot ask for std in isolation and a std-only build is unreachable (code).

- 2026-04-19 (dea892ec) statement: sf_has_simd is a derived capability set only
  on a 64-bit native backend with the right vector ISA — arm64 requires NEON and
  x64 requires SSE2 + SSSE3 + SSE4.1 (not SSE2 alone) — while the 32-bit backends
  and the emulator get no sf_has_simd, so SIMD is a 64-bit-native-only capability
  (code).

- 2026-04-18 (f98d3458) statement: the SIMD bring-up implements baseline SIMD
  only — relaxed-SIMD opcodes are detected separately and rejected at decode as
  relaxed SIMD is not supported, a boundary that still held at the end of the
  SIMD bring-up (x64 done) (code).

- 2026-07-25 (d4dc0be9) pitfall: a second front end kept only to demonstrate a
  configuration is not exercised by the work that changes that configuration,
  so it drifts out of date while still being cited as evidence (code).

## Moves

- 2026-07-25 (d4dc0be9) dropped: the separate minimal `no_std` front-end
  crate: a front end maintained purely as a size demonstration went stale,
  and the same floor is now expressed as a feature configuration of the CLI
  that the test suite actually runs (sourced)
