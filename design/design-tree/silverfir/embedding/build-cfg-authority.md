- Backend and OS-abstraction target gating routes through `build.rs`, which
  defines the supported arch-by-os matrix and emits a single `sf_*` cfg
  vocabulary (`sf_backend_*`, `sf_os_*`, `sf_has_posix`, ...) that the engine's own
  compile-time selection reads instead of raw `target_arch`/`target_os`; the
  exception is the hosted WASI host-syscall layer, which reads raw `target_os` /
  `target_arch` directly to bind platform syscalls (`build.rs`).

## Moves

- 2026-04-07 (3693bde6) replaced [[raw-target-cfgs]]: raw target_arch/target_os
  checks scattered across the source could not introduce a bare-metal `none`
  target or a shared posix path without editing every gated site; routing all
  target gating through build.rs gives the source one sf_* vocabulary and a single
  place to define the supported arch x os matrix (diff).
