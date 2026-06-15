- The active backend and OS are selected directly in source with raw
  `cfg(target_arch = ...)` and `cfg(target_os = ...)` attributes at each
  platform-dependent site.

## Moves

- 2026-04-07 (3693bde6) replaced by [[build-cfg-authority]]: raw
  target_arch/target_os checks scattered across the source could not introduce a
  bare-metal `none` target or a shared posix path without editing every gated
  site; routing all target gating through build.rs gives the source one sf_*
  vocabulary and a single place to define the supported arch x os matrix (diff).
