- The only 32-bit Arm backend is `armv7a`, named for the A-profile it targets;
  build.rs routes every `target_arch = arm` build to `sf_arch_armv7a` with no
  distinction between A-profile and Thumb/M-profile targets.

## Moves

- 2026-04-10 (1881a660) replaced by [[arm32]]: a single armv7a module hardwired
  to A-profile could not host Thumb-only M-profile targets, and build.rs mapped
  every target_arch=arm to armv7a; renaming the module to arm32 and splitting
  build.rs by target (thumbv* -> sf_arch_thumbm, else sf_arch_armv7a) gives the
  32-bit Arm backend one encoding-neutral home and a target-driven profile
  selection (diff).
