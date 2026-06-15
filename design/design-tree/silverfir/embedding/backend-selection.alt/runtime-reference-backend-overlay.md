- A process-global AtomicU8 ReferenceBackendMode (Disabled / Emu64 / Emu32)
  selects whether the emulator/reference backend is forced; setters mutate it at
  runtime, and Disabled falls back to Emu64 when the emulator is the only compiled
  backend.

- The emulator backend is a single NativeBackend::Reference variant gated on the
  sf_emulator feature, whose register-budget preset is chosen at runtime from the
  active ReferenceBackendMode.

- The selected target architecture is exposed through
  sf_arch_arm64 / sf_arch_armv7a / sf_arch_thumbm / sf_arch_x64 cfgs.

## Moves

- 2026-04-08 (76e9c612) replaced [[emulator-always-compiled]]: compiling the
  reference emulator unconditionally bloated every build, including the minimal
  no_std target that has a real native backend and never needs it; making it an
  opt-in feature keeps the emulator available where it is wanted (spectest, the
  WASI CLI, --emu64/--emu32) while letting production builds drop it entirely
  (diff).

- 2026-04-18 (f98d3458) replaced by [[backend-selection]]: backend selection
  used a process-global mutable AtomicU8 ReferenceBackendMode
  (Disabled/Emu64/Emu32) settable at runtime to force the emulator on a host
  that also had a real native backend, which carried a mutable global in a
  no_std engine and let the backend differ from the compiled target; selection
  moves entirely to build time, where exactly one sf_backend_* cfg is set (the
  old sf_arch_* cfgs are renamed to sf_backend_*) and the emulator becomes a
  first-class build-selected backend (backend-emu64 / backend-emu32 features)
  like any native one (diff)
