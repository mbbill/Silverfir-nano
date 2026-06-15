- Exactly one native backend is compiled into a build, chosen at compile time
  by the target arch through `sf_backend_*` cfgs; the build links only that
  backend's code, never all of them (`NativeBackend`).

- A target with no supported backend still compiles; selecting the active
  backend returns a runtime error rather than failing the build
  (`active_native_backend`).

- The emulator/reference backend is an opt-in, default-off Cargo feature; a
  production build that has a real native backend drops the emulator entirely,
  while spectest and the WASI CLI enable it for `--emu64` / `--emu32`
  (`sf_backend_emu64`).

## Moves

- 2026-04-18 (f98d3458) replaced [[runtime-reference-backend-overlay]]: backend
  selection used a process-global mutable AtomicU8 ReferenceBackendMode
  (Disabled/Emu64/Emu32) settable at runtime to force the emulator on a host
  that also had a real native backend, which carried a mutable global in a
  no_std engine and let the backend differ from the compiled target; selection
  moves entirely to build time, where exactly one sf_backend_* cfg is set (the
  old sf_arch_* cfgs are renamed to sf_backend_*) and the emulator becomes a
  first-class build-selected backend (backend-emu64 / backend-emu32 features)
  like any native one (diff)
