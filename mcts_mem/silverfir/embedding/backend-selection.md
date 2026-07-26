- A build's backend is its target ISA, not a choice made among the compiled-in
  backends: exactly one `sf_backend_*` cfg is set, the build links only that
  backend's code, and no value, parameter, or type names which backend is
  active (`sf_backend_arm64`).

- Asking for an engine on an ISA that has no backend is refused at build time,
  naming the target and the supported ISAs, rather than producing an engine
  that reports unavailability at instantiation.

## Moves

- 2026-04-18 (f98d3458) replaced [[runtime-reference-backend-overlay]]: backend
  selection used a process-global mutable AtomicU8 ReferenceBackendMode
  (Disabled/Emu64/Emu32) settable at runtime to force the emulator on a host
  that also had a real native backend, which carried a mutable global in a
  no_std engine and let the backend differ from the compiled target; selection
  moves entirely to build time, where exactly one sf_backend_* cfg is set (the
  old sf_arch_* cfgs are renamed to sf_backend_*) and the emulator becomes a
  first-class build-selected backend (backend-emu64 / backend-emu32 features)
  like any native one (code)

- 2026-07-25 (7dec6c6a) dropped: building an engine for an ISA with no backend:
  both engines are ISA-specific, so such a build yields only an engine that
  fails at instantiation, and the build script is the earliest place that can
  say so by name (sourced).

- 2026-07-25 (ac353dbc) dropped: backend selection as a modelled choice (the
  NativeBackend enum and the active_backend parameter threaded through the
  compile pipeline): it was load-bearing only while the emulator could stand in
  for a backend other than the host's, and without that a target has exactly
  one backend, leaving a univariant enum and a parameter carrying no
  information (sourced).
