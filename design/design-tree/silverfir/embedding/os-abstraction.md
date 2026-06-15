- All host OS primitives (executable-memory allocation, guard-page reservation,
  W^X toggling, instruction-cache flush) are reached through a single
  `runtime/os` module; no other runtime file declares OS externs or dispatches
  per-OS at the call site (`alloc_executable`).

- Exactly one OS submodule is active per build, selected by `sf_os_*` cfgs;
  adding a host is adding one submodule, and the bare-metal `none` submodule
  binds the same API to embedder-provided extern shims rather than syscalls.

## Facts

- 2026-04-26 (c5b1b808) statement: The bare-metal embedding contract
  (statically-reserved code arena via sf_os_alloc_executable plus a barrier-only
  finish-write) is exercised by independent MCU families: the RP2350 (Pico 2)
  firmware satisfies it for both Cortex-M33 (dsb;isb) and Hazard3 RV32
  (fence rw,rw; fence.i) cores from one source, and the ESP32-C6 firmware
  satisfies it for a separate RV32 chip — confirming the embedder owns only
  executable memory and CPU barriers, not engine state (diff).

## Moves

- 2026-04-07 (b6d6e3de) replaced [[inline-os-syscalls]]: each runtime file
  (code_buf, guard_pages, trap_signal) declared its own
  mmap/VirtualAlloc/pthread_jit_write_protect_np externs and dispatched per-OS
  inline, so adding a target meant editing every syscall site and there was no
  embedder-shim seam for a hosted-OS-free target; isolating every host primitive
  behind one os/ module makes adding a target a matter of adding one submodule
  and lets bare-metal (`none`) bind the same API to embedder-provided extern
  shims (diff).
