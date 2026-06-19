- The guard-page SIGSEGV/SIGBUS handler holds the ucontext layout, register
  surgery, sigaction wiring, the JIT-range trap table, the install latch, and the
  signal-storm counter together in a single trap_signal module with per-platform
  code selected by inline cfgs.

## Moves

- 2026-04-07 (11b835a2) replaced by [[traps]]: the guard-page signal handler held
  every platform's ucontext layout and register surgery inline in one file, so the
  OS-agnostic trap table was entangled with per-(arch x os) frame parsing;
  splitting the platform half into os/signal/ modules behind a single
  install_platform_handler entry point lets trap_signal own only the trap table,
  storm counter, and install latch, and a new platform is one more signal
  submodule (code).
