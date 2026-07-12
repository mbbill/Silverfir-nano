- Each WASI preview1 syscall is its own host struct implementing the VM's
  external-function trait plus a `WasiFunction` trait that supplies
  guest-memory accessors (read_u32/write_u32/read_bytes/...) over the calling
  module's memory 0.

- Syscalls are split into per-category module files (args, env, fd, fs, proc,
  random, time, misc) under `wasi_snapshot_preview1`, registered together
  against the VM.

- Host-side WASI state (args, environment, preopens) lives in a `WasiContext`
  held behind `Rc<RefCell>`, and a `WasiRuntime` owns that context for
  registration.

## Facts

- 2025-06-23 (40c1e696) rationale: a WASI syscall reaches guest linear memory
  through the calling module's memory at index 0; all reads and writes are
  little-endian and bounds-checked against the memory's current length,
  faulting (`Errno::Fault`) on out-of-range access — so a rebuilder validates
  every buffer against live memory length rather than trusting guest-supplied
  pointers (code).

## Moves

- 2025-06-24 (d6614342) replaced by [[wasmtime-wasi-backed]]: delegate WASI to
  the mature wasmtime-wasi implementation instead of maintaining a hand-rolled
  one (code).
