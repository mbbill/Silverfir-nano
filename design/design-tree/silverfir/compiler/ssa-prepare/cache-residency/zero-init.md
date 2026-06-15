- Wasm-mandated zero-initialization of non-parameter locals is emitted
  callee-side at function entry and gated on a read-before-write liveness
  analysis; every store for a provably-written-before-read local is elided
  (`emit_zero_init_locals`, `init_locals`).

- A structured must-set analysis over the well-nested control flow computes, for
  each non-parameter local, whether it is definitely written on all paths from
  function entry before any read at entry (control-depth-0) scope; a local not
  definitely written has an observable initial zero, and this `reads_before_write`
  fact is carried per-local from planning to backend lowering.

## Facts

- 2026-03-19 (996c342b) pitfall: leaving a cached local's register uninitialized
  at entry is unsound on 4-byte-GP targets — the 32-bit legalization pass tracks
  each register's storage type by storage-flow analysis and `save_all_cached_locals`
  stores every cached local to the frame before a call, so an Undefined register
  reaching that point has no inferable type and the legalizer rejects the
  function; on 32-bit targets every non-parameter cached local is
  zero-initialized unconditionally, while 64-bit targets keep the elision because
  all GP registers have a single width (diff).

## Moves

- 2026-04-06 (94946b38) replaced [[caller-side-zero-init]]: the caller blindly
  zeroed the callee's whole local prefix at every call site (and the C/emulator
  entry pre-zeroed it), initializing locals the callee provably writes before
  reading; moving zero-init into the callee at function entry and gating it on a
  read-before-write liveness analysis elides every store for a provably-written
  local (diff).
