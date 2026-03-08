# Lockstep Differential Debugging

This document describes the debugging infrastructure that should be built for
the native backend refactor and future backend work.

It is intentionally about debugging architecture, not performance
optimizations.

## Goal

When base and native diverge, we want to stop at the first meaningful
checkpoint and inspect that exact point immediately.

The system should:

- use the battle-tested base backend as ground truth
- compare base and native incrementally
- avoid producing huge trace logs
- work on long-running workloads like CoreMark
- remain usable on slower targets such as RISC-V
- stay out of normal release builds and avoid binary-size growth there

## Core Model

This should be a **streaming lockstep differential runner**, not a
"record-everything then diff later" system.

The intended flow is:

1. Run base until checkpoint `k`.
2. Snapshot canonical state.
3. Run native until checkpoint `k`.
4. Snapshot canonical state.
5. Compare immediately.
6. If equal, continue to checkpoint `k + 1`.
7. If not equal, stop at the first mismatch.

This keeps memory bounded and avoids giant trace files.

## Binary Size and Build Policy

This infrastructure must not pollute normal builds.

Requirements:

- All lockstep code is behind a dedicated feature.
- That feature implies `std`.
- Normal release and embedded builds do not include it.
- Debugging can still use optimized builds by enabling the feature in release.

So the intended build split is:

- normal runtime build: no lockstep support, no size cost
- optimized debug build: lockstep support enabled

## Shared Contract

Both base and native must use one shared checkpoint contract.

The contract has four parts:

1. checkpoint mode
2. checkpoint identifiers
3. canonical snapshot format
4. resumable stepping API

### Checkpoint modes

The system should support coarse-to-fine narrowing:

- `Function`
  - function entry and function exit only
- `Control`
  - function boundaries plus control/call boundaries
- `Dense`
  - every lowered op boundary

Expected workflow:

1. run whole program in `Function` mode
2. locate the first bad function or invocation
3. rerun that function in `Control` mode
4. if needed, rerun a narrowed region in `Dense` mode

### Checkpoint identifiers

Each checkpoint must be stably identified by:

- function index
- site ordinal within that function
- checkpoint kind
- lowered op index if applicable

This is the coordinate system that lets every tool speak the same language.

### Canonical snapshot

Snapshots should compare canonical state only:

- call depth
- relevant frame words
- globals
- memory page hashes
- trap state

Do not record full memory at every checkpoint by default.

Use page hashes first. If a mismatch appears, rerun the narrowed region with
more detail if necessary.

### Resumable stepping API

Both backends need a resumable API:

- advance until the next planned checkpoint
- return the checkpoint id and canonical snapshot
- or return finished state

This is the foundation for the lockstep runner.

## Implementation Order

The agreed order is:

1. Shared checkpoint contract and ids.
2. Native-side resumable/checkpoint API.
3. Base-side resumable/checkpoint API.
4. Lockstep harness that interleaves both backends and compares snapshots.
5. Coarse-to-fine checkpoint modes in real debugging workflows.

## Native Backend Requirements

The lockstep infrastructure must respect the native backend design invariant:

- after lowered IR, native must not reason about abstract stack or TOS state

So native checkpoints must be expressed in terms of lowered IR entries and
canonical snapshots, not reconstructed stack-state metadata.

For native specifically:

- checkpoint sites must be chosen from lowered IR
- debug mode may force extra hard boundaries in native code generation
- group splitting is acceptable in lockstep debug mode
- production grouping must remain unaffected

## Base Backend Requirements

Base is the ground truth backend.

The base side should expose the same checkpoint ids and snapshot format, even if
its stepping implementation is simpler.

## What This Infrastructure Is For

This should make later work much safer:

- backend cleanup
- native codegen optimization
- moving more Rust helpers to native code
- experimenting with grouping policy
- supporting more architectures

The intended end result is:

- first divergence is detected early
- the failing function/op checkpoint is known immediately
- no giant logs are required
- the system is usable even on weaker targets
