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

1. checkpoint invariants
2. checkpoint identifiers
3. compare snapshot format
4. resumable stepping API

## Checkpoint Invariants

Checkpoint design must start from invariants, not from "where can we stop
easily".

The invariants are:

1. A checkpoint must be a semantic commit point.
   - Previous effects are already committed.
   - The next semantic action is well defined.
2. Resume from a checkpoint must not change execution behavior.
   - Checkpoint mode must capture live backend state exactly.
   - It must not spill or materialize extra state just to make comparison
     easier.
3. Comparison data must be backend-independent.
   - Do not compare backend-private register or cache state.
   - Do not compare abstract stack/TOS metadata after lowered IR.
4. Checkpoint density is a narrowing strategy, not a default mode.
   - Start sparse.
   - Narrow only after the first failing function or invocation is known.

This means the lockstep system must keep two separate views of checkpoint
state:

- resume state
  - exact live backend state needed to continue execution
- compare state
  - canonical semantic state used only for equality checks

### Checkpoint modes

The system should support coarse-to-fine narrowing:

- `Function`
  - function entry and function exit only
- `Control`
  - function boundaries plus selected control/call/helper boundaries
- `Scoped`
  - only for an already narrowed function or region
  - still only at semantically valid commit points

Expected workflow:

1. run whole program in `Function` mode
2. locate the first bad function or invocation
3. rerun only that function in `Control` mode
4. if needed, rerun a narrowed region in `Scoped` mode

The system should not use arbitrary per-op checkpoints as a default. That is
too intrusive, too slow, and often compares backend-private transient state
instead of semantic state.

### Checkpoint identifiers

Each checkpoint must be stably identified by:

- function index
- invocation or call-depth context if needed
- site ordinal within that function
- checkpoint kind
- semantic op index if applicable

This is the coordinate system that lets every tool speak the same language.

### Canonical snapshot

Snapshots should compare canonical semantic state only:

- call depth
- trap state
- return state
- globals
- memory page hashes

Do not compare:

- local-cache registers
- TOS/cache registers
- backend-private temporary registers
- `spill_depth`
- `current_tos_slots`
- raw backend frame/cache layout

Logical operand-stack values may be compared later, but only at checkpoint kinds
where both backends can expose them in a common semantic form.

Do not record full memory at every checkpoint by default.

Use page hashes first. If a mismatch appears, rerun the narrowed region with
more detail if necessary.

### Resumable stepping API

Both backends need a resumable API:

- advance until the next planned checkpoint
- return the checkpoint id and canonical compare snapshot
- or return finished state

The live resume state is backend-private and should not be part of the compare
contract.

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

- checkpoint stops may capture native registers and backend-private state
- checkpoint resume must continue from that exact captured state
- debug mode may force extra hard boundaries in native code generation
- group splitting is acceptable in lockstep debug mode
- production grouping must remain unaffected
- native must not introduce spill/materialize behavior solely for lockstep
  comparison

## Base Backend Requirements

Base is the ground truth backend.

The base side should expose the same checkpoint ids and snapshot format, even if
its stepping implementation is simpler.

The initial implementation should start with function-boundary checkpoints only.

That gives the most useful first narrowing step at the lowest runtime cost:

- identify the first mismatching function invocation
- rerun only that function with more checkpoints

The base backend remains the ground truth, but it should expose the same sparse
checkpoint contract rather than a trace-everything interface.

Current status:

- the first implementation should treat function-boundary mode as
  function-entry plus final completion compare
- finer "function exit before return" checkpoints should not be added until
  they are mapped from semantic/frontend identity instead of backend-lowered
  op indices
- otherwise the debugger risks reintroducing exactly the backend-coupling bugs
  it is supposed to catch

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
