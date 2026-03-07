# Native Backend Roadmap

This document captures the current architectural conclusion for Silverfir-nano's
next major backend evolution.

It is intentionally forward-looking. [DESIGN.md](./DESIGN.md) explains the
current fast interpreter and why it works. This document explains what should
come next, why the current hybrid design is not the final shape, and what work
should happen in what order.

This document is intentionally about architecture, not short-term performance
tuning inside the current JIT. Near-term optimizations to the current backend
may still happen, but they are not part of this roadmap.

## Why This Document Exists

Recent benchmark and profiling work suggests that the current micro-JIT is no
longer limited mainly by missing small peepholes.

The important pattern in [../benchmarks/wasi/results.md](../benchmarks/wasi/results.md) is:

- Silverfir can beat Winch on several compute-heavy workloads.
- Silverfir still falls well behind on `mandelbrot`, `c-ray`, and `stream`.
- Those three are all loop-kernel / memory-kernel workloads and cluster around
  roughly 60-80% of Winch.

This matters because a few tiny local codegen tweaks are unlikely to close a
30% gap across all three. We need to reason about the architecture from a
higher level.

## Current Situation

Today the project has three related execution modes:

- Fast interpreter base mode.
- Fast interpreter fusion mode.
- Current micro-JIT mode.

The current native backend code now lives under `vm/native/`, but it is still
shaped like a
hybrid handler-dispatch system:

- It reuses the fast interpreter's lowered IR.
- JIT groups compact multiple IR ops into one instruction slot.
- Group exits still dispatch like handlers.
- Non-JIT ops still use the existing handler world.

The current fast path also still depends heavily on compiler-specific ABI help:

- `musttail` for guaranteed tail-call dispatch.
- `preserve_none` for the fixed register ABI and zero/near-zero prologue cost.

That design was a good stepping stone. It is probably not the clean final
architecture.

## Main Conclusions

### 1. The remaining gap is structural, not just peephole-sized

For the lagging workloads, the common problem is not only "some op is still not
JITed".

What profiling has shown:

- `mandelbrot` spends most time in a small number of hot JIT loop groups.
- `c-ray` spends most time in a small number of hot JIT loop groups plus call
  overhead.
- `stream` is also dominated by a handful of hot JIT loop groups.

So the common problem is that these workloads are already inside JITed code, but
that code still retains too much interpreter-shaped overhead:

- repeated dispatch at loop boundaries
- repeated memory metadata loads and bounds-check setup
- hybrid transitions between JIT code and handler-style code

### 2. `preserve_none` should become an optimization, not a requirement

`preserve_none` is great where available, but it is not portable enough to be a
core architectural dependency.

It is currently only available on a limited set of targets, while this project
also cares about:

- RISC-V
- ARM32
- MCU-like targets
- tiny systems where portability matters more than peak host-side speed

The right long-term position is:

- `musttail` remains essential for handler-threaded dispatch.
- `preserve_none` is optional and target-specific.

### 3. Hot-path opcodes should stop being ordinary compiler ABI functions

If a hot opcode is entered through an ordinary C or Rust function under the
normal ABI, then even a tiny bridge stub does not solve the real problem.

You still pay:

- call / return overhead
- prologue / epilogue
- compiler-chosen spills and restores
- loss of the VM's intended register residency

Therefore:

- Cold and complex operations may remain real helper functions.
- Hot opcodes should increasingly become generated native code entries.

### 4. The current micro-JIT should evolve into a separate native backend

The clean end state is not "a native backend that still behaves like a hidden
subsystem of the fast interpreter".

The clean end state is:

- `interp/` for the handler-based interpreter backends
- `native/` or `jit/` as a sibling backend that owns its own VM ABI and emits
  native code directly

This removes several conceptual mismatches at once:

- the JIT is no longer "just one more kind of handler"
- direct code-to-code chaining becomes natural
- `nh` can disappear from the native backend
- compiler ABI quirks stop defining the backend architecture

## Target Backend Split

The desired split is:

### `interp/`

Existing handler-based fast interpreter family:

- base
- fusion

Properties:

- still uses handler dispatch
- still uses `musttail`
- may use `preserve_none` where supported
- may also support a slower normal-ABI + `musttail` build on targets that lack
  `preserve_none`

This backend family remains valuable:

- it is the fallback when native code generation is unavailable or undesirable
- fusion is still the second-best option when JIT cannot be used

### `native/` or `jit/`

New native backend with a self-defined VM ABI.

Properties:

- does not rely on `preserve_none`
- should not rely on `musttail` for internal hot-path control flow
- should not require a C compiler for hot-path execution
- owns native entry generation for singleton ops, fused groups, and bridge stubs
- prefers direct code-to-code flow instead of handler dispatch

## Native Backend Principles

### One VM ABI defined by us

The native backend should define its own internal ABI:

- `ctx`
- `pc`
- `fp`
- hot locals
- TOS registers
- any temporary/cached state that belongs to the native backend

This ABI should be defined by the backend, not by Clang's calling convention.

### No `nh`

`nh` exists because the current fast path is shaped like a chain of compiler ABI
handler calls.

In the native backend, direct entry points are the right model. That means:

- `nh` is no longer needed
- one more register becomes available
- every entry/exit shape simplifies

### Hot code is generated, cold code is bridged

The native backend should separate operations into two classes.

Hot path:

- arithmetic
- comparisons
- local get/set/tee
- loads/stores
- simple control flow
- other small frequently executed ops

These should be emitted as native code entries, not ordinary ABI functions.

Cold path:

- host/import calls
- `call_indirect`
- `memory.grow`
- trap slow paths
- other complex or infrequent operations

These can remain Rust helpers behind bridge stubs.

### Bridge stubs are for cold transitions only

Bridge stubs are still useful, but only at the boundary to cold helpers.

They should not sit on the hot path for every opcode. If they do, the normal ABI
prologue/epilogue cost comes back and defeats the point.

### Direct entry pointers instead of handler-only thinking

Every kept instruction in the native backend should conceptually have a native
entry address:

- singleton stub
- fused group
- bridge stub to a cold helper

Once that exists, several later optimizations become natural:

- direct JIT-to-JIT branch chaining
- direct fallthrough from one native entry to the next
- mixed native-to-cold transitions without re-entering the old handler model

## What Happens to the C Handlers

Most current `handlers_c` bodies are small and mechanically express simple
semantics.

That suggests the long-term direction should be:

- stop depending on C handlers for hot-path execution
- generate their equivalent native code directly
- keep only complex and cold helpers in Rust

This solves two problems at once:

- removes the hot-path dependency on `preserve_none`
- removes the hot-path dependency on a C compiler

The current C handlers are still useful during migration as:

- a correctness oracle
- a reference semantics source
- fallback backend material

They should not be deleted early. They should be demoted gradually from "main
execution engine" to "reference/fallback implementation".

## Platform Story

The intended backend matrix is:

### ARM64 / x86_64 with executable memory

Preferred order:

1. native backend
2. fast interpreter fusion/base

### Targets without `preserve_none` but with executable memory

Preferred order:

1. native backend
2. fast interpreter built with normal ABI + `musttail`

### Targets without executable memory

Preferred order:

1. fast interpreter fusion/base if available on that target
2. fast interpreter built with normal ABI + `musttail`

Important point:

- missing `preserve_none` should hurt performance
- it must not block correctness or portability

In other words, `preserve_none` should be an optimization feature of the
handler-based fast interpreter, not a project-wide requirement.

## Structural Work

### 1. Direct code-to-code chaining

Move away from handler-style exit dispatch for native code.

Goal:

- JIT/native entry -> JIT/native entry directly where possible
- only branch to bridge stubs for cold helper paths

### 2. Native singleton stubs

Do not require fusion/grouping for every benefit.

The native backend should be able to emit:

- one-op native stubs
- multi-op groups

That lets the fast path stop depending on C singleton handlers.

### 3. Complete backend separation

The module move to `vm/native/` is only the first structural step. The backend
still needs to be completed as a true sibling backend family rather than a
native code generator that happens to reuse fast-interpreter assumptions.

This is not just a directory cleanup. It is the point where the project stops
treating native code generation as "a special kind of fast-interpreter handler"
and starts treating it as its own backend family.

## Shared Pipeline That Should Stay Shared

The backend split should not duplicate the middle of the pipeline.

These should remain shared:

- Wasm decode
- stack tracking
- neutral lowering / IR construction
- finalization
- branch target metadata
- correctness and differential test infrastructure

Only the final execution backend should differ:

- handler-based fast interpreter
- fusion
- native backend

## Validation Requirements

Because this migration changes core execution machinery, every step should keep
the existing validation discipline:

- differential base-vs-JIT/native tests where applicable
- `spectest`
- `coremark`
- targeted workload checks for:
  - `mandelbrot`
  - `c-ray`
  - `stream`
- profiling with `samply-for-ai`

This is especially important when replacing handler-based execution with native
singleton stubs and bridge transitions.

## Recommended Migration Order

The intended order is:

1. Keep the current architecture stable enough to serve as the migration base.
2. Design the native backend's self-owned VM ABI.
3. Use the new sibling `native/` backend module as the migration target and
   continue moving architectural ownership out of the fast-interpreter world.
4. Add native singleton stubs for already-supported hot ops.
5. Add bridge stubs for cold Rust helpers.
6. Allow direct native-entry chaining.
7. Broaden native coverage until it no longer depends on C handlers for hot ops.
8. Keep `interp/` as the handler-based base/fusion fallback family.

## Non-Goals for the First Native Backend Pass

The first pass does not need to solve everything.

It does not need:

- full coverage of every Wasm opcode
- immediate deletion of the C handler backend
- immediate deletion of fusion
- full CFG-region fusion on day one
- perfect cross-platform parity on every ISA backend

It does need:

- a clean VM ABI not defined by `preserve_none`
- native singleton/group entries for hot ops
- cold helper bridges
- a clear portability story

## Short Version

The current micro-JIT proved that runtime native code generation is worth doing.
It also exposed the limits of keeping that code generator embedded inside a
handler-threaded `preserve_none` interpreter architecture.

The next step is not "more and more native special cases that still behave like
the old fast-interpreter JIT".

The next step is:

- keep `interp/` for handler-based base/fusion execution
- build a separate native backend with its own VM ABI
- generate hot-path code directly
- bridge only to cold Rust helpers
- make `preserve_none` an optimization, not a requirement

That is the path toward a cleaner architecture, broader portability, and the
next meaningful performance step beyond the current hybrid model.
