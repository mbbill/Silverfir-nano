- A control-transfer local call is an explicit CFG construct in MachineIR with its
  post-call continuation as a first-class machine-IR block; a direct local call
  splits its block into a pre-call block and a continuation block, flushing dirty
  cached locals before the transfer and reloading in the continuation
  (`lower_call_internal`).

- Compiled local calls are one unified MachineIR `Call` terminator carrying a
  call target that is either a compile-time-known local callee or a
  table/runtime-resolved callee; MachineIR has no arch-specific call variants and
  each backend lowers a single call path (`MachineTerminator::Call`,
  `MachineCallTarget`).

- The local-call ABI splits each dynamic register bank into a volatility contract
  (volatile caller-saved prefix, preserved callee-saved suffix, backend scratch);
  values may stay in preserved-lane registers across a local call, a trailing scalar
  parameter suffix passes in abstract GP/FP argument lanes, and the caller switches
  and restores a relative frame delta around the native call instead of writing a
  per-call link record (`is_preserved_dynamic_reg`, `frame_delta`).

- The callee-side fallback return region is pinned to frame slots [0, result_count),
  independent of the callee's local count and operand-stack base; direct, indirect,
  and tail calls all agree on result placement without a per-call result-base pointer.

- A single-result function may return its value in an abstract scalar return lane
  (recorded by an SSA `ReturnScalar` terminator and a per-function `MachineReturnAbi`)
  instead of publishing to the canonical return frame slot, and a single-result
  scalar call keeps its result as a live SSA value carried in a register through the
  call's continuation edge, both gated per backend; otherwise results travel through
  frame slots (`scalar_return_lanes`).

- Helper-backed operations are explicit runtime boundaries described by typed
  machine-level call contracts (explicit arg/result registers, clobbers, continuation
  behavior); a transparent helper that falls through lowers as an ordinary
  machine-level call, while a control-transfer helper returns explicit native resume
  state for the backend to resume natively. Runtime-boundary call metadata and helper
  extern targets are deduplicated into a machine-module sidecar const pool referenced
  by id (`ConstPoolBuilder`).

- Each function carries a runtime record derived once from its frame plan (frame
  regions plus the return ABI); execution below MachineIR resolves frame
  geometry from it without re-reading the frame planner (`MachineFunctionAbi`).

- call_indirect is lowered into a chain of machine blocks that resolve the target
  inline against a cached function-view table — a bounds check, null-reference check,
  and signature check each branch to a trap block, then dispatch splits local from
  external targets sharing one continuation; a funcref table classified at
  instantiation as fixed-size/private/local-only lowers its sites through a compact
  in-MachineIR fixed dispatch view, reverting to the generic path (clearing native
  code for recompilation) if mutated at runtime (`lower_call_indirect`,
  `TableDispatchMode`).

- return_call / return_call_indirect / return_call_ref are lowered as tail
  terminators that repack arguments into the callee frame prefix from slot 0, reuse
  the caller's own frame, and transfer with a jump; a chain of tail calls runs in
  constant stack. The frame-reuse transfer is realized only for compiled-local
  (SF-to-SF) targets, with non-local targets falling back to a runtime call followed
  by a return (`lower_tail_call_internal`).

## Facts

- 2026-03-13 (013fd297) pitfall: lowering hard-rejects any call, boundary, or return
  reached with a non-empty live transient set, forcing the frontend to publish all
  live SSA to slots before every boundary; this is the invariant that lets the backend
  skip stack reconstruction (code).

- 2026-05-13 (a31b7b9a) rationale: register scalar returns are gated per backend by a
  `scalar_return_lanes` flag and apply only to a single non-V128 scalar; a backend
  opts in once its physical return-lane ABI is wired (ARM64 first: GP in x1, FP in d0),
  and `derive_return_abi` falls back to publishing slot 0 otherwise, so callers and
  callees always agree on scalar-vs-frame return ABI from the shared function ABI (code).

- 2026-04-17 (75603283) rationale: the constant-stack frame-reuse tail transfer is
  realized only when the callee is a compiled local function; a direct tail call to a
  non-local target lowers as a runtime call + Return, and indirect/ref tail calls branch
  at runtime on whether the resolved target is compiled-local, so tail calls give the
  constant-stack guarantee only for compiled-to-compiled transfers (code).

- 2026-03-30 (edbc310e) rationale: foreign C-ABI argument/result registers are not
  extra MachineIR register classes and may overlap caller-saved transient or scratch
  registers (never the four fixed roles); the overlap is safe because by the time a
  foreign boundary is reached shared lowering has made dynamic MachineIR state
  unavailable — transients are required dead and cached locals already published (code).

- 2026-05-16 (f752fd7b) measurement: specializing fixed local-only call_indirect tables
  dropped Lua func155 block codegen from 5020 to 3424 instructions (sourced).

## Moves

- 2026-04-16 (9ff58dcd) replaced [[split-call-terminators]]: keeping CallDirect and CallIndirect as two separate terminators forced every backend to carry two near-identical call-lowering paths and duplicated the shared caller-stub / status-check / frame-transfer contract; collapsing them into one Call terminator that carries a MachineCallTarget::{Direct(func_id), Indirect{target,entry}} keeps MachineIR portable with no arch-specific call variants and lets each backend lower a single lower_call against one contract (code).

- 2026-03-13 (013fd297) replaced [[scratch-register-helper-call]]: the current helper set reads and writes canonical frame spans directly through metadata, so helpers operate on frame regions named by the metadata instead of being forced through a per-call scratch-base register, which becomes an optional escape hatch (sourced).

- 2026-05-13 (adc74515) replaced [[frame-and-call-link-abi]]: the old JIT-to-JIT local-call ABI treated dynamic-register order as an allocation preference with no semantic class boundary, so any value live across a local call had to be published to a frame slot before the call and reloaded after — registers could not carry values across a local call at all, and every call also reserved a backend-private host-stack call-link record holding caller frame pointer and result base; the new ABI splits each dynamic bank into a JIT-internal volatility contract (volatile caller-saved prefix, preserved callee-saved suffix, backend scratch), passes a trailing scalar parameter suffix in abstract GP/FP argument lanes described by `MachineFunctionAbi::param_locs` / `MachineCallArgs`, replaces the absolute base pointers and call-link record with a relative `frame_delta` the caller switches and restores around the native call, and fixes the callee fallback-result region at frame slots [0, result_count) so direct, indirect, and tail calls agree on result placement without any per-call link record (code).

- 2026-05-14 (fec6b497) replaced [[frame-slot-call-results]]: a call result written only to a frame slot must be reloaded by any consumer and cannot participate in register residency or sink planning; producing a live SSA result for single-result scalar calls lets the result stay in a register across the continuation edge and lets call -> local.set_cache sink directly into the cache register, eliding the store/reload pair (code).

- 2026-05-15 (e7402d3e) replaced [[publish-all-indirect-call-args]]: eagerly storing every indirect-call argument to the callee frame before dispatch wastes a store per arg on the local fast path where the args could pass in registers, and forced a reload-shaped frame round trip; capturing live args and threading them as register lane args through the dispatch cluster keeps the local indirect-call hot path register-resident, while the runtime-dispatch fallback block still publishes the carried args to the frame before the runtime helper (code).
