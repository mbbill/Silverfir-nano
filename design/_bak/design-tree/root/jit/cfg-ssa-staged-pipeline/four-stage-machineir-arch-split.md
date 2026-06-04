# Four stages with a MachineIR + per-arch backend split

The pipeline is **Wasm → Semantic IR (`wasm/`) → SSA-IR (`middle/`) → MachineIR
(`machine/`) → native (`arch/`)**. Semantic IR keeps Wasm-specific structure
(structured control markers, abstract locals, semantic calls, typed results,
`max_stack_height`) intact so stack-machine artifacts never leak into the
backend; SSA-IR in `middle/` is the optimization layer (constant folding, copy
propagation, sink planning); MachineIR is a target-independent CFG of
instructions with its own validator; per-arch backends only select target
encodings.

The MachineIR ↔ arch boundary is the portability boundary: a new architecture is
a new backend, not a codegen rewrite. Shared emit/compile logic is factored into
`ArchBackend` + `CompilerCore` + a shared streaming pipeline so a new target is
mostly encoding + ABI. A reference `emulator` backend executes MachineIR directly
and serves as a non-host correctness oracle (`emu64` / `emu32` configs).

This node opens one sub-problem in this bounded tree (`i64-legalization-layer/`):
on 32-bit targets, where 64-bit values get split into register pairs.

## In practice

Must:
- Keep the four layers distinct: `wasm/` (Semantic IR), `middle/` (SSA-IR),
  `machine/` (MachineIR), `arch/*` (native backends).
- Confine all target-specific encoding to `arch/<target>/`; MachineIR ops carry
  Wasm or shared-JIT semantics only.
- Add a MachineIR op only when its semantics are platform-independent across all
  targets; if different targets may choose between a native instruction and a
  helper fallback for the same operation, that choice lives below MachineIR.
- Run every backend (arm64, x86_64, armv7a/emu32, riscv64/32, emu64) through the
  same MachineIR and pass spectest behind it.
- Keep the `emulator` backend able to execute any MachineIR program produced by
  the shared pipeline, so it stays usable as a correctness oracle.
- Validate MachineIR with `machine/validate.rs` before handing it to a backend.

Must not:
- Let Semantic-IR artifacts (`pre_height`, generic variants, `read_t0` helpers)
  appear in MachineIR or any backend.
- Add a MachineIR op merely because one backend currently needs a helper call.
- Make adding a new architecture require touching the middle end or MachineIR
  definitions.
- Perform SSA-level optimizations (constant folding, copy propagation moved up
  from MachineIR peepholes) inside a per-arch backend.

## Ground rules — i64-legalization-layer
Must:
- 64-bit GP targets must bypass the legalization layer entirely — their pipeline
  output is byte-identical with the layer present or absent.
- `emu32` and every real 32-bit backend must consume the same finalized 32-bit
  MachineIR contract, so shared legalization bugs surface on `emu32` rather than
  on hardware.
- After 32-bit finalization, no `GpI64` storage, params, or scalar i64 widths may
  survive — validator-enforced, not convention.

Must not:
- Must not thread pair-lowering complexity through stages that 64-bit targets
  share (the cost of being 32-bit is paid only on the 32-bit path).
- Must not let individual backends invent their own i64 splitting — the split is
  decided once, above the per-arch layer.
