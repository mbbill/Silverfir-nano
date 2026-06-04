# CFG + SSA staged pipeline

The native backend is a staged optimizing compiler rather than a single linear
pass: **Wasm → Semantic IR → SSA-IR → MachineIR → native code**. The
intermediate representation is a control-flow graph in SSA block-parameter form.
The frontend stays structured long enough to make cheap global-ish choices;
slots, transient budget, and call barriers are made explicit before lowering; a
target-neutral MachineIR carries the fixed-shape lowering; and each architecture
backend selects final instructions late. This is the structure the project runs
on today.

The pipeline owns its own VM ABI and emits code with direct code-to-code
chaining between blocks and functions. `preserve_none` is treated as an optional
optimization where the target supports it, not a structural requirement.
Registers are allocated over a bounded transient window only; canonical locals
and deep stack/call payloads already have fixed homes.

This option opens two coupled sub-problems (an AND-branch): how the IR pipeline
is structured (`pipeline-ir-structure/`) and how registers are allocated across
it (`register-allocation-model/`). They stay in one ordered branch because the
SSA-IR definition constrains the register model.

## In practice

Must:
- Lower through the four stages Wasm → Semantic IR → SSA-IR → MachineIR → native,
  with the IR in CFG/SSA block-parameter form.
- Keep MachineIR target-neutral; each architecture backend lowers every
  MachineIR op it supports natively, and a new target is a new backend behind
  MachineIR, not a change to the middle end.
- Restrict register allocation to a bounded transient window; canonical locals
  and deep Wasm-stack / call payloads must already have stable frame-slot homes
  before allocation.
- Use direct code-to-code chaining between blocks/functions and define the
  backend's own internal VM ABI.
- Carry register residency across loop boundaries (cached locals pinned to
  registers) so hot loops do not re-enter dispatch or re-setup memory metadata
  at each boundary.
- Pass spectest on the structure's primary target before promotion, and validate
  across the supported backends.

Must not:
- Make `preserve_none` a structural requirement of the hot path; a target that
  lacks it must still produce correct, native code (at a performance cost only).
- Enter hot opcodes through an ordinary C/Rust ABI function whose
  prologue/epilogue and spills would destroy register residency.
- Require a full general-purpose register allocator over all values; only the
  transient window participates in allocation.

## Ground rules — pipeline-ir-structure
Must:
- MachineIR is the shared/arch boundary: every MachineIR op represents Wasm or
  shared-JIT behavior that is meaningful for **all** backends — each backend
  lowers it natively or via a fallback, never neither.
- Each stage does its job at its own altitude: the frontend preserves Wasm
  structure (and preprocesses what later stages shouldn't reverse-engineer);
  the middle makes homes, budgets, and call boundaries explicit; machine
  lowering is one-pass; the arch layer only selects encodings.
- A stage may consume only what earlier stages made explicit — no stage may
  reach forward into target-specific facts that belong below MachineIR.

Must not:
- Must not add a MachineIR op for a single backend's helper strategy; if targets
  may differ between native instruction and helper fallback, that choice lives
  below MachineIR.
- Must not let a later stage rediscover or re-derive structure an earlier stage
  already had (e.g. reverse-engineering loops from a flat CFG).

## Ground rules — register-allocation-model
Must:
- The middle-end must guarantee that everything needing a register fits the
  configured dynamic budget at **every** program point; a machine-stage
  allocation failure is by definition a middle-end bug, never a backend concern.
- Exactly two consumers share the dynamic banks: single-use SSA values (Wasm
  stack operands — consumed once, then dead) and cached locals. Any answer must
  account for both inside one budget.
- The chosen model must keep one-pass machine lowering sufficient — dead-input
  reuse plus a bounded transient window, no graph-coloring/global RA downstream.

Must not:
- Must not push allocation policy below MachineIR: backends select encodings and
  own only their scratch pool; they never re-decide who owns a register.
- Must not let any value class escape the model (every value is a single-use SSA
  transient, a cached local, or frame-resident — no fourth class).
