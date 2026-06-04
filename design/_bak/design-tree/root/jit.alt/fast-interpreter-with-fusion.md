---
status: abandoned
---
# Fast interpreter with instruction fusion

Execute Wasm by interpreting a linearized opcode stream, optimized to be a
near-JIT-speed pure interpreter that needs no JIT — buying portability to any C
target and a tiny `no_std`/embedded footprint. This was the project's first
execution strategy, shipped at the initial commit.

The design is a stack-machine interpreter built from four interlocking
sub-decisions, each its own sub-problem below: tail-call dispatch with `musttail`
+ `preserve_none` leaf handlers (`interpreter-dispatch/`); staying stack-based
rather than converting to a register machine (`ir-model/`); profile-guided
instruction fusion that fuses `local.get` with hot successors to delete most
dispatches (`reducing-dispatch-overhead/`); and register windows for the top
stack values and the hottest locals so fused hot sequences run with zero memory
traffic (`keeping-stack-values-in-registers/`, `keeping-hot-locals-in-registers/`).

It is kept rather than deleted because its sub-tree was genuinely explored and
produced facts, and the register-residency vocabulary it established (TOS window,
L0/L1/L2 hot-local cache) was inherited wholesale by the JIT design.

## In practice

Must:
- (While in force) the hot handler chain must be generated C reached through a
  Rust→C trampoline, because the `musttail` + `preserve_none` dispatch form is
  not expressible in stable Rust.
- The operand model must stay stack-based, with fusion driven from a profile of
  `local.get` and its hot successors.
- The top stack values (TOS window) and the hottest locals (L0/L1/L2) must be
  kept in registers so fused hot sequences emit zero frame memory traffic.

Must not:
- Must not be the live execution backend: this strategy is abandoned in favor of
  jit.md. Its build system and entry points were removed, not left dormant.
- Must not convert the operand stack to a register machine; automatic fusion
  depends on the stack model.

## Ground rules — interpreter-dispatch
Must:
- Pick exactly one dispatch mechanism for the interpreter's hot loop.
- Preserve the register residency the rest of the interpreter design depends on
  (the TOS-window and L0/L1/L2 registers must survive each dispatch).

Must not:
- Mix dispatch mechanisms on the hot path.
- Choose a mechanism whose per-dispatch overhead spills the threaded register
  state.

## Ground rules — ir-model
Must:
- Choose one operand model for the whole interpreter.
- Keep the model compatible with automatic fusion (the compiler must be able to
  optimize across fused operations).

Must not:
- Adopt a model whose operands are loaded from the instruction stream in a way
  that forces the compiler to assume aliasing across fused ops.

## Ground rules — keeping-hot-locals-in-registers
Must:
- Give the hottest locals a fixed register residency that survives across the
  handler chain.
- Keep the chosen mechanism compatible with fusion (hot-local ops must be fusable).

Must not:
- Leave hot-local access on the frame-memory path once fusion has removed the
  dispatch cost.

## Ground rules — keeping-stack-values-in-registers
Must:
- Keep the topmost operand-stack slots in registers threaded through the handler
  chain.
- Bound the handler-variant cost to be linear in window depth (depth-specific
  variants), relying on the statically-known stack height.

Must not:
- Let the chosen mechanism require a register-permutation number of handler
  variants.

## Ground rules — reducing-dispatch-overhead
Must:
- Reduce the number of dispatches executed per unit of work (fewer handler jumps
  per Wasm-instruction sequence).
- Keep the dispatch-reduction mechanism grounded in the real dispatch-frequency
  profile of workloads.

Must not:
- Trade dispatch reduction for a regression in branch prediction or register
  residency established by the dispatch and register-window decisions.
