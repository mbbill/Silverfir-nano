- The fused-instruction set is discovered automatically rather than
  hand-written: candidate N-grams of executed/decoded instruction sequences are
  counted, and a greedy pass selects the highest-dispatch-savings patterns and
  emits the fused-handler table, rejecting any candidate that cannot fit the
  three-slot encoding budget or whose stack effect falls outside the supported
  TOS pop/push set.

- Two discovery paths coexist: a host-side runtime profiler that captures
  sliding-window N-grams of executed handler sequences, and an offline static
  tool that parses a Wasm binary without executing it, weighting N-grams by
  loop-nesting depth and propagating weights through an extracted call graph.

## Facts

- 2026-02-22 (4bb1de83) rationale: automated discovery is only practical because
  the stack-machine model makes fusion mechanical (concatenate handler bodies,
  let the compiler optimize); the discover tool generalizes well because LLVM's
  Wasm backend emits consistent instruction sequences across programs (author).

- 2026-02-16 (194dbd17) rationale: candidates are discovered from a basket of
  workloads, not one — each workload's N-gram counts are normalized to
  frequencies, combined, then scaled back to counts, so size differences do not
  let one workload dominate; the inclusion threshold is a percentage of total
  instructions, not an absolute count (diff).

- 2026-03-01 (3b2933b5) rationale: the static tool weights each N-gram by
  loop-nesting depth and propagates interprocedurally through a call graph so
  hot-loop sequences dominate without running the workload, complementing the
  runtime profiler; a compare tool diffs static-vs-dynamic candidate sets on
  equal footing (diff).

## Moves

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (diff).
