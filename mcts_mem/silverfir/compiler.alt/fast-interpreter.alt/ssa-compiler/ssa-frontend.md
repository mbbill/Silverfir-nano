- The frontend decodes WebAssembly into SSA form directly, maintaining a
  value map from locals and parameters to their current SSA value and
  inserting phi nodes at control-flow merges (`SsaBuilder`).

- Pure operations are not emitted immediately as linear instructions but
  accumulated as expression trees on an expression stack; trees are
  materialized to SSA only at barriers (calls, stores, branches, memory and
  control operations).

- Trees left unmaterialized after an unconditional branch are simply dropped,
  giving dead-code elimination for free without a separate pass.

- A call materializes all pending argument trees (and, for call_indirect, the
  callee table index) to SSA values before emitting its single Call
  instruction, then pushes the result values back as fresh materialized trees.

- After construction finishes, the SSA function runs through a frontend
  SSA-to-SSA optimization pipeline (instruction fusion, unreachable-block
  elimination, phi-predecessor cleanup, trivial/dead phi elimination, dead-code
  elimination) before the middle stage consumes it.

## Facts

- 2025-10-10 (e6646469) rationale: SSA construction omits formal dominance
  analysis (no immediate-dominator tree, no dominance-frontier computation as
  in Cytron et al.) because WebAssembly's structured, reducible control flow
  makes every merge point syntactically explicit and every loop header and
  back-edge known from the opcode stream, so phi placement needs no dominance
  frontiers; loops use the incomplete-phi technique — a phi per live local is
  created at the loop header carrying only the entry edge, and the back-edge
  source is appended when a br/br_if/br_table to the header is encountered
  (code).

- 2025-10-12 (28d6692d) pitfall: a call (or call_indirect) is one SSA
  instruction producing several result values, and all of those results must
  carry the same instruction-index ValueOrigin (the call's own index), not
  index+i, since index+i points at instruction slots holding no instruction
  and corrupts the origin map that liveness and register allocation read
  (code).

- 2025-10-19 (4ea03b39) rationale: an SSA-reconstruction pipeline is built
  rather than executing the already-LLVM-optimized wasm directly because
  WebAssembly's mutable locals break SSA form (forcing re-analysis to recover
  def-use) and a runtime can do runtime-specific optimizations LLVM cannot —
  speculative/profile-guided inlining, bounds-check elimination from known
  memory layout, redundant-load elimination from runtime alias info,
  cross-module inlining, and physical register allocation; all major engines
  (V8, SpiderMonkey, Wasmtime) likewise reconstruct SSA — full reasoning in
  [[ssa-frontend.fact/why-reconstruct-ssa]] (sourced).

- 2025-10-22 (639b766b) rationale: SSA IR keeps a 1:1 mapping with
  WebAssembly bytecode — operations with separate Wasm opcodes (e.g.
  struct.get / struct.get_s / struct.get_u) stay separate SSA variants even
  when structurally similar — and instruction merging is not done at the SSA
  layer (code).

- 2025-10-22 (639b766b) rationale: the 1:1 SSA-to-bytecode mapping is kept to
  preserve a faithful, debuggable correspondence with the source bytecode
  (sourced).

- 2025-11-26 (c05efabe) rationale: the frontend's SSA structural validator
  (each value defined once, reachable blocks terminated, branch targets in
  range, value references valid, phi sources match predecessors, return arity)
  is gated behind debug/test cfg so it compiles out of release builds — it is
  a debug-time invariant net catching malformed SSA the frontend itself
  produced, not a runtime check on the guest program, so it must impose zero
  release cost (code).
