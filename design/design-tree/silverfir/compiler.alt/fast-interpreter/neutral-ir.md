- Wasm lowers to a neutral, backend-agnostic IR (`Vec<IrOp>`) carrying a purely
  semantic op kind plus a resolved D-variant, pre-height, and linear/alt targets
  — no handler pointers and no encoding-level pattern data — that resolves stack
  management once (TOS variants, spill/fill insertion, hot-vs-frame local
  mapping).

- The interpreter (base), static-fusion, and micro-JIT backends all consume that
  single IR and produce a common resolved instruction (handler plus IR kind for
  finalizer encoding), each falling back to 1:1 base resolution for ops it
  cannot optimize; the unified backend degrades gracefully JIT -> fusion -> 1:1.

## Facts

- 2026-03-04 (507655cf) rationale: the IR exists because stack-state management
  was duplicated three times (interpreter builder, static fusion, micro-JIT) so
  adding one opcode touched 4-5 places, and the handler-coupled TempInst was
  unfit as a neutral representation; the IR makes one stack_effect table the
  source of truth, makes the D-variant part of instruction identity, and makes
  spill/fill first-class fusible ops rather than group boundaries (diff).

- 2026-03-04 (507655cf) rationale: redesigning static fusion onto concrete IR
  sequences eliminates the Wasm-level lookahead matcher (each pattern IS a
  concrete register-specific instruction sequence), and lets discovery profile
  actual executed IR so only hot variant combinations get handlers, versus
  blindly generating four handlers per pattern (diff).

- 2026-03-05 statement: a fused handler's TOS depth-variant is the first op's
  variant already resolved in the IR (`add_D1` and `add_D2` are distinct
  instructions), so fusion matching reads `ir[pos].variant` directly; the
  predecessor Wasm-level matcher instead ignored variants and recomputed depth at
  runtime via `ref_depth = h + max(0, push - pop)`, which the resolved-variant IR
  makes unnecessary — the wrapper derives h%4 from the variant and indexes the TOS
  registers as `t[(h%4 - p + 4) % 4]`; a reimplementation of fusion matching must
  not reintroduce a runtime ref_depth computation, since the variant is
  instruction identity (author).

- 2026-03-06 (2944da01) rationale: the finalizer's branch-target legality moved
  from a single structural boolean to a three-way compaction disposition because
  the bool could not distinguish a removed structural marker a branch may legally
  be retargeted to from an internal-only removed slot that must never be a branch
  target, so the finalizer silently skipped past either when remapping a branch;
  the three-way form makes the distinction explicit and asserts a branch never
  lands on an internal-only removed op (diff).

## Moves

- 2026-03-05 (2c7ce3f3) replaced [[neutral-ir.alt/tempinst-pipeline]]:
  stack-state management was triplicated across the interpreter builder, static
  fusion, and the JIT, and the handler-coupled TempInst could not serve as a
  neutral representation; lowering Wasm to one neutral IR resolves stack
  management once and lets all three backends share a single pipeline with
  graceful degradation (diff).

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (diff).
