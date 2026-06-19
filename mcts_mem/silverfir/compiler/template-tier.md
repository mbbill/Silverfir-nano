- A second native codegen tier — a single-pass streaming template JIT — emits
  machine code directly from the Wasm opcode stream without ever materializing
  semantic, SSA, or MachineIR vectors for the function; it is selected per
  function when an estimate of the full pipeline's RAM
  (`bytecode_size * COMPILE_EXPANSION_FACTOR`) exceeds the configured
  `compiler_ram_budget_bytes`, with the full optimizing pipeline used otherwise
  (`template`).

- The template tier supports only a restricted subset (homogeneous scalar result
  types, control depth <= 16, GP unit 4 or 8 bytes, GP budget >= 2) and rejects a
  function it cannot handle with a distinct exhaustion error rather than
  mis-compiling it; the emulator backends do not implement it.

## Facts

- 2026-04-30 (becb63ba) rationale: the template tier exists to keep oversized
  functions compilable on RAM-constrained devices without bringing back an
  interpreter — the full pipeline's peak per-function memory grows roughly
  linearly with bytecode size (the expansion factor is pinned at 160x), so an
  embedded target sets `compiler_ram_budget_bytes` (Pico2/RP2350 uses 500 KB) and
  any function whose estimate exceeds it takes the template path instead of
  OOMing, while hosted builds default the budget to u32::MAX so the path is never
  taken and full-optimization behavior is preserved; because it never builds
  MachineIR, branch resolution uses an intrusive forward-patch chain over emitted
  jump placeholders instead of the optimizing path's CFG parallel-move machinery
  (code).
