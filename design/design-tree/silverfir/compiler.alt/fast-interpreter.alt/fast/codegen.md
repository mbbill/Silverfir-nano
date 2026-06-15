- The fast backend's handler set is declared in a single declarative
  specification from which the build generates the C wrappers, the Rust extern
  declarations, and the wasm-opcode-to-handler map, rather than three hand-kept
  parallel lists.

- How each instruction packs its operands (slot indices, branch targets,
  immediates) into the instruction word is defined declaratively as part of the
  same handler specification — shared encoding patterns plus per-handler inline
  field schemas — from which the build generates both the Rust encode/decode
  functions and the matching C decode macros; a layout change touches only the
  schema.

- Each base WebAssembly op's computation is defined exactly once as a semantic
  macro that is the single source of truth: standalone handlers expand it directly
  and the fusion generator composes the same macros into fused handler bodies, a
  fused handler computing its constituents identically to the per-opcode handlers.

- A handler may be implemented in C or Rust, selected per handler by a flag in the
  spec; a C handler is a FORCE_INLINE function inlined into the trampoline
  translation unit (the whole dispatch chain optimizes together), while handlers
  that must touch Rust-side runtime state (call stack, store, module) stay in Rust.

## Facts

- 2025-12-03 (e76be08e) statement: the fast backend's handler spec deliberately
  omits the XIR backend's register-permutation system — the fast interpreter is
  stack-based with no register allocation, so each wasm op maps to one handler
  rather than to a family of (operation, register-permutation) variants; the schema
  is a flat per-op TOML plus a separate fused-pattern table, in contrast to XIR's
  arity-pattern permutation expansion (author).

- 2025-12-11 (3d6ab597) rationale: the instruction word was made uniform and
  schema-driven to keep the Rust emitter and the C handlers in lockstep — the typed
  layout (a dedicated alt branch-target pointer plus two u32 and one u64 immediate
  with hand-coded bit packing scattered across emitter and handlers) was replaced by
  three uniform u64 immediates with all packing generated from the schema; branch
  targets now ride in a general immediate instead of the removed alt field (diff).

- 2025-12-07 (ae18d62e) rationale: C handlers are FORCE_INLINE functions #included
  into the trampoline translation unit before the generated wrappers, so they
  inline directly into the preserve_none/musttail dispatch wrappers in one TU where
  LLVM (with thin-LTO and sibling-call optimization) can optimize the whole dispatch
  chain together; the migration targets handlers whose hot path benefits from C
  branch-prediction hints (unlikely() on trap checks) the Rust extern handlers could
  not express as directly (diff).

- 2025-12-07 (176413c2) rationale: the Rust-vs-C handler split is drawn at access
  to Rust-side runtime state — purely computational handlers (arithmetic, memory
  load/store, float, comparison, conversion, and the branch ops whose frame fixup
  is just a slot-shift in C) move to C, while handlers that must touch the call
  stack, store, or module (return, call, br_table, the table/data drop ops) stay in
  Rust because that state is not reachable from the C side (diff).

- 2026-06-14 rationale: portability drives moving the hot handlers into C
  (single-translation-unit, inlined into the trampoline), not a benchmark win —
  the Rust-impl + C-wrapper path needs tail-call and preserve_none (C-only)
  together with cross-language LTO between C and Rust, and cross-language LTO does
  not work on Win64, a hard deployment limiter; a single-TU C implementation
  removes the LTO dependency entirely, and also removes the Rust->C argument
  write-back friction (Rust args pass by copy, so writing a result back needs
  pointers, which inlining alone does not solve), with the single-TU inlining
  performance following as a consequence rather than a motivation (author).

- 2026-02-06 (d12e0e31) rationale: each base numeric op's computation is defined
  exactly once as a semantic macro and both the standalone handlers and the fusion
  generator expand it, so a fused handler is guaranteed to compute its constituents
  identically to the per-opcode handlers; this commit extended the macros from a
  partial set to the full numeric op set (trapping div/rem, rotl/rotr, all
  comparisons, clz/ctz/popcnt, parameterized load/store, conversions, full float
  ops) (diff).

- 2026-02-06 (d12e0e31) pitfall: the trapping semantic macros (div/rem) are
  written as a flat statement sequence rather than wrapped in a do{...}while(0) —
  the flat expansion is critical for LLVM's guard-check elimination, i.e. wrapping
  the trap guards in a block defeats the compiler's ability to fold redundant
  bounds/zero checks across composed fused handlers (diff).

- 2026-02-08 (da03882b) measurement: the recovered fast-interpreter design note
  quantifies the flat-expansion pitfall — wrapping the semantic macros in a
  do{}while(0) (giving them their own scope) breaks LLVM's guard-elimination proof
  and costs ~30%; this is the measured cost behind the "flat, no scoping — NEVER
  wrap in do{}while(0)" rule the trapping macros are written to satisfy (diff).

- 2025-08-16 (516ffe11) pitfall: an opcode with no fast-backend handler must fail
  the build, not be silently aliased to a no-op — the opcode->handler map
  previously fell back to op_nop on its default arm, so an unsupported instruction
  would build into a stream that quietly skipped it; the map now returns
  WasmError::Invalid on the default arm so building a function with an unhandled
  opcode fails loudly (diff).

- 2025-08-16 (73afbb5c) pitfall: float->int range checks must use exact
  power-of-two boundaries, not the integer extremes cast to float — i64::MAX as f64
  rounds up to 2^63 so a value exactly 2^63 would wrongly pass an inclusive upper
  bound; the check compares against the literal 2^63 with a strict reject, the f32
  range checks widen to f64 first so the f32 representation of i32::MIN/MAX does not
  lose precision, an unsigned-i64->f32 convert casts directly to avoid double
  rounding, and the saturating truncations test is_nan() rather than !is_finite()
  so infinities saturate to the integer bounds instead of collapsing to 0 (diff).

- 2026-06-14 rationale: cross-language LTO couples the toolchain versions —
  with LTO enabled the compilers emit LLVM IR instead of object files, so
  clang and rustc must use the SAME LLVM version for the IR to link; this
  version lock-step, on top of the Win64 failure and the
  preserve_none/musttail C-only requirement, is the deeper limit of the
  Rust-impl-plus-C-handler approach — an interpreter that must avoid this
  coupling cannot use the C trampoline at all (author).

## Moves

- 2025-12-03 (e76be08e) replaced [[hand-maintained-handler-tables]]: adding one
  fast-interpreter instruction previously required editing four places kept in
  sync by hand — the extern-declaration macro in handlers.rs, the
  DEFINE_OP_WRAPPER list in vm_trampoline.c, the map_handler match arm in
  ir_builder.rs, and the impl_ function — with no single source of truth; a
  declarative handlers.def from which the C wrappers, Rust extern declarations,
  and the wasm-op-to-handler map are all build-generated removes the three-way
  manual synchronization (diff).
