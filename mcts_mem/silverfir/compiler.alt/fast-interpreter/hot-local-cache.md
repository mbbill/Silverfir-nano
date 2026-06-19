- The three hottest locals are cached in three dedicated `preserve_none`
  register arguments (l0/l1/l2), orthogonal to the TOS window; hotness is
  computed at compile time by a single bytecode pass counting local accesses
  weighted per loop-nesting level, and a function prologue physically
  index-swaps the hot locals to indices 0-2; they map to the dedicated
  registers regardless of original position.

- A register-resident local goes only through dedicated `local_*_lN` fused
  variants; a generic `local_*` fused op rejects when its index remaps to a
  hot-local slot, keeping the register cache and the generic fusion path
  mutually exclusive on the same slot.

## Facts

- 2026-02-22 (4bb1de83) statement: the hot-local weight is 10x per loop-nesting
  level, so a double-nested local access counts 100x an unnested one; the top
  three locals by weight become l0/l1/l2, computed in a single bytecode pass with
  no CFG or SSA — from the design paper docs/INTERPRETER_DESIGN.md (deleted
  78b1f6d6, content at 4bb1de83) (sourced).

- 2026-02-22 (4bb1de83) rationale: fusion removes local-access dispatch but not
  the underlying frame memory traffic, and local access is ~38% of instructions,
  so three extra register arguments cache the three hottest locals; their
  spill/fill folds into existing call/return handlers and `local_get/set/tee_lN`
  are first-class fusion operands so fused hot-local ops compile to pure
  register-to-register (zero instructions) (sourced).

- 2026-03-01 (a05de669) rationale: hotness-based local reordering — profiling
  local-access frequency at load time then physically swapping the hottest
  locals to register-mapped positions — is claimed novel to the author's survey;
  wasm3, WAMR, wasmi, LuaJIT, CPython, HotSpot all access locals at original
  indices via frame memory (sourced).

- 2026-02-18 measurement: CoreMark's top-3 functions account for 96% of all
  local accesses, and weighted-optimal register-cache coverage is l0 = 22.1% of
  accesses, +l1 = 19.0% marginal (l0+l1 = 41.1%), leaving 58.9% generic — the
  diminishing-marginal data behind capping the cache at three hot locals; full
  per-function table in [[hot-local-cache.fact/local-access-coverage]] (sourced).

- 2026-02-18 measurement: an A/B test (SF_FUSION_DISABLED + SF_HOT_LOCAL_MODE)
  found the register cache alone, with fusion OFF, gives ~0% benefit on CoreMark —
  l0+l1, l0-only, and neither all land within ~2% noise (avg 3358/3394/3331) —
  yet with fusion ON, l0-aware patterns gave ~10% over non-l0 fusion; the
  standalone-vs-fused gap was an unresolved contradiction at the time (a fused
  [get_l0,X,Y,Z] saves the same dispatch as [get,X,Y,Z], the only difference being
  a register read vs an fp[idx] read), implying the cache's value is realized
  solely when fusion folds the local access into a register-to-register op — a
  future rebuilder must not ship the register cache without fusion expecting a win
  (sourced).

- 2026-02-18 (296a4898) pitfall: the `l0 = fp[0]` invariant must hold for every
  function because call/return handlers unconditionally spill/fill l0 through
  fp[0]; `init_l0` is emitted even with no hot local and even at
  frame_size==0 (where fp[0] is the return_pc slot), since skipping it would let
  the call-frame spill overwrite return_pc with the register's stale zero and
  corrupt the stack (code).

## Moves

- 2026-03-07 replaced by [[compiler]]: the interpreter's preserve_none
  handler-threaded model and its embedded micro-JIT retained interpreter-shaped
  overhead and could not port to RISC-V/ARM32/MCU targets, so a native
  code-generation backend owning its own VM ABI replaced the whole interpreter
  execution era (code).
