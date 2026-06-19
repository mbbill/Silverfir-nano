- A function-level local-cache preference table, built before lowering, decides
  which canonical locals are cache-resident across the whole function; transient
  register pressure and local-cache residency are governed by separate budgets,
  and cache, spill, and sink decisions are owned by separate passes run over the
  already-lowered SSA, with each block's boundary live-ins derived after lowering
  rather than chosen before it (`analyze_local_cache_prefs`).

- Hot canonical locals (GP and FP, each its own bank) are bound to fixed cache
  registers with dirty/valid tracking; a cached local access is a register move,
  and dirty cache registers are published to their frame slots before clobbering
  boundaries and reloaded after.

## Facts

- 2026-02-18 (309d0db2) pitfall: the single-register hot-local cache could not
  hold more than one hot local because its `Option<u32>` signature was
  type-incapable of carrying a second; caching N hot locals needs a sequence of
  prologue transpositions `fp[N]<->fp[K_N]` whose per-register effective indices
  are computed by composing the earlier swaps (each swap can move a later swap's
  target) and whose Wasm-local-to-physical-slot mapping applies every
  transposition in order, to keep `fp[]` addressing correct (code).

- 2026-03-06 (40f9b026) pitfall: the loop-depth multiplier in the
  local-frequency scan cannot be tracked by decrementing on every Wasm `end`
  opcode (it then drops loop depth at the first nested block/if end,
  undercounting locals used deeper in the loop); a boolean control stack
  recording loop-vs-non-loop per frame is needed so loop depth is decremented
  only at the matching loop `end` and preserved across `else` (code).

- 2026-03-13 (4ae8509d) rationale: dirty-tracked cached-local writeback (saving a
  cached local before a clobbering boundary only when its dirty bit is set) was
  dropped here for a correctness issue — writeback fell back to unconditionally
  saving every cached local — and was reintroduced later once the underlying
  defect was fixed; dirty-tracked writeback is live at HEAD, so this is a
  temporary regression of a feature the design kept, not an abandoned idea
  (sourced).

- 2026-03-13 (09aef490) pitfall: helper-backed boundaries (external calls and
  runtime helpers) are slot-based and may clobber cache registers, so they too
  must flush dirty cached locals before the call and reload after; the initial
  lowering emitted the helper call without this synchronization and only direct
  calls had it (code).

- 2026-03-15 (392f8a0c) rationale: the ARM64 backend backs FP transients with
  caller-saved physical FP registers (D3-D7, D16) and FP cached locals with
  callee-saved physical FP registers (D8-D14): transients never need to survive
  a helper or local JIT-to-JIT call so they can be caller-saved, while cached
  locals are persistent and must survive calls so they must sit in the
  callee-saved range preserved by the shared prologue/epilogue (code).

- 2026-03-15 (f94559c7) rationale: tying the FP local cache to callee-saved
  registers only left most of the register file unused; splitting it into a
  callee-saved tier (free across helper calls) and a caller-saved tier
  (spilled/reloaded around helper calls by the portable lowering, which already
  saves every in-use cached local regardless of physical register) exposes the
  whole file, and ordering the cache so usage-weight-sorted hot locals land in
  the free callee-saved slots keeps the common case cost-free (code).

- 2026-03-17 (73f298e0) rationale: reloading every cached local from its frame
  slot at entry loaded slots for non-parameter locals that hold no
  caller-written value (wasted loads, and the slot is not the local's initial
  zero); threading per-local analysis facts (is_param, reads_before_write) from
  the planner lets entry load only parameters, materialize the wasm-mandated
  zero in-register only for observably-zero non-params, and skip both for
  non-params provably written before read (code).

- 2026-03-18 (3778de1c) rationale: top-N-by-access-weight cached-local selection
  cannot respect that a true `i64` consumes two GP budget units on a 32-bit
  target; replacing it with a 0/1 weighted knapsack over the GP budget-unit
  capacity charges each local its width-dependent unit cost and maximizes cached
  access weight within the real register budget (code).

## Moves

- 2026-04-03 (8aab7e14) replaced by [[per-block-residency]]: the old middle
  owned cache and spill decisions across several passes over a whole-function
  local-cache preference table and then reconstructed block live-ins from
  already-lowered cache ops; whole-function hotness is misleading because cache
  usefulness is region/CFG-local, so the middle is rebuilt to choose cache
  residency and transient spills jointly in one pass against a single shared
  dynamic-bank budget, with each block's entry boundary known up front from the
  plan rather than reconstructed afterward (code).
