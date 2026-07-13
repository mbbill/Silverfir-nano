- A joint planner decides which canonical locals are worth keeping resident in
  dedicated cache registers in each region of a function, by minimizing a
  single weighted cost function that trades access benefit against call tax and
  per-boundary transition cost, subject to per-region register capacity
  (`ALGORITHM4`).

- The region tree the allocator optimizes over comes directly from Wasm's
  well-nested structured control flow: the root plus one region per Wasm loop
  (blocks and ifs create no region), with no SCC discovery or heuristic region
  formation.

- Residency is decided per region rather than per block; the chosen resident
  set is stable across edges instead of churning ensure/drop traffic at block
  boundaries that disagree.

- The planner scores locals by per-block access count, feeding the region-tree
  DP; it respects separate GP and FP budgets and charges an `i64` local as two GP
  units on 32-bit GP targets.

- Locals proven written-before-read at entry scope skip cache-register
  zero-initialization, while locals that may be read before a write
  materialize the Wasm-mandated zero (`reads_before_write`).

- A separate lane-assignment phase maps the chosen residents to physical
  register lanes with sticky inheritance, leaving holes after drops instead of
  compacting, and remapping a resident's lane only when unavoidable.

- Live transient SSA values and resident cached locals share one per-bank dynamic
  budget; for each bank `live transients + resident cached locals <= total
  dynamic budget` must hold at every program point and block boundary; GP and FP
  hot canonical locals are kept in separate GP and FP local-cache banks.

- Canonical local accesses keep frame-slot identity throughout the middle-end;
  the middle-end publishes exact per-block cached-slot rows (never storage
  kinds), and which fixed cache register mirrors which slot is decided at
  machine lowering — register-cached locals are mirrors of their canonical slot
  homes, never replacements for them.

## Facts

- 2026-04-06 (0b5d2ea0) statement: each local is solved as an independent
  per-local tree DP over the region tree, bottom-up with Lagrangian capacity
  prices (a few iterations) and a final feasibility projection; this is cheap
  because the region tree (root plus one node per loop instruction) is small, so
  no SCC discovery or heuristic region formation is needed (code).

- 2026-04-03 (8aab7e14) rationale: each non-entry block picks one canonical
  incoming edge whose exit boundary it inherits for free (single-predecessor ->
  that predecessor; loop header -> the backedge, not the cold preheader; ordinary
  merge -> the hotter predecessor); only the non-canonical edges pay boundary-repair
  cost, which keeps hot loop boundaries free of ensure/drop traffic (sourced).

- 2026-04-06 (c452a0ff) pitfall: the solver's per-region capacity headroom must
  account for the worst-case live-transient pressure point inside a region, but
  the op plan recorded only each op's before-state; capturing the after-state too
  closed a peak-pressure gap where the highest-pressure point fell after an op's
  entry snapshot and was under-counted (code).

- 2026-05-15 (e7402d3e) pitfall: the undamped per-region price step `1/cap(R)`
  let tiny-capacity regions oscillate between overfull and empty across price
  iterations, so the final iteration could land on a zero-price state that ignores
  capacity competition and produced an unstable cache layout; the step is now
  damped by an iteration factor so prices settle instead of ringing (code).

- 2026-05-15 (e7402d3e) pitfall: the feasible-state projection DP initialized
  every capacity level of its base row to 0.0, so a reserved-but-unused
  region-capacity level was reachable for free and the best-value backtrack could
  reconstruct a weaker resident set; only used=0 is now feasible at the base row
  (the rest stay NEG_INFINITY) so the backtrack reconstructs the best capacity
  choice — a separate slip from the price-step oscillation, in Step 6's knapsack
  base case (code).

- 2026-04-08 (47daba23) pitfall: a cached `i64` local on a 32-bit GP target lives
  in a register pair, so a cached `local.get`/`local.tee` cannot be lowered as a
  source-alias of a single cache register the way it can on 64-bit; the alias
  optimization carved an aliased single register that the `i64` slot load/store
  then mis-read under high register pressure, so on gp32 an `i64` cached get/tee
  now materializes a real linear pair from the frame slot instead of aliasing the
  cache (code).

- 2026-04-13 (a3a7a102) measurement: holding a fully-materialized per-block
  local-access region for every block dominated planner memory; recomputing the
  current block's region on demand from a reused scratch buffer and retaining only
  a compact per-block summary (ranked slots + entry/hot scores) cut ~36M of
  compile-time memory with no change to the residency scoring it feeds (code).

- 2026-04-28 (a50023c5) rationale: the joint planner had been over-built with
  several speculative passes reachable only from `#[cfg(test)]` code (a per-op
  OpInfo table, a whole-program entry-stack-shape pass, entry-stack/transient-symbol
  region analyses, and a block-open scoring model with read-first/write-first/reuse
  bonuses); the production region solver needs only per-CFG-block local access
  counts, so these passes and the whole hotness/ranking scoring were deleted and
  the summary collapsed to a single per-slot access_count. Lesson: a future
  revisit of block-open admission policy should start from the region solver's
  actual inputs, not resurrect this scoring scaffold that scored values production
  never consumed (code).

- 2026-04-06 (a50a44d4) rationale: the region DP (recovered draft
  middle/ALGORITHM4.md, consolidated as docs/ALGORITHM4.md) derives per-loop
  entry/exit frequency as header `block_weight / assumed_trip_count` (default 8,
  root = 1), and the constant is deliberately left inaccurate — it only sets the
  ratio between per-iteration benefit and per-entry transition cost, so no real
  trip-count profiling is warranted; a future implementor should not invest in
  measuring loop trip counts to improve residency, only in the relative weighting
  (sourced).

- 2026-04-07 (41a02194) rationale: ALGORITHM4 supersedes ALGORITHM2 (one global
  resident set) and ALGORITHM3 (root set plus per-loop override), which treated
  residency stability as a structural constraint and so ignored ensure/drop
  transition cost and produced massive boundary churn; the global-set and
  root+loop-override cases now fall out of the single region-tree cost objective
  rather than being hand-coded — recovered design doc in [[cache-residency.fact/algorithm4]]
  (sourced).

- 2026-04-14 (a6f07b08) measurement: a block-local SSA CSE pass for LocalGetSlot
  reads was added then reverted within a day; the motivation is that ALGORITHM4
  prices a hot read-only local out of residency under a tight GP budget (e.g.
  x86_64), and the post-regalloc MIR peephole cannot recover the redundant loads
  because the backend allocator has already reused the load's destination
  register before the next read — the pass deduped repeated slot reads into one
  SSA value per block behind a pressure-guard hoist cap, reportedly passing all 11
  WASI benchmarks with +4-14% gains (Lua highest) and shrinking code size
  everywhere (code).

- 2026-06-14 rationale: the LocalGetSlot CSE pass was reverted for a correctness
  issue — machine/ is handed linear SSA plus a local cache, but the local cache
  is the deliberately non-linear part of the model, so deduping its reads into
  one long-lived SSA value needs real lifetime tracking, which drags the engine
  back toward the traditional register-allocator problems (spill/interference/
  coloring) nano was built to avoid; the gain did not justify reopening that door
  (sourced).

- 2026-05-24 (30779e5d) pitfall: ALGORITHM4's entry-block capacity counted only
  operand-stack transient pressure and ignored that incoming parameters arrive
  pinned in their GP/FP argument lanes until consumed, so on a tight GP budget
  (x86_64: 7 allocatable minus 4 argument lanes) cap(R_root) over-admitted
  residents the machine layer could not bind; the entry block now lifts peak
  GP/FP pressure by the simulated argument-lane footprint before computing its
  cap, and arm64's wider register file is why the bug stayed hidden there (code).

- 2026-06-20 correction: the 2026-04-08 (47daba23) Fact above was reversed on
  2026-04-27 (65ccf38f) — gp32 `i64` cached `local.get`/`local.tee` again
  source-aliases both cache lo/hi registers rather than materializing a real
  linear pair. The "from the frame slot" wording in that Fact was also
  imprecise: the Apr-8 fix snapshotted from the cache registers, not the frame
  slot; only the non-cached `LocalGetSlot` path actually loads from the frame
  (code).

- 2026-06-24 measurement: an ALGORITHM4 edge/call/trip policy sweep over
  coremark, sha256, c-ray, and lua found that raising call tax to
  `algorithm4:call=1.5` reduced final native instruction count on all four
  modules versus the current default (unweighted average -1.027%, total
  instructions -1.371%), while lowering call tax, removing call tax, removing
  edge cost, and the tested edge multipliers all regressed final native code;
  seven edge-policy cases improved middle metrics while worsening final native
  instructions, so cache-residency tuning must rank by final disassembly rather
  than SSA/MIR/cache-op counts (code).

- 2026-06-24 measurement: the primal-final-extraction experiment (normal price
  iterations, then recomputing final DPs with zero lambdas before feasible
  extraction) tied lua but regressed coremark by 24 native instructions (+0.157%)
  and sha256 by 8 (+0.049%), exactly matching the `simplify_no_prices` result on
  the same corpus (code).

- 2026-06-24 rationale: because primal final extraction and no-price extraction
  produced the same final-code regressions on coremark and sha256, the current
  dual-priced final extraction is not the first ALGORITHM4 component to simplify
  or fix on this corpus (code).

- 2026-06-24 measurement: removing ALGORITHM4 price iterations alone
  (`simplify_no_prices`) was nearly neutral but not beneficial on coremark,
  sha256, and lua — lua was unchanged, while coremark regressed by 24 native
  instructions and sha256 by 8 — so final feasible extraction carries most of
  the value on this small corpus but price iteration removal is not yet a safe
  simplification (code).

- 2026-07-12 statement: the named diagnostic residency policies behind the
  2026-06-24 sweep facts (simplify_no_prices, simplify_no_edge_cost, the
  diagnostic static/none policies) were retired from SF_CACHE_POLICY once the
  experiment campaign concluded; the parameterized algorithm4 form
  (trip/iters/edge/call) remains, so those dated sweep results are historical
  record and no longer reproducible by policy name (code).

- 2026-07-12 measurement: objective-term ablations on the middle-v2 pipeline —
  with the price-free solver, per-region adaptivity beats one global static
  resident set by 3.4% on arm64 coremark and 1.5% on armv7 qemu coremark
  (concentrated in multi-phase functions such as BZ2_compressBlock); removing
  call tax adds ~156k native instructions across the 9-module corpus; removing
  edge cost adds ~30k; disabling local caching entirely costs 43% of arm64
  coremark — every objective term is load-bearing even where the pricing was
  not (sourced).

- 2026-07-12 (748c8416) statement: the `algorithm4:iters=` parameter was
  retired with the pricing deletion recorded at
  [[cache-residency/capacity-search]]; the parameterized policy form is now
  trip/edge/call (code).

- 2026-07-13 pitfall: the policy-sweep harness (explore_cache_policies.py)
  runs a prebuilt target/debug CLI and never rebuilds it; two captures taken
  against a stale binary measured nothing and read as "corpus-neutral" until a
  probe function traced the discrepancy. Rebuild the debug CLI before every
  capture (sourced).

- 2026-07-13 measurement: qemu-user scores are comparable only within one VM
  session — the same armv7 binary scored 2,528±34 in one colima session and
  1,751-2,089 across runs in the next; byte-identical builds A/B level inside
  a session while the absolute level moved 20%+ between sessions (sourced).

- 2026-07-13 pitfall: the ir-dump file layout is NOT the executed layout —
  live JIT code places each function at a different base than
  native_code.bin's concatenation (per-function shifts of ~0x4d4-0x68c on the
  sha256 module), so symbolizing sampled PCs against dump offsets attributed
  every hot instruction to the wrong function and produced an entire
  plausible-but-false stall analysis. Symbolize runtime samples against a
  live capture (selfsample dylib: thread_get_state + rwx-region dump) or
  pattern-match block bytes to derive per-function shifts (sourced).

## Moves

- 2026-04-06 (0b5d2ea0) replaced [[per-block-residency]]: choosing a resident set
  independently per block minimized per-block access cost but ignored transition
  cost at edges, so the cache set churned at almost every block boundary; solving
  residency per region on the Wasm loop tree by minimizing one weighted cost
  (benefit minus call tax minus per-boundary transition cost subject to
  per-region capacity) makes whole-function stability and loop-specific overrides
  emerge instead of being fought (code).
