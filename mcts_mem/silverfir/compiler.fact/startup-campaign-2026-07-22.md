The July 22 startup campaign used the FFmpeg module (14,290 functions) to
separate compiler CPU work from parallel wall time while preserving eager
compilation. Evidence comes from the Codex session log at
`2026-07-22T08:25:46Z` through `08:34:37Z`, Criterion result directories, and
the listed commits.

**Measured outcome.**

- The original local sf-nano FFmpeg startup measurement was 5.831 s, while the
  final committed eight-worker implementation repeatedly measured 1.54-1.77 s;
  the conservative summary is about 1.7 s. The comparable local Wasmer
  Cranelift result was 2.051 s, so eager sf-nano was about 14-16% faster on this
  workload (sourced).

- A temporary build with `MAX_MACHINE_WORKERS = 0` measured 6.92, 6.63, and
  6.78 s (6.78 s median, 6.59 CPU-s median). The same code with eight workers
  measured 1.68 s median, a 4.0x wall-time speedup, while consuming about 10.1
  CPU-s because worker setup, duplicated ABI views, allocator contention, and
  fragmented constant pools are real costs (sourced).

- Across all seven startup workloads, the full run measured sf-nano at 40.919
  ms (bz2), 33.060 ms (pulldown-cmark), 574.554 ms (SpiderMonkey), 1,771.539 ms
  (FFmpeg), 3.323 ms (CoreMark), 12.018 ms (Argon2), and 1.955 ms (ERC20).
  sf-nano beat at least one local Cranelift integration on five of seven and was
  19% faster geometrically than the chosen Cranelift comparison set (sourced).

**Contributions retained.**

- `11761c7` removed the dominant container/algorithmic pathologies: quadratic
  cleanup rescans, whole-function scratch for block-local work, tree sets for
  tiny resident sets, repeated type vectors, linear primitive-pool interning,
  and replicated per-block metadata. The profile fell from 14.22 to 8.79 s,
  about 38% overall (sourced).

- `9d349d1` bounded sink-planner scratch by a block's value range rather than
  absolute whole-function SSA ids. It was not cleanly isolated; the campaign
  estimated 1-3% overall with a much larger late-block memory reduction
  (uncertain).

- `ca56b4b` removed a duplicate release-mode semantic walk used to predict
  cache boundaries and derived them from emitted SSA instead. The removed pass
  occupied about 8% of the profile; replacement work left an estimated 6-8%
  overall gain (sourced).

- `4454fa8` introduced bounded hosted eager workers without changing the
  all-functions-compiled-before-return contract. Four workers moved the
  observed wall time from about 7.77 to 2.53 s (3.1x, 67% lower); `e5ee16e`
  raised the cap from four to eight and moved 2.1-2.4 s to 1.68 s (26-31%
  lower) on the ten-core host (sourced).

- `766f2f5` removed temporary scalar operand/result/alias/budget vectors during
  SSA rewrite: the phase was about 26% faster, estimated 2-4% overall
  (sourced).

- `3753936` replaced dense cache-layout metadata with compact integer lanes and
  dense lookup: the phase was about 5% faster and matrices about 4x smaller,
  under 1% overall (sourced).

- `22350bc` counted planner pressure directly from the type stack instead of
  copying type slices: the planner phase was about 31% faster, estimated 2-3%
  overall (sourced).

- `884d7e0` replaced per-block SSA-use tree maps with sorted flat vectors, added
  a dead-value worklist, avoided duplicate operand decoding, and kept common
  operands on the stack. Alternating A/B runs measured 8-10% overall
  (sourced).

- `0f4a8df` built cache rows in place, referenced parent exit rows rather than
  cloning them, and reused occupancy/addition scratch: the phase was about 12%
  faster, estimated 1-2% overall (sourced).

**Durable lessons.**

- Report eager compiler wall time separately from CPU work. Worker scaling is
  a wall-time optimization with a measurable CPU/memory tax, not proof that the
  serial pipeline itself became cheaper (sourced).

- The highest-value serial wins removed repeated traversal and allocation at
  ownership boundaries between stages; they did not require replacing the
  structured streaming design or introducing a heavyweight optimizer
  (sourced).

- Historical absolute times collected after long builds were thermally
  distorted (an old checkpoint moved from its original 14.2 s profile to 22-31
  s on the warm machine). Old changes were therefore ranked by normalized
  phase/profile deltas, and exact percentages were used only for recorded
  alternating A/B runs (sourced).

**Later loop-peephole regression and partial recovery.**

- After the campaign, loop-frame reuse scanned all blocks and repeatedly used
  linear loop-membership tests before rejecting loops whose exits had no
  compatible reload. On Pulldown-cmark this pass occupied about 75% of active
  serial compiler samples. Commit `fa131723` checks exits first and uses dense
  membership; controlled same-binary measurements fell from about 259 to 59.8
  ms parallel and 375 to 138.9 ms serial without disabling eager compilation
  or the optimization (sourced).

- The post-fix seven-workload run measured 64.561 ms (bz2), 59.104 ms
  (pulldown-cmark), 730.580 ms (SpiderMonkey), 3,365.5 ms (FFmpeg), 4.573 ms
  (CoreMark), 14.672 ms (Argon2), and 2.297 ms (ERC20); the last three use
  isolated all-engine reruns because their first full-run samples were visibly
  heat-skewed. Against the fastest measured Cranelift integration per workload,
  the ratios were 2.98x, 1.85x, 1.50x, 0.95x, 1.00x, 0.86x, and 0.54x
  respectively, or 1.20x slower geometrically (sourced).

- This is only a partial recovery: compared with the campaign's recorded
  last-good sf-nano results above, post-fix startup remains 1.58x, 1.79x,
  1.27x, 1.90x, 1.38x, 1.22x, and 1.17x slower, or 1.45x geometrically.
  Pulldown ablations measured about 39.9 ms with both recent loop passes off,
  47.6 ms with address hoisting alone, and 59.8 ms with both passes plus the
  exit-first repair, so valid-candidate frame-reuse work and address hoisting
  remain measurable startup costs rather than the original compiler pipeline
  itself becoming inherently quadratic (sourced).

- A follow-up serial bz2 profile found another traversal pathology rather than
  inherent code-generator cost: dead block-parameter elimination occupied
  14.9% self / 17.4% inclusive because it scanned instructions per parameter
  and the whole CFG per fixed-point round. Commit `70e165e0` replaced both
  layers with one block scan and a flat reverse-dependency worklist. Repeated
  serial startup fell from 67.180 ms to 56.587 and 57.066 ms (about 15%), and
  the pass fell to 1.0% self / 3.2% inclusive. Against a same-binary Wasmtime
  Cranelift rerun at 45.078 ms, Nano's bz2 ratio narrowed from 1.51x to 1.27x
  without changing eager or serial compilation policy (sourced).

- A second follow-up found loop-frame reuse and loop-address hoisting each ran
  a whole-CFG reachability DFS for every backward edge. Commit `8a89ce77`
  replaced those repeated queries with one shared SCC analysis. Serial bz2
  moved again from 57.066 ms to 51.909 and 52.621 ms (about 8-9%); the DFS
  hotspot disappeared, address hoisting fell from 9.3% to 6.7% inclusive, and
  frame reuse from 6.4% to 4.4%. Together with `70e165e0`, serial bz2 improved
  from 67.180 ms to about 52.3 ms (22%) and the ratio to same-binary Wasmtime
  Cranelift narrowed from 1.51x to about 1.16x (sourced).

- The cooled/isolated seven-workload serial rerun after `70e165e0` and
  `8a89ce77` measured 51.912 ms (bz2), 123.66 ms (pulldown-cmark), 2,309.2 ms
  (SpiderMonkey), 8,213.9 ms (FFmpeg), 4.112 ms (CoreMark), 14.212 ms
  (Argon2), and 2.332 ms (ERC20). Relative to the prior serial Nano row, the
  improvements were 22.7%, 15.2%, 14.5%, 33.9%, 14.7%, 16.2%, and 3.5%,
  respectively: 1.215x faster geometrically, or 17.7% less compile/instantiate
  time (sourced).

- Against the faster serial Cranelift integration recorded for each workload,
  the new Nano ratios are 1.16x, 1.27x, 0.82x, 0.63x, 1.07x, 0.94x, and 0.62x.
  Nano now wins SpiderMonkey, FFmpeg, Argon2, and ERC20 and is 0.90x the
  fastest-Cranelift time geometrically (about 10% faster); the remaining
  serial catch-up targets are Pulldown, bz2, and CoreMark (sourced).

- A Pulldown profile then found middle-end CFG cleanup at 12.46% inclusive,
  with `remove_blocks` alone at 6.38%. Single-predecessor merging compacted and
  renumbered every program side table after each individual merge, then
  restarted the cleanup fixed point. Commit `4b801ebb` computes predecessor
  counts once, absorbs each eligible goto chain without changing block ids, and
  compacts all tombstoned successors once. Pulldown repeatedly moved from
  123.66 ms to 111.8-114.1 ms (about 9%); cleanup fell to 4.14% inclusive and
  `remove_blocks` to 1.63% (sourced).

- The post-`4b801ebb` seven-workload serial row was 51.158 ms (bz2), 112.41 ms
  (pulldown-cmark), 2,192.1 ms (SpiderMonkey), 8,037.3 ms (FFmpeg), 4.136 ms
  (CoreMark), 14.306 ms (Argon2), and 2.340 ms (ERC20). Relative to the
  immediately preceding row, bz2, Pulldown, SpiderMonkey, and FFmpeg improved
  1.5%, 9.1%, 5.1%, and 2.2%; the three small cases moved by less than 0.7% in
  either direction. The geometric improvement was 2.4%. Ratios to the fastest
  recorded serial Cranelift integration are now 1.15x, 1.16x, 0.78x, 0.62x,
  1.08x, 0.94x, and 0.62x, or 0.88x geometrically. Pulldown's remaining gap
  narrowed from 27% to 16% without adding a general allocator or changing the
  eager policy (sourced).

- A post-cleanup Pulldown profile exposed a smaller instance of the same
  repeated-traversal failure mode at the backend boundary:
  `terminator_uses_reg` occupied 3.06% self-time, and 97.54% of its samples
  came from ARM64 backend construction. The backend was traversing each block
  terminator once for every MachineIR register merely to discover which
  physical registers the terminator read. Commit `678501db` introduced one
  canonical terminator source-register visitor and made backend construction
  traverse each terminator once. The verification profile put ARM64 backend
  construction at 0.65% total and its remaining `terminator_uses_reg` sample
  came from a peephole pass, not backend construction (sourced).

- The post-`678501db` seven-workload serial row was 49.481 ms (bz2), 106.54 ms
  (pulldown-cmark), 2,112.7 ms (SpiderMonkey), 7,544.7 ms (FFmpeg), 3.909 ms
  (CoreMark), 13.665 ms (Argon2), and 2.314 ms (ERC20). Improvements over the
  post-`4b801ebb` row were 3.3%, 5.2%, 3.6%, 6.1%, 5.5%, 4.5%, and 1.1%,
  respectively, or 4.2% geometrically. Approximate ratios to the fastest
  recorded serial Cranelift integration are now 1.11x, 1.10x, 0.75x, 0.58x,
  1.02x, 0.90x, and 0.61x (0.84x geometrically); bz2, Pulldown, and CoreMark
  are the remaining near-parity catch-up cases (sourced).

**Serial compiler comparison correction.**

- Parallel startup is not the stable measure of intrinsic pipeline cost. With
  `SF_NANO_BENCH_SERIAL=1`, Nano's eager compiler measured 67.180 ms (bz2),
  145.780 ms (pulldown-cmark), 2,701.3 ms (SpiderMonkey), 12,421 ms (FFmpeg),
  4.819 ms (CoreMark), 16.962 ms (Argon2), and 2.417 ms (ERC20). CoreMark,
  Argon2, and ERC20 use isolated cooldown reruns after the full matrix's two
  multi-minute FFmpeg phases visibly heated the machine (sourced).

- The benchmark's Wasmtime adapter already omits Wasmtime's
  `parallel-compilation` feature. For a true serial comparison, the temporary
  Wasmer adapters additionally set their Cranelift and Singlepass Rayon pools
  to one thread. Against the faster serial Cranelift integration per workload,
  Nano's ratios were 1.51x, 1.50x, 0.96x, 0.96x, 1.26x, 1.12x, and 0.64x,
  respectively: Nano won SpiderMonkey, FFmpeg, and ERC20 and was 1.09x slower
  geometrically across all seven. The parallel table remains useful for
  end-user wall time, but must not be used as the compiler-efficiency headline
  (sourced).

**Natural-loop correction.**

- The SCC optimization fixed repeated reachability traversal but retained a
  semantically weak latch predicate: every numerically backward edge whose
  endpoints shared an SCC was treated as a natural-loop backedge. A dump of
  Pulldown's post-optimization MachineIR showed 11,674 blocks, 1,325 alleged
  loops, and 1,124,306 expanded block memberships. Function 26 alone had 2,629
  blocks, 556 alleged loops, and 877,011 memberships; the generated regions
  contained 4,486 non-laminar overlap pairs. This was repeated work caused by
  misclassification, not register allocation or an inherently superlinear
  compiler pipeline (sourced).

- Requiring the candidate header to dominate the latch reduced the same module
  to 289 natural loops and 4,710 memberships (about 239x fewer); the resulting
  loop sets were laminar with maximum nesting depth four. In the verification
  profile, `natural_loop_nodes` disappeared from the sample and the replacement
  graph analysis occupied about 3.1% inclusive in a low-sample run, versus
  9.15% self-time for loop expansion before the correction (sourced).

- The final serial seven-workload run measured 48.164 ms (bz2), 91.752 ms
  (Pulldown-cmark), 1,882.9 ms (SpiderMonkey), 7,020.9 ms (FFmpeg), 3.861 ms
  (CoreMark), and 13.895 ms (Argon2). Those are improvements of 2.7%, 13.9%,
  10.9%, 6.9%, and 1.2% for bz2 through CoreMark; Argon2 moved 1.7% in the
  opposite direction within the host's small-case noise. ERC20 was also noise
  dominated: identical-code isolated means swung 3.077, 2.227, and 2.358 ms,
  so no directional claim is supported. Pulldown is now about 6% faster than
  the roughly 97 ms fastest recorded serial Cranelift integration rather than
  about 10% slower (sourced).

- A runtime guard compared Mandelbrot after the loop correction against the
  exact parent: the complete post-peephole MachineIR and all emitted function
  code sizes were byte-for-byte identical. The apparent 2.9% execution
  movement in a warm full-matrix run was therefore environmental rather than a
  generated-code change (sourced).

- The next 7,421-sample serial FFmpeg profile found no register-allocation
  hotspot. Preparation occupied 36.0%, MachineIR lowering 32.4%, MachineIR
  optimization 16.9%, native emission 3.6%, and remaining decode/setup work was
  distributed. One avoidable target-policy cost remained inside block-local
  optimization: the 32-bit-only signed pair-multiply recovery pass consumed
  about 30% of block-local peephole time while running on ARM64. Gating it by
  GP width moved the full serial FFmpeg criterion mean from 7,020.9 to 6,943.8
  ms (1.10%) and reduced block-local peephole share from 7.71% to 6.51%
  (sourced).

- The dominance correction changed the result of an earlier rejected
  experiment. Sharing predecessor/SCC inputs had not helped while each pass
  still expanded hundreds of thousands of pseudo-loop memberships; once the
  corrected predicate reduced those closures about 239x, constructing the
  predecessor/dominance graph twice became visible. Sharing one immutable graph
  across the adjacent address-hoisting and frame-value-reuse passes moved
  serial Pulldown from 91.287 to 88.748 ms (2.78%). Bz2 at 46.757 ms and
  FFmpeg at 6,909.7 ms moved favorably but within noise, while SpiderMonkey was
  unchanged (sourced).

- A follow-up tested whether the memmove recognizer's repeated construction of
  temporary edge-argument vectors was a remaining container-level startup
  problem. Comparing edge arguments directly removed up to six small
  allocations for a fully matched candidate, but three serial bz2 means were
  46.591, 46.441, and 46.441 ms versus the 46.757 ms accepted baseline; the
  repeat comparisons reported no statistically significant change. The
  experiment was reverted. The remaining bz2 gap is therefore not explained by
  these small matcher-local allocations and should be pursued in the aggregate
  lowering, cache-planning, and peephole costs (sourced).

- Another follow-up replaced `incoming_param_owns_reg`'s scan of the per-cell
  parameter-state vector with a dynamic-register-indexed ownership bitmap. The
  source-level scan was a plausible hidden cost in the otherwise fixed-budget
  allocator, but three serial bz2 means were 47.038, 46.871, and 46.566 ms
  against the 46.757 ms accepted baseline, and every Criterion comparison was
  statistically flat. The experiment and its extra mirrored state were
  reverted. Incoming parameter ownership is too small or too short-lived on
  this workload to explain the remaining compile-time gap (sourced).

- The residency solver's feasible-state extraction was 7.35% inclusive in the
  serial bz2 profile and materialized both the floating-point values and
  boolean decisions as full `(locals + 1) x (capacity + 1)` matrices for every
  region. Backtracking needs the complete decision matrix, but each value row
  reads only its predecessor. Retaining a dense decision matrix while rolling
  two value rows preserved the exact knapsack and tie-breaking policy and
  removed about half of its matrix initialization (sourced).

- Controlled long-sample bz2 B/A/B means for the rolling rows, exact parent,
  and rolling rows again were 45.421, 46.548, and 44.786 ms, a repeatable
  2.4-3.8% reduction with significant transitions. The seven-workload serial
  breadth run measured 44.495 ms (bz2), 88.073 ms (Pulldown-cmark), 1,826.2 ms
  (SpiderMonkey), and 6,835.5 ms (FFmpeg); SpiderMonkey improved 3.0%
  significantly, while Pulldown and FFmpeg moved favorably by 0.8% and 1.1%.
  The three millisecond-scale cases remained noise dominated. In the
  verification profile, feasible extraction fell from 7.35% to 2.93%
  inclusive and the whole joint-plan builder from 11.58% to 7.32% (sourced).

- Removing the per-block `shrink_to_fit` calls during SSA rewrite was tested
  because final prepared-SSA compaction repeats those calls after cleanup and
  optimization. Serial bz2 measured 44.583 ms versus 44.786 and 44.495 ms for
  the accepted rolling-DP parent, inside noise. The experiment was reverted:
  it did not reduce startup, while retaining the early shrink bounds transient
  rewrite memory before the final whole-program compaction (sourced).

- MachineIR lowering then exposed a small allocation-pattern cost:
  `append_entry_cache_params` constructed an owned one- or two-element vector
  for every value, mapped its ownership metadata, and immediately drained it
  into the destination block-parameter vector. Directly appending scalar or
  GP32-pair parameters reduced this helper from 7.18% to 1.93% of
  `lower_function`. Controlled serial bz2 direct/parent/direct means were
  43.829, 44.416, and 43.856 ms, a repeatable 1.26-1.32% reduction (sourced).

- The full seven-workload serial breadth run after direct parameter appends
  measured 43.773 ms (bz2), 88.154 ms (Pulldown-cmark), 1,799.6 ms
  (SpiderMonkey), 6,685.6 ms (FFmpeg), 3.624 ms (CoreMark), 12.979 ms
  (Argon2), and 2.117 ms (ERC20). Relative to the rolling-DP row, the four
  large cases moved by -1.6%, +0.1%, -1.5%, and -2.2%; bz2's controlled A/B/A
  remains the primary attribution evidence, while the breadth run confirms
  there is no workload regression (sourced).

- The next lowering profile found that `dynamic_reg_available` and
  `is_linear_value_reg` together still occupied 3.40% self-time. Nano's
  allocator remained fixed-budget and local; the cost came from linearly
  scanning all cache bindings and unpublished incoming parameters for every
  candidate register. A compact per-dynamic-register non-linear reservation
  count made both ownership queries indexed while preserving the existing
  cache/parameter policy and live-value safety check (sourced).

- Controlled serial bz2 indexed/parent/indexed means were 42.664, 43.965, and
  42.624 ms, a repeatable roughly 3.0% improvement. In the verification
  profile, `dynamic_reg_available` fell from 1.80% to 0.19% self-time and
  `is_linear_value_reg` from 1.60% to 0.09%, confirming that the scan loops
  disappeared rather than moving elsewhere (sourced).

- The corresponding seven-workload breadth run measured 44.498 ms (bz2),
  86.077 ms (Pulldown-cmark), 1,782.0 ms (SpiderMonkey), 6,641.3 ms (FFmpeg),
  3.649 ms (CoreMark), 13.241 ms (Argon2), and 2.097 ms (ERC20). Bz2 had two
  severe high outliers and the millisecond-scale cases remained noisy, so the
  long A/B/A bz2 result is the attribution evidence; Pulldown, SpiderMonkey,
  FFmpeg, and ERC20 moved by -2.36%, -0.98%, -0.66%, and -1.25% respectively
  with no substantial-workload regression (sourced).

- The block-local peephole profile then showed `RawVec` growth below
  store-to-load forwarding and load-to-load reuse. Both passes intentionally
  run twice, but each invocation discarded its small tracking vector. Moving
  the two trackers into the existing per-function `BlockOptCtx` reused
  capacity across passes and blocks without combining passes or changing
  invalidation/order semantics; the verification profile no longer found
  either pass beneath `RawVec::grow_one` (sourced).

- Controlled serial bz2 scratch/parent/scratch means were 42.664, 42.985, and
  42.654 ms, a repeatable roughly 0.75% reduction below Criterion's default
  practical-change threshold. The seven-workload breadth run measured
  42.985 ms (bz2), 84.925 ms (Pulldown-cmark), 1,770.8 ms (SpiderMonkey),
  6,489.5 ms (FFmpeg), 3.588 ms (CoreMark), 12.702 ms (Argon2), and 2.051 ms
  (ERC20). FFmpeg improved 2.29% significantly, while Pulldown, SpiderMonkey,
  CoreMark, and ERC20 moved favorably by 0.6-1.7%; bz2's controlled A/B/A is
  the attribution evidence (sourced).

- 2026-07-23: entry-cache Ensure-versus-Reserve publication still called the
  scalar first-touch classifier once per cached local, repeatedly walking the
  same block prefix. Classifying the whole entry row in one block scan
  preserved the scalar semantics and measured about 2.4% faster on serial bz2
  and 1.27% on serial FFmpeg against the exact parent (sourced).

- 2026-07-23: the cache-dirty dataflow recomputed every block after every
  change even though only successors can become stale. Replacing those global
  rescans with a successor worklist preserved the same monotone transfer
  function and measured 5.13% faster on serial bz2 and 2.78% on serial FFmpeg
  against the exact parent (sourced).

- 2026-07-23: loop-address hoisting repeatedly scanned the whole block set to
  rediscover the same natural-loop membership. Retaining the already-derived
  loop structure for the later hoist decisions removed those repeated scans;
  exact-parent serial bz2 improved 6.11%, while FFmpeg was statistically
  neutral (a favorable 0.54% point estimate). The change was retained because
  it removes input-scaled repeated work and passed the full release suite
  without changing generated output (sourced).

- 2026-07-23 rejected follow-ups: borrowing cache-layout rows instead of
  cloning them, using binary search for short cache rows, and preallocating the
  decoded-op sliding buffer did not survive exact alternating measurement.
  Unstable sorting in the region solver regressed serial startup by 2.83%.
  All four experiments were reverted rather than accumulating speculative
  container-level changes (sourced).

- 2026-07-23 breadth checkpoint after batched entry classification, dirty
  propagation worklisting, and loop-scan removal: the seven serial Nano means
  were 37.329 ms (bz2), 83.418 ms (Pulldown-cmark), 1,687.4 ms
  (SpiderMonkey), 6,177.4 ms (FFmpeg), 3.424 ms (CoreMark), 12.232 ms
  (Argon2), and 2.032 ms (ERC20). Against single-threaded Wasmer Singlepass,
  the per-row gaps were 10.869x, 9.845x, 8.775x, 8.571x, 9.864x, 8.906x,
  and 5.209x (8.670x geometric mean). Nano's geometric mean improved 3.83%
  from the preceding full matrix, but the remaining order-of-magnitude gap
  confirms that the campaign still needs structural aggregate-work reductions
  rather than only tiny local container changes (sourced).

- 2026-07-23: incoming-edge GP cache layout refinement deliberately keeps two
  ascending rounds, but the second round rescanned every join. The replacement
  preserves that exact schedule while revisiting only a row that improved in
  round one or a successor whose predecessor exit became stale after its last
  visit; the bitmap is allocated lazily only after the first improvement.
  FFmpeg's complete SSA/MachineIR dump remained byte-identical (14,290
  functions, 4,019,624 MIR ops, 32,515,260 code bytes). A controlled
  exact-parent Criterion pair measured 6.571 versus 6.714 seconds (2.1%
  faster); bz2's first matched pair measured 36.52 versus 37.41 ms (2.4%
  faster), but later thermally disturbed samples were inconclusive, so FFmpeg
  is the primary attribution evidence. All 355 release tests passed (sourced).

- 2026-07-23 rejected: the adjacent initial store-to-load forwarding and
  load-to-load reuse passes were factored into independent per-instruction
  transfers and composed into one block traversal, while their later
  post-fusion and post-copy-propagation reruns stayed in place. A focused test
  pinned the subtle sequential-state case, and FFmpeg's full SSA/MachineIR dump
  was byte-identical. The first 30-sample bz2 pair favored the combined scan
  by 1.60% (36.07 versus 36.66 ms), but the repeat reversed slightly (37.77
  versus 37.49 ms). FFmpeg likewise favored the candidate in the first
  long-sample order (6.056 versus 6.450 s) but reversed when the order was
  swapped (6.739 versus 6.592 s), with severe thermal outliers. The refactor
  was fully reverted: removing one of four memory-value traversals is visible
  in a profile but did not produce a reproducible end-to-end gain large enough
  to justify the extra transfer abstraction (sourced).

- 2026-07-23: the opcode macro generated a separate `Result`-constructing
  match arm for every valid discriminant. On AArch64, `Opcode::try_from`
  consequently occupied 3,992 bytes as a jump table plus duplicated result
  writes and accounted for 2.01% of serial FFmpeg profile samples. Matching
  the same complete set of valid discriminants in one arm and then performing
  a checked representation conversion reduced the function to 116 bytes and
  its self time to 0.72%; inclusive decoder time fell from 5.35% to 3.79%.
  FFmpeg's complete SSA/MachineIR dump remained byte-identical and all 355
  release tests passed. Exact-parent serial bz2 was neutral in both orders
  (36.32 versus 36.48 ms, then 36.21 versus 36.20 ms); thermally noisy FFmpeg
  samples were non-regressing but are not used to claim an end-to-end
  percentage. The change was retained on the direct causal profile, exact
  output equivalence, and four-line implementation surface (sourced).
