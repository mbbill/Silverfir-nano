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
