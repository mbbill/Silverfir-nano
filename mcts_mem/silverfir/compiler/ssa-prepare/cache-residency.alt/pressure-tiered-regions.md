- Each region's owned blocks are split into child pseudo-regions ("pressure
  tiers"): maximal runs of consecutive blocks whose interior peak exceeds the
  region's pressure floor. A tier sits at the same loop depth as its parent,
  and its boundary frequency is once per execution of the parent's body rather
  than a loop's per-trip rate.

- Capacity keeps its shape — `budget − max(interior peak over the region's own
  blocks)` — but is computed over a region that no longer contains the spikes,
  and a tier holds exactly the blocks that raised the old maximum.

- The unchanged tree DP gains a third option per cell that it could not express
  before: stay resident through the spike (consuming the tier's smaller
  capacity), be *sheltered* around it (one publish and one reload at the tier
  boundary, losing the accesses inside it), or not be resident in the region.

- An economic gate splits only regions whose accessed-cell demand exceeds the
  capacity today's rule gives them; where residency already fits, a tier admits
  nothing and is pure boundary cost.

- The plan rows, IR vocabulary, rewriter, and machine layer are untouched: a
  tier boundary is an ordinary region-boundary repair emitting the existing
  `CellEnsureCache`/`CellDropCache`/`CellReserveCache`.

## Facts

- 2026-07-25 rationale: a tier is placed at its parent's loop depth so that
  per-block benefit weights are unchanged by the split (code)

- 2026-07-25 rationale: because a tier captures exactly the blocks that raised
  the old maximum, the calm remainder's capacity rises and no block's capacity
  falls (code)

- 2026-07-24 measurement: rejected. Deterministic static gate (`--compile-only`
  native code size, 10-module WASI corpus, three targets, no timing): arm64
  +0.14..+2.47%, armv7 +1.10..+2.01% on the 4 of 10 modules that compile,
  x86_64 +1.84..+4.91% on 9 of 10. No module improved on any target (code).

- 2026-07-24 measurement: the economic gate removed most of the arm64 cost
  (lz4 +3.85% → +0.95%, c-ray +1.17% → +0.48%) and moved the tight targets by
  under 0.1pp — on armv7/x86_64 the gate FIRES, because capacity really is
  binding there, and the design still loses. The refutation is strongest
  exactly where the design was supposed to win (code).

- 2026-07-24 pitfall: unit-count feasibility is not lane-assignment
  feasibility. Every block still individually fit its own interior peak, yet
  armv7 failed 6 of 10 modules in `allocate_cache_binding` because an i64
  cached cell needs a register PAIR on a 32-bit target; x86_64, equally tight
  at 7 allocatable lanes but without pairs, failed only 1 of 10. Modelling
  i64-pair adjacency plan-side is a precondition for widening capacity on any
  32-bit backend (code).

- 2026-07-24 pitfall: the region-max rule is also the channel through which the
  entry block's incoming-param register footprint constrains residency — the
  peak lift in `compute_joint_plan` only binds anything by way of that maximum.
  Splitting the lifted entry block into a tier over-admitted at once ("middle
  cache demand exceeded available dynamic lanes after canonical register params
  were frame-published"); pinning the entry block out of tiering was necessary
  and not sufficient (code).

- 2026-07-24 measurement: the loop bodies grew MORE than the modules did, which
  removes the one defence whole-module size leaves open (cost paid on cold
  boundary paths, benefit collected in loops). Per-block emitted bytes from the
  dump's region table, loop depth from natural loops over the MachineIR CFG,
  x86_64: in-loop bytes vs module total — coremark +6.47/+2.79%, sha256
  +6.41/+3.64%, lz4 +5.64/+3.63%, bzip2 +5.48/+4.05%, stream +5.14/+2.39%,
  fib +5.83/+4.06%; blocks at depth >= 2 grew +3.22..+8.17%. The added code
  lands where it executes most. A static gate CAN answer the hot-path question
  — weight per-block code size by loop depth instead of summing the module
  (code).

- 2026-07-24 measurement: middle-end op counts and native code size moved in
  OPPOSITE directions — x86_64 coremark MIR −1.1% while native code +4.6%. The
  middle end does remove frame operations and the native code still grows;
  rank residency changes on native code within one build config, never on
  middle-end counts (code).

- 2026-07-24 measurement: the cruder variant of the same idea — replace the
  region-max headroom with per-block ENTRY pressure and leave interior overflow
  to discipline's `ensure_capacity` clamp — is not merely worse but infeasible:
  it fails during native lowering on 10 of 10 modules at armv7 and x86_64, and
  3 of 10 at arm64. The clamp prefers spilling transients to dropping
  residents, and its alias-discounted accounting is not the machine's, so
  over-admission is never absorbed (code).

- 2026-07-24 measurement: on arm64 the capacity constraint is not binding at
  all — the maximally permissive entry-pressure rule changes code size by at
  most 0.022% across the 7 modules that survive it, and 4 are byte-identical.
  That bounds the whole capacity-model line of work on arm64, and means the
  target hides the entire benefit while still exposing part of the hazard
  (code).

- 2026-07-24 statement: the governing reading, not yet independently tested —
  ALGORITHM4's benefit term prices a resident cell's accesses but not the code
  residency itself costs (establishment loads, boundary publishes, cached-cell
  block params, added pressure at lowering), so any change that collects the
  capacity residual by admitting more residents pays more than it saves on a
  scarce budget. Same shape as the `algorithm4:call=0` probe, which cut SSA
  frame ops 82% on coremark while raising machine-level cell-home traffic 30%
  and code size 5.9% (uncertain).

- 2026-07-24 pitfall: arm64 native code size carries a ~4-byte per-module ASLR
  jitter (sha256 53448 once, 53452 five times over 6 runs) from
  address-dependent constant materialization; treat sub-0.01% deltas as noise.
  armv7 and x86_64 were bit-stable across repeats (code).

## Moves

- 2026-07-24 replaced by [[cache-residency]]: splitting a region's pressure
  spikes into priced sub-regions widens the calm blocks' capacity as intended
  and still costs native code on every target — and costs it fastest inside
  the loop bodies — because admitting the extra residents costs more than the
  frame accesses it removes; the region-max headroom rule stays, and the residual it leaves on tight budgets is not
  collectable without first pricing residency's own code in the objective
  (code).
