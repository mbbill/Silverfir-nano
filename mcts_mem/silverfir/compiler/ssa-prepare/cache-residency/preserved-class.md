- The volatile/preserved class of a cached local is a whole-function
  nomination made by the residency solver: a local is preserved-class when its
  trip-weighted survivable-call tax relief, summed over the regions where it
  has any access benefit, amortizes the backend's per-lane save/restore
  overhead; winners are clamped to the bank's preserved-lane capacity in
  budget units by descending relief with slot index as tiebreaker, and
  reference-typed locals are never nominated (`nominate_preserved`).

- A survivable call is a direct call to a local JIT body — the one call shape
  the machine carries preserved-lane caches across. Indirect dispatch
  (including fixed-local-only tables) and non-local calls are value barriers.

- Nominated residents pay call tax at survivable calls at a residual policy
  factor (the safepoint publish of a dirty cache) instead of the full
  publish-plus-reload freight; volatile residents and barrier calls pay full
  freight.

- The plan's call model is class-aware: the exact walker and the rewriter both
  keep nominated residents resident across survivable calls and schedule no
  post-call re-ensures or backedge repair loads for them; every other call
  clears the resident cache on both sides of the shared discipline.

- First-touch requirement classification (Ensure versus Reserve) treats every
  call as a barrier, survivable or not; class-aware survival governs residency
  only.

- The nomination is published to the machine as the whole-function
  preferred-preserved bit; at every survivable call the machine re-checks the
  physically assigned register's class before carrying, and a nominated
  value-holding cache found in a non-preserved lane is a hard internal error
  — a plan-contract tripwire, not a recoverable state.

- The backend declares the per-lane nomination overhead alongside its
  preserved capacity; a backend with zero preserved capacity makes the class
  inert.

## Facts

- 2026-07-13 (7db708a6) measurement: the class-aware plan contract improved
  all nine corpus modules simultaneously — bzip2 −813, c-ray −1,203, coremark
  −643, lua −3,418, lz4 −500, mandel −1,009, sha256 −509, speedtest1 −18,562,
  stream −745 native instructions (net −27,402, −2.6%); arm64 coremark
  35,969±190 versus 35,891±373 before (level); compile-RAM peaks
  +2.46%/+3.63%/+1.67% on coremark/lua/speedtest1; armv7 output byte-identical
  (preserved capacity 0, class inert) with a same-environment qemu A/B level
  (sourced).

- 2026-07-13 measurement: the synthetic ceiling probe (hot loop with one
  direct local call, 7 hot locals, 6 preserved lanes) went from 14 frame ops
  per iteration to 8 — exactly the capacity-limited floor of 7 safepoint
  publishes plus 1 reload for the local that does not fit a preserved lane
  (sourced).

- 2026-07-13 (7db708a6) pitfall: first-touch requirement classification must
  stay a call barrier even for nominated survivors — classifying through a
  survivable call makes the walker's repair say Reserve while the final-SSA
  requirement derivation says Ensure, and the machine's edge contract check
  then rejects handing a reserved (value-free) lane to a row demanding a real
  value (first hit on bzip2's function 53) (code).

- 2026-07-13 rationale: flipping the old preference bit alone could not
  recover cross-block reloads — the machine's carry works only within the
  call's own block, because the plan rows (then class-blind) forced dropping
  carried caches at the next block boundary; the synthetic probe recovered
  only the same-block counter reload (14 to 13 per iteration) until the plan
  itself modeled survival. Coherence across solver pricing, plan rows, and
  machine execution is the load-bearing property (sourced).

- 2026-07-13 (7db708a6) statement: fixed-local-only indirect dispatch is a
  value barrier here, but the machine layout sim deliberately still credits it
  for lane-assignment stability (a lane kept assigned across the call so the
  post-call reload lands in the same lane); narrowing that layout
  classification to direct-only measured +1,314 instructions on sqlite. Value
  survival and lane stability are different questions with different call
  sets (code).

- 2026-07-13 rationale: the 32-bit backends hold the biggest unclaimed win —
  x86_64 (R14,R15), arm32 (R5-R7,R9), and riscv (s0,s1,s6-s11) bulk-save
  physically callee-saved registers in their prologues yet model them as
  volatile, so every call still kills every cache while the save cost is
  already paid; enabling a preserved class there needs an arm64-style lazy
  per-body save (an internal body-to-body ABI change), not just a config
  entry. armv7 and the planned RP2350 port are the main beneficiaries
  (sourced).

- 2026-07-13 measurement: an instrumented compile of the nine-module arm64
  corpus counted zero nominated-resident drops at the call-site physical
  class re-check and zero scratch-lane evictions of live caches — the
  class-mismatch lazy-reload safety net never fires in practice, so on this
  corpus the plan's survival promise is de facto machine-guaranteed. The
  soft layout bias (preference-mismatch cost + preferred-first ordering in
  the layout sim) is sufficient alignment; the 32-bit i64-pair exposure is
  theoretical until Phase C opens preserved classes there (sourced).

- 2026-07-13 measurement: a falsification experiment pinned WHY the
  class-mismatch net never fires — it is arithmetically unreachable under
  the shipped nomination clamp, not lucky. An adversarial module (15
  simultaneously-live transients exhausting the volatile lanes, a
  non-nominated hot local as preserved-lane squatter, six nominees crossing
  a survivable call in an inner loop) fires it zero times; raising the GP
  nomination clamp by one (7 nominees against 6 preserved lanes) fires it
  exactly once per call site, and execution stays correct through the lazy
  reload (sourced).

- 2026-07-13 rationale: the unreachability is an identity, enforced from
  three sides — the bank's plan budget equals its physical lane count
  (volatile + preserved), the nomination clamp equals the preserved-lane
  count, every lane-assignment point places preference-first, and the
  block-entry preference gate un-squats non-nominees from preserved lanes
  at every boundary — so a nominee can only be denied a preserved lane if
  one of those invariants is broken. The net's standing value is as a
  correctness tripwire for such breakage (today it recovers silently, with
  no counter or assert) and as the forward path for Phase C, where 32-bit
  i64 pairs need two adjacent preserved lanes and adjacency fragmentation
  makes denial genuinely reachable by design (sourced).

- 2026-07-13 statement: the machine's dead-cache reload at a cache GET is
  not only a divergence net — it is the plan's intended mid-block
  first-touch establishment mechanism: the rewriter emits a cache GET with
  no prior ensure when residency begins mid-block, and the machine
  materializes the lane lazily from the frame at that GET (12-1069
  establishment loads per corpus module, all plan-intended; zero were
  divergence) (code).

- 2026-07-13 (8be69337) measurement: Phase C landed on arm32 — 2 preserved
  lanes (EABI callee-saved R9, R5) with the arm64-style lazy per-body save
  at overhead 3. armv7 qemu coremark improved +4.2% (interleaved A/B, six
  pairs, six wins, 2502±27 vs 2400±44); lua fib level; spectest all pass.
  The static shape of the win: MIR ops fell corpus-wide (lua −578, sqlite
  −462 — call-crossing publish/reload traffic gone) while native bytes rose
  ~0.5-1.4% from incidental-clobber saves in cold body preludes — the
  save cost moved off the hot path, exactly what the trip-weighted
  nomination prices. x86_64 and riscv remain preserved=0 (sourced).

- 2026-07-13 (8be69337) statement: on gp32 the machine layout places by
  preference before width — an unnominated i64 pair placed width-first
  could occupy the two-lane preserved run and strand a nominee in a
  volatile lane, the exact state the survivable-call carry rejects as a
  broken plan contract. On 64-bit targets every cached width is 1, so the
  ordering is provably identical there (arm64 MIR byte-equality held on
  the nine-module corpus) (code).

- 2026-07-13 statement: the sha256 −16% regression this contract initially
  shipped with was fully resolved — the cause was not the class contract but
  the M4 same-address store→load pipeline hazard it exposed by removing a
  covering frame reload; with [[counter-forwarding]] eliminating the chain,
  sha256 reaches 277.3±0.8 MB/s under this contract (above the 271 pre-class
  record) and the carried-register loop finally shows its win (sourced).

- 2026-07-13 (e333bec8) statement: Phase C must model 32-bit i64-pair
  adjacency in the plan (nominate only what the preserved segment can
  actually hold contiguously) rather than reintroducing a silent recovery —
  the carry re-check errors on a volatile-lane nominee, so an
  adjacency-blind clamp would fail loudly at the first fragmented layout
  (code).

## Moves

- 2026-07-13 (7db708a6) replaced [[static-cross-count-preference]]: the static
  unweighted cross-count threshold (7, never benchmarked or retuned) gated a
  machine-side carry that could only rescue same-block re-ensures — the plan's
  class-blind kill-at-call had already scheduled the cross-block reloads — so
  a hot loop crossing one direct call paid publish-drop-reload per iteration
  for every cached local; making the class a solver nomination priced in the
  residency objective and a plan-level survival contract removed those reloads
  corpus-wide, all nine modules improving (net −27,402 native instructions)
  (sourced)

- 2026-07-13 (e333bec8) dropped: silent lazy-reload recovery for a nominated
  value-holding cache found in a volatile lane at a survivable call: the
  falsification experiment proved the state arithmetically unreachable under
  the shipped nomination clamp, so reaching it means the budget/clamp/
  preference-first identity was broken upstream — silent recovery hid exactly
  that breakage; the carry re-check now raises a hard internal error (code)
