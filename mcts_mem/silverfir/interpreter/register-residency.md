- Only fixed-role registers plus a fixed, small set of value registers
  participate in the dispatch contract: the accumulator and the pinned-local
  registers. There is no fill/spill scheme.

- Two locals per function — the link-time most-referenced slots — are
  register-resident across the whole body (the l0 and l1 classes): reads cost
  zero instructions, writes go to both the register and the frame slot
  (write-through), and calls and returns carry both offsets in call cells and
  return records. The registers reload from their slots at every chain entry;
  the slow path never sees them. Register- or flash-constrained targets link a
  reduced class set that drops l1, taking a three-operand op from 100 variants
  to 48.

- One accumulator register carries span-1 temp edges: when a value
  producer is immediately followed by its sole consumer in the same
  control region and its destination was not folded into a local, the
  producer keeps the result in the accumulator and the consumer reads it
  there — both sides cost zero instructions for that operand.

- Every native value handler computes its result into the accumulator;
  memory-destination variants store from it. A consumer reading the local
  written by the immediately preceding same-region instruction reads the
  accumulator instead of the frame slot; the slot store still happens.

- Accumulator marking is a droppable hint: the predecoder retro-marks
  pairs with flag bits while slot fields stay valid, and the linker
  honors a pair only when both sides link to native handlers, stripping
  the hints otherwise — the slow path never knows the accumulator exists.

- The accumulator crosses activation boundaries: the native return path
  stages its result copy through it, a single-result call's first
  result arrives in it, and an adjacent caller-side consumer reads it
  there; the driver relays it through the entry/exit state around slow
  exits, sentinel returns, and host calls.

## Facts

- 2026-07-26 measurement: a third pinned local fits under the hard 512 KB
  assert at 495,204 bytes, 483.6 KB, once canonical operand order has freed
  the room; the naive six-class space models at 556 KB and would panic at
  instantiation (code)
- 2026-07-26 measurement: the third-ranked local is touched by 7.7-39.1% of
  dispatches across the corpus, mean ~21%, against the 15.9% measured for l1
  (code)
- 2026-07-26 measurement: loops that have NO carried local pinned and would
  gain one from a third pin carry 34.2% of bzip2's dispatches, 26.7% of
  sha256's, ~9% of c-ray's and sqlite's, and under 1% everywhere else (code)
- 2026-07-26 measurement: the l2 experiment measured CoreMark-neutral --
  1,161,795 cycles per iteration against 1,172,447 over six interleaved
  samples each -- with the score a wash at ~8,000 despite a 44% larger engine
  (code)
- 2026-07-26 measurement: sha256 read about +2.8% and bzip2 flat, though bzip2
  carried the corpus's strongest predictor at 34.2% (code)
- 2026-07-26 rationale: carried-pin coverage does not predict pinning value
  either (code)
- 2026-07-26 statement: the experiment avoided any call-protocol change by
  restricting the third pin to LEAF bodies, so a caller never holds one, and by
  riding the callee's l2 slot in call-cell `b` bits 19..30, free because an
  argument base needs only 19 (code)
- 2026-07-26 measurement: without that, forcing every call that touches an l2
  function onto the slow path cost CoreMark 36% at 5.6M slow exits, and the
  leaf restriction alone did not help because the exits come from callers
  calling INTO l2 bodies (code)
- 2026-07-26 pitfall: the arm64 MovPair handler selected its operands with a
  catch-all that assumed the class set closed at L1, so a third class fell into
  the load-from-slot arm whose base register is only loaded for slot operands
  and read a stale value -- wrong results, no crash. Catch-all arms over the
  operand classes are a third recurring failure shape beside bit-packing
  collisions and hand-counted branch deltas (code)

- 2026-07-23 measurement: adding the single accumulator moved CoreMark
  4227.7±39.4 → 5454.6±48.9 (+29%) with the dispatch count unchanged —
  per-dispatch cost fell ~2.1 → ~1.65 cycles. Only an estimated 15-20% of
  dispatches carry accumulator flags, so the win is dominated by the
  killed store→load forwarding chains through the frame, not by the
  removed instructions (code).

- 2026-07-23 measurement: corpus temp depth at semantic ops — ≤1 covers
  95.1% aggregate (95.1-98.6% per benchmark), ≤2 covers 98.5%; both-temp
  binops are 0.1%. A second accumulator slot buys ~3.4pp of coverage for
  ~2.7× the handler-variant space; width 1 was chosen (code).

- 2026-07-23 rationale: the governing law, from the xir post-mortem —
  registers are free, statically addressing them is what costs: every
  register-resident operand class multiplies the handler set, which is
  how xir reached >10k handler permutations at 8 usable registers
  (sourced).

- 2026-07-23 measurement: write-through local read-after-write measured
  +0.7%±1.5% on CoreMark (interleaved paired runs) — no significant
  effect; kept because it costs zero handler variants and zero
  instructions (code).

- 2026-07-23 rationale: why write-through is flat — adjacent local reads
  mostly feed predicted branches and sit off the critical path; the chain
  that matters is the loop-carried local dependency cycle (written in one
  iteration, read in the next), which no adjacency-window scheme can
  capture; only a local that stays register-resident across the loop body
  (the l0 class) breaks that cycle (uncertain).

- 2026-07-23 measurement: the BrTable index operand's accumulator edge is
  worth ~8% of CoreMark by itself (5107 → 5529 on a cold machine when
  restored): br_table is the chain's dominant mispredicting branch, and a
  register-resident index shortens every misprediction resolve (code).

- 2026-07-23 pitfall: that edge was silently lost during the write-through
  refactor — the strict-adjacency rework dropped the one marking site that
  needs to run after boundary materialization — and interleaved A/B
  against a sibling variant could not see it because both variants shared
  the loss; pair against the committed baseline, not only against
  siblings (code).

- 2026-07-24 measurement: the l0 class measured +15.3% on CoreMark in
  interleaved pairs against the committed baseline (every pair +799 to
  +927 points); formal cold runs 6485.3±100.5 — ABOVE the historical
  full-fusion interpreter's 6,251 peak, with ~3,100 generated handlers
  (~110 KB) against its 2.9 MB pattern library (code).

- 2026-07-24 measurement: loop-depth-weighted l0 selection (10^depth
  over link-detected back edges) measured 11% WORSE than unweighted
  static counts on CoreMark (interleaved pairs, 6371 vs 5663): static
  depth is blind to branch probability, so a scratch local in one rare
  arm of a switch inside a loop outweighs the local the whole function
  leans on; breadth of use is the better hotness proxy absent profile
  data (code).

- 2026-07-24 measurement: the earlier entry's mechanism clause ("a
  scratch local in one rare arm of a switch") was an unverified reading
  and the per-function diagnostic disproved it: all five CoreMark
  functions that flip picks under depth weighting flip between two
  BROADLY-used candidates, and in the clearest cases the weighting
  displaced a frequently-written local (33r/17w, 21r/17w) with a
  read-mostly one (39r/3w, 36r/1w). The l0 payoff is breaking the
  loop-carried store→load chain, which requires the WRITTEN local;
  read-mostly slot loads are independent and hidden by the
  out-of-order core (code).

- 2026-07-24 statement: a write-biased selector (score = reads +
  k·writes) is the untested follow-up this diagnostic suggests; depth
  weighting with deduplicated per-header brackets picks identically to
  the naive-bracket version, so bracket inflation was not the cause
  (code).

- 2026-07-24 measurement: the write-biased selector (reads + 2·writes)
  flips only two cold functions' picks on CoreMark and measured -3±230
  paired — inconclusive, so the simpler unweighted count stays; the
  write-bias hypothesis remains open for workloads where it would flip
  a hot function (code).

- 2026-07-24 measurement: the l1 class (second pinned local, x14;
  variants 5·5·4 = 100, ~220 KB handlers; 32-byte return records carrying
  both caller offsets) decomposes on CoreMark as +10.7% true residency
  gain MINUS a 3.7% infrastructure tax paid by all code (larger handler
  set pressuring the I-cache, five extra instructions per call/return
  pair) = +7.7% net, consistent in every interleaved A/B/C round. The
  gross gain matches the corpus top-2 heat statistics (l1 static
  references are 78% of l0's); a naive net-only measurement under noise
  had read +3-4% and looked like a bug until decomposed (code).

- 2026-07-24 measurement: the +10.7%/-3.7%/+7.7% decomposition above was
  measured in a session whose machine conditions were later shown
  unreliable (the same binary swung 3684-6815 within minutes; fixed
  A-then-B run order penalizes the second binary under monotone
  degradation). The only clean cold-machine formals bound the l1 net at
  +1.2% (l0 6485.3±100.5 vs l1 6562.9±160.4, overlapping errors) — the
  true l1 net is somewhere in +1..+3% and UNPROVEN against its 2×
  handler size until a quiet-machine measurement exists (code).

- 2026-07-24 measurement: timing-independent dispatch-count modes
  (exact counters, immune to machine noise): per CoreMark iteration
  354,918 dispatches total, 69,234 involve the L0 class (19.5%) and
  56,444 involve L1 (15.9%) — dynamic L1/L0 engagement is 81.5%,
  matching the 78% static prediction, so selection, classing, and
  linking are exonerated (code).

- 2026-07-24 measurement: combining the engagement counts with the
  clean wall-clock bounds (+1..3% net), equally-engaged l1 dispatches
  pay roughly 6× less per engagement than l0's — the open question is
  now a payoff asymmetry, not an engagement gap (code).

- 2026-07-24 statement: hypothesis, NOT yet tested — the payoff
  asymmetry is critical-path structure: the hottest local carries the
  loop's longest store→load dependency cycle and breaking it pays
  wall-clock; the second local's cycles overlapped that chain and were
  already slack, so pinning them removes work the out-of-order core was
  hiding anyway. Testable prediction: a loop with two INDEPENDENT
  carried chains (two accumulators) should show l1 paying like l0
  (uncertain).

- 2026-07-24 measurement: quiet-machine verdict (6 order-balanced
  rounds after the antivirus scan storm subsided, all six positive,
  base ~5570±25 / l0 ~6475±90 / l1 6748±48): l0 over base +16.3% —
  independently re-verifying the original +15.3% — and l1 over l0
  +4.2% (median +4.0%, range +1.9..+6.9%). Combined with the 81.5%
  engagement parity, an l1 engagement pays roughly a quarter of an l0
  engagement: the payoff asymmetry is real, not a measurement artifact
  (code).

- 2026-07-24 measurement: the payoff asymmetry is now pinned by a
  three-microbench matrix (base/l0/l1 ladders, order-balanced): with two
  equal INDEPENDENT carried chains, l0 alone gains only 2.7% (the other
  chain still binds) while l1 completes the set for 25%; with a
  NON-carried second local (fed from the first), l0 alone gains 16% and
  l1 adds ~0%. Pinning pays if and only if it breaks a BINDING
  loop-carried chain — value follows chain criticality, not reference
  counts, which explains CoreMark's l1 at +4% despite 81.5% engagement
  (code).

- 2026-07-24 measurement: a "short" carried chain is not slack — an
  unpinned carried local is store-forwarding-bound (~6 cycles) whatever
  its ALU length (2-op carried chain benched like a 4-op one); and an
  unpinned loop COUNTER did not bind either bench because its
  adjacent write-then-read pattern is already covered by write-through
  acc — the accumulator and pinning mechanisms compose (code).

- 2026-07-24 statement: the identified tax remedy is demand-driven
  handler emission — emit only the variants the instance's linked code
  actually classes, collapsing the 100-variant set back toward the used
  subset (code).

- 2026-07-24 pitfall: two l1 bring-up defects were caught by the
  differential net, not by inspection: a patch hunk that silently failed
  to apply left call cells in the old packing (always assert automated
  edits), and the Select pinned-dst force-slow guard from the l0 round
  captured CoreMark's hottest select once l1 widened the pinned set —
  fixed by giving Select real pinned-dst variants (code).

- 2026-07-24 rationale: the historical l0/l1/l2 rejection was about the
  class-explosion of MULTIPLE register-resident classes; a single
  link-chosen class at 4·4·3 = 48 variants per op stays inside the
  variant budget and needed no predecoder involvement at all (sourced).

- 2026-07-24 measurement: the float accumulator (v16, caller-saved, so
  entry/exit paths never save it; never live across an exit because
  float acc producers only bail to traps) moved the float benchmarks
  mandel −22%, c-ray −22%, stream −17% in one step: float results stop
  round-tripping through the integer accumulator and frame, float loads
  land in the register, float stores leave from it. Acc pairing became
  domain-checked in the predecoder (producer result domain must equal
  consumer operand domain; call results are integer-domain by the
  raw-bits return copy; MovSlot/Select class integer, falling back to
  slots for float values) (code).

- 2026-07-24 measurement: float-pinned locals (v17/v18 as the float
  register file for the same two pinned slots; a slot is float-mode only
  when every writer is float-domain, mixed-writer slots are unpinnable,
  and wrong-domain reads demote to the slot at link) measured stream
  −41%, mandel −12%, lua fib −2.9%, c-ray and CoreMark neutral; the
  synthetic zero-float call microbench pays +5% — two extra untaken
  test-branches per call/return pair, ~1.2 cycles, the design's floor
  (code).

- 2026-07-24 pitfall: int→FP register transfers are expensive enough on
  this core to dominate small handlers: unconditionally reloading the
  float twins with fmov on every call/return cost the zero-float call
  microbench 10%. The fix stamps a per-callee flag into call-cell bit 31
  and a per-caller flag into the recorded l0 offset's bit 0 (both bits
  structurally free), gating the transfers off the integer path
  entirely (code).

- 2026-07-24 pitfall: branch POLARITY on the hot path is a first-class
  cost: with the integer case as the branch TARGET the gate still cost
  11% (two taken branches per pair, fetch redirects); inverting so
  integer code falls through and the fp block sits out of line past the
  dispatch branch cut it to the ~5% instruction-count floor (code).

- 2026-07-24 pitfall: two stage-2 bring-up defects were caught by tests,
  not inspection: a hand-counted tbz delta landed the integer return
  path on the skip branch (recursive_fib exhausted the call stack), and
  the callee-fp flag at cell bit 31 leaked into the argument-base
  extraction (spectest call_indirect exhausted the stack) — hand-counted
  branch deltas and bit-packing collisions are this emitter's two
  recurring failure shapes (code).

- 2026-07-23 measurement: short-span local reads (within ≤2 dispatches of
  the write) are 30.6% of consumed local reads; per-function top-1 locals
  cover 27-30% of them — the open follow-up (a write-through accumulator
  for just-folded local writes, or a single register-local class)
  (code).

- 2026-07-24 pitfall: the first call-result-in-acc build loaded the
  accumulator with a dedicated load in the return handler after the
  result-copy loop; a call-recursive microbench ran 6-8% slower in 4/4
  paired rounds — the native return path tolerates no extra
  store-forwarded load. Staging the existing result copy through the
  accumulator instead (it already loads each result once) removed the
  entire regression at zero added instructions (code).

- 2026-07-24 measurement: call-result-in-acc measured neutral on
  CoreMark (median +0.0% over 6 quiet paired rounds vs the committed
  baseline) and ~1% faster on the lua benchmark (3/4 paired rounds);
  kept at zero cost — the removed consumer load is store-forward slack
  the out-of-order core hides, so the win is deferred to in-order
  targets (code).

- 2026-07-24 pitfall: the call-result-in-acc consumer marking was dead
  on arrival — the link strip pass required the producer to have a
  handler-table entry, but Call/CallIndirect are wired by the fixup
  pass and are absent from the table, so every call-consumer hint was
  stripped and the "neutral" measurement above measured a disengaged
  mechanism. Diagnosed while wiring native call_indirect; the fix
  exempts call producers from the table lookup (every call flavor —
  native, slow, host — delivers result 0 through the accumulator
  relay, so the pairing is always sound) (code).

- 2026-07-24 measurement: with the strip exemption in place,
  call-result-in-acc engages for real: CoreMark +2.2% in 4/4 quiet
  paired rounds against the same baseline that previously measured
  neutral (code).

- 2026-07-24 measurement: the pinned-global experiment (hottest written
  global register-resident write-through in a dedicated register,
  get/set handler rows swapped in by a link special-case, per-entry
  reload) measured ≈0 on CoreMark and ~1% slower on lua, and still ≈0
  vs the committed baseline after relocating its handlers behind the
  hot families. Static counts explain the ceiling: the only mutable
  global real clang modules carry is the shadow stack pointer, at 2:1
  set:get (lua 438/227, coremark 29/14); write-through sets save
  nothing (one extra register move), and gets save one L1-hot slack
  load. Register residency pays for loop-carried read-write chains, and
  no real-module global is one (code).
- 2026-07-25 measurement: ranking locals by the existing unweighted static
  reference count, the third-ranked slot is the most WRITE-heavy of the top
  three on 7 of 8 corpus modules (write share 0.27-0.42) and over 99.6% of its
  write weight is loop-carried, while the read-mostly base pointer that made
  depth-weighting 11% worse first appears at rank FOUR (write share 0.02-0.17)
  (code)
- 2026-07-25 measurement: reference coverage flattens at the same rank the
  write character does -- top2 to top3 adds 10-14 percentage points of weighted
  local references, top3 to top4 adds 2.9 -- so a third pinned local is
  supported and a fourth is not (code)
- 2026-07-25 measurement: a third pinned local would cut modeled memory traffic
  2.4-7.0%, and 6.2-12.5% combined with making the pinned register
  authoritative instead of write-through; it is gated on emitted code size,
  since a sixth operand class overflows the native code buffer (code)
- 2026-07-25 measurement: bzip2's chosen l0 has write share 0.00 -- a pure base
  pointer holding the most valuable register -- while its rank-3 local carries
  5.1% weighted writes, all loop-carried (code)
- 2026-07-26 measurement: the function-wide pin pick is frequently wrong for the
  loop that actually runs -- the share of loop-executing dispatches whose
  loop-hottest local is neither pinned slot is 99.9% on mandelbrot, 59.6% bzip2,
  41.5% lz4, 35.7% stream, 26.4% CoreMark, and under 15% on sha256, sqlite,
  lua-sunfish, lua-json and c-ray (code)
- 2026-07-26 measurement: restricting the reference census to LOOP BODIES raises
  that coverage on mandelbrot from 0.1% to 99.9% and changes the pick for
  functions carrying 100% of mandelbrot's and sha256's dispatches, 37.6% of
  sqlite's and 35.3% of CoreMark's -- and measured -0.1% on mandelbrot and a
  -0.2% median over the corpus. Reference-count coverage is not a proxy for
  pinning value, now the third independent confirmation that value follows chain
  criticality (code)
- 2026-07-26 measurement: the picks that rule swaps are read-mostly on BOTH
  sides -- write share 0.03 to 0.03 on CoreMark, 0.21 to 0.17 on sqlite, 0.26 to
  0.29 on mandelbrot (code)
- 2026-07-26 rationale: a reference-ranked selector cannot reach the
  loop-carried WRITTEN local the mechanism needs (code)
- 2026-07-26 measurement: the existing unweighted reference count already pins a
  LOOP-CARRIED local in most hot loops -- 100% of loop dispatches on mandelbrot
  and lz4, 98.2% lua-sunfish, 96.9% lua-json, 87.5% sqlite, 85.6% lua-fib, 80.3%
  c-ray, 70.1% stream, 63.2% CoreMark, 58.3% sha256, 41.1% bzip2 (code)
- 2026-07-26 rationale: what selector headroom remains sits in bzip2, sha256 and
  CoreMark (code)
- 2026-07-26 measurement: hot loops carry 2.3 to 3.1 loop-carried locals each,
  mean 2.7, against two pinned registers (code)
- 2026-07-26 rationale: the binding limit on pinning is capacity, not selection
  (code)
- 2026-07-26 rationale: that result also prices the expensive form out. Per-loop
  REPINNING is representable and cheap to make sound -- write-through leaves the
  slot authoritative, so a switch is one reload and needs no liveness proof --
  but it would raise the same coverage metric this experiment just showed does
  not convert, so it must not be built until a criterion exists that predicts
  value rather than references (code)
- 2026-07-25 pitfall: the write-biased pin score (reads + 2*writes) was
  recorded inconclusive, but that verdict was reached on CoreMark alone, where
  the traffic model also predicts ~0. The model puts it at -8.0% of memory
  traffic on mandelbrot with two pins and -14.7% with three, so the experiment
  was run on the one benchmark that cannot show the effect (code)

- 2026-07-26 measurement: the share of dynamic local-slot accesses a function's
  top-k most-referenced locals carry, over the corpus, counted by denying every
  op a native handler so the Rust executor sees the whole stream and the counts
  are exact — at k=2, which is what ships: CoreMark 46.7%, sha256 42.5%, bzip2
  32.9%, stream 65.4%, mandelbrot 42.1%, lz4 39.2%. At k=4:
  78.8/63.5/53.6/80.2/78.9/65.6. At k=8: 93.6/85.5/72.5/95.1/100/88.3. At k=16:
  99.0/99.6/88.9/100/100/100 (code)

- 2026-07-26 rationale: that census is engagement headroom only. It raises the
  same reference-coverage metric per-loop repinning was already priced out for
  raising without converting, and pinned write-through means a wider set removes
  loads and not stores — so a wider pinned class is not justified by it, and the
  open question stays the payoff asymmetry rather than the engagement gap (code)

- 2026-07-26 measurement: the register file is not what bounds the pinned set on
  arm64. x1 through x6 are never mentioned anywhere in the generated engine (x18
  is the reserved platform register, and x0 appears only in the entry
  trampoline), so eight pinned locals fit the contract as it stands; four need no
  structural growth either, since the return record's fourth word carries only a
  16-bit offset and a call cell's third field is unused. What bounds the set is
  the governing law's handler multiplication (code)

- 2026-07-30 measurement: hot loops carry about ONE independent loop-carried
  chain each, corpus-wide, once strided induction variables are excluded --
  effective chains per innermost loop 0.96-1.03 across nine modules against 3.0-4.9
  carried locals, with eff>=2 at 0.1-11.4% and eff>=3 at 0.1-8.5%. So a second pin
  has no second chain to break in most hot loops and the CoreMark l1 verdict is the
  corpus shape, not a benchmark artifact; a third pin has almost nowhere to convert
  by this mechanism. Method, validation against a hand-read mandelbrot kernel, a
  corrected intermediate result, and the upper-bound caveat are in
  [[register-residency.fact/independent-carried-chains-2026-07-30]] (code)
- 2026-07-30 rationale: that closes the chain-criticality case for a wider pinned
  set but not the load-removal case -- the top-k census's extra coverage was
  dismissed because read-mostly slot loads are hidden by the out-of-order core,
  which is a wide-core argument, and every pinned-local timing to date was taken on
  one M4 P core (code)
