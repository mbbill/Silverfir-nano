The July 22 execution campaign targeted one explicit invariant: preserve eager
compilation and make sf-nano match or beat at least one comparable Cranelift or
V8 result on every registered execution workload. Measurements below are from
the Codex session log, saved Criterion estimates, generated-code dumps, and
`samply-for-ai` profiles on the same Apple Silicon host.

**Initial ranking.**

At `e5ee16e`, the 20-workload execution sweep (milliseconds, lower is better)
showed that large compute kernels were already competitive, while short loops,
tail recursion, root-call setup, and tiny kernels exposed systematic overhead:

| Workload | sf-nano | Wasmtime CL | V8 |
|---|---:|---:|---:|
| counter-local | 0.275 | 0.250 | 0.253 |
| counter-param | 0.770 | 0.251 | 0.260 |
| counter-global | 0.708 | 0.464 | 0.284 |
| fibonacci-rec | 1.914 | 2.905 | 1.506 |
| fibonacci-iter | 0.506 | 0.503 | 0.513 |
| fibonacci-tail | 1.850 | 0.626 | 0.302 |
| sort | 18.669 | 18.454 | 19.868 |
| prime-sieve | 16.442 | 16.833 | 21.396 |
| matrix-mul | 28.998 | 40.083 | 52.790 |
| nbody | 4.146 | 5.024 | 4.280 |
| argon2 | 68.895 | 77.641 | 56.782 |
| tiny-keccak | 0.0156 | 0.0076 | 0.0065 |
| mandelbrot | 11.698 | 11.786 | 11.811 |
| spectralnorm | 6.080 | 6.304 | 7.099 |
| compression | 4.665 | 4.708 | 4.965 |
| word-count | 0.535 | 0.449 | 0.460 |
| JSON parse | 2.938 | 2.610 | 2.835 |
| reverse-complement | 0.0244 | 0.0162 | 0.0142 |
| regex-redux | 0.0323 | 0.0181 | 0.0181 |
| bulk-ops | 0.558 | 0.463 | 0.453 |

(sourced)

**Retained changes and measured effects.**

- Direct self-tail calls become register shuffles plus a backedge when the ABI
  and local/reference safety preconditions hold (`1a6c8e9`). Fibonacci-tail
  fell from about 1.85 ms to 0.253 ms (-86.3%); nine frame stores and three
  parameter reloads per iteration became two arithmetic instructions plus the
  branch (sourced).

- Typed loop values remain SSA/block parameters and cache/edge state is
  coalesced with one linear predecessor index (`b933a0e`). Counter-param fell
  from 0.749 to 0.254 ms (-66%); the final hot loop is `sub` plus `cbnz`.
  The same change improved tiny-keccak about 11%, spectralnorm 7%, argon2 4%,
  and sort 2.5% in the sweep (sourced).

- Constant rotates lower as immediate/native rotates across backends rather
  than materialize-constant + negate + register rotate. Tiny-keccak fell from
  about 13.73 to 11.87 us (-13.5%) before later changes (sourced).

- ARM64 reclaimed already-preserved x29/x30 capacity and paired callee-save
  stores/loads. The wider dynamic bank reduced Keccak spill traffic; paired
  saves moved word-count from 484.6 to 460.6 us (-5.0%) and sort from about
  18.2 to 11.0 ms (-40%) where repeated internal calls had paid scalar
  prologue/epilogue costs (sourced).

- Store-owned invocation stack/context reuse with exact revisions moved
  regex-redux from 30.393 to 19.199 us (-36.8%); this was the largest
  high-level design bug uncovered and is recorded in
  [[runtime/invocation-cache]] (sourced).

- Re-running store-to-load forwarding after copy propagation and carrying an
  invariant context load through a loop moved counter-global 617 -> 560 -> 415
  us, beating the 461 us Cranelift reference (sourced).

- Direct bounds-checked mem0 calls to libc, instruction-point liveness on that
  infallible path, dead loop-parameter elimination, cached helper targets, and
  shorter/reused bounds proofs moved bulk-ops from roughly 558 to 456.8 us in
  the focused run. A matched sampled run measured sf-nano 498.46 us versus V8
  499.34 us, with 96.7% of sf-nano time inside the same platform
  `memset`/`memmove` implementation (sourced).

- ARM64 inlines small conditional and unconditional edge transfers beside the
  branch rather than bouncing through a distant edge-stub tail. Word-count
  moved from about 496-501 to 451.5 us in the clean profile (about 8-10%)
  (sourced).

- A strict whole-function matcher recognizes Rust's canonical overlap-safe byte
  copy and replaces both scalar directions with MachineIR `MemoryCopy`. JSON
  parse moved 2.729 -> 2.553 ms (-6.4%), while reverse-complement moved 16.43
  -> 5.37 us (over 3x) because the same library loop dominated it (sourced).

- Loop passes carry proven-unchanged frame/context values and hoist only
  invariant native address arithmetic. Regex-redux moved from about 18.69 us to
  17.77-18.06 us after loop-residency and base-address reuse; the final clean
  result met both published references (sourced).

- SSA constant absorption accepts select value operands and bulk-memory
  operands; carrying the newly freed unchanged frame value through the loop
  moved word-count from 477.48 to 450.38 us (-5.7%), 2.9% faster than the V8
  reference (sourced).

- ARM64 fuses an all-ones XOR feeding an adjacent AND into flag-setting `BICS`
  when the transient is dead. The 73%-hot Keccak round block shrank from 944 to
  844 bytes (25 instructions); tiny-keccak moved from 7.52-7.66 to 6.80-6.85 us
  (-6-10%), faster than the 7.25 us Cranelift reference (sourced).

**Rejected or corrected experiments.**

- A global liveness-only save set for every semantic helper broke native
  array/GC/ref/table behavior. Full caller-clobbered preservation is required
  for semantic helpers; only raw infallible libc bulk calls use the reduced set
  ([[runtime/call-boundaries/preserved]]) (sourced).

- Reclaiming cache capacity from an alias-aware second pressure pass compiled
  the small target but overcommitted lowering temporaries in bz2 and FFmpeg.
  Bounding it to one lane was still unsafe for FFmpeg, so the refinement was
  removed instead of adding module exceptions (sourced).

- Sharing one flags result across adjacent regex boolean materializations made
  regex 17.6% slower: on Apple Silicon a fresh `cmp` can break the dependency
  chain and let the boolean computations overlap. Fewer instructions are not a
  win when NZCV becomes the serialized critical path (sourced).

- Several regex experiments changed code size but not time and were removed:
  invariant constant/multiply reuse, duplicate-CMP removal, lexicographic
  control-flow fusion, shared-address record loads, paired record memory ops,
  and partial store-to-load forwarding. Apple Silicon hid those scalar
  instructions under memory/branch latency (sourced).

- Deleting apparently dead cached-cell entry loads was incorrect because those
  loads establish mutable local state used on later paths; the benchmark
  trapped immediately. Cache state cannot be treated as ordinary dead linear
  SSA at MachineIR (sourced).

- The first loop-address proof checked only instruction definitions, not edge
  bindings, and made JSON parsing non-terminating. The retained proof requires
  the identical value on every loop entry/backedge and hoists address arithmetic
  only, never the memory access (sourced).

- Profiling without live JIT symbols produced a convincing but false attribution
  to an unexecuted sorting function. Dump-file concatenation is not live JIT
  layout; optimization work must use the JIT symbol stream or exact live
  function/block addresses (sourced).

**Measurement discipline.**

- Very short cases and long sequential suites are sensitive to host load and
  code-layout features. One full run's late half produced intervals as wide as
  9.4-14.9 ms for a normally 6.2 ms spectralnorm result; those values were
  discarded. JIT-symbol instrumentation also changed generated layout and was
  removed for final timing (sourced).

- When the remaining difference was a few percent, competitors were run from
  the same binary/host state and confidence intervals or alternating runs were
  used. Bulk's matched sampled parity is more informative than comparing two
  isolated sub-millisecond means under different host load (sourced).
