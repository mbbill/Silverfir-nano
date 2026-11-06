# Handler layout: what is pinned, what is still on the table

**Status: phase pinning LANDED with the perf-fix series (2026-08-01); the
optimization project below is NOT started.** This file exists so that work
can resume after the RuntimeWorld PR merges without re-deriving the
evidence. The measurements referenced here live in
`mcts_mem/silverfir/interpreter/dispatch.md` (Facts, 2026-08-01 entries)
and in the perf-CI runs of the `dev/probe-*`, `dev/pad-*` and
`dev/fix-align*` branches.

## The established facts

1. The generated dispatch engine is one contiguous `global_asm!` blob:
   ~10,300 handler labels, no per-handler alignment. Any byte-size change
   upstream of a handler shifts every handler after it.
2. Interpreter benchmark scores depend on the ABSOLUTE placement of the
   hot handlers, not only on their content. 48 bytes of unreachable
   padding moved x64-windows sha256 from -14.83% to +0.27% and
   arm64-linux stream-Scale by +103.49% against the same baseline.
3. The giant swings exist only on micro-kernels whose hot set is a
   handful of handlers (stream-*, sha256, coremark). They are threshold
   effects — an L1-I set, branch-target entry, or fetch-window collision
   between two hot addresses is either present on every iteration or not
   at all. Programs that touch hundreds of handlers dilute this to a few
   percent.
4. Per-handler alignment does NOT stabilize results; it re-rolls every
   address. Measured twice (16-byte and 32-byte variants): swings up to
   +89% / -30% in both directions, plus 8-35% of engine text as padding.
   Refuted; do not retry. (The line guard below is a different mechanism:
   it pads only where a handler would straddle a line.)
5. What LANDED instead: two `.p2align 6` boundaries, one at the blob base
   and one between the prelude/stubs and the first handler (at most 128
   bytes on any target). With both sides of a comparison carrying these
   boundaries, link-layout and prelude-size changes cannot move handler
   phase at all. Only a handler-body size change shifts later handlers —
   and that is the one placement change a diff author actually made, and
   should measure.
6. Emission ORDER was already tuned once (hot families first; moving a
   cold family between two hot ones measured ~5% three separate times) —
   see the dispatch node's Items and Facts.

## Line guard (LANDED on x86-64 and arm64-linux 2026-11-05; off on Apple)

`Caps::handler_line` keeps every handler that fits in one 64-byte line
inside a single line: an assembler-evaluated `.fill` before the handler
pads to the next line exactly when the handler would otherwise straddle
one, and adds nothing otherwise. The assembler computes the padding
from the handler's end label at layout time, so the generator needs no
instruction lengths. x86-64 and arm64 other than Apple enable it; Apple
arm64, arm32 and RISC-V pack as before (arm32 and RISC-V are
flash-constrained targets, and CI measures them only for correctness).

This is not the per-handler alignment refuted in fact 4. That moved
every handler to a new phase and left line crossings to chance; the
guard removes one specific cost -- the second fetch on every entry into
a straddling handler -- for all hot handlers at once. Unguarded, 4,713
of 9,711 x86-64 handlers and 4,652 of 10,084 arm64 handlers crossed a
line; guarded, none of those that fit do. Text cost: the x86-64 handler
region grows from 356 KB to 460 KB, arm64 from 326 KB to 417 KB.

Evidence on x86-64:

- Local Intel Xeon, same code shifted by 0-56 bytes of padding ahead of
  the handlers: spectralnorm (wasmi rust-benchmarks) measured 128-154 ms
  unguarded, tracking how many of its 19 hot handlers cross a line
  (4-14), and 113-115 ms guarded at every shift; counter-global
  1.71-2.29 ms unguarded, 1.68-1.78 ms guarded.
- Local CoreMark, 5 interleaved rounds against 92fa012: the guard alone
  +5.0%; the instruction-count series that preceded it (-10.4% retired
  instructions) alone -2.7%; both together +9.8%. Instruction savings
  only paid off once placement stopped re-rolling under them.
- CI run 42001110918 (ec7edf9 against 92fa012, all 46 jobs passing, no
  REGRESSION): x64-linux on EPYC 9V74 CoreMark +47.9%, sha256 +46.2%,
  lz4 +52.7% / +55.4%, mandelbrot +48.3%, c-ray +26.2%, lua-fib +22.1%,
  lua-json +8.6%; x64-windows CoreMark +26.1%, lua-fib +24.8%,
  lz4-decompress +46.8%. The size of the gain depends on the CPU's
  front end: Zen-class runners gain most, the local Xeon least.

Evidence on arm64, CI run 42012501097 (64-byte guard against none):
Neoverse-N2 gained modestly with no regression (bzip2 +1.6%,
funcref-exported-table +1.8%, lua-json +1.3%, wasmi fibonacci-rec
+2.0%, json_parse +3.0%). The Apple M1 runner went the other way
(lua-fib -8.6% and lua-json -4.4% in the primary run, not reproduced by
the cross-run confirmation; c-ray -6.2%, stream-Triad -17.1% as
placement rows). Apple builds therefore pack as before. An open
hypothesis, never measured: Apple cores fetch through 128-byte lines, so
a 64-byte guard only adds padding and a 128-byte one might pay. 9311f27
tried that guard, but none of its darwin interpreter perf jobs got a
macOS runner; the next measurement should compare a 128-byte guard with
none on arm64-darwin alone.

Both CI-confirmed x86-64 regressions of 2026-11-05 (spectralnorm,
counter-global) were placement draws of exactly this kind: identical
instruction counts, and reproduced or cured by padding alone. The guard
does not decollide BTB or L1-I set conflicts; those remain the subject
of the placement project below.

## The unstarted project: profile-guided handler placement

The pinning above makes measurements reproducible; it deliberately does
not choose a GOOD layout, only a stable one. The upside available:

- Realistic target: the ~5-10% class on real workloads (the ~5%
  emission-order swings are the calibration point). The +100% swings on
  micro-kernels are per-benchmark lottery prizes and cannot all be won
  simultaneously — a placement that decollides one kernel's hot handful
  can collide another's.

Method sketch:

1. Build with the existing `interp-count` feature to collect per-handler
   execution counts, ideally per-workload, and (if extended) hot
   handler-to-handler transition pairs.
2. Order and place so the hot core stays compact (small icache footprint)
   and the most frequent handler pairs never share a cache set / BTB-
   indexed address bits. The emission-order list in `interp_gen/mod.rs`
   is the knob; the phase boundaries guarantee the result is measurable.
3. Tune per microarchitecture that matters (CI runners: Zen-class x64,
   Cobalt-100 arm64, M-series darwin), verify on the rest. Set indexing
   and BTB organization differ; expect one compromise order, not one
   optimum.
4. Trust arm64-linux magnitudes over x64 when a single number is needed
   (runner-speed variance fact in
   `mcts_mem/silverfir/runtime/cross-instance-identity.md`).

Pitfall to respect (recorded 2026-08-01): until a candidate layout is
compared against the SAME phase-pinned baseline over 2+ runs, any
measured "gain" is indistinguishable from a lucky draw — that is exactly
how the alignment probes produced fake +64%/+89% improvements.
