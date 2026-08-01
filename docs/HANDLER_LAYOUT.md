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
   Refuted; do not retry.
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
