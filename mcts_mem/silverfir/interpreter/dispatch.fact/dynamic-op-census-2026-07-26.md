A dynamic op and adjacent-pair census over the eleven-workload wasi corpus,
taken to supply the dynamic-population evidence that fusion-family selection
was otherwise choosing without. Method: a throwaway build denies every op a
native handler except Return, which the driver requires, and the two call
flavours, which the cross-function fixup wires outside the handler table. The
existing per-op slow-exit counter then IS an exact dynamic op histogram, and a
per-cell counter beside it gives the adjacent-pair census and lets every
i32.add be classified by what consumes it. Predecode does not depend on handler
availability, so the instruction stream, the op mix and the branch outcomes are
identical to a normal run; only the timing differs. Shares are within-run
ratios, never counts compared across builds.

Per-op share of dispatches, mean over the corpus with the extremes named:
i32.add 18.6% (33.7 stream, 27.8 bzip2, 27.0 sha256, 1.9 c-ray); all loads
0-26.4% (26.4 sqlite, 22.8 bzip2, 21.8 lz4, 20.1 coremark); all stores 0-14.9%
(14.9 lz4, 11.7 stream, 10.3 bzip2); materialization movs 0-15.4% (15.4
coremark, 13.2 sha256, 11.7 lua-sunfish, 0 stream); fused compare-branches
0.3-16.7%; unfused branches 0-13.4%; unfused compares at most 1.5%, so compare
fusion is close to saturated; globals 0-2.1%, nonzero only on lua and sqlite
and there the shadow stack pointer; select at most 3.9%.

Composition of i32.add, the dominant op, by what consumes it. Constant-address
adds whose result is a temp feeding an ADJACENT memory op are 19.5% of all
dispatches on stream, 7.3 bzip2, 4.6 sha256, 2.8 lua-fib, 2.3 lua-sunfish and
lua-json, 1.5 coremark, 0.7 sqlite, and about zero on lz4, mandelbrot and
c-ray. The same shape with its consumer two to eleven cells away adds 7.8% on
stream and under 0.6% everywhere else. Two-slot adds that the existing fusion
rule declines are at most 1.8% (sha256) and 1.1% (lz4). The remainder, 3-13%
of dispatches in every module, is pointer and index bookkeeping that writes a
LOCAL rather than a temp, which no address fold can absorb because the written
local outlives the memory op.

Top adjacent fallthrough pairs by mean share of dispatches, with the workloads
that carry them: i32.add to i32.add 4.27% (8.0 sha256, 7.8 stream, 7.1 lz4);
f64.mul to f64.add 2.50% (21.7 mandelbrot, 5.8 c-ray); i32.add to i32.load
2.32% (5.8 bzip2); f64.add to f64.mul 1.91% (17.4 mandelbrot); i32.add to
f64.load 1.66% (18.2 stream); i32.add to i32.load8_u 1.64% (6.6 bzip2);
i32.load to i32.add 1.54%; i32.shl to i32.add 1.38%; i64.load to i64.store
1.35% (8.4 lz4); i32.shr_u to i32.and 1.33% (5.0 lua-fib); i32.load to
i32.load 1.31% (5.2 lua-fib); f64.add to f64.add 1.31% (10.1 c-ray); f64.mul
to f64.mul 1.28% (9.8 c-ray).

The dynamic ranking does not match the static one the candidate families were
sized from. The two heaviest dynamic shapes are ALU-to-ALU, whose family was
sized worst on handler cost, and load-to-consumer; and the single heaviest pair
anywhere is f64.mul to f64.add at 21.7% of mandelbrot's dispatches, which is
one multiply-add identity rather than a family.

Adjacent slot-to-slot copy pairs, counted before the pinned-destination
exclusion, are 3.3% of CoreMark dispatches and 5.7% of sha256's. Link-time
pairing removes 0.053%. The gap is the exclusion, not the pairing rule: the
locals these copies write are the hot ones, which are exactly the pinned ones.
