The pinned-local payoff law says a pin converts only where it breaks a binding
loop-carried chain, and the three-microbench matrix put a second pin at ~25%
when a loop carries two INDEPENDENT chains and at ~0% when the second local is
fed from the first. Hot loops were separately measured to carry 2.3-3.1 carried
locals each against two pinned registers, which reads as capacity pressure. The
quantity that decides between those two readings -- how many of those carried
locals are on INDEPENDENT chains -- had not been measured, and the l1 verdict it
would qualify was reached on CoreMark alone.

Measured statically over the benchmark corpus. `wasm-tools print` emits one
instruction per line, so loop nesting is recoverable with a frame stack;
innermost means no `loop` appears anywhere inside, including through an
intervening block, which is how LLVM shapes nested loops. Within an innermost
loop body a carried local is one both read and written there, an edge b -> a is
a `local.set/tee a` whose value came from a `local.get b` in the same
straight-line run, and independent chains are the weakly-connected components
over carried locals. Strided induction variables are excluded: a carried local
whose every write is a self-update by a constant or by a loop-invariant local is
a counter, and the tree already measured that an unpinned counter does not bind
because its adjacent write-then-read is covered by write-through acc. Loops are
weighted 10^loopdepth, the convention already used here.

Effective (non-induction) independent carried chains per innermost loop:

    module      loops   carried   chains   effective   eff>=2   eff>=3
    coremark      150      3.40     1.69        1.02     11.4%     8.5%
    bzip2         276      4.11     1.56        1.00      0.1%     0.1%
    sha256        116      3.38     1.56        0.99      7.3%     6.0%
    stream        128      3.49     1.62        0.96      8.5%     6.4%
    mandelbrot    114      4.90     1.97        1.00      0.6%     0.5%
    lz4           148      3.64     1.48        0.97      8.3%     4.9%
    c-ray         140      2.96     1.39        0.98      5.2%     4.1%
    lua           571      3.31     1.44        0.96      8.5%     0.6%
    sqlite       2150      3.04     1.04        1.03      2.8%     0.0%

Carried locals per loop reproduce the 2.3-3.1 range at 3.0-4.9, but only about
one of them is independent, everywhere. So a hot loop offers a second pin no
second chain to break, and the CoreMark l1 result is the corpus shape rather
than a benchmark artifact. At eff>=3 of 0.1-8.5% a third pin has almost nowhere
to convert by this mechanism at all.

The method was validated against a body read by hand before the corpus was run.
mandelbrot's kernel inlines into main and its deepest innermost loop is the
2x-unrolled complex square: locals 11 and 12 are zr and zi, 13 and 14 their
squares, with edges 13<-{11,14}, 14<-{12,13}, 12<-{11,13}, 11<-{12,14}. The
analysis returns one coupled component of those four plus the iteration counter
alone, so effective 1 -- which is what the measured ~0% l1 gain on mandelbrot
already said.

One intermediate result was wrong and is recorded so the correction is not lost.
With induction detection restricted to CONSTANT strides, bzip2 read 1.55
effective chains and 55.0% of loops at eff>=2, which looked like the sharp
exception the corpus headroom facts predict for it. Reading the actual body of
`BZ2_blockSort`'s loopdepth-7 loop showed both survivors were strided counters
whose stride is a loop-invariant local rather than a constant (7 -= 17, 12 -=
15). Generalizing the stride rule collapsed bzip2 to 1.00 and 0.1%, in line with
everything else.

Two bounds on the numbers. The edge rule is a backward scan inside one basic
block, so a dependency routed through memory or across a branch is missed --
that under-counts edges and therefore OVER-counts independence, making these
figures an upper bound, which only strengthens a result of about one. And loop
hotness is static nesting depth, which the tree already records as blind to
branch probability; sqlite's 2150 loops in particular are mostly cold.

What this does not touch is the load-removal case. The top-k census shows the
two shipped pins carry 32.9-65.4% of dynamic local-slot accesses and four would
carry 53.6-80.2%, and the reason those extra loads were dismissed is that
read-mostly slot loads are independent and hidden by the out-of-order core.
That is a wide-core argument, and every pinned-local timing to date was taken on
one M4 P core. On a narrower core those loads are not hidden, and the same facts
that price a wider pinned set out here would not.
