The cell-size probe showed footprint costs nothing in time, which was read as
removing the reason to shrink a cell at all. That reading weighs only the
performance axis. Cells are the interpreter's dominant runtime allocation, well
past the engine's own code, so halving one is a memory result in its own right
on the targets this runtime is built for:

    module        funcs      cells   at 32B   at 16B    saved
    coremark         92     20,204     631K     316K     316K
    bzip2            33     28,301     884K     442K     442K
    sha256           21     10,153     317K     159K     159K
    lz4              23     12,288     384K     192K     192K
    mandelbrot       74     17,280     540K     270K     270K
    stream           80     18,031     563K     282K     282K
    c-ray           123     22,004     688K     344K     344K
    lua             778    130,951   4,092K   2,046K   2,046K
    sqlite         1,423    471,015  14,719K   7,360K   7,360K

CoreMark's 631K of cells against a 339K engine sets the proportion: the cell
array is the larger of the two, and it scales with the module while the engine
does not. Cell counts here are static instruction counts plus one pad cell per
function, so they track the linker's array length rather than measuring it.

That reframes the target from "must gain time" to "must not lose time", and
three probes on one M4 P core price the ways of reaching 16 bytes. Each keeps
CoreMark's crcfinal at 0x33ff, and each is a redundant-but-correct edit to the
real handlers rather than a synthetic kernel: a 16-bit truncation of an offset
that already fits changes no value, and a reload of a field into the register
that already holds it changes no value.

    probe                            paired mean   rounds down   verdict
    +3 ALU (ubfx on operand path)         -2.63%         5 / 5   real cost
    +2 memory uops (redundant loads)      -2.10%         4 / 5   soft
    narrow reads (ldrh, 0 ALU)            -0.62%         4 / 8   neutral

The first two are the two halves of packing three fields into one word and
extracting them, and they cancel. Removing two memory uops buys about 2.1% and
the three extractions give back about 2.6%, so pack-and-extract does not pay
for itself in time -- and the ALU figure is the solid one of the pair, measured
5/5 down with no overlap between the two distributions, while the memory-uop
figure was taken on a thermally drifting machine at 8.5% baseline spread.

The third probe is the one that matters. Reading each field with its own
halfword load adds no ALU and holds the memory-uop count where it already is:
today's paired `ldp` of two offsets plus the destination `ldr` is three memory
uops, and three `ldrh` is also three. It measures -0.62% with the sign split
4/8, against the +3-ALU probe's clean 5/5 -- so the difference between the two
is a signal, not scatter. A 16-byte cell whose fields are read narrow is
therefore performance-neutral, and that, not packing, is the shape to build.

What a 16-byte cell still has to resolve is the payload that does not fit eight
bytes. Three slot offsets fit in 48 bits; a branch fits as a 16-bit operand plus
a 32-bit cell-relative target, at the price of one `add` against the live cell
base to restore the absolute form worth -4.80%; a load or store fits as base,
32-bit offset, and destination; and the MovPair cell, which already packs two
destinations into one word, fits exactly. What does not fit is a call, whose
cell already consumes all three payload words including a full callee-cells
pointer, and a constant wider than 16 bits. Calls are 2.18% of CoreMark
dispatches, so a side table indexed by cell index costs them two ALU and one
load where it will not be felt. Wide constants are 5-10% of constant operands,
and forcing those cells to the slow path is the mechanism already used for a
memory offset too wide for a 32-bit host.
