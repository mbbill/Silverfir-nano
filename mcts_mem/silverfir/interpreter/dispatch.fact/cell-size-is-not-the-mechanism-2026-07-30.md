Descriptor packing was justified by two effects at once: it folds three operand
fields into one load, and it takes the cell from 32 bytes to 16. The second
effect is the one that reads as decisive, because the resource on record as
binding is the interpreter's own instruction stream at 2.7-3.2 of ~4.1 memory
uops per dispatch against 0.05-0.31 for the guest's data. The two are
separable, and the size half was never measured on its own.

It is measured by driving the cell the other way. Doubling the cell to 64 bytes
leaves the field-load count and every dependency chain exactly as they are and
changes only the footprint, so it prices size alone. The change is the cell's
alignment, the cell-stride constant, the four stride sites in the arm64
generator, and the five in the driver; CoreMark's crcfinal stays 0x33ff on both
sides, so the two engines compute the same thing.

Five alternating CoreMark runs, one M4 P core, iterations/sec:

    cell size    mean      spread
    32 bytes     7,890.2     2.0%
    64 bytes     7,872.4     0.7%

Doubling the cell costs 0.23%, inside the run-to-run spread. Cell footprint is
not a binding resource on this benchmark -- a hot loop's cells fit L1D either
way -- so shrinking a cell from 32 bytes to 16 buys nothing by itself, and the
16-byte target that the packing-fit analysis was built around is worth
approximately zero.

What survives is the field-load count, and dropping the size target removes
every obstacle that the 16-byte form had run into. A 32-byte cell has room for
one packed word of slot offsets at offset 8 alongside two full 64-bit words at
16 and 24, so constants of any width, memory offsets, BrTable bases, the
Call/CallIndirect payload that already consumes all three words, and above all
the branch target stay exactly as they are. Branch targets keeping their
absolute post-link form matters most: converting them to a packed cell index
would have re-added the dependency link whose removal is on record at -4.80% of
CoreMark cycles, and no side table is needed for calls or for wide constants.

That leaves the change as one trade on the hot slot/slot/slot path, where the
two cell-field loads today are an `ldp` of a and b plus an `ldr` of the
destination:

    before   ldp x10, x11, [x19, #8]  +  ldr x12, [x19, #24]
    after    ldr x9, [x19, #8]  +  three ubfx

One instruction and two memory uops removed against three ALU uops added. The
sign is not known: a cell load prices at 2-22% and a dispatch-path ALU at
0.5-4.9%, but the ALU figure was measured for an ALU inserted INTO the
dispatch chain between the handler-word load and the branch, whereas these
three sit on the operand path where there is slack. So the modelled +4 to +9%
that assumed both halves is not supported; the remaining half has to be
measured, not modelled.
