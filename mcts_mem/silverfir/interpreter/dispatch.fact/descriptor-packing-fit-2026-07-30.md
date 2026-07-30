True descriptor packing is recorded as open and as the change that genuinely
reduces the uop count where cell-field pairing does not. It folds the cell's
three operand fields into one word, which also takes the cell from 32 bytes to
16 and halves the instruction stream the chain reads -- the resource measured as
binding at 2.7-3.2 of ~4.1 memory uops per dispatch against 0.05-0.31 for the
guest's own data. Whether it can be done at all is a question about field
widths, answered here statically over the corpus.

    module      slots  slot bits   max cells  index bits   const<=16b  <=20b
    coremark       43         10       4,439          14        90.4%  97.1%
    bzip2         183         12       8,753          15        93.7%  97.8%
    sha256         43         10       4,482          14        93.7%  95.6%
    stream         43         10       4,482          14        89.6%  91.0%
    mandelbrot     43         10       4,482          14        94.7%  96.4%
    lz4            43         10       4,482          14        93.5%  96.1%
    c-ray          43         10       4,482          14        92.2%  95.7%
    lua            40         10      10,921          15        94.2%  95.4%
    sqlite         65         11      22,473          16        95.3%  97.9%

A layout of a:16, b:16, c:16 and 16 bits of flags holds every field.

Slot fields are pre-scaled byte offsets, so their width follows the frame size.
The widest declared params-plus-locals in the corpus is bzip2's 183, needing 12
bits once scaled; the temp range adds the function's maximum operand-stack
depth on top, which LLVM output keeps small. Sixteen bits addresses 8,192 slots,
three orders of magnitude of headroom. This one figure is estimated rather than
exact: the declared count is read from the module, the temp range is not
simulated.

Branch targets are absolute cell addresses after linking and can never pack, so
a packed cell has to carry a cell INDEX instead. The largest single function in
the corpus is sqlite's at 22,473 cells, and an unsigned 16-bit index reaches
65,535.

Constants are the only field that does not always fit, because a Const operand
holds its VALUE inline. 89.6-95.3% of constant operands fit signed 16 bits and
91.0-97.9% fit 20. The remainder needs a side-table index, which is the
mechanism already used where a 64-bit memory offset consumes the bits a packed
index would occupy. That costs those cells back the load the packing removes,
but constant operands are only 4.4-19.9% of loop-depth-weighted instructions and
only 5-10% of those overflow, so 0.2-2% of instructions pay it.

The probe that priced the other half of the change is what removed it from the
design. Storing the handler word as a 32-bit offset would shrink the cell
further but costs one dependent ALU between the handler-word load and the
dispatch branch, and that ALU measures 0.5% to 4.9% depending on benchmark. It
is also unnecessary: an 8-byte handler word plus one 8-byte packed word is
already 16 bytes, so the packing alone reaches the target size while the
dispatch branch's dependency chain stays exactly as it is.
