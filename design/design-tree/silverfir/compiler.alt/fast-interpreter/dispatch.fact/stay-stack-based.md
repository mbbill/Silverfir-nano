commit: 4bb1de83

The interpreter keeps the Wasm stack-machine IR rather than converting locals to
SSA virtual registers (the wasm3 / WAMR approach). Two reasons drove staying
stack-based: (1) virtual registers live in memory not hardware registers,
creating load-to-use / store-to-load chains with no real liveness-aware
allocator; (2) on a stack machine fused operand locations are compile-time
constant stack offsets, so concatenating N handler bodies lets the C compiler
eliminate intermediate loads/stores automatically — register-machine operands
are runtime indices loaded from the instruction stream, creating aliasing
barriers the compiler cannot see through, so register-machine fusion needs
hand-written per-pattern handlers or combinatorial specialized variants. This is
the paper's central thesis, backed by clang -O3 godbolt measurements showing the
register-machine form needs 3-5x more instructions for the same fused sequence.
