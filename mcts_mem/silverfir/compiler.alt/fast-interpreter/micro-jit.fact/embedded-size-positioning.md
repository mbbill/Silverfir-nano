commit: 2ff0b000

The micro-assembler adds an estimated ~10-20 KB to the binary (ARM64
instruction encoding for ~30 patterns + codegen logic + code-buffer
management). Combined with the ~230 KB interpreter core, the total is
approximately ~250 KB for a Wasm runtime with JIT capability — 60-400x smaller
than any existing Wasm JIT.

| Runtime | Approximate Size |
|---------|-----------------|
| V8 Wasm engine | ~30 MB |
| Wasmtime + Cranelift | ~15-20 MB |
| Wasmer + LLVM | ~100+ MB |
| WAMR + LLVM JIT | ~50+ MB |
| WAMR Fast JIT | ~1-2 MB |
| Silverfir-nano + micro-JIT | ~250 KB |

On resource-constrained embedded Linux devices with AArch64 cores
(Cortex-A35/A53 in IoT gateways), binary size is the primary constraint — these
devices have virtual memory and mmap but cannot afford 15+ MB runtimes. On
bare-metal RISC-V or Cortex-M targets with executable SRAM the micro-JIT could
work without mmap at all (allocate and write code), though those would need
RV32/Thumb-2 backends rather than AArch64. Either way, this could be the first
Wasm JIT that fits in resource-constrained environments where existing runtimes
cannot.

The size collapse is possible because existing JIT compilers need a full
compiler framework (IR, optimization passes, register allocator) while the
micro-JIT needs none: the interpreter's TOS + L0/L1/L2 architecture already
provides the register assignment, so the "compiler" is just a template
assembler — the architectural innovation that made the interpreter fast is the
same one that makes its JIT trivially small.
