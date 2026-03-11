// Build-time verification of preserve_none register mapping.
// The build script compiles this to assembly and parses the output
// to verify that argument registers match the JIT's Reg enum.
//
// If Clang/LLVM changes the preserve_none register assignment,
// the build will fail with a clear error message.

#include "vm_trampoline.h"

// Same signature as OpHandler — 11 arguments.
// Each argument is stored to a predictable offset so the build script
// can parse the assembly and determine which register holds each arg.
PRESERVE_NONE
void __jit_abi_probe(PARAMS) {
    volatile uint64_t* out = (volatile uint64_t*)0xAB10;
    out[0]  = (uint64_t)ctx;
    out[1]  = (uint64_t)pc;
    out[2]  = (uint64_t)fp;
    out[3]  = l0;
    out[4]  = l1;
    out[5]  = l2;
    out[6]  = t0;
    out[7]  = t1;
    out[8]  = t2;
    out[9]  = t3;
    out[10] = (uint64_t)nh;
}
