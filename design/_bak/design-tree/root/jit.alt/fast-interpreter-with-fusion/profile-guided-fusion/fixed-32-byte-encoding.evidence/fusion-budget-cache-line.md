---
commit: 4bb1de8
---
Every interpreter instruction is encoded as four 64-bit words (32 bytes): one
word holds the handler function pointer, the remaining three are immediate slots
handlers use freely for local indices, constants, branch targets, memory offsets,
or TOS-encoded stack offsets. On its face this is wasteful — a simple `i32.add`
needs zero immediates and most single Wasm ops use at most one or two, so versus a
variable-length encoding it burns ~4x the memory per instruction.

Fusion converts the width from wasteful to dense. Fusion discovery filters
candidate patterns by immediate capacity: any sequence whose combined immediates
exceed three slots is rejected at discovery time, so every fused pattern that
reaches the final instruction set is guaranteed to fit, and because fused patterns
pack several instructions' operands into the three slots, the slots are
well-utilized (a 3-instruction `get_const_add` uses all three; longer fusions
share or reuse fields). After fusion removes roughly two-thirds of dispatches, the
surviving instructions have their immediate slots filled rather than empty, so the
stream is dense despite the wide format. The fixed width also gives a branchless
fixed-stride decode and clean cache alignment: 32 bytes means exactly two
instructions per 64-byte cache line, with no straddling or padding.

The deciding property is that the ≤3-slot budget is the same constraint fusion
discovery already enforces, so the wide fixed encoding and the fusion pipeline
reinforce each other. The flip side — only 3 immediate slots, a finite pattern
set, a workload-dependent offline discovery step — is exactly the encoding
limitation the later micro-JIT was built to escape.
