---
commit: 4bb1de8
---
A dispatch scheme where each opcode handler is an ordinary function called
through a table of function pointers gives each handler its own branch-target-buffer
entry (good for prediction) and lets the C compiler optimize each handler
independently (no whole-function register pressure). But every dispatch is an
ordinary call/return: the handler pays a prologue and epilogue, and the platform
ABI forces caller/callee-saved registers to be spilled across the call boundary.
That spilling is fatal to this interpreter's design, because the whole approach
depends on the TOS window and the L0/L1/L2 hot-local cache staying physically
resident in registers across every handler boundary — and a normal call spills
exactly those values on the hot path.

This is the structural reason function-pointer-table dispatch was passed over in
favor of tail-call dispatch. Tail calls keep the same per-handler BTB
independence and per-handler compiler optimization, but with `musttail` +
`preserve_none` they eliminate the prologue/epilogue and the ABI spills entirely,
so the threaded register arguments survive the dispatch. The deciding property is
not branch prediction (function-pointer tables predict fine) but register
residency: an ordinary call discards it, a guaranteed tail call preserves it.
