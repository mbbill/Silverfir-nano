---
commit: 4bb1de8
---
Profiling on Instruments (macOS) and VTune (Windows) showed that tail-call
dispatch — each opcode an independent leaf function ending in a `musttail`,
`preserve_none` tail call — achieves >95% branch prediction. The single shared
dispatch point of a `switch`/jump-table interpreter thrashes the branch-target
buffer and suffers 95%+ misprediction (consistent with the Ertl/Gregg interpreter
literature); threaded/computed-goto dispatch lands in between at ~50–60%. This is
the measurement that selected tail-call dispatch over switch, computed-goto, and
function-pointer-table dispatch. It is a hardware-microarchitecture measurement,
and the conclusion is tied to having each handler get its own BTB entry.
