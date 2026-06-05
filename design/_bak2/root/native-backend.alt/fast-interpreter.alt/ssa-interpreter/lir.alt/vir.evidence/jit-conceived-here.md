---
commit: 50b75e8e
---
The backend design doc introduces VIR explicitly to serve two backends:
"Key Insight: Register allocation happens ONCE (SSA→VIR), then both
backends consume the result." The second backend is a native-code JIT —
named here for the first time in the project — consuming the same VIR with
a second vreg→physical-register allocation pass. The interpreter's window
problem and the JIT's register problem are framed as the same problem at
different binding times.
