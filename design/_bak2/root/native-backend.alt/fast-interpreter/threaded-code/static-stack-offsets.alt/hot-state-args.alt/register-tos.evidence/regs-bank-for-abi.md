---
commit: 240fb3d8
---
The header comments record the why for moving lanes out of argument
registers into a pointer-passed bank: on Win64's four-register ABI, seven
scalar args force per-call stack traffic, and the by-value `Next` return
goes through memory. One `Regs*` keeps every handler call at three register
args on all targets; depth is widened to 64-bit to avoid zero-extend
shuffles. The lanes become memory-resident, mutated in place — trusting the
compiler to keep the bank hot.
