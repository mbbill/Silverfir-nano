---
commit: 296a489, a50a44d
---
The engineer first tried a static local/dynamic register split and "got quite good
results," but it hit a ceiling: it could not assign a register when a local lives
in a hot loop. Separately, `PRESSURE.md` (measured across 9 WASI benchmarks)
documented the static model's waste: a fixed two-bank split per register class
decided once per function and never changed — ARM64 had 13 GP local-cache regs
(x23–x28, x9–x15) plus 9 GP transient regs — so if a block only used 2 transient
values, the remaining 7 sat idle and could not be reused for local caching, while
local caching was simultaneously capped at a function-wide set.

The hot-loop register-assignment ceiling is the deciding fact that drove
abandoning the static split for a unified dynamic budget; the idle-transient waste
is the corroborating measurement. Part of this is a hard structural limit
(hot-loop locals) and part is a measured inefficiency (idle banks).
