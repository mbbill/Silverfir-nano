commit: 959fc3c2

Recovered design document (docs/LEGALIZATION.md) quantifies why the old late
legalizer was abandoned: ~1000 lines of infrastructure existed solely to recover
information already available during lowering -- ~400 lines of storage-flow analysis
to rediscover register types lost after lowering, ~200 lines of hi-half companion-
register tracking, and ~400 lines of GP-bank compaction to re-pack the inflated
register set. The new design eliminates all three: the lowerer knows the LIR value
types (no storage-flow analysis), pair instructions carry their operands explicitly
(no hi-half tracking), and the planner budgets 2 GP lanes per i64 (no compaction).

The contract: above MachineIR everything stays Wasm-shaped and scalar (semantic IR,
planning, LIR, locals, params, returns, call slots, frame layout); the lowerer is the
single place that maps one i64 LirValue to a (lo, hi) machine-register pair, and a
cached i64 local consumes 2 GP cache registers.
