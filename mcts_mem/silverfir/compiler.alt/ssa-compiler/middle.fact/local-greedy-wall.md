commit: 0fa96d4e

The state-machine/planner allocator it replaced was a per-block single-pass
local-greedy forward allocator, and that locality was the wall. The eight
diagnosed failures:

- P1: calls flushed ALL live registers — slot-based call ops spilled every
  live-after value.
- P2: no copy coalescing — SameReg constraints emitted defensive copies.
- P3: only constants were rematerializable — missed GlobalGet, AddImm.
- P4: a static cost model with no loop-depth weighting — every spill cost the
  same.
- P5: no live-range splitting — values were spilled and loaded whole.
- P6: "first predecessor wins" edge reconciliation was frequency-blind.
- P8: inclusive live-range endpoints caused false interference at ParCopy
  points.

The chosen successor is a regalloc2-inspired bundle allocator with backtracking
and live-range splitting: half-open [from,to) intervals fix the false
interference, loop-depth-weighted spill weight allocates hot values first,
extended rematerialization covers GlobalGet/AddImm, and — the structural payoff
— bundle merging IS phi coalescing, so SSA deconstruction moves post-RA and
ParCopy stops being a special case (a worked example cut 6 moves/iteration to
0). The validated state machine was kept but demoted from driving allocation to
verifying it; the all-caller-saved calling convention was deliberately left
unchanged for a separate future project.
