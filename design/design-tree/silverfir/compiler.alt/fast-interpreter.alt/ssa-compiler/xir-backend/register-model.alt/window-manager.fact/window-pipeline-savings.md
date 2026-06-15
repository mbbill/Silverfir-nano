2025-10-26, author statement.

The 3-slot window keeps the hottest VRegs resident via a score-based eviction
order: score = use_count * 10^loop_depth / sqrt(live_range_len), computed from
per-vreg metadata. The author reports that the combined pipeline — batched
window stores, score-based eviction, dead-store elision, and branch
fall-through — cut window operations by ~30% and total bytecode by 7-8%.
