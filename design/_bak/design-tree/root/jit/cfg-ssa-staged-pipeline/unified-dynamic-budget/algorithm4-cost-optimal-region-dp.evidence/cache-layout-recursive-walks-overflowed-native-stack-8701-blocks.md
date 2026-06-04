---
commit: ae6fcc9
---
The cache-layout planning that realizes ALGORITHM4 in the machine backend
(`lower_cache_layout.rs`) used recursive walks over long CFG / idom chains. A
synthetic `single_fn_200k.wasm` input creates 8,701 SSA blocks in a single
function, and that was enough to overflow the Windows native thread stack — the
crash was in MachineIR cache-layout planning, not in the region solver itself.
The fix replaced the recursive traversals with explicit stacks and added
regression tests for 12,000-block linear CFGs.

This is the concrete scar behind ALGORITHM4's region/idom traversals: the
formulation is cheap in operations, but the recursive *implementation* of its
companion cache-layout walks did not scale to pathological block counts on a
constrained native stack. It is an implementation-robustness fact (deep recursion
on adversarial inputs), distinct from the algorithm's asymptotic cost.
