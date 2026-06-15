- call_indirect and call_ref lowering publishes all argument values to the callee
  frame (publish_call_args_to_frame) before entering the indirect-dispatch cluster;
  the local dispatch path then reloads them from the frame.

- An indirect call never produces a live SSA scalar result; results are delivered only
  through frame slots.

## Moves

- 2026-05-15 (e7402d3e) replaced by [[call-lowering]]: eagerly storing every indirect-call argument to the callee frame before dispatch wastes a store per arg on the local fast path where the args could pass in registers, and forced a reload-shaped frame round trip; capturing live args and threading them as register lane args through the dispatch cluster keeps the local indirect-call hot path register-resident, while the runtime-dispatch fallback block still publishes the carried args to the frame before the runtime helper (diff).
