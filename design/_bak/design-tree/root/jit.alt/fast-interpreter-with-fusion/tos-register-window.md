# TOS register window (top-N stack values in registers)

Even with `sp` in a register, touching stack memory causes load/store stalls. The
top-of-stack window maps the top N stack slots (`t0`–`t3`) to register arguments
threaded through the whole handler chain; `preserve_none` plus tail-call dispatch
keep them physically resident across handlers.

This is only expressible because the design stayed stack-based: static Wasm
verifiability gives the compile-time stack height at every point, so each handler
is emitted as a depth-specific variant (e.g. `i32_add_D2`). The window needs only
N variants per handler — linear in window depth, not a register permutation.

## In practice

Must:
- Map the top N stack slots to the `t0`–`t3` register arguments, threaded through
  the handler signature and kept resident by `preserve_none` + tail-call dispatch
  (see the sibling interpreter-dispatch/ subtree).
- Emit one depth-specific handler variant per reachable stack depth (the variant
  count is linear in window depth, e.g. `i32_add_D2`).
- Use a 4-deep window, sized against a measured spill/fill cost rather than
  guessed (see facts/tos-spill-fill-overhead-3-percent-coremark.md).

Must not:
- Let the window cost a register-permutation number of handler variants (the
  linear depth-variant bound is the defining property and depends on staying
  stack-based).
