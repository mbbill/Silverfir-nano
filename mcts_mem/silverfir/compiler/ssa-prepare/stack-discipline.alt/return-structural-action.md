- The shared discipline engine returns its already-classified
  `StructuralAction` to the emit-side caller. SSA primitive emission extracts
  `PrimitiveFill { pop, push }` from that result instead of dispatching through
  `stack_effect` again.

- This preserves the shared structural table as the only policy source, but
  makes the action enum a live value across the generic driver boundary.

## Moves

- 2026-07-23 removed: in the wasmi startup harness's
  fat-LTO bench profile, the candidate's `__text` grew from 3,702,156 to
  3,933,912 bytes. Sixteen alternating quick serial bz2 pairs averaged
  38.627 ms for the exact parent and 39.404 ms for the candidate, a 2.01%
  regression. The implementation was fully reverted; [[stack-discipline]]
  remains in force (sourced)
