- Phi nodes are eliminated by attaching a parallel copy to each CFG edge, stored
  on the function as a map from edge key (pred → succ) to a ParCopy; during
  SSA-to-MIR lowering each phi source contributes a (dst, src) copy to the
  parallel copy on its predecessor-to-block edge.

- The verifier checks that every edge-attached PARCOPY names blocks that are
  genuinely adjacent in the CFG.

## Moves

- 2025-11-06 (f8a05906) replaced by [[phi-elimination]]: a critical edge
  (predecessor with multiple successors into a successor with multiple
  predecessors) has no block to host its copies, so edge-keyed PARCOPY could not
  place them unambiguously; splitting critical edges into landing-pad blocks
  first makes every edge non-critical, so each PARCOPY lands either at the end of
  a single-successor predecessor or the start of a single-predecessor successor
  (code).
