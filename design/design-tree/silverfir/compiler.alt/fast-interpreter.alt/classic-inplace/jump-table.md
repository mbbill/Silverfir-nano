- The per-function branch jump table is built by the function-body validator as
  a byproduct of the single validation walk, not by a separate interpreter pass:
  while type-checking each block and branch the validator emits one entry per
  branch (br, br_if, br_table, if, else) recording the branch's opcode offset,
  its resolved target program counter, the operand-stack slots to drop, and the
  value count to carry across; the interpreter only consumes it.

- Forward branch targets, unknown when the branch opcode is seen, are resolved
  by parking each branch's slot index as a pending slot on the control frame it
  targets and patching it when that frame's end (or else) is reached; loop
  branches resolve to the loop's recorded start PC, and a branch leaving the
  function points one opcode before the function's final end.

- Each entry also records the index of the next jump-table entry at or after its
  branch target; the interpreter advances from one branch site to the next
  relevant entry by index instead of rescanning the body for the next branch at
  run time.

## Facts

- 2024-02-03 (0305ca11) pitfall: the else block's entry first computed
  stack_offset and arity from val_stack.len() minus the frame height, but an
  else carries the if block's exact result type and leaves the stack as-is when
  it falls through to its end, so its entry must use stack_offset 0 and arity 0;
  the height-derived computation produced a wrong drop count for the fallthrough
  (diff).

- 2024-03-11 (6a196f2c) pitfall: when a branch is validated inside an
  already-unreachable control frame the operand stack height is indeterminate,
  so the entry's drop-count cannot be computed from the live height; the
  validator stores a usize::MAX sentinel as the stack offset for branches in
  dead code instead of underflowing the live-height subtraction (diff).

- 2024-03-15 (5350829a) pitfall: jump-table target program counters and a
  function's code offset are module-absolute byte offsets, not
  code-section-relative; the code parser must add the section's base offset to
  each function's code offset or every cross-function branch target is wrong
  (diff).

- 2024-03-15 (5350829a) pitfall: when linking each entry to the next entry at or
  after its target pc, the search predicate is >= (not >): an entry whose target
  pc lands exactly on another entry's pc must select that entry, otherwise a
  branch to a label coinciding with a jump-table slot skips it (diff).

- 2025-10-05 (22db348a) rationale: entries are appended in parsing order, leaving
  them sorted by source program counter; linking each branch to the entry at or
  after its target PC binary-searches that sorted array (forward jumps search the
  suffix after the entry, backward jumps the prefix up to it) rather than
  linear-scanning, cutting link time from O(n^2) to O(n log n) for the same
  result (diff).
