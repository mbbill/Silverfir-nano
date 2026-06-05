- The validation walk doubles as branch-target precomputation: while
  type-checking a function the validator builds a per-function jump table
  (`JumpTable`) that a later interpreter will consult so a branch never has
  to rescan code at run time.

- Each branch produces one table entry recording where it jumps to and how the
  operand stack is reshaped on the jump (`JumpTableEntry`): the resolved target
  offset, the number of stack slots to drop beneath the carried values
  (stack_offset), and the count of values carried across the branch (arity).

- Targets are resolved by deferral, not by a second scan. A forward branch
  (`br`/`br_if`/`br_table`, and the `if`/`else` constructs) pushes an entry with
  an unresolved target and registers that entry's slot index as pending on the
  control frame it targets. When that frame's `end` (or `else`) is reached, the
  decoder's "next opcode offset" patches every pending slot. A backward branch
  to a loop resolves to the loop's first post-header opcode, captured when the
  loop frame is pushed.

- A branch that exits the whole function targets one byte before the function's
  final `end`, an interpreter-facing adjustment so a function-level branch lands
  on the terminating opcode rather than past it.

- After the function is fully decoded (`on_decode_end`), the entries are linked:
  each entry's next-index is set to the following table entry whose own opcode
  lies past this entry's resolved target, scanning forward for forward branches
  and backward for loop branches. The next-index is what an interpreter follows
  to find the next live branch from a jump destination.

## Facts

- 2024-02-02 (ba69c050) rationale: the table is built on the validation pass
  rather than its own pass because validation already tracks the stack
  heights and arities a branch needs (diff).

- 2024-02-03 (0305ca11) pitfall: the else edge was first treated as a
  value-dropping branch with a computed stack offset and arity; an else
  block keeps the stack as-is, so both fields must be zero (diff).
