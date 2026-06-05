commit: 0305ca11

The jump-table entry emitted for an `else` first computed a stack_offset and
arity from the else frame's label types and current value-stack height, the same
way a `br` entry is computed. That is wrong for `else`: an `else` block has the
same type as its `if` block and falls straight through to the instruction after
the matching `end`, carrying the operand stack unchanged. The fix sets the
`else` entry's stack_offset and arity to zero — jump to after `end`, reshape
nothing. The lesson is that not every control-flow edge in the table is a value-
dropping branch: the `if`→`else`→`end` fall-through edges move control without
collapsing the stack, and reusing the branch arithmetic for them corrupts the
reshape.
