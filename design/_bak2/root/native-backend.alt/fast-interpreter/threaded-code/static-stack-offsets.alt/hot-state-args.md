- Every handler shares one fixed signature carrying the VM's hot state as
  arguments: ctx, pc, the operand-stack pointer, linear-memory base and size,
  and the locals base — kept in registers across the entire chain by the
  `preserve_none` convention.
- Operand values live in memory at the stack pointer; no value lanes travel
  in registers between handlers.
- The IR builder consumes the validator's jump table: branch instructions
  carry stack_offset and arity copied from its entries, and br_table targets
  derive from it.
