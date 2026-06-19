- The compile-time stack tracker (`SlotTracker`) tracks the location of every
  operand-stack value as a typed Slot (Local / Operand / Temp), not just a
  height.

- Each fast instruction encodes the absolute frame slot index of its destination
  and source operands; handlers read and write operands by slot index relative
  to the frame pointer, with no separate operand stack.

- Constants and preserved locals are allocated compile-time temp slots and
  emitted as a dense constants blob initialized into the frame; operand and temp
  slot placeholders are fixed up to absolute indices by the finalizer.

- Locals, operands, and temps all live in one flat frame addressed from fp;
  there is no distinct stack pointer in the handler signature.

## Facts

- 2025-12-06 (b7b5dc6a) statement: the frame is laid out
  [params][locals][temps][operand stack] — temps sit immediately after locals so
  their base is known at compile start, while the operand-stack base depends on
  the final temp count and is unknown until compilation finishes; the builder
  emits Operand and Temp slots as relative placeholders against fixed marker
  bases and the finalizer rewrites them to absolute frame indices once the temp
  count is final (code).

- 2026-06-14 rationale: encoding an instruction's destination as an independent
  output slot (rather than forcing the result to overwrite its first operand)
  exists because overwrite-first caused real codegen problems — it forced extra
  copies to preserve a still-live first operand, and each such copy needs its own
  handler, growing the handler count, which is exactly the cost an interpreter
  pays per dispatch; a separate dest slot removes those copies (sourced).

## Moves

- 2025-12-06 (b7b5dc6a) replaced [[sto-no-stack]]: STO addressing still copied
  locals and constants through the operand stack and re-derived operand positions
  from one stack-top offset; tracking each value's actual slot
  (Local/Operand/Temp) at compile time lets local.get and const emit no
  instruction at all (the
  consumer reads the source slot directly) and lets every operation carry
  explicit dest/src slot indices, eliminating the redundant copies the
  offset-only encoding could not avoid (code).

- 2025-12-14 (b9499733) replaced by [[sp-stack-machine]]: the slot model encoded
  an absolute frame slot index for every operand of every instruction and tracked
  operand-stack positions plus temp allocation at compile time; switching to an
  implicit operand stack addressed via a stack pointer drops all operand-slot
  encoding and the compile-time slot/temp bookkeeping, leaving handlers to read
  sp[-1]/sp[-2] and adjust sp (code).
