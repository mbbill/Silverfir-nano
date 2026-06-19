- A standalone module validator type-checks a parsed module on the target
  before instantiation: entity type references, function-body operand/result
  types, control-flow nesting and branch targets, start-function signature,
  tag result types, and export-name uniqueness (`Validator`).

- The standalone validator is a separable component selected at build time
  (`sf_module_validator`); the compilation pipeline independently re-derives
  the types it needs while decoding, and a build can verify with the validator,
  with the decoder's own checks, or both.

- Each local function is validated by implementing the decoder's handler trait
  and running the decode pass over its code (imported functions are skipped);
  validation rides the same single decode of the body that other consumers do.

- Validation is the spec's stack-typing algorithm: a value-type stack plus a
  control-frame stack where each frame records its result/param types and the
  stack height at entry; values are pushed and popped against per-instruction
  type expectations ([[stack-pop-expectation]]).

- Unreachable code is handled stack-polymorphically through an `Unknown` value
  type that is the spec's bottom: `Unknown <= T` for every type but never the
  reverse; a pop in unreachable code returns `Unknown` and satisfies every
  later constraint.

- While walking a body the validator tracks the peak operand-stack depth and
  stashes that max height onto the function's `FunctionSpec` alongside its jump
  table; a backend can size the operand stack without re-walking the body.

## Facts

- 2024-02-01 (57ae0b79) pitfall: the stack-polymorphic underflow guard compared
  current_frame.height() == val_stack.len(), letting a pop past the frame's base
  slip through when the stack had grown unevenly; it must be height() >= len()
  so any descent to or below the frame base traps as a stack underflow (code).

- 2024-02-01 (57ae0b79) pitfall: ControlFrame::new hardcoded height: 0 and
  unreachable: false, discarding the arguments its callers passed, so every
  pushed frame recorded a zero base height and a reachable state; the
  constructor must store the passed-in height and unreachable for
  stack-polymorphic typing to be correct (code).

- 2024-01-30 (80306a9a) rationale: the value-type stack uses an Unknown variant
  (added as ValueType 0x41) that pop returns once a frame is marked unreachable
  and that compares-equal to any expected type — this is how the spec's
  stack-polymorphism in unreachable code is implemented (code).

- 2024-03-07 (c3c22fac) pitfall: the memarg alignment immediate is a base-2
  logarithm of the alignment, not a byte count; validating it as align >
  size_of::<T>() instead of 2.pow(align) > size_of::<T>() accepts misaligned-hint
  encodings the spec rejects as invalid (code).

- 2024-03-09 (ffd9b838) pitfall: marking a control frame unreachable must
  truncate the operand-stack model back to the frame's base height, not merely
  set a flag; leaving stale entries above the base makes later instructions
  type-check against a non-polymorphic stack and miss invalid modules the
  unreachable rule should accept-then-reset (code).

- 2024-03-14 (e952da4e) pitfall: entering a block must pop the block type's
  parameters off the operand stack before pushing the control frame; the loop
  arm pushed the frame without popping params while block and if did, so a loop
  with a non-empty parameter type validated against a wrong stack (code).

- 2024-03-15 (ed2e28e0) pitfall: an entity's declared type must be read from
  whichever LinkableData arm holds it (Imported or Spec); reading it only via
  the local Spec arm errors for any imported entity, so a body referencing an
  imported global, table, or function failed validation (code).

- 2024-03-15 (2c3f1683) pitfall: an else-less if whose result types are
  non-empty was wrongly rejected; the implicitly-empty false path must leave the
  same stack as the true path, so the correct rule compares the block's params
  against its results, not its results against empty (code).

- 2024-03-15 (b22db5c9) rationale: the validator never re-checks an immediate's
  variant — the decoder has already guaranteed the variant matches the opcode,
  so the per-arm wrong-variant branches are dead and collapse to an unreachable
  extraction; a malformed immediate is the decoder's responsibility, never the
  validator's (code).

- 2024-03-17 (c18c21b2) rationale: limited resources keep their values in native
  units (pages for memory), the page-to-byte conversion moved out of spec
  construction into allocation, and each kind carries a per-kind default ceiling
  (memory 65536 pages, table u32::MAX entries) bounding both min and max (code).

- 2024-03-19 (7254b65a) pitfall: a RETURN targets the function body's outermost
  control frame, so its expected types come from control_frames[0], not the
  innermost frame the earlier code read (code).

- 2024-03-19 (42f220b8) pitfall: pop_vals must collect the value types actually
  popped from the type stack, not the expected types it was given; under
  stack-polymorphism the popped type can be Unknown (in unreachable code) where
  the expected type is concrete, and propagating the expected type loses that
  distinction (code).

- 2024-03-19 (0033e67d) rationale: reference-type agreement is enforced across
  table operations — table.init requires the element segment's element type to
  equal the destination table's type, and table.copy requires the src table type
  to equal the dst table type — read through Element::value_type and
  Table::value_type rather than assuming funcref (code).

- 2024-03-19 (89e33fd9) rationale: table.fill validates that the popped
  reference value's type equals the target table's declared element type instead
  of accepting any reference (is_ref) (code).

- 2025-06-20 (01ea329c) pitfall: computing 2usize.pow(align) for the memarg
  alignment check can overflow when a malformed module encodes a large exponent,
  so the exponent must be range-checked (> 63 rejected) before the shift (code).

- 2025-10-05 (8a6e5f01) pitfall: operand-stack typing must pop in true top-down
  order — array.new_fixed pushes [init-value, length] so the i32 length is
  popped first and the element value second; array.set pushes [arrayref, index,
  value] so it pops value, then index, then arrayref; popping in the wrong order
  types the wrong slots (code).

- 2025-10-05 (593ff0ab) rationale: the "function must be declared in an element
  segment" rule for ref.func applies only inside function bodies (where the
  validator enforces it); in constant-expression initializers the wasm 3.0/GC
  relaxation drops it — ref.func in an initializer implicitly declares the
  function (code).

- 2025-10-05 (9b25df61) rationale: a memory/table instruction's
  address/index/offset/size operand is i64 when the referenced entity is 64-bit
  and i32 otherwise, determined per-instruction from the entity's is64 flag; for
  two-endpoint bulk ops the size is i64 only when both endpoints are 64-bit, and
  the static memarg offset widens to a u64 LEB that must still fit in u32 for a
  32-bit memory (code).

- 2025-10-05 (38eb8a20) statement: Unknown is the spec's bottom type — in
  unreachable code a pop returns Unknown so it satisfies every later type
  constraint, which is how stack-polymorphic typing is implemented (code).

- 2025-10-05 (2b011ba6) statement: limit flags 0x04/0x05 encode 64-bit limits
  (memory64/table64) read as u64 LEBs; a Limits records whether it is 64-bit
  (is64), and that flag drives whether the corresponding memory/table is
  addressed by i64 or i32 operands (code).

- 2026-02-14 (a8528504) rationale: the design doc gates the validator
  (~3,580 LOC) off by default because trusted modules need no validation and
  the original fast interpreter did not consume validator outputs (no jump
  table, no stack-height precomputation), trading away ~3,580 LOC of binary
  for the trusted-input fast path (sourced).
