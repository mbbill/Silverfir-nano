- The handler context holds Store, current function, current module and the
  shared evaluation environment as NonNull raw pointers, exposing each through an
  unsafe accessor that fabricates a reference with an unbounded lifetime.

- The error slot and register file are plain fields and handlers receive the
  context by mutable reference (&mut Ctx).

## Moves

- 2025-10-15 (0d24ab09) replaced by [[handler-ffi]]: the old Ctx stored
  Store/Module/Func as NonNull and handed out unbounded-lifetime references
  through unsafe accessors so that a handler could re-borrow ctx mutably (e.g. to
  set the error), which defeated the borrow checker across the FFI boundary;
  making the fields real &'a borrows and moving the mutable state (error, regs)
  into Cell/RefCell lets every handler take &Ctx and mutate through interior
  mutability with no unbounded-lifetime unsafe (code).
