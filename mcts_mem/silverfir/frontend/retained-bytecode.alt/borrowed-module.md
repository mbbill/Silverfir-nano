- Module, Runtime, and the per-item structures carry a 'a lifetime parameter;
  Module::new takes a Cow<'a, [u8]> and the module borrows function code, init
  expressions, and data from that source binary.

- Payload owns a Cow<'a, [u8]>; advance_and_split_at can hand back
  either a borrowed or an owned slice depending on the payload's variant.

## Moves

- 2024-02-05 (ba01d633) replaced by [[retained-bytecode]]: holding function
  code as a Cow borrowed from the input binary tied Module, Runtime, and every
  contained item to the binary's lifetime, which made storing a parsed module
  independently awkward; making the module own its code as Rc<[u8]> drops all
  the lifetime parameters and lets a module be held and shared freely while
  Payload becomes a pure borrowing cursor over input (code).
