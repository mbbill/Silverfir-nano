- Function code and constant init expressions are retained as owned
  reference-counted `Rc<[u8]>` rather than slices borrowed from the parse
  buffer; the module owns its code, carries no lifetime parameter, and can be
  held and shared independently of the source binary.

- The two retained uses are distinguished at the type level by separate
  newtypes (`Bytecode` for function code, `ConstExpr` for constant init
  expressions) though both still wrap `Rc<[u8]>`.

## Facts

- 2024-02-16 (da94bda8) rationale: retained bytecode is owned rather than
  borrowed precisely so it outlives parsing and can be re-evaluated at
  instantiation; the constexpr parser changed its return type from a borrowed
  slice to the owned form for this reason (code).

## Moves

- 2024-02-05 (ba01d633) replaced [[borrowed-module]]: holding function code as
  a Cow borrowed from the input binary tied Module, Runtime, and every
  contained item to the binary's lifetime, which made storing a parsed module
  independently awkward; making the module own its code as Rc<[u8]> drops all
  the lifetime parameters and lets a module be held and shared freely while
  Payload becomes a pure borrowing cursor over input (code).
