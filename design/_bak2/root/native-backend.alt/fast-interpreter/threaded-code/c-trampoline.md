- Handler chaining runs through a minimal C trampoline compiled under
  cross-language LTO: Rust op bodies (`impl_*`) inline into C wrappers
  (`op_*`), and the chain into the next handler compiles to a tail jump —
  stable Rust alone cannot guarantee tail calls.
- On clang the chain call carries an explicit `musttail` attribute
  (`SF_MUSTTAIL`); other compilers rely on sibling-call optimization.
- Handlers use the `preserve_none` calling convention: no callee-saved
  registers, maximal argument registers — the chain never pays
  save/restore.
- Hot handler bodies (arithmetic, memory, float, control flow, calls) are
  implemented in C beside the trampoline; Rust keeps the slow paths and the
  IR builder.
