- Dispatch is subroutine-threaded: each instruction carries a handler function
  pointer; a handler executes its op and chains into `next->handler`; chains
  end at an explicit terminal op.
- Branch targets are resolved instruction pointers (a fallthrough and an alt
  target per instruction), not indices — no offset-preserving filler ops. The
  code array is therefore patched only after its final allocation and is
  pinned for its lifetime.
- A Rust IR builder translates wasm bytecode into the instruction stream.
- Traps never unwind through the chain: a trapping op sets a shared trap flag
  on the context and ends the chain; the trampoline's caller converts the
  flag into an error.
- Handler variants are generated at build time from a TOS configuration;
  op semantics live once in shared C macros (`semantics.h`) used by both
  plain and fused handlers.
