- The interpreter is a compilation target, not a special case: the XIR
  target descriptor exposes 8 physical registers, and the same allocator
  that serves native targets allocates into them.
- The 8 registers travel between handlers as `preserve_none` function
  arguments; register selection is baked statically into permuted handler
  variants — dynamic (runtime-indexed) register selection is avoided
  entirely.
- Spilling is explicit XIR instructions (slot↔register load/store) placed at
  compile time; no runtime window manager exists.
- Handler permutations are generated only for combinations real code
  exercises; usage reports and permutation calculators keep the handler
  count bounded.
- Hot handler bodies (memory, fused, spill/copy ops) are implemented in C
  beside the dispatch wrappers; the rest stay in Rust.
