---
status: unexplored
---
# Tree-walking interpreter

Execute Wasm by decoding it to an AST / structured form and interpreting that
structure directly, walking nodes and recursing into children. The simplest
possible execution strategy.

Never built in this project — it was passed over at the first commit in favor of
a linearized fast interpreter. Kept as a node only to mark that the fork existed.

## In practice

Must:
- (Were this chosen) execution would dispatch per AST node, recursing through
  structured control nodes rather than over a linear opcode stream.
- Must still clear the spectest gate (root.all/correctness-validation.md) like
  any execution strategy.

Must not:
- Must not be relied on for the "fast" half of the project why: per-node dispatch
  and recursion overhead are assumed categorically too slow for the hot path.
