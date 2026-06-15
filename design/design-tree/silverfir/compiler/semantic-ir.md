- The first IR keeps Wasm structure intact rather than lowering it: structured
  control markers (block/loop/if/else/end), abstract locals, semantic direct
  and indirect calls, typed result information, and max stack height all
  survive into Semantic IR (`semantic_ir`).

- Semantic IR carries no frame slots, cache registers, or transient budgets;
  it is an optimization-friendly representation choice, not an optimization
  pass — later passes reason about loops, calls, and locals without
  reconstructing them from low-level code.

- Fallthrough is implicit by instruction order; only branching ops
  (if/else/br/br_if/br_table) carry explicit branch targets, rather than every
  op carrying generic control side-channels.

## Facts

- 2026-03-06 (c58d3a92) rationale: the decode/placement split moves
  backend-specific decisions out of the decode pass — the stack tracker no longer
  carries the hot-locals array and decode no longer emits the prologue or
  hot-vs-frame local opcodes — so a single semantic decode can feed different
  backend placements (diff).

- 2026-03-06 (80645597) statement: semantic IR carries abstract TOS-cache
  management markers (CacheSpill/CacheFill) rather than concrete lowered
  spill/fill opcodes, which only the backend-lowering step emits — the frontend
  layer holds semantic markers, the lowered layer holds concrete
  placement-resolved ops (diff).

- 2026-06-14 rationale: semantic-IR leaf inlining (replacing a call to a small
  leaf callee with the callee's body before lowering) is deliberately left
  disabled. nano is a streaming compiler that compiles function-by-function (not
  block-by-block), so a function's whole working set must fit at once; inlining
  grows function size and thus peak compile-time memory, which loses on the
  footprint-constrained esp32/pico2 targets. The gain is marginal regardless:
  ALGORITHM4 does not place hot-local swaps optimally, so a bigger function with
  more locals after inlining gets worse local-cache allocation, eroding the
  inlining win — so enabling it is not worth the footprint cost (author).

## Moves

- 2026-03-12 (2ea0bb68) replaced [[semantic-ir.alt/generic-control-side-channels]]:
  a generic next/alt side channel on every semantic op duplicated fallthrough
  into every instruction and could not distinguish a branching op from a
  straight-line one, forcing later preparation to rediscover control meaning
  from generic side fields; making fallthrough implicit by order and attaching
  explicit targets only to branching ops (if's false target, else's end target,
  br/br_if/br_table targets) keeps the semantic layer honest — only branching
  ops carry control targets (diff)
