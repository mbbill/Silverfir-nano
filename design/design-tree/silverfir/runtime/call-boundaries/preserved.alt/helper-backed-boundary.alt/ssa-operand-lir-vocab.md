- LIR runtime-boundary primitives (memory.grow, table.grow, segment lifecycle)
  are a distinct Runtime op carrying SSA-value args and results, separate from
  generic Leaf ops.

- Publishing and reloading a transient SSA value to/from its canonical operand
  slot are distinct Spill and Fill ops, separate from the ReadSlot/WriteSlot ops
  that access canonical local slots.

## Moves

- 2026-03-13 (013fd297) replaced by [[helper-backed-boundary]]: native lowering
  must not reconstruct stack or frame publication on its own, so every boundary op
  must already carry only canonical frame spans with all live SSA published to
  slots before it, rather than SSA operands (author).
