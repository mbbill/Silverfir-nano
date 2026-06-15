- Each block's entry cache lanes are assigned by sorting its entry cached slots
  into a global first-appearance order and giving them sequential dynamic
  registers (a compact prefix) (`target_entry_cache_params`).

- Cached locals are assumed to occupy a contiguous prefix of the dynamic bank;
  dropping a local compacts the survivors down.

- A local resident on both sides of an edge can still be renumbered to a
  different lane when an earlier-ordered local changes membership, turning into an
  avoidable register move.

## Moves

- 2026-04-06 (366923b2) replaced by [[lane-assignment]]: compacting each block's
  entry cache lanes into a sequential prefix by global slot order renumbered a
  still-resident shared local to a different lane whenever an earlier-ordered
  local was added or dropped, causing avoidable cross-edge moves; assigning lanes
  top-down with sticky inheritance and leaving holes after drops keeps a shared
  local on the same lane across edges (diff).
