- A spill load moves one value from a spill slot into a physical register and a
  spill store moves one value from a physical register to a spill slot, each
  carrying a single (register, slot) pair.

- Each spill move lowers to one XIR instruction whose register rides in an
  immediate field; every spilled value costs a separate handler dispatch.

## Moves

- 2025-11-26 (dda758f9) replaced by [[register-model]]: a single register-slot pair
  per instruction meant one handler dispatch per spilled value, and dispatch count
  is the interpreter's dominant cost; batching up to three pairs into one
  instruction (registers carried in the handler's permutation signature, slots in
  the immediate fields) collapses several spill moves into a single dispatch
  (diff).
