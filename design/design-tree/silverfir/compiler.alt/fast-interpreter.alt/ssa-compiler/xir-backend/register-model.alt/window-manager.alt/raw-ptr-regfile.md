- The handler trampoline passes the register file to each operation as a raw
  pointer.

- Operations read and write registers through unchecked pointer arithmetic, with a
  length carried in the context only under debug assertions for bounds-check
  assertions.

## Moves

- 2025-10-13 (82b6303a) replaced by [[window-manager]]: raw-pointer register access
  scattered unsafe blocks and unchecked add(index) through every handler; moving
  the file into Ctx as a Vec gives all register access automatic bounds checking
  with the raw-pointer handler argument retired to dead weight (diff).
