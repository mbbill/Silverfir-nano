- The interpreter executes the middle IR through a small register window: a
  few hot slots hold the most valuable vregs; all other vregs live in a
  per-frame backing file in memory.
- Window residency is a compile-time choice driven by IR metadata — usage
  frequency weighted by loop depth; short-lived values stay cold.
- The backend's executable form is XIR: a linear instruction stream in which
  each instruction binds a generated handler; lowering assigns window slots
  and emits fill/spill moves only where operands or results are cold.
- Slot eviction picks victims by score, and liveness analysis skips the
  spill store entirely when the evicted value is already dead.
- Fusion candidates are discovered by a profiler counting executed handler
  sequences, not chosen by intuition.
