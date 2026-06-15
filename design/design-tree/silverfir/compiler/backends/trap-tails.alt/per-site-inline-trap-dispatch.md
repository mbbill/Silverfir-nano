- On arm64, riscv32, riscv64, and x86_64, MachineTerminator::Trap lowers by
  inlining the full trap dispatch (set up the context and trap-code arguments,
  call raise_trap) directly at the trap site, duplicating the dispatch body at
  every trapping terminator on those backends; only arm32 branches its Trap
  terminator to a shared per-kind label.

## Moves

- 2026-05-16 (91e898fe) replaced by [[trap-tails]]: inlining the full
  trap-dispatch sequence (argument setup + raise_trap call) at every trapping
  terminator duplicates that code at each site; branching to one shared trap
  label per kind emits the dispatch body once per function and turns each trap
  site into a single branch (diff).
