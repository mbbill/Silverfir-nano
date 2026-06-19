- Each helper call carries a scratch MachineReg holding the base of a writable native
  scratch area, and helper inputs/outputs flow through explicit memory accesses around
  that scratch base.

## Moves

- 2026-03-13 (013fd297) replaced by [[call-lowering]]: the current helper set reads and writes canonical frame spans directly through metadata, so helpers operate on frame regions named by the metadata instead of being forced through a per-call scratch-base register, which becomes an optional escape hatch (sourced).
