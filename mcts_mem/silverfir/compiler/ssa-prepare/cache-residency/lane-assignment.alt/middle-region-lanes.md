- Abstract lane indices are assigned in the middle-end by a region-sticky
  pass: one frozen (slot, lane) map per region above a per-region floor, with
  sticky inheritance into nested regions and preserved-suffix promotion for
  preferred residents; the published entry rows carry the lane, and machine
  lowering is a reader mapping lane indices to physical registers.

- Synthesized blocks (bridges, boundary repairs, the entry repair) derive
  their lane rows emit-side from the predecessor's published row.

## Moves

- 2026-07-12 (30aac662) replaced by [[lane-assignment]]: with identical
  MachineIR op inventories and improved edge parallel-move counts the
  pure-middle lane assignment still inflated native code because register
  choice couples to transient allocation, scratch borrowing and trace
  selection that only the machine sees; lane placement returned below LIR
  while the structure-derived preference and requirement signals stayed in
  the middle (code)
