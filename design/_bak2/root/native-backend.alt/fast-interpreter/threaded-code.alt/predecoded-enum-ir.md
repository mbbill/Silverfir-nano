- The IR is a Rust enum (`FastOp`) of fixed-size variants: one decode pass
  turns opcodes and LEB immediates into enum values, executed by a match-based
  eval loop.
- Branch targets are stored as instruction indices; block/loop emit
  offset-preserving filler ops so the indices line up.
- Locals are addressed as absolute offsets from the frame base.
- Ops the IR does not support fall back per-op to the baseline in-place
  interpreter; the spec suite stays green with the fast path on or off.
