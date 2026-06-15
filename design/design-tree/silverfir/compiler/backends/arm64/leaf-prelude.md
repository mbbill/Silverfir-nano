- The arm64 body prelude pushes the x29/x30 link-save pair only when the
  function needs a body host frame; a leaf body that makes no native call and
  clobbers no preserved dynamic register returns directly through the caller's LR
  with no prologue/epilogue stack traffic (`has_body_host_frame`).

## Moves

- 2026-05-13 (c5fe4fc8) replaced [[unconditional-link-save-prelude]]: the
  unconditional link-save prelude cost a stp/ldp pair on every function including
  leaves that make no native call and clobber no preserved dynamic regs; gating
  it on a per-function body-host-frame predicate lets a leaf body return directly
  through the caller's LR with no prologue/epilogue stack traffic (diff).
