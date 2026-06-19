- Every arm64 function body emits an x29/x30 link-save prelude
  (stp x29,x30,[sp,#-16]!) at body entry unconditionally, and every terminal
  sequence pops that pair before the native ret.

- Trap stubs assume the link save is already present; they call raise_trap
  without saving the link register themselves.

## Moves

- 2026-05-13 (c5fe4fc8) replaced by [[leaf-prelude]]: the unconditional
  link-save prelude cost a stp/ldp pair on every function including leaves that
  make no native call and clobber no preserved dynamic regs; gating it on a
  per-function body-host-frame predicate lets a leaf body return directly through
  the caller's LR with no prologue/epilogue stack traffic (code).
