---
status: abandoned
---
# Computed-goto / threaded dispatch

The classic threaded-code technique: each handler ends with an indirect jump
(`goto *next_label`) to the next handler's label, giving each opcode its own
dispatch site instead of one shared switch. All handlers still live inside one
function.

## In practice

Must:
- Emit a distinct indirect-jump dispatch site at the end of each handler, so each
  opcode gets its own branch-target-buffer entry.

Must not:
- Funnel dispatch through one shared site (that is switch dispatch).
- Be treated as a substitute for tail-call dispatch: because all handlers share
  one function, the whole loop carries combined register pressure and no handler
  is independently optimizable.
