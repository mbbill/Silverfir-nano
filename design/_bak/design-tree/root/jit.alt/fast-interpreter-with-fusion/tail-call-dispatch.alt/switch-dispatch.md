---
status: abandoned
---
# Switch / jump-table dispatch

A single `switch` on the opcode inside one big interpreter loop — the textbook
interpreter shape. Every opcode dispatches through one shared branch site, and all
handler code lives inside one function.

## In practice

Must:
- Route every opcode through a single shared `switch`/jump-table dispatch site.
- Serve as the baseline against which every other dispatch option is measured.

Must not:
- Give any opcode its own dispatch/branch site (a single shared site is the
  defining property).
- Be relied on for hot interpretation: the single shared dispatch site thrashes
  the branch-target buffer, and the giant function carries whole-function register
  pressure.
