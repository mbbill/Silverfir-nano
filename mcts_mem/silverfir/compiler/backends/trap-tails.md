- Each trapping terminator branches to one shared per-kind trap label emitted
  once per function rather than inlining the full trap-dispatch sequence
  (argument setup plus the raise_trap call) at every trapping site
  (`ensure_trap_label`).

## Facts

- 2026-03-28 (dfdac079) measurement: replacing per-site inlined trap handlers
  (~56 bytes at every trapping site) with a single shared trap stub per kind that
  TrapIf/Trap branches to cut ARMv7-A Coremark code size from 340 KB to 162 KB
  (2.1x), the dominant size win of the shared-pipeline migration (code).

## Moves

- 2026-05-16 (91e898fe) replaced [[per-site-inline-trap-dispatch]]: inlining the
  full trap-dispatch sequence (argument setup + raise_trap call) at every
  trapping terminator duplicates that code at each site; branching to one shared
  trap label per kind emits the dispatch body once per function and turns each
  trap site into a single branch (code).
