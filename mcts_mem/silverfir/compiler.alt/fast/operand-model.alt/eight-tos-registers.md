- The TOS cache holds the top eight operand-stack values in registers t0-t7; the
  top register for a depth is (depth-1) % 8.

- Each instruction has up to sixteen depth variants (D1-D16), keeping depths 1-16
  register-resident before spilling.

- The full handler signature is thirteen parameters: five interpreter-state
  params (ctx, pc, fp, mem, memsz) plus eight TOS-cache registers (t0-t7); there
  is no dedicated local-register cache.

## Moves

- 2026-01-20 (8136fd44) replaced by [[operand-model]]: PRESERVE_NONE provides 12
  GPRs; five go to interpreter state (ctx, pc, fp, mem, memsz), so capping the TOS
  cache at four registers leaves three reserved for a future hot-locals cache
  instead of consuming the whole register file (code).
