- The dispatch signature passes the per-frame VM register file by pointer:
  the handler takes the context, the program counter, the register-file base
  pointer, the value-stack pointer, and the memory-0 base/size pointers
  (`regs_base`).

- regs_base is a contiguous register file holding params, locals, and SSA
  temps; every VM value lives in this in-memory file and is reached through the
  pointer.

- No VM values are carried in CPU argument registers across tail-calls; only
  the context, program counter, register-file base, value-stack pointer, and
  memory-0 base/size pointers are passed.

## Moves

- 2025-09-17 (e85e902d) replaced by [[dispatch-abi]]: a regs_base
  pointer-to-memory handler signature cannot carry VM values in CPU registers
  across musttail tail-calls, so a hot window of VM values v0..v3 is passed by
  value in argument registers across the whole tail-chain, the abstract working
  set staying in registers with zero prologue/epilogue between handlers (diff).
