- The compiled native module keeps its full MachineIR module and ABI resident for
  the lifetime of the compiled code; MachineIR is never released after native code
  is emitted.

## Moves

- 2026-04-09 (c329abab) replaced by [[machine-ir]]: MachineIR is only consumed during native code emission (and by the emulator backend, which executes it directly), so retaining it on the runtime module wastes memory after compilation; making it optional and dropping it once native code is emitted frees that footprint on memory-constrained targets (code).
