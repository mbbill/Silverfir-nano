- `CompilerCore` borrows the function body as a mandatory
  `function: &'a MachineFunction` and `ArchBackend::new` requires one; there is
  no representation for a body that carries no MachineIR, leaving a MachineIR-free
  (streaming template) emission to fabricate a throwaway empty `MachineFunction`
  and widen its borrow with `unsafe core::mem::transmute` to construct a backend.

- Backend code reads `core.function` directly (`core.function.id`,
  `core.program.blocks`, fp-width and block-layout init) on the assumption a real
  `MachineFunction` is always present; there are no fallible MachineIR accessors
  guarding the no-MachineIR case.

## Moves

- 2026-04-30 (af719dc5) replaced by [[emission-core]]: the backend emission-core
  constructor type could not express a function body with no MachineIR, so the
  template tier was forced to fabricate a throwaway empty `MachineFunction` and
  widen its borrow with `unsafe core::mem::transmute` to satisfy
  `ArchBackend::new(compiled, function: &MachineFunction)`; the input is now a
  `FunctionBody<'a>` (`Mir(&MachineFunction)` | `Template { func_id }`) that
  `CompilerCore` owns, `ArchBackend::new` takes the `CompilerCore` directly, and
  MachineIR-specific accessors return an error in the template case instead of
  dereferencing a fake function — the input must model 'no MachineFunction' as a
  first-class case, not be coerced into one with unsafe (code)
