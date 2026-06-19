- The shared backend emission core models the function body source as a
  `FunctionBody<'a>` enum (`Mir(&MachineFunction)` | `Template { func_id }`) and
  exposes only fallible MachineIR accessors that return an internal error in the
  template case; a body that carries no MachineIR is a first-class input rather
  than a borrow that must always point at a real function (`FunctionBody`).

## Facts

- 2026-04-30 (af719dc5) statement: the same per-backend encoders, labels, patch
  tables, and tail regions serve both the MachineIR tier and the MachineIR-free
  template tier, so the emission core is genuinely body-source-agnostic — a
  future revisit must keep "no MachineFunction" a first-class input case and not
  return to the prior band-aid of a fabricated placeholder `MachineFunction`
  widened by `unsafe transmute` (code).

## Moves

- 2026-04-30 (af719dc5) replaced [[borrowed-machinefunction-input]]: the backend
  emission-core constructor type could not express a function body with no
  MachineIR, so the template tier was forced to fabricate a throwaway empty
  `MachineFunction` and widen its borrow with `unsafe core::mem::transmute` to
  satisfy `ArchBackend::new(compiled, function: &MachineFunction)`; the input is
  now a `FunctionBody<'a>` (`Mir(&MachineFunction)` | `Template { func_id }`)
  that `CompilerCore` owns, `ArchBackend::new` takes the `CompilerCore`
  directly, and MachineIR-specific accessors return an error in the template
  case instead of dereferencing a fake function — the input must model 'no
  MachineFunction' as a first-class case, not be coerced into one with unsafe
  (code)
