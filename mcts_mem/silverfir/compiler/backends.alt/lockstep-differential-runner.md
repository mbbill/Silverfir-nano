- The fast interpreter runs alongside the native backend over the same program,
  with both engines pausable at the same instruction boundary.

- At each pause the two engines' canonical state — linear memory, globals, and
  the value stack — is compared, and the first divergence localizes the codegen
  fault to an instruction.

- State is compared as a stream, checkpoint by checkpoint, rather than recorded
  whole and diffed at the end.

- Checkpoints are placeable at function, control-transfer, or per-instruction
  granularity (ProgramCheckpointPlan).

## Facts

- 2026-03-08 (cdc717a4) rationale: the fast interpreter was already a trusted
  correctness baseline, so running it as ground truth against the native backend
  and diffing at a shared pause point promised exact per-instruction bug
  localization without authoring a separate reference engine (sourced).

## Moves

- 2026-03-08 (e25f8063) replaced by [[emulator]]: keeping a separate interpreter
  and the native backend running side by side in instruction-lockstep,
  synchronizing their step boundaries and paused state at every instruction,
  proved too complex and surfaced too many issues to be worth it, so it was
  abandoned early on feasibility grounds rather than on the differential idea
  being unsound (sourced)
