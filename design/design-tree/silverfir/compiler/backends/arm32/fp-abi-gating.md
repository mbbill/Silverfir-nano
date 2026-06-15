- The arm32 FP register/ABI plan is gated on a build-time `sf_fp_dp` cfg: the FP
  dynamic budget and the callee-saved FP save/restore are present only when the
  target has double-precision FP; the same arm32 backend compiles for
  FPU-less cores with an FP budget of 0 (`sf_fp_dp`).

## Facts

- 2026-04-10 (ec83df34) statement: sf_fp_dp is derived in build.rs from the
  target — ARMv7-A is treated as always having VFPv3-D16+ (it pairs with the
  IDIV extension) so sf_arch_armv7a sets sf_fp_dp, while sf_arch_thumbm does not,
  since many Cortex-M cores are single-precision-only or have no FPU, deferring
  DP-FP enablement to a future per-core feature (diff).

## Moves

- 2026-04-10 (ec83df34) replaced [[vfp-always-present]]: an unconditional
  VFPv3-D16 plan cannot target an FPU-less core: it would allocate FP lanes that
  do not exist and VPUSH/VPOP nonexistent D-registers in the prologue; gating the
  FP dynamic budget and callee-saved FP save/restore on a build-time sf_fp_dp cfg
  lets the same arm32 backend compile for cores without double-precision FP
  (budget 0) (diff).
