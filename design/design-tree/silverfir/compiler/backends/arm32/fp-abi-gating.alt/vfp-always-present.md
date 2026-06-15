- The arm32 register plan unconditionally provides the full VFPv3-D16 FP dynamic
  bank (D3-D15) and always emits VPUSH/VPOP of callee-saved D8-D15 in the shared
  prologue/epilogue; there is no build axis for an FPU-less target.

## Moves

- 2026-04-10 (ec83df34) replaced by [[fp-abi-gating]]: an unconditional
  VFPv3-D16 plan cannot target an FPU-less core: it would allocate FP lanes that
  do not exist and VPUSH/VPOP nonexistent D-registers in the prologue; gating the
  FP dynamic budget and callee-saved FP save/restore on a build-time sf_fp_dp cfg
  lets the same arm32 backend compile for cores without double-precision FP
  (budget 0) (diff).
