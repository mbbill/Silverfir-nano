- Compiled local Wasm calls are two distinct MachineIR terminators:
  `CallDirect{callee, callee_frame_base, caller_result_base, continuation}` for a
  compile-time-known local callee, and `CallIndirect{callee_target, callee_entry,
  callee_frame_base, caller_result_base, continuation}` for a table/runtime-resolved
  callee whose entry is a runtime register value.

- Each backend lowers the two call terminators through separate per-arch entry points
  (lower_call_direct / lower_call_indirect), even though their caller-stub,
  frame-transfer, and post-call status-check contracts are otherwise identical.

## Moves

- 2026-04-16 (9ff58dcd) replaced by [[call-lowering]]: keeping CallDirect and CallIndirect as two separate terminators forced every backend to carry two near-identical call-lowering paths and duplicated the shared caller-stub / status-check / frame-transfer contract; collapsing them into one Call terminator that carries a MachineCallTarget::{Direct(func_id), Indirect{target,entry}} keeps MachineIR portable with no arch-specific call variants and lets each backend lower a single lower_call against one contract (code).
