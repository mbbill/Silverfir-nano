- A Call SSA instruction never defines an SSA result value; every call result is
  delivered into canonical frame slots and read back from there by later instructions.

- A call is always a sink barrier: a value produced by a call can never sink into a
  cached local's register.

## Moves

- 2026-05-14 (fec6b497) replaced by [[call-lowering]]: a call result written only to a frame slot must be reloaded by any consumer and cannot participate in register residency or sink planning; producing a live SSA result for single-result scalar calls lets the result stay in a register across the continuation edge and lets call -> local.set_cache sink directly into the cache register, eliding the store/reload pair (diff).
