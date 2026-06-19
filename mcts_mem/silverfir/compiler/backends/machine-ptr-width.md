- Machine pointer and word widths are parameterized by the selected backend's
  GP register width, not the host compiler's `usize`; the same shared
  lowering produces correct memory-access and word widths when targeting a
  32-bit ABI (e.g. armv7a, 4-byte GP) from a 64-bit host
  (`machine_ptr_width`, `gp_reg_width`).

- NativeContext and ABI-struct field offsets and strides used by lowering and
  the emulator are computed from the selected backend's GP unit size rather than
  the host Rust struct layout (`native_runtime_abi_layout`).

## Facts

- 2026-04-27 (ae557569) rationale: on 32-bit GP targets an i64 result is a
  register pair, and when its high half is dead the high-half computation is pure
  waste in hot i64 loops; a backward per-instruction MachineIR liveness analysis
  marks i64-pair ops whose high-half result is demanded by no later computation,
  letting 32-bit GP lowering emit only the low half for selected pair
  add/sub/and, extend32_s (lowered to a single move), and small-count right
  shifts (code).

## Moves

- 2026-03-18 (3778de1c) replaced [[host-usize-ptr-width]]: shared lowering must
  key memory-access and word widths off the backend ABI it is targeting, not
  the host compiler's usize, because armv7a (4-byte GP) lowering runs on 64-bit
  hosts; keying off size_of::<usize>() cannot express a 32-bit target's pointer
  width from a 64-bit host build (code).
