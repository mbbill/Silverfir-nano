- The MachineMemWidth for pointer-width loads/stores of NativeContext and
  ABI-struct fields is chosen from the host compiler's
  `core::mem::size_of::<usize>()`: U64 on a 64-bit host, U32 on a 32-bit host.

## Moves

- 2026-03-18 (3778de1c) replaced by [[machine-ptr-width]]: shared lowering must
  key memory-access and word widths off the backend ABI it is targeting, not
  the host compiler's usize, because armv7a (4-byte GP) lowering runs on 64-bit
  hosts; keying off size_of::<usize>() cannot express a 32-bit target's pointer
  width from a 64-bit host build (diff).
