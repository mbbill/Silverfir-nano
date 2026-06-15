- Every memory-accessing operand (load/store memarg, memory.size/grow/fill/
  init/copy) carries a single reserved byte that the decoder reads and the
  validator requires to be zero; a non-zero reserved value is malformed.

- Only one memory (implicit index 0) is addressable; the validator rejects a
  module that declares more than one memory, and every memory operation reads
  the single memory 0 directly.

- The memarg of a load/store is decoded as (align, offset) with no
  memory-index field.

## Moves

- 2024-01-29 (a719e961) replaced [[byte-shape-immediates]]: raw byte-shape
  variants conflated operands of different meaning behind one tag (the single
  U32 tag served funcidx, localidx, globalidx and tableidx indistinguishably),
  so the decoder now names each immediate by its operand role and the handler
  reads meaning without re-deriving it from the opcode (diff).

- 2025-10-01 (b7d4218c) replaced by [[immediates]]: the reserved-byte encoding
  could address only memory 0 (it rejected any non-zero value), so it could not
  express WebAssembly 3.0 modules that declare and operate on more than one
  memory (diff).
