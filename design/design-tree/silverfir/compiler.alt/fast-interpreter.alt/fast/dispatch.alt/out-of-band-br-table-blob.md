- br_table entry data (target offsets, stack drops, arities) is built into a
  separate per-function auxiliary byte blob by a BlobBuilder, length-prefixed and
  4-byte aligned.

- A br_table instruction stores the blob offset of its table in an immediate; its
  handler reads the entry count and the selected entry's fields from the blob via
  unaligned reads off a fast_blob_base pointer.

- The blob base pointer is carried in the Context and saved/restored per call in
  each CallFrame's caller_blob_base, letting the running function locate its own
  br_table data.

## Moves

- 2025-12-10 (df2532aa) replaced by [[dispatch]]: the separate br_table blob
  required its own heap allocation plus a fast_blob_base pointer threaded through
  the Context and every CallFrame; storing each table's entries inline as data
  pseudo-instructions right after its br_table keeps all per-function data in the
  single instruction array, eliminating the blob allocation and the blob-base
  pointer entirely (diff).
