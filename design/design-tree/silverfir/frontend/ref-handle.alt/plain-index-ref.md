- A reference is a plain index word Ref(usize) whose only special value is the
  all-ones null sentinel; every non-null value is a raw index with no kind
  information in the word itself.

- The kind of a reference (funcref vs externref) is known only from the
  accompanying RefType, never from the index word.

## Facts

- 2024-03-13 (7c57034e) pitfall: a default/zero reference aliases store index
  0, which is a valid instance, so uninitialized table slots cannot use
  Ref::default(); they are filled with the all-ones (usize::MAX) null sentinel
  and call_indirect must trap on a null slot before resolving it as a store
  index (diff).

## Moves

- 2024-03-12 (35d0c137) replaced [[module-local-bare-usize-ref]]: a module-local
  reference index cannot be resolved against the flat global store without
  re-adding the module's range base, so references now carry the global store
  index directly and are wrapped in a newtype that cannot be confused with a raw
  integer (diff).

- 2025-10-04 (eac9a06d) replaced by [[ref-handle]]: a plain usize index
  carried only a null sentinel and could not distinguish a GC-heap reference
  from an inline i31 value from a funcref, nor encode an i31 payload inline;
  tagging the high bits lets one word carry all reference kinds without a
  separate type word (diff).
