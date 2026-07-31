- The value model splits two representations: an internal untyped word used
  during execution ([[raw-word]]) and a separate tagged externally-facing
  `Value` used only at API boundaries (arguments, results, host calls),
  converted to and from the raw word at the edge ([[external-value]]).

- A WebAssembly value type is a structured descriptor ([[value-type]]) whose
  reference case is a `RefType` (nullability flag plus a heap-type hierarchy),
  not a flat enumeration of reference shapes.

- A reference value at runtime is a single tagged index word
  ([[ref-value]]) whose high bits distinguish the funcref, GC, i31,
  extern, and host hierarchies; the null reference is the all-ones sentinel,
  never a valid index.
