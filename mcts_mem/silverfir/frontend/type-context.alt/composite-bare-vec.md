- The module stores its type definitions as a bare owned field
  type_definitions: Vec<Rc<DefType>>, exposed as a slice.

- Subtyping over concrete type indices takes the type definitions as an Option:
  with None it conservatively cannot confirm concrete-to-abstract subtyping;
  a no-context call (is_subtype_of) only matches concrete types exactly.

## Moves

- 2025-10-04 (d93a9392) replaced [[function-type-vector]]: a vector of function
  types could not represent the 3.0 unified type/index space: struct and array
  composite types, subtyping (supertype indices and finality), or recursive type
  groups all share one index space with functions (code).

- 2025-10-05 (5dc7bbc5) replaced by [[type-context]]: type-index resolution is
  module-local and is needed inside subtyping checks, but a bare owned Vec field
  could not be threaded by value and the optional-context subtyping path could
  silently return a wrong answer when no context was passed; a cheaply-clonable
  TypeContext makes the module scope a first-class value that subtyping now
  always receives (code).
