- The parser walks the module binary section by section in a single forward
  pass, reading a section id and length, carving out that section's byte
  range, and dispatching to a per-section routine (`parse_module`).

- Section order is enforced as the pass runs: each section id must be strictly
  greater than the previous, except that custom sections may appear anywhere
  and the data-count section is allowed to precede the code and data sections.
  Out-of-order sections are rejected as malformed.

- Parsing is zero-copy where it can be: byte ranges that outlive the parse
  (code bodies, init/offset expressions, data segments) are handed out as
  `Cow` slices that borrow the original module bytes when the input was
  borrowed, and own a copy only when the input itself was owned.

- The module is assembled through a mutable builder that accumulates each
  entity vector as sections are parsed, then is finalized into an immutable
  module (`ModuleBuilder`).
