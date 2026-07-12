- The fallible error type is a flat enum whose variants each carry the message
  String and a captured Backtrace inline.

- Callers discriminate and read error contents by matching directly on the
  error enum's variants.

## Moves

- 2025-10-13 (04cd73c2) replaced by [[boxed-string-error]]: the flat enum's
  message and backtrace inflated every Result-returning frame's stack footprint
  on the hot success path; boxing the payload shrinks WasmError to one pointer
  (code).
