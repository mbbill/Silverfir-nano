- A parsed module is an immutable bundle of the binary version plus one vector
  per entity kind: function types, functions, memories, tables, globals,
  element segments, and data segments, with an optional start-function index
  (`Module`).

- Every entity kind (function, table, memory, global) is one struct that can
  represent either an imported or a locally-defined instance; there is no
  separate type for the imported case.

- Function bodies are not decoded into instructions at parse time: a local
  function stores its declared locals and its raw code bytes as a borrowed
  slice, to be decoded on demand.

- The builder shrinks each entity vector to fit before finalizing the module,
  so the long-lived module carries no spare capacity.
