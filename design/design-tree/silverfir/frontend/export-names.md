- Each exportable entity carries a set of export names (`ExportNames`); one
  entity exported under several distinct names is representable, and the set also
  rejects a duplicate at insertion.

- Export-name uniqueness is checked against one name pool spanning all entity
  kinds; a function and a table cannot share an export name.

## Facts

- 2024-03-14 (9b254645) pitfall: uniqueness was checked with a fresh set per
  entity kind (functions, then tables, then memories, then globals), which let a
  function and a table share an export name; the spec requires names unique
  across the whole module, so one shared name pool must span all kinds (diff).

## Moves

- 2024-03-14 (9b254645) replaced [[single-export-name]]: the same entity can be
  exported under several distinct names, which a single optional export name
  could not hold; the name set also rejects a duplicate at insertion to enforce
  export-name uniqueness (diff).
