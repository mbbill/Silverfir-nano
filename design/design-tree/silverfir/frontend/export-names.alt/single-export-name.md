- Each exportable entity carries at most one export name as an Option<String>.

- An entity is matched for import resolution by equality against that single
  name.

## Moves

- 2024-03-14 (9b254645) replaced by [[export-names]]: the same entity can be
  exported under several distinct names, which a single optional export name
  could not hold; the name set also rejects a duplicate at insertion to enforce
  export-name uniqueness (diff).
