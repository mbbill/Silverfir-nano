- Instantiation builds every instance kind in local vectors and applies all
  element and data segments against those locals before touching the store.

- Instances are committed to the store only after every segment succeeds; a
  failure during segment application leaves the store unchanged.

## Moves

- 2024-03-13 (055cac01) replaced by [[local-vector-processing]]: the spec
  requires segments applied before a failing one to persist, which an
  instantiation that appended every instance to the store only after all element
  and data segments succeeded could not express (code).
