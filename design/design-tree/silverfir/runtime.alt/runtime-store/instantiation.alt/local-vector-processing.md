- Each instance kind is created in its own local vector and bound/initialized
  in place by iterating that owned local vector, then appended to the store per
  kind right after its own processing (interleaved with the next kind's
  processing), with `add_module` last; processing reads the owned local vectors,
  not the store's per-module slice.

## Moves

- 2024-03-13 (055cac01) replaced [[local-vector-processing.alt/all-or-nothing]]:
  the spec requires segments applied before a failing one to persist, which an
  instantiation that appended every instance to the store only after all element
  and data segments succeeded could not express (diff).

- 2024-03-15 (b22db5c9) replaced by [[instantiation]]: binding the module
  instance and evaluating segment initializers need the instances already live
  in the store, so cross-references resolve through the store's per-module slice
  rather than through pre-append local vectors (diff).
