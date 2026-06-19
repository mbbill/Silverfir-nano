- The runtime registry holds loaded modules in a flat vector; a module carries
  a reference-counted shared name list, and a lookup scans for the module whose
  list contains the requested name.

- A module and its instance share that one reference-counted name list;
  registering a further name (`register_as`) appends to the shared list, and the
  one module is then reachable under every name in the list.

## Moves

- 2024-03-14 (fd959de0) replaced [[module-registry.alt/single-name-registry]]:
  a module must be reachable under several registered names (the spec's
  register directive adds names to an already-loaded module), which a registry
  keyed by exactly one name could only fake by re-inserting the whole module
  under each name as a separate entry with no shared identity (code).
