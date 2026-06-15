- The runtime registry maps each module name to its `Rc<Module>` in a HashMap
  keyed by that single name.

- A second registration of an already-present name is rejected as a duplicate
  (`RuntimeError::ModuleExists`); there is no way to reach one module under
  multiple names. A lookup is a registry get on the requested name.

## Moves

- 2024-03-14 (fd959de0) replaced by [[module-registry]]: a module must be
  reachable under several registered names (the spec's register directive adds
  names to an already-loaded module), which a registry keyed by exactly one
  name could only fake by re-inserting the whole module under each name as a
  separate entry with no shared identity (diff).
