- Module lowering receives prepared SSA as a borrowed whole-module slice; every
  function's prepared SSA and the semantic IR stay live until the whole module
  finishes lowering.

## Moves

- 2026-04-09 (c329abab) replaced by [[machine-lowering]]: a borrowed whole-module SSA slice ties every function's prepared SSA to one lifetime so none can be freed until lowering finishes; taking ownership of the lowering inputs lets each function's SSA (and the semantic IR, now taken and dropped) be released as soon as it is lowered, cutting peak compile-time memory (diff).
