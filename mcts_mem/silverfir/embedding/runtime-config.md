- Runtime sizing (executable code-arena size, linear-memory page ceiling,
  Wasm operand/call stack size, compile-time RAM budget, compilation
  parallelism) and the execution tier are one value the embedder builds
  and hands to an engine (`Config`).

- An engine holds a validated configuration and copies it into every
  instance it creates. Two engines coexist with different budgets and
  different tiers, and configuring one cannot disturb the other.

- Budgets are read from the instance in hand, never from ambient state.

- The hosted defaults reproduce the former fixed numbers. The bare-metal
  (`sf_os_none`) defaults are zeroed, and engine construction rejects a
  zeroed budget naming the field.

## Facts

- 2026-07-25 rationale: a write-once global cannot express two
  differently-configured engines in one process, and it made the
  configuration a property of the program rather than of the thing being
  configured (code).

- 2026-07-25 statement: the check that a bare-metal embedder configured
  anything moved from each reading call site to engine construction, so
  it reports a missing budget before a module is touched rather than as a
  trap inside a guest (code).

- 2026-07-25 measurement: reaching the budgets needed no new parameter on
  the hot paths — the module instance already travels everywhere they are
  read, so carrying the configuration there covered all ten sites (code).

- 2026-07-25 pitfall: the global's storage needed an `UnsafeCell` behind a
  hand-written `Sync` impl and a three-state atomic to be sound against a
  concurrent reader; a value passed by the embedder needs none of it
  (code).

## Moves

- 2026-07-25 replaced [[write-once-global]]: configuration that is
  process-wide cannot describe two engines at once, and being write-once
  made a second call an error rather than an ordinary thing to do (code)
