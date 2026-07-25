- Which engine runs a module is described by three separate types: a
  settable mode the embedder chooses, the kind that mode resolves to, and
  a reported runtime engine that also carries the ISA name.

- Every case of all three is the JIT. The interpreter is reached only by
  the embedder constructing a different instance type, and the selector
  has no case for it.

- Resolving the selector in a build without the JIT yields a runtime
  error naming the missing backend. An engine the build cannot run is
  still a value the embedder can hold and pass around.

- The reported engine folds the ISA into the engine value. The two axes
  cannot be asked about independently.

## Moves

- 2026-07-25 replaced by [[engine-selection]]: three separate types
  described one choice and none of them could name the interpreter, so an
  interpreter-only build reported its engine as unavailable instead of as
  the interpreter, and no build could fold the choice away (code)
