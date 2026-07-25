- Each engine exposes its own instance type, and they share no surface.
  The JIT's takes an import list of typed callbacks and calls exports by
  name with values; the interpreter's takes one raw dispatch closure and
  calls exports by index with u64 words.

- The interpreter's instance borrows its module for its whole life. An
  embedder has to keep the module alive separately.

- Converting between the two shapes — signature lookup, raw-to-value
  conversion, building the caller's view of memory — is the embedder's
  code, written again in each one.

- Which engine an embedder wants is visible in its source. The engine
  choice is a compile-time branch around two different programs, not a
  value.

## Moves

- 2026-07-25 replaced by [[engine-transparent-api]]: an embedder had to
  write one code path per engine and hand-roll the interpreter's raw host
  boundary itself, which every embedder in the tree had duplicated (code)
