- The permutation spec lists, per signature, an explicit array of permutations
  as concrete {inputs, outputs} window-slot index tuples; the build step expands
  exactly those listed permutations into handlers.

- The permutation set is fixed for three registers; adding registers means
  hand-editing every signature's permutation list.

## Moves

- 2025-11-29 (005fae86) replaced by [[handler-generation]]: listing every
  register permutation by hand in arity_patterns.toml was tractable for three
  registers but explodes to hundreds of entries per signature at eight (512 for
  3-input forms); a rule-based schema (num_inputs, output_type
  any/in_place/first_input, input_constraint non_overlapping, plus a global
  register count) lets build.rs generate the permutations programmatically for any
  register count (code).
