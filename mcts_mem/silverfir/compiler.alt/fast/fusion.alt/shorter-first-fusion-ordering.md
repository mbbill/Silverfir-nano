- At each position the peephole tries the pair fusion rules first; only if no pair
  matches does it try the triple (3-gram) rules, which are appended at the tail of
  the loop body.

- The longest fusion windows are 3-gram (e.g. local.get; i32.const; i32.shl ->
  shli_local_imm); no 4-gram superinstructions are produced.

- A contained shorter rule matches and advances past its operands first; a longer
  window whose prefix is an existing shorter rule can never form.

- An inbound alt branch target inside a candidate window vetoes that window;
  fusion never spans a label.

## Moves

- 2025-09-13 (d4600172) replaced by [[old-quickening-peephole]]: a shorter rule
  matched first under pair-first ordering, so 4-gram superinstructions could never
  form (e.g. local.get; i32.const; i32.shl; i32.add was consumed by the 3-gram
  shli_local_imm before the 4-gram i32_add_shli_local_imm could match);
  longest-match-first is required to emit them (code).
