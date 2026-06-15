- Adjacent opcodes are fused into super-instructions at build time by a
  maximal-munch (longest-pattern-first) scan over the decoded opcode stream,
  falling back to per-opcode dispatch when no pattern matches; a fused run does
  the combined work of its constituents in one tail-call hop instead of one hop
  per opcode.

- A fused handler reuses the single-opcode handler ABI: each fused op declares a
  pop/push stack effect that drives the same per-depth register-variant wrappers
  over one impl, and a fused op whose read and write operands occupy overlapping
  stack positions aliases them to the same register (read-before-write).

- Which adjacent-opcode sequences become super-instructions is discovered
  automatically: a pattern trie built from one max-window profiling run captures
  exact N-gram prefix counts, and a greedy selector picks a globally non-redundant
  set (subtracting each chosen pattern's count from its strict prefixes) under the
  immediate-encoding budget, then emits the complete fused-pattern table.

- Discovery profiles the raw, unfused instruction stream: a global fusion-disabled
  flag makes the builder emit one handler per opcode during the discovery run, and
  candidate selection sees the original opcode sequence, unbiased by whatever
  fusions already exist.

- Each fused C handler body is composed by the build from the same per-opcode
  semantic macros the standalone handlers expand, computing its constituents
  identically to the per-opcode handlers.

## Facts

- 2025-12-06 (b7b5dc6a) rationale: build-time super-instruction fusion was
  removed entirely when the slot-tracking operand model eliminated the local/const
  copies the old fused super-instructions existed to collapse, then reintroduced as
  a maximal-munch OpFuser once the SP-based operand model made dispatch count the
  dominant cost again — fusion pays off when dispatch count dominates, not when it
  only collapses copies a smarter operand model can already elide (diff).

- 2026-06-14 rationale: the multi-op-read consumer abstraction (OpStream) was
  added so a fusion pass can read more than one decoded instruction at a time —
  it is the foundation for consuming a super-instruction window rather than one
  opcode per step (author).

- 2025-12-13 (a2b353da) rationale: which adjacent-opcode sequences become
  super-instructions is chosen empirically — a built-in sequence profiler records
  sliding windows of executed handlers and ranks them by reduction potential
  ((sequence_length-1) * frequency), so the fusion patterns are driven by measured
  hot sequences in real workloads rather than guessed (diff).

- 2025-12-15 (a258d56c) rationale: fusion exists to cut dispatch count — each
  fused handler does the combined work of its constituent opcodes in one tail-call
  hop instead of one hop per opcode, so the win scales with (constituent count - 1)
  times the run's frequency (diff).

- 2025-12-15 (10fc8467) statement: fusion candidates exclude sequences with
  control flow in non-terminal positions — a branch may appear only as the last
  opcode of a fused run, so the fuseable-sequence ranking filters out windows with
  mid-sequence control flow (diff).

- 2025-12-15 (c258b03d) pitfall: a fused run that ends in a branch must be
  registered in the finalizer's second-pass branch-target patch set, and because
  its 64-bit target field can land in imm1/imm2 (offset+memidx already fill imm0)
  rather than imm0 like a plain branch, the re-encode must propagate all three imm
  words; a pattern added with compilable handlers but left out of the patch set and
  re-encoded only imm0 leaves its branch target at the first-pass null placeholder
  — every branch-terminated fused pattern is part of the finalizer's correctness
  boundary (diff).

- 2026-02-06 (701894ef) rationale: the fusion code generator stopped hand-listing
  each fusible opcode (a ~20-op table that panicked on anything else) and instead
  classifies ops by category (pure binop / trapping binop / pure unary / load /
  store, plus the special local/const/br_if cases), with the opcode-name mapping
  collapsed to a mechanical snake_case-to-UPPER_SNAKE; this lets any base numeric op
  participate in a fused pattern without per-op generator edits, made possible
  because the semantic macros now cover the whole op set (diff).

- 2026-01-25 (1dad15de) rationale: fused handlers reuse the single-opcode
  two-layer ABI rather than a separate calling convention — each fused op declares a
  tos_pattern (pop/push) that drives generation of the same per-depth register-
  variant wrappers over one impl, the variant is selected by the fused op's output
  stack depth, and a fused op whose read and write operands occupy overlapping stack
  positions aliases them to the same TOS register (read-before-write) (diff).

- 2026-02-06 (2716f6bc) rationale: the auto-discovered fused patterns are split out
  of the hand-authored handler spec into a separate fused-pattern file the build
  merges in; discover-fusion writes the complete regenerated set directly to that
  file and no longer dedups against existing patterns because it owns the whole
  file, closing the loop discover-fusion -> overwrite -> build, with the only
  cross-reference being reserved op names so generated fused names cannot collide
  (diff).

## Moves

- 2026-02-06 (478aee26) replaced [[hand-written-fused-handlers]]: each fused
  super-instruction required its C body hand-written in fused.c and a matching
  hand-written Rust matcher and encoder in fusion.rs/fusion_emit.rs, so every
  new fused pattern was three hand-edited artifacts that had to stay in sync;
  moving to a build-time generator (gen_fusion.rs) that emits fusion.rs,
  fusion_emit.rs, and the fused C handlers from the declarative `[[fused]]`
  entries in handlers.toml — composing the C bodies from SEM_*
  base-instruction semantic macros in semantics.h, with each encoded field
  tagged by the 'from' index of the pattern element it comes from — makes a
  fused pattern a single declarative table entry, generating all three
  artifacts mechanically (diff).

- 2026-02-06 (fbbce862) replaced [[profiler-ranked-fusion-selection]]: the
  profiler only printed a flat frequency ranking of fuseable windows (scored
  length-1 times frequency) that the author then read and hand-coded into
  `[[fused]]` patterns, which double-counted overlapping sequences and could
  not reason about prefix relationships; the discover-fusion command builds a
  pattern trie that captures exact counts for every N-gram prefix from one
  max-window run and greedily selects a globally non-redundant set,
  subtracting each chosen pattern's count from all its strict prefixes
  (prefix-overlap adjustment) and enforcing the 192-bit (3x64)
  immediate-encoding budget, then auto-generates the complete `[[fused]]` TOML
  entry (name, encoding fields, TOS pattern) instead of a human transcribing
  the ranking (diff).
