- Every fused super-instruction's C handler body is written by hand in fused.c.

- The fusion pattern matcher and the per-pattern operand encoder are written by
  hand in builder/fusion.rs and builder/fusion_emit.rs.

- Adding a fused pattern means editing the C handler, the Rust matcher, and the
  Rust emitter in lock-step.

- A fused pattern's encoding fields list bit widths only, with the source pattern
  element implied by hand-written code rather than a declared 'from' index.

## Moves

- 2026-02-06 (478aee26) replaced by [[fusion]]: each fused super-instruction
  required its C body hand-written in fused.c and a matching hand-written Rust
  matcher and encoder in fusion.rs/fusion_emit.rs, so every new fused pattern
  was three hand-edited artifacts that had to stay in sync; moving to a
  build-time generator (gen_fusion.rs) that emits fusion.rs, fusion_emit.rs,
  and the fused C handlers from the declarative `[[fused]]` entries in
  handlers.toml — composing the C bodies from SEM_* base-instruction semantic
  macros in semantics.h, with each encoded field tagged by the 'from' index of
  the pattern element it comes from — makes a fused pattern a single
  declarative table entry, generating all three artifacts mechanically (code).
