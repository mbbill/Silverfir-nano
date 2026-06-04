---
commit: 4bb1de8
---
Stable Rust provides neither guaranteed tail calls (`musttail`) nor the
`preserve_none` calling convention that the zero-prologue handler chain requires.
Nightly's `become` / `explicit_tail_calls` is incomplete and has no `preserve_none`
equivalent. Consequence (an infeasibility fact, not a benchmark): the hot handler
chain cannot be written in stable Rust at all, so it is generated **C** reached
through a Rust→C trampoline — the project is ~91% Rust but its core dispatch loop
is not, and any non-C handler pays a wrapper-to-Rust call. This is a constraint of
the toolchain at the time, and is exactly the kind of "the design is infeasible
under these constraints" fact that pins an option's cost.
