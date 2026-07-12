- The full per-(operation, type, register-permutation) handler set is generated
  at build time from a declarative spec: each operation declares its
  arity-signature and scalar types, and a rule-based schema (num inputs, output
  placement, input constraints, global register count) lets the build step
  expand the register permutations programmatically for any register count
  (`arity_patterns.toml`, `build.rs`).

- Each operation's compute logic is hand-written once as a canonical
  implementation taking pointer arguments under a uniform signature with an
  explicit destination pointer; the generator emits only thin per-permutation
  wrappers that bind those pointers to the right slots, and adding an operation
  does not duplicate its body across permutations.

- Operations encode their type in the operation name (i32_add, i64_add) rather
  than being parameterized by type, following WebAssembly's own fixed-signature
  opcode design.

- The LIR-physical-register-to-XIR-permutation mapping is generated from the
  same spec that generates the handlers; the mapping stays in lockstep with
  the handler set at any register count and enumerates every register
  arrangement the allocator can emit (including same-register arrangements).

## Facts

- 2025-10-19 (e75a34dd) rationale: generating the handler variants from one DSL
  rather than hand-writing them keeps trap handling and operation/type/slot
  coverage consistent and makes adding a new type or operation a spec edit; the
  type is encoded in the operation name following WebAssembly's fixed-signature
  opcode design (code).

- 2025-10-20 (18d821be) rationale: canonical handler implementations come in two
  flavors behind one FFI-compatible signature — pure-computation operations in C
  for inlining into the wrappers, runtime-touching operations in safe Rust
  through the store — both driven by the same generated permuted wrappers, so the
  C-vs-Rust split is a property of the operation, not of the dispatch (code).

- 2025-10-25 (624771a6) rationale: select is a single type-agnostic handler
  rather than one per value type, because every type's select is the same 64-bit
  conditional copy of the operand word; collapsing them removed redundant
  permutations where the per-type code was byte-identical (code).

- 2025-11-10 (2520eaa7) pitfall: the permutation spec must enumerate every
  (input, output) register arrangement the allocator can emit or lowering hits an
  unreachable arm; the store set, which had assumed the two inputs occupy
  different registers, gained the same-register arrangements (code).

- 2025-11-13 (59b83416) pitfall: a 3-input operation must enumerate the
  duplicate-register arrangements (e.g. op(v0,v0,v1) ... op(v2,v2,v2)), not just
  the 6 all-distinct ones, because register allocation can legally assign one
  physical register to two or three operands — a 3-input op needs all 27 input
  arrangements (code).

- 2025-11-10 (593bd2cf) rationale: unlike arithmetic handlers whose register
  slots are baked into the permutation, the spill-load/spill-store/copy handlers
  choose which register slots to touch from their immediate fields at run time,
  so the generator hands them pointers to all abstract registers rather than a
  permutation-shuffled subset (code).

- 2025-11-10 (67a4d17e) pitfall: memory load/store width was once built into a
  per-width handler-name string fed to the mapping — names that were never in the
  handler spec, so a sub-width request hit the generated 'Unknown op' panic and
  left sub-width memory ops broken; lowering now always requests one load/store
  handler and carries width as a runtime size code (code).

- 2025-10-10 (aba1636a) statement: in the first generation every handler was
  hand-written Rust behind a macro, declared one at a time in the C trampoline
  header and surfaced to Rust by a bindgen shim; the build-time generator
  replaced this hand-maintained one-declaration-at-a-time scheme (code).

- 2025-11-27 (6a751e31) pitfall: multiply-add fusion is spec-compliant only if
  the fused handler avoids a hardware FMA — a true fma rounds once, while wasm
  requires the mul and add to round separately, so the madd handler forces the
  product through a volatile intermediate to stop the C compiler contracting it
  back into an FMA; integer madd has no rounding concern and is always safe
  (code).

- 2026-02-12 (b0f04750) rationale: the single return handler was split by result
  arity (return_void / return_one / return 2+) so the common cases avoid per-call
  overhead — the same per-operation specialization principle the backend applies
  to register permutations, extended to the return-arity axis (code).

- 2025-10-23 (a7a48d92) pitfall: the generated tail-call must thread the next
  instruction pointer, not the current one — the wrapper computed `next` but then
  tail-called it with the old `pc` still in the argument pack, so every handler
  re-ran with the wrong program counter; the fix passes `next` explicitly as the
  new pc (code).

- 2025-10-13 (885c3d4b) pitfall: the trapping float-to-int truncation range check
  cannot compare the float against the integer bound itself — floats lack the
  precision to represent every integer near the bound, so the guard must compare
  against the next representable value, and each (source-float-width,
  target-int-width) pair needs its own boundary constant (code).

- 2025-10-25 (3ffe176a) pitfall: the trap boundaries for non-saturating
  float-to-int truncation are off the obvious integer limits — unsigned conversions
  must accept the half-open (-1.0, max] range, and f64-to-i32 signed must use a
  bound just below -2^31 because f64 can represent values there that still truncate
  in range (code).

- 2025-10-13 (bbabb35d) pitfall: the wasm f32/f64 nearest operation rounds halves
  to even (banker's rounding), not half-away-from-zero, so Rust's round() is the
  wrong primitive and round_ties_even() is required (code).

- 2025-10-25 (c40b82b5) pitfall: wasm f.min/f.max must return NaN if either operand
  is NaN, whereas C fminf/fmaxf return the non-NaN operand; the handlers explicitly
  test isnan on both inputs and produce NaN (code).

- 2025-12-05 (75b2b40c) pitfall: f32/f64 min/max must return the WebAssembly
  canonical NaN (quiet NaN with all-zero payload) when either operand is NaN, not
  the C NAN macro whose bit pattern is implementation-defined; the canonical NaN is
  constructed from explicit bits (code).

- 2025-10-27 (b3083bda) pitfall: the C signed-remainder handlers must special-case
  INT_MIN % -1 and return 0 before evaluating lhs % rhs — on the host the C `%`
  there is undefined behavior and the CPU's division traps on the overflow, whereas
  WebAssembly defines rem_s(INT_MIN, -1) = 0; the divide-by-zero guard alone does
  not cover this boundary (code).

## Moves

- 2025-10-24 (d097f067) replaced [[named-permutation-generator]]: the old
  generator named each permutation with an opaque string (V0V1_V0) and decoded it
  through per-signature hardcoded match arms that mapped window slots onto a fixed
  canonical layout (pv0=lhs, pv1=rhs, result implicitly to pv0), which could not
  express an output landing in a slot chosen independently of the inputs;
  declaring each permutation as explicit inputs[] and outputs[] slot vectors and
  giving every handler a uniform signature with an explicit pv_dst output pointer
  makes the result slot a first-class free parameter and removes the special-cased
  shuffling (code).

- 2025-11-10 (11f72dc7) replaced [[split-comm-noncomm-families]]: the commutative
  family kept only 3 of the operand arrangements by exploiting operand-order
  freedom and neither specialized family enumerated same-register arrangements, so
  the allocator could emit register combinations that had no handler; one Sig_2_1
  family enumerating all 9 register permutations (including the same-register
  V0V0V0/V1V1V1/V2V2V2 cases) covers every allocator output, at the cost of
  generating the full handler set for commutative ops too (code).

- 2025-11-29 (11a2cdff) replaced [[hand-written-reg-mapping]]: the hand-written
  mapping enumerated one match arm per register permutation for 3 registers and
  could not scale to 8 (e.g. 512 arms for 3-input signatures); generating the
  mapping functions in build.rs from the same arity_patterns.toml that generates
  the handlers keeps the LIR-register-to-XIR-permutation mapping in lockstep with
  the handler set at any register count (code).

- 2025-11-29 (005fae86) replaced [[explicit-permutation-spec]]: listing every
  register permutation by hand in arity_patterns.toml was tractable for three
  registers but explodes to hundreds of entries per signature at eight (512 for
  3-input forms); a rule-based schema (num_inputs, output_type
  any/in_place/first_input, input_constraint non_overlapping, plus a global
  register count) lets build.rs generate the permutations programmatically for any
  register count (code).
