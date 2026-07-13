- A post-optimize SSA pass removes the in-loop "memory counter reload" idiom
  (store `counter+1` to a cell, byte-store at `buffer[counter]`, reload the
  same cell to feed the `!= B` exit compare — clang's `ctx->len` pattern):
  when the reload's value is provably the just-stored value, the reload is
  deleted and the consuming `local.set` is repointed at a recomputation of
  the stored expression (`counter_forward.rs`).

- A cell is `(root local, constant offset, width)` with the root assigned at
  most once, in the entry block; windows are single-block store-to-load spans
  containing no call and no unclassifiable memory write.

- Intervening stores are admitted by two oracles: constant-offset stores from
  the same root with provably disjoint byte ranges, and (i32 cells, at most
  one per window) the counter byte-store `base + counter` where `base` is a
  reaching-definition-unique alias of `root + delta` and
  `delta + B <= cell offset`.

- The bound `B` is proven by closed induction over the counter's reaching
  definitions: every definition observable at the byte-store is a constant
  `< B` or a guarded cell load (immediately stored to the counter and
  compared `!= B` by the block terminator, with the `== B` successor
  reassigning a constant `< B` before any counter read); the induction treats
  calls on the reset path as counter-transparent (locals are unreachable
  from callees).

- The rewrite recomputes instead of forwarding: the stored value must be
  `Add(local.get S, const)` with `S` unwritten through the window, and the
  load's result must have exactly one consumer, a `local.set`; the load is
  replaced by a fresh get-add pair. Values stay single-use.

## Facts

- 2026-07-13 measurement: the pass removes the sha256 hot-loop reload and
  restores 277.3±0.8 MB/s under the preserved-class default (from 225 broken,
  267 pre-regression; Cranelift 249) with corpus impact confined to sha256
  (−12 native instructions, all other modules byte-identical); compile-RAM
  +2.50%/+3.63% coremark/lua within the +10% ceiling; spectest 257/31/0
  (sourced).

- 2026-07-13 measurement: on Apple M4 the same-address store→load chain is a
  pipeline hazard beyond its one load: native microbenchmarks reproduced the
  JIT gap (948 vs 702 cycles per 64-byte chunk) and isolated the law — the
  in-memory counter chain is the carrier (registerizing it recovers ~90%),
  ANY extra load per iteration restores full speed (even one dependent on the
  chain), an independent 4-cycle ALU delay recovers ~78%, a predictable exit
  branch ~57%; strb/line adjacency, str/ldr addressing forms, and publish
  scheduling were all measured irrelevant. PMU (xctrace custom instrument):
  the slow shape shows +129 cycles/chunk of MAP_STALL_DISPATCH and +1.5
  conditional-branch mispredicts per chunk; memory-order violations equal
  (sourced).

- 2026-07-13 rationale: the pathology surfaced when the preserved-class
  contract removed a per-iteration frame reload that had been accidentally
  covering it — the old code was fast because a redundant load acted as
  medicine. Optimizations that delete loads from tight loops carrying a
  same-address store→load chain can expose this class of stall; the scan
  found the enabling condition (chain with ≤2 other loads in the loop) at
  only 13 of 737 in-loop reload sites in the corpus, sha256's being the only
  hot one (sourced).

- 2026-07-13 measurement: corpus scan of the reload idiom — 27 strict counter
  patterns, 676 plain in-loop same-address reloads (245 with zero alias
  hazard), 1828 straight-line cases across the 9 modules. The hottest
  non-sha256 counter-shaped sites (bzip2 bit-writer `s->numZ`, lz4 `*token`)
  need pointer-provenance reasoning, not counter bounds; the hazard-free
  in-loop sites (c-ray f64 struct temporaries, a luaV_execute chain) need a
  value-duplication mechanism to satisfy linearity. Both are recorded doors,
  not part of this pass (sourced).

- 2026-07-13 pitfall: the middle SSA's linear single-use discipline is
  enforced by a debug-only validator (`validate_linear_op_uses`) and assumed
  by machine register ownership — release builds do not check it. The first
  forwarding implementation repointed the load's uses at the stored value,
  passing release compiles while violating the contract (caught on sqlite in
  debug); a legal rewrite must materialize a fresh value (sourced).
