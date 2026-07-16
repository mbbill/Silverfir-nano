- Indexed memory-address fusion is a shared MachineIR peephole: it recognizes
  the lowering's recognizable address-computation-plus-load/store sequence
  (base+index, the zero-extended UXTW variant, and an optionally absorbed
  positive i32 Wasm offset) and rewrites it into first-class portable
  IndexedLoad/IndexedStore ops each backend maps to its best addressing mode
  (`fuse_indexed_memory`).

- IndexedLoad/IndexedStore carry base, index, an index-extend mode (None or
  ZeroExtend32), an i32 offset, width and load-extension, with semantics
  dst <- mem[base + extend(index) + offset] (`MachineIndexExtend`).

- When the wasm address is a materialized constant (move-imm feeding the
  extend), the pass folds the entire address computation into the original
  load/store's immediate offset instead of forming an IndexedLoad/Store,
  keeping the memory base as the base operand; the fold is exact
  (base + zext32(imm) + wasm_offset) and requires the constant and extend
  registers dead after the memory op (`try_fuse_const_addr`).

## Facts

- 2026-03-23 (0a30b592) statement: when IndexedLoad/IndexedStore were
  introduced only ARM64 mapped them — x86_64 and armv7a emitted todo!() and the
  emulator returned an internal error — so the portable op existed ahead of
  three of its four consumers (code).

- 2026-03-23 (af139e58) measurement: enabling the fusion produces correct,
  smaller code (fewer MachineIR ops), but the size reduction shifts function
  layout unpredictably and regressed cache/alignment-sensitive benchmarks
  (sha256 -17%, coremark -3%); the pass was briefly gated off behind that
  observation and re-enabled one commit later once a stable-base emit strategy
  plus page-boundary-aware function alignment removed the regression (code).

- 2026-03-23 (6b4ba56e) measurement: for IndexedLoad/IndexedStore with
  offset!=0 the offset is folded into the INDEX register and the original
  memory base is kept as the load/store base operand, never base+extend(index)
  computed into a scratch used as the base; modern CPUs tag speculative
  store-to-load forwarding on the base register, and a freshly-computed scratch
  base loses the address predictor's confidence and stalls the load 5+ cycles —
  measured as a 17% regression on SHA-256 on Apple Silicon with the
  scratch-base form; x86_64 is exempt because its displacement is part of the
  instruction encoding, not a computed register (sourced).

- 2026-03-25 (c99ef2f0) pitfall: the earlier optimization of using the load's
  GP destination register as the scratch holding the adjusted index (to break
  false-dependency chains between consecutive loads) is unsafe — when the
  IndexedLoad's dst maps to the same machine register as its base, materializing
  the index into dst clobbers the base before the load reads it and corrupts the
  address; the emit reverted to always taking a fresh pool scratch (code).

- 2026-04-07 (5e59319b) measurement: fusing consecutive IndexedLoad/IndexedStore
  ops that share (base,index,extend) into one shared 'add Xs,Xb,Wi,UXTW' plus N
  immediate-offset accesses ('burst' fusion) is a measured loss on Apple Silicon
  M-series for integer-load-bottlenecked workloads (coremark 32044 burst-on vs
  34048 off, +6.3%): the M-series macro-fuses 'add x,x,#imm' with the following
  'ldr w,[base,x]' into one AGU op and renames 'mov w,w' at zero latency, so the
  mov+add+ldr-reg sequence executes as a single load, while the burst form's
  'add ...,UXTW' does not fuse with the load AGU and adds a dependent cycle to
  the whole group; the result motivates reanimating burst fusion only on
  microarchitectures without macro-op fusion / move elimination (code).

- 2026-07-16 (95fec85d) measurement: constant-address folding fires mostly in
  wasm libm and static-data code — in c-ray, pow's polynomial coefficient
  tables (the module's fourth-hottest function at 7.8% of samples) go from
  mov-imm + register-indexed load to a single scaled-imm ldr; c-ray 4000x4000
  improved 1989 -> 1970 ms mean over 5/5 interleaved rounds with the DVFS band
  checked, narrowing the same-session gap to V8 from 3.2% to 2.2% (code).

- 2026-07-16 (95fec85d) pitfall: the fold hands linear-memory accesses to the
  store-forwarding and load-reuse passes in the same base+imm Load/Store shape
  as frame accesses, and those passes' invalidation was frame-tuned — exact
  same-base range overlap only, with IndexedStore and bulk-memory ops
  invalidating nothing; the first folded build reused a load across an
  aliasing indexed store (memory_redundancy.wast caught it). Pre-fold this was
  latent only because every linear-memory address was recomputed into a fresh
  register per access, so tracked entries never survived to a stale reuse. The
  fold is only sound together with conservative alias rules in both passes:
  cross-base stores kill entries unless one side is a runtime-owned base
  (frame, context), and unknown-offset stores (indexed, bulk memory, table
  writes) kill every non-runtime-owned entry (code).

## Moves

- 2026-03-23 (0a30b592) replaced [[arm64-uxtw-indexed-fusion]]: a
  backend-private fusion form could only ever serve ARM64; lifting the fusion
  into the shared peephole as portable IndexedLoad/IndexedStore MachineIR ops
  makes one address-mode contract every backend maps to its best addressing mode
  (code)
