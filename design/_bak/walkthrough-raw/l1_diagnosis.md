# L1 Register Cache — Diagnosis Progress

## Status (2026-02-18)
L0+L1 implemented and passing all tests (88 spectest, coremark, lua fib_small).
CoreMark baseline (l0+l1+fusion): 7014

## What's Done (implementation)
- Full plumbing: l1 added to handler signature (10 params: ctx,pc,fp,l0,l1,t0,t1,t2,t3,nh)
- Builder: find_hot_locals returns top-2, dual transposition remap_local, init_l1 emitted
- Handlers: init_l1, local_get_l1, local_set_l1, local_tee_l1 (C impls + auto wrappers)
- Call/return: l1 spill (fp[1]=l1) and fill (l1=fp[1]) in call_local/return_epilogue
- Fusion: l1 exclusion guard added to gen_fusion_match.rs (rejects fused patterns where remap==1)
- Fusion TOML: 500 patterns, 160 l0-aware, 0 l1-aware

## Diagnosis Step 1: Per-handler coverage (DONE)
With profiler (fusion disabled, identity mapping for l0/l1), 308M instructions:
- L0 ops: 28M (9.05% of instructions, 24% of all local accesses)
- L1 ops: 22M (7.07% of instructions, 18% of all local accesses)
- Generic local ops: 68M (22.18% of instructions, 58% of all local accesses)
- L0+L1 covers 42% of local accesses, 58% remain generic

## Diagnosis Step 1b: Per-function local access profiles (DONE)
Added per-function, per-local-index runtime counters (C-side 2D array, keyed by init_l0 PC).
**Key result** — CoreMark top 3 functions (96% of all local accesses):

| Func | % of total | top-1 local (%) | top-2 locals (%) | # locals used |
|------|-----------|-----------------|-------------------|---------------|
| #6   | 44.7%     | local[0] 18.9%  | +local[3] 33.8%   | 23            |
| #5   | 26.6%     | local[2] 25.2%  | +local[4] 49.7%   | 19            |
| #10  | 24.4%     | local[3] 26.0%  | +local[2] 47.9%   | 10            |

**Weighted optimal coverage (all functions):**
- OPTIMAL l0: 26.1M / 118.1M = 22.1%
- OPTIMAL l1: 22.5M / 118.1M = 19.0% marginal (l0+l1 = 41.1%)
- REMAINING: 69.5M / 118.1M = 58.9% still generic

## Diagnosis Step 4: A/B test — register cache without fusion (DONE, SUSPICIOUS)
Added env var controls: `SF_FUSION_DISABLED=1`, `SF_HOT_LOCAL_MODE={none,l0,l0l1}`
Verified env vars work (debug prints confirm correct mode + fusion state).

**Results (fusion OFF, 3 runs each):**
```
                   Run 1    Run 2    Run 3    Avg
l0+l1:            3383     3316     3374     3358
l0 only:          3389     3406     3388     3394
neither:          3362     3306     3325     3331
```
All within noise (~2%). **Register cache alone gives ~0% benefit.**

## OPEN QUESTION — Does Not Make Sense Yet
The A/B test shows l0 register cache gives ~0% benefit without fusion.
But with fusion ON, l0-aware patterns give ~10% improvement over non-l0 fusion.
This is contradictory because:
1. Fusing `[local_get_l0, X, Y, Z]` saves the same dispatch count as `[local_get, X, Y, Z]`
2. The only difference inside the fused handler is `*p_l0` (register) vs `fp[idx]` (memory)
3. If that difference is zero for standalone handlers, it can't be non-zero inside fused handlers

**Possible explanations to investigate:**
- The "neither" identity mapping might still benefit from l0/l1 handlers for locals 0,1 (e.g., func #6 where local[0] IS the hottest — identity l0 = real l0 for that function)
- Something about how the test modes interact with init_l0/init_l1 and the remap
- The 10% l0 improvement attribution may be wrong — maybe it came from something else entirely
- Need to verify with more controlled experiment or assembly inspection

## Instrumentation Added
- `sf-nano-core/src/vm/interp/fast/mod.rs`: `hot_local_mode()` reads `SF_HOT_LOCAL_MODE` env var, `is_fusion_disabled()` reads `SF_FUSION_DISABLED` env var (both gated on `feature = "wasi"`)
- `sf-nano-core/src/vm/interp/fast/builder/mod.rs`: mode 0/1/2 dispatch for hot local selection
- `trampoline/vm_trampoline.c`: per-function per-local-index counters (FAST_PROFILE_ENABLED)
- `handlers_c/const_local.c`: PROFILE_LOCAL_ACCESS in generic local handlers, PROFILE_SET_FUNC in init_l0
- `sf-nano-core/src/vm/interp/fast/profiler.rs`: `take_func_local_profiles()` FFI
- `sf-nano-cli/src/discover_fusion.rs`: per-function local profile reporting + 1-gram handler counts
- Debug eprints in `hot_local_mode()` and builder (active in current build — remove when done)

## Key Files Modified (l1-specific, beyond l0)
- `sf-nano-core/build/fast_interp/op_classify.rs` — l1 ops classification
- `sf-nano-core/build/fast_interp/gen_fusion_match.rs` — l1 exclusion guard + l1 position checks
- `sf-nano-core/build/fast_interp/gen_fusion_c.rs` — l1 ops use *p_l1
- `sf-nano-core/src/vm/interp/fast/builder/hot_local.rs` — find_hot_locals returns top-2
- `sf-nano-core/src/vm/interp/fast/builder/stack.rs` — hot_local_1_idx_eff, has_l1(), dual remap_local
- `sf-nano-core/src/vm/interp/fast/builder/dispatch.rs` — l1 handler selection
- `sf-nano-core/src/vm/interp/fast/builder/emitter.rs` — emit_init_l1
- `sf-nano-core/src/vm/interp/fast/builder/mod.rs` — effective k1 computation, emit init_l1, mode dispatch
- `sf-nano-core/src/vm/interp/fast/builder/temp_inst.rs` — Handler type with l1 param

## Build Notes
- `cargo clean` REQUIRED after changing build scripts
- Use `--profile release-with-debug` for samply profiling (release has strip=true)
- Env vars: `SF_FUSION_DISABLED=1` disables fusion, `SF_HOT_LOCAL_MODE={none,l0}` controls mapping
- Profile build: `cargo build --release --features profile --bin sf-nano-cli`
- Discovery: `cargo run --release --features profile --bin sf-nano-cli -- discover-fusion --top 500 --window 5 -o handlers_fused.toml --workload benchmarks/coremark/coremark.wasm --workload benchmarks/lua/lua.wasm benchmarks/lua/fib_small.lua`
