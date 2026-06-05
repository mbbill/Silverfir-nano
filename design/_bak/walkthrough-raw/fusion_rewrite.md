# Fusion Rewrite Plan

## Core Insight
After IR lowering, variants are resolved. `add_D1` and `add_D2` are DIFFERENT instructions.
JIT and fusion both fuse IR instructions — JIT dynamically, fusion statically.
The fused handler's variant = first op's variant from IR. No runtime variant computation.

## What's Wrong Now
The current fusion implementation is identical to the old Wasm-level system:
- Ignores variants during matching
- Computes variant at runtime from `ref_depth = h + max(0, push - pop)`
- Uses `get_register_args_for_pattern(pop, push, variant_idx)` for wrapper generation
- The wrapper convention assumes variant encodes reference depth based on (pop, push), but the first op's variant encodes differently for push vs pop ops

## The Fix: Two Changes

### 1. Wrapper generation for fused handlers
Replace `get_register_args_for_pattern(pop, push, variant_idx)` with a fused-specific version.

**Key formula**: derive `h%4` (entry height mod 4) from variant_idx:
- Pop-first pattern (first op is binop/set/tee/drop): `h%4 = (variant_idx + 1) % 4`
- Push-first pattern (first op is const/get/global_get): `h%4 = variant_idx`

Then compute register indices directly:
- Input at position p (from TOS at height h): `t[(h%4 - p + 4) % 4]`
- Output at position p (from TOS at height h+net): `t[((h%4 + net) - p + 4) % 4]`
  where net = push - pop (can be negative, use modular arithmetic)

**Register mapping per (pop, push):**
- pop2_push1 (net=-1): lhs=t[(h%4-2+4)%4], rhs=t[(h%4-1+4)%4], dst=lhs
- pop1_push1 (net=0): src=t[(h%4-1+4)%4], dst=src
- pop0_push1 (net=+1): dst=t[h%4]
- pop2_push0 (net=-2): addr=t[(h%4-2+4)%4], val=t[(h%4-1+4)%4]
- pop1_push0 (net=-1): src=t[(h%4-1+4)%4]
- pop1_push2 (net=+1): src=t[(h%4-1+4)%4], dst0=src, dst1=t[h%4]
- pop0_push2 (net=+2): dst0=t[h%4], dst1=t[(h%4+1)%4]

### 2. Fusion matching: use ir[pos].variant directly
In `gen_fusion_ir_match.rs`, replace the `ref_depth` computation with:
```rust
let variant_idx = (ir[pos].variant as usize).saturating_sub(1);
let handler: Handler = handler_lookup::XXX[variant_idx];
```
No formula needed — the variant is already in the IR.

## File Organization
Move all fusion runtime code to `sf-nano-core/src/vm/interp/fast/fusion/`:
- `mod.rs` — module root, `resolve_fusion()`
- Include generated `fast_fusion_ir_match.rs`
- Currently in `builder/fusion.rs` — move out of builder

## Build Script Changes
- `gen_c_wrappers.rs`: use `get_fused_register_args()` for fused handlers (new function)
- `tos_config.rs`: add `get_fused_register_args(pop, push, variant_idx, first_op_is_push)`
- `gen_fusion_ir_match.rs`: use `ir[pos].variant` directly, remove ref_depth

## How to determine first_op_is_push
Check the first element of `fused.pattern`:
- Push ops: `i32_const`, `i64_const`, `local_get`, `local_get_l0/l1/l2`, `global_get`
- Pop ops: everything else (binops, unops, local_set, local_tee, stores, loads, br_if, if_, drop)
Note: local_tee is pop-first (reads TOS, writes to local, keeps TOS)
Note: loads are pop-first (pop addr, push value — net 0 but starts with pop)

## Testing
1. All 88 spec tests pass with fusion
2. CoreMark runs correctly
3. WASI benchmarks pass (mandelbrot, stream, lua, coremark)

## Current State
- 190 patterns, spec tests pass, CoreMark 6560
- Working but using old approach (ref_depth formula)
