# vm/wasm Boundary Audit

Scope: `crate::vm::wasm` (`pub(crate) mod wasm` inside `pub mod vm`). The reachable
visibility ceiling for everything here is **pub(crate)** — nothing in this subtree can
escape the crate. Any `pub` field/item is therefore over-declared by definition.

---

## 1. SUMMARY

| File | #items | #over_exposed | #crate-contract |
|------|-------:|--------------:|----------------:|
| `mod.rs` | 7 | 4 | 3 |
| `context.rs` | 20 | 17 | 2 |
| `control.rs` (ORPHANED) | 14 | 14 | 0 |
| `decode.rs` | 14 | 13 | 1 |
| `inline.rs` | 2 | 2 (dead) | 0 |
| `sir/mod.rs` | 3 | 1 | 2 |
| `sir/common.rs` | 13 | 7 | 6 |
| `sir/primitive_op.rs` | 3 | 1 | 2 |
| `sir/semantic_ir.rs` | 26 | 18 | 12 |
| **TOTAL** | **102** | **77** | **28** |

Tier totals across all files:

| Tier | Count |
|------|------:|
| crate-contract | 28 |
| wasm-internal | 27 |
| file-local | 16 |
| dead (orphaned `control.rs` + dead `inline.rs` fns) | 16 |
| (non-over-exposed crate-contract already correct) | 28 |

Headline: of 102 items, **77 are over-exposed**. 16 of those are dead (whole orphaned
`control.rs` + 2 unused `inline.rs` fns). The genuine surviving external surface is **28
crate-contract items**, concentrated in `sir/` (the SIR types) plus `decode_to_semantic_ir`
and `CompileContext::with_value_types`.

---

## 2. THE REAL EXTERNAL CONTRACT

The pub(crate) surface that genuinely crosses the `vm/wasm` boundary. This is the module
boundary; it survives.

### `sir/semantic_ir.rs` — the SIR data contract (frontend → middle)
- **SemanticProgram** (struct) — `vm::build`, `vm::middle` (mod/cfg/slot_ssa/rewrite::function/joint_plan::build). THE wasm→middle type.
- **SemanticOpKind** (enum) — `vm::middle` cfg/slot_ssa/rewrite::function/joint_plan::build + tests. Central op enum, exhaustively matched.
- **SemanticOp** (struct) — `vm::middle` cfg/joint_plan::build/tests::helpers.
- **SemanticCatchClause** (struct) — `vm::middle` cfg/slot_ssa/rewrite::function (travels in `TryTable.catches`).
- **ensure_prepare_supported** (method) — `vm::middle` (mod.rs:57). Fan-in 1, SIR→prepare handshake.
- **validate** (method, both cfg twins) — `vm::middle` (mod.rs:56) + tests::helpers.
- Data fields (kept accessible, see §4/§5): SemanticCatchClause.{tag_idx,payload_arity,forwards_exn,stack_drop,target}; SemanticOp.kind; SemanticProgram.{params,results,local_count,max_stack_height,ops,local_types,result_types,op_result_types}.

### `sir/primitive_op.rs` — shared leaf-op vocabulary
- **PrimitiveOpKind** (enum) — `vm::middle` (optimize/slot_ssa/cfg/sink_plan/cleanup/rewrite::{function,edge}/ssa_ir::ir/joint_plan::{build,init_locals}) + `vm::machine` (lower_*/gp32) + tests. The widest fan-in in the module.
- **stack_effect** (fn) — `vm::middle` optimize/rewrite::function/ssa_ir::ir/joint_plan::build.

### `sir/common.rs` — branch targets
- **SemanticTarget** (struct) + **new** + **index** — `vm::middle` slot_ssa/cfg/rewrite::function/tests::helpers.
- **BrTableEntry** (struct) + fields target/stack_drop/arity — `vm::middle::rewrite::function`.

### `context.rs` — decode driver input
- **CompileContext** (struct) — `vm::build` (build.rs:98/139). Construction site.
- **with_value_types** (method) — `vm::build` (build.rs:139). Sole constructor, fan-in 1.

### `decode.rs` — frontend entry point
- **decode_to_semantic_ir** (fn) — `vm::build` (build.rs:137). Fan-in 1, single-function module contract.

Shape: the contract is essentially **the SIR types + two pipeline-stage entry points**
(`decode_to_semantic_ir`, `CompileContext::with_value_types`). Two consumers only:
`vm::build` (drives compilation) and `vm::middle` (consumes SIR).

---

## 3. OVER-EXPOSURE WORKLIST

### `mod.rs`
- [ ] `sir` (mod, 9): pub(crate) -> private — no external code names `sir::`; re-exports in same file keep working. **medium**
- [ ] `common` (use-reexport, 12): pub(crate) -> pub(in crate::vm::wasm) — consumers control.rs/inline.rs only. **high**
- [ ] `context` (mod, 16): pub(crate) -> pub(in crate::vm::wasm) — only decode.rs uses it. **high**
- [ ] `inline` (mod, 18): pub(crate) -> pub(in crate::vm::wasm) — referenced only by decode.rs. **high**

### `context.rs` (CompileContext)
- [ ] `types` (field, 17): pub -> pub(in crate::vm::wasm) — read only in context.rs + decode.rs. **high**
- [ ] `store` (field, 18): pub -> pub(in crate::vm::wasm) — read only in vm/wasm. **high**
- [ ] `params` (field, 19): pub -> pub(in crate::vm::wasm) — read only decode.rs:1223. **high**
- [ ] `local_count` (field, 20): pub -> pub(in crate::vm::wasm) — decode.rs:552,1225. **high**
- [ ] `results` (field, 21): pub -> pub(in crate::vm::wasm) — decode.rs:539/980/1224. **high**
- [ ] `local_types` (field, 24): pub -> pub(in crate::vm::wasm) — decode.rs:1227/1243. **high**
- [ ] `result_types` (field, 26): pub -> pub(in crate::vm::wasm) — decode.rs:541/1228. **high**
- [ ] `resolve_block_type` (method, 52): pub(crate) -> private — callers only in context.rs:67,132. **high**
- [ ] `resolve_block_type_from_imm` (method, 65): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1730/1744/1753. **high**
- [ ] `resolve_block_result_types` (method, 73): pub(crate) -> private — callers only in context.rs:94,143. **high**
- [ ] `resolve_block_result_types_from_imm` (method, 89): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1731/1745/1754. **high**
- [ ] `resolve_type_index` (method, 100): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1145/1178/1191/1202. **high**
- [ ] `resolve_func_type` (method, 108): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1006/1167. **high**
- [ ] `resolve_tag_type` (method, 118): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1106/1801. **high**
- [ ] `resolve_try_table_block_type` (method, 130): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1790. **high**
- [ ] `resolve_try_table_result_types` (method, 138): pub(crate) -> pub(in crate::vm::wasm) — decode.rs:1791. **high**

### `decode.rs` (all file-local; drop pub(crate))
- [ ] `SemanticBuilder` (struct, 36): pub(crate) -> private — referenced only in decode.rs. **high**
- [ ] `SemanticBuilder::current_index` (method, 42): pub(crate) -> private. **high**
- [ ] `SemanticBuilder::push` (method, 46): pub(crate) -> private — decode.rs:565. **high**
- [ ] `SemanticBuilder::patch_target` (method, 52): pub(crate) -> private — decode.rs:570. **high**
- [ ] `SemanticBuilder::patch_br_table_target` (method, 68): pub(crate) -> private — decode.rs:580. **high**
- [ ] `SemanticBuilder::patch_try_table_catch_target` (method, 83): pub(crate) -> private — decode.rs:783 (no wrapper). **high**
- [ ] `SemanticBuilder::finish` (method, 98): pub(crate) -> private — decode.rs:1222. **high**
- [ ] `DecodeContext` (struct, 515): pub(crate) -> private — never in entry-point signature. **high**
- [ ] `DecodeContext::new` (method, 531): pub(crate) -> private — decode.rs:3006. **high**
- [ ] `DecodeContext::current_index` (method, 559): pub(crate) -> private. **high**
- [ ] `DecodeContext::push_op` (method, 564): pub(crate) -> private. **high**
- [ ] `DecodeContext::patch_target` (method, 569): pub(crate) -> private. **high**
- [ ] `DecodeContext::patch_br_table_target` (method, 574): pub(crate) -> private — decode.rs:780. **high**

### `sir/mod.rs`
- [ ] `common` (mod, 10): pub(crate) -> pub(in crate::vm::wasm) — all consumers inside vm/wasm. **high**

### `sir/common.rs`
- [ ] `SemanticIndex` (struct, 8): pub(crate) -> pub(in crate::vm::wasm) — only control.rs/decode.rs name it. **high**
- [ ] `SemanticIndex::new` (method, 12): pub(crate) -> pub(in crate::vm::wasm). **high**
- [ ] `SemanticIndex::as_usize` (method, 17): pub(crate) -> pub(in crate::vm::wasm). **high**
- [ ] `BrTableEntry.target` (field, 51): pub -> pub(crate) — read in middle + decode/inline. **high**
- [ ] `BrTableEntry.stack_drop` (field, 52): pub -> pub(crate). **high**
- [ ] `BrTableEntry.arity` (field, 53): pub -> pub(crate). **high**

### `sir/semantic_ir.rs` (15 fields: pub -> pub(crate); 2 helpers -> private)
- [ ] `SemanticCatchClause.tag_idx` (field, 28): pub -> pub(crate) — module/validator/functions.rs:590, op_decoder.rs:266. **high**
- [ ] `SemanticCatchClause.target` (field, 32): pub -> pub(crate) — cfg.rs:303, slot_ssa.rs. **high**
- [ ] `SemanticOp.kind` (field, 38): pub -> pub(crate) — cfg.rs:236, rewrite::function:400. **high**
- [ ] `SemanticProgram.local_count` (field, 171): pub -> pub(crate). **high**
- [ ] `SemanticProgram.max_stack_height` (field, 172): pub -> pub(crate) — joint_plan/build:226. **high**
- [ ] `SemanticProgram.ops` (field, 173): pub -> pub(crate) — rewrite::function:400. **high**
- [ ] `SemanticProgram.local_types` (field, 178): pub -> pub(crate) — rewrite::function:350. **high**
- [ ] `SemanticProgram.result_types` (field, 182): pub -> pub(crate) — rewrite::function:374. **high**
- [ ] `SemanticProgram.op_result_types` (field, 188): pub -> pub(crate) — rewrite::function:374. **high**
- [ ] `semantic_op_result_arity` (fn, 192): pub(crate) -> private — only in-file caller validate@324. **high**
- [ ] `SemanticProgram::requires_simd` (method, 212): pub(crate) -> private — only in-file caller ensure_prepare_supported@229. **high**

### `inline.rs` (dead — see §5; delete, not just tighten)
- [ ] `retain_inline_candidate` (fn, 70): **delete** (zero callers repo-wide). **high**
- [ ] `inline_calls_in_function` (fn, 252): **delete** (zero callers; module's only entry point). **high**

### `control.rs` (orphaned — delete entire file; 14 items, see §5). **high**

### NEEDS HUMAN CHECK (medium confidence)
- [ ] `sir/primitive_op.rs` `result_type` (fn, 308): pub(crate) -> pub(in crate::vm::wasm) — sole caller semantic_ir.rs:222, cfg-gated `#[cfg(not(sf_has_simd))]`; name is common, sole consumer conditional. **medium**
- [ ] `sir/common.rs` `SemanticTarget::pending` (method, 38): pub(crate) -> pub(in crate::vm::wasm) — decode-only; owning type is crate-contract. **medium**
- [ ] `sir/common.rs` `SemanticTarget::is_pending` (method, 43): pub(crate) -> pub(in crate::vm::wasm) — decode-only. **medium**
- [ ] `semantic_ir.rs` SemanticCatchClause.{payload_arity(29),forwards_exn(30),stack_drop(31)}: pub -> pub(crate) — confirmed via construction site, no direct middle read of these specific fields. **medium**
- [ ] `semantic_ir.rs` SemanticProgram.{params(169),results(170)}: pub -> pub(crate) — name collides with enum-variant fields; classification by construction. **medium**
- Note: `sir/common.rs` `SemanticTarget::index` (33) flagged **medium** but kept pub(crate) — no positively-confirmed external caller, but owning type is contract. Conservative: leave as-is.

---

## 4. TYPE SURFACE OUTLIERS

Types with a wide surface but little/none used **outside vm/wasm**:

- **CompileContext** (context.rs): 7 fields + 10 methods; external surface = just the type
  name + `with_value_types` (1 method, by `vm::build`). All 7 fields read only by decode.rs;
  9 resolvers internal (2 file-local). → Fields to pub(in crate::vm::wasm); 2 resolvers
  private; rest pub(in crate::vm::wasm). Essentially a vm/wasm-internal data-bag whose only
  justified pub(crate) surface is the type + constructor.

- **SemanticBuilder** (decode.rs): 6 methods + 1 field, **0** used externally → struct+impl private.

- **DecodeContext** (decode.rs): 5 methods + 11 fields, **0** used externally → struct+impl private.

- **SemanticIndex** (sir/common.rs): 2 methods, both used but **all inside vm/wasm** → lower
  whole type + both methods to pub(in crate::vm::wasm).

- **SemanticTarget** (sir/common.rs): 4 methods; only `new`+`index` are the cross-wasm
  contract. `pending`/`is_pending` are decode-only bookkeeping leaking into a contract type →
  consider lowering those two to pub(in crate::vm::wasm) (medium).

Types whose wide surface IS genuinely used (no narrowing, keep pub(crate)): **PrimitiveOpKind**
(intrinsic shared IR enum, ~150+ variants constructed in decode, matched crate-wide),
**SemanticOpKind**, **SemanticProgram** (all 8 fields read by middle/), **SemanticOp**,
**SemanticCatchClause** (all 5 fields part of contract), **BrTableEntry** (all 3 fields used).
For these the only action is the redundant-`pub`-keyword demotion (§5), not surface trimming.

---

## 5. SMELLS & ANOMALIES

### Module not declared (orphaned / dead file)
- **`control.rs` is ORPHANED.** No `mod control;` anywhere under sf-nano-core/src (only
  unrelated `arch/{arm32,arm64,x86_64}/mod.rs`). The file is not compiled; all 14 items
  (BlockKind, ControlFrame + 5 pub fields, ControlStack + frames + 5 methods) are dead.
  `ControlStack` has no constructor and is never instantiated. **Action: delete the file.**
  Worth a `git log` check to confirm it was planned-but-never-integrated scaffolding.

### Dead code behind blanket suppression
- **`inline.rs` is effectively a dead module.** Its only two pub(crate) entry points —
  `retain_inline_candidate` (70) and `inline_calls_in_function` (252) — have **zero callers
  repo-wide** (fan-in 0, not 1). The file carries a module-scope `#![allow(dead_code)]`
  (line 10) that masks this — exactly the band-aid CLAUDE.md forbids. **Action: confirm the
  inliner is not about to be wired in, then delete the module + the allow attribute.**

### `pub` (fully-public) fields/items inside a pub(crate) subtree (redundant width)
Because `vm/wasm` is pub(crate)-capped, every `pub` field below is strictly broader than its
reachable visibility. None can be seen outside the crate; demote to pub(crate) (or narrower):
- `context.rs`: 7 `pub` fields on CompileContext (17-26) — all read only within vm/wasm → pub(in crate::vm::wasm).
- `sir/common.rs`: BrTableEntry.{target,stack_drop,arity} (51-53) → pub(crate).
- `sir/semantic_ir.rs`: SemanticCatchClause 5 fields (28-32), SemanticOp.kind (38), SemanticProgram 8 fields (169-188) → pub(crate).
- `control.rs`: ControlFrame 5 fields + ControlStack.frames — dead, deleted with the file.

### Rc / RefCell / Cell across the boundary
- **None.** No file exposes Rc/RefCell/Cell in any boundary signature. SIR types are plain
  value/Copy data (SemanticIndex/SemanticTarget are Copy u32 newtypes; CompileContext is
  Clone/Copy over shared borrows + scalars). No shared-mutable lifetime crosses the boundary.

### Fan-in-1 (single external consumer) — NOT relocation candidates
All three are clean pipeline-stage boundaries, keep in place:
- `decode_to_semantic_ir` → only `vm::build` (build.rs:137); build.rs is the driver, decode is its stage.
- `CompileContext` / `with_value_types` → only `vm::build`; build.rs is the construction site.
- `ensure_prepare_supported` → only `vm::middle` (mod.rs:57); SIR→prepare handshake.

### Dead-at-pub(crate) helpers (should be private, not deleted)
- `semantic_ir.rs` `semantic_op_result_arity` (192) — only in-file caller validate@324.
- `semantic_ir.rs` `SemanticProgram::requires_simd` (212) — only in-file caller ensure_prepare_supported@229 (don't confuse with unrelated `Module::requires_simd`).
- `context.rs` `resolve_block_type` (52), `resolve_block_result_types` (73) — file-local.

### Other anomalies
- **API asymmetry in decode.rs:** `patch_target` and `patch_br_table_target` each have a
  DecodeContext pass-through wrapper, but `patch_try_table_catch_target` is called directly on
  `self.builder` (decode.rs:783) with no wrapper. Stylistic, not a correctness issue.
- **Inventory drift:** the original inventory expected `DecodedModule`/`ParsedModule`/
  `decode_module` under `decode`; the real entry is `decode_to_semantic_ir`. Guessed names
  do not exist.
- **Conditionally-dead:** `primitive_op::result_type` sole consumer is `#[cfg(not(sf_has_simd))]`;
  it may be unreferenced on sf_has_simd builds. Conditionally, not unconditionally, dead — do
  not delete.

---

## 6. INTERNAL EDGES (wasm-internal dependency graph)

Within `crate::vm::wasm`, who depends on whom (via `super::`/`sir::` paths):

```
decode.rs    ──> context (CompileContext)
decode.rs    ──> common  (SemanticIndex, SemanticTarget, BrTableEntry)
decode.rs    ──> primitive_op (stack_effect)
decode.rs    ──> semantic_ir (SemanticOp/Kind, SemanticCatchClause, SemanticProgram)
decode.rs    ──> inline (helpers)        [inline currently dead]
inline.rs    ──> common  (BrTableEntry, SemanticTarget)
inline.rs    ──> primitive_op (stack_effect)
inline.rs    ──> semantic_ir (SemanticProgram)
context.rs   ──> (self-contained; reads its own fields)
sir/semantic_ir.rs ──> common (super::common: SemanticTarget)
sir/semantic_ir.rs ──> primitive_op (super::primitive_op: stack_effect, result_type)
control.rs   ──> common  (SemanticIndex, SemanticTarget)   [control orphaned/dead]
mod.rs       re-exports: common, primitive_op, semantic_ir (from sir/)
```

Sinks (depended on, depend on nothing in-module): **common**, **primitive_op**, **context**.
Roots (nothing in-module depends on them): **decode** (the only live driver), plus the two
dead files **inline** and **control**.

**No dependency cycles** among submodules. The live graph is a clean DAG rooted at `decode`,
which fans into `context` + the three `sir/` leaves. `sir/semantic_ir` → `sir/common` and
`sir/primitive_op` is the only intra-`sir` edge (also acyclic).
