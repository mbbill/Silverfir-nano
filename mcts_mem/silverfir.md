- Silverfir is a `no_std` WebAssembly engine (current codebase: Silverfir-nano): the only dependency it
  requires at runtime is `alloc`, and all heap traffic routes through a
  single tracked-allocation facade (`tracked_alloc`) rather than `alloc`
  directly.

- The engine is self-contained: it carries its own binary decoder, validator,
  value model, reference and GC heaps, and in-tree WASI host layer rather than
  delegating any of these to external WebAssembly or WASI crates.

- Execution has two engines, chosen per instance by the embedder: the
  native JIT (the default — every Wasm function compiles to machine code
  before it runs) and a folded-stack-machine interpreter tier; see
  [[execution-model]] and [[interpreter]].

- Verification and code generation both run on the target device from the
  shipped `.wasm` artifact; the deployment input is portable Wasm, never a
  relocatable machine-code blob.

- A module is compiled eagerly: instantiating a module compiles all of its
  local functions to native code up front (`ensure_module_compiled`), not
  lazily on first call.

- The engine targets the WebAssembly 3.0 feature set: garbage collection,
  exception handling, baseline and relaxed SIMD, tail calls, 64-bit memories
  and tables, multiple memories, typeful references, and extended constant
  expressions.

- The compiler is a fixed multi-stage pipeline lowering Wasm bytecode through
  three intermediate representations to native code: Wasm → Semantic IR →
  prepared SSA-IR → MachineIR → native code.

- One shared frontend, middle-end, and register allocator drive every native
  backend; ISA-specific choices are confined to the final native-emission
  stage.

## Facts

- 2026-02-14 (a8528504) rationale: the design targets zero runtime dependencies
  for minimal binary size — num_enum/thiserror/anyhow/smallvec/log/env_logger and
  others are removed and hand-reimplemented (TryFrom<u8> by hand, Display by hand,
  Vec instead of smallvec, no-op log macros), leaving only build-time tooling
  (sourced).

- 2026-02-14 (a8528504) rationale: the engine was founded single-module-only —
  the Instance owns its tables/memories/globals directly, sf-core's multi-module
  Linkable/LinkableData abstraction and every Rc<RefCell<>> wrapper are dropped,
  and HashMap import lookup is replaced by a Vec linear scan (few imports in
  practice, avoids hashbrown); the stated payoff was eliminating refcount/borrow
  overhead and a dependency while simplifying lifetimes, host functions arriving
  instead through the external-fn-pointer hook (sf-nano.md design doc) (sourced).

- 2026-02-14 (a8528504) rationale: the engine was first staged to WebAssembly
  2.0 (MVP + bulk-memory, reference-types, multi-value, sign-extension),
  explicitly excluding GC/3.0 to keep the binary minimal — the design doc
  budgeted GC at ~1,500 LOC across gc_heap/gc_type_check/ref_ops/type_context, so
  struct/array types, the subtyping hierarchy, recursion groups and GC ref ops
  were deferred and ref handles initially carried only funcref/externref; the 3.0
  feature set was added in later windows (sourced).

- 2026-02-14 statement: the codebase was restarted as Silverfir-nano, porting
  the fast single-pass compiler-interpreter forward from the Silverfir-rs
  codebase and leaving its three coexisting execution backends behind; the
  design lineage is continuous across the restart (sourced).
