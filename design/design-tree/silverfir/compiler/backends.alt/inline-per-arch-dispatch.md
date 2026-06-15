- The shared module-build pipeline (vm/build) and the eval entry (native_eval)
  select the per-architecture `compile_module` and `eval` implementations with
  inline `cfg(sf_arch_*)` branches at the call sites, threading each backend's
  arch-specific CompiledEntry type through directly.

## Moves

- 2026-04-07 (12aa736b) replaced by [[backends]]: the module-build and eval
  paths each branched on the active backend with inline sf_arch_* cfgs in
  vm/build.rs and native_eval, leaking ISA gating into the shared pipeline;
  projecting every backend's per-function compile result into one
  CompiledArchEntry shape and routing compile/eval through arch::dispatch_*
  keeps every caller free of sf_arch_* cfgs and confines arch selection to
  arch/mod.rs (diff).
