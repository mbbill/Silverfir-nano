- Each Store owns at most one reusable native invocation stack and one
  `NativeContext`. Evaluation takes the cache entries out, resets per-call
  error/escape/stack state, and returns them on every normal or error result;
  re-entrant evaluation sees an empty cache and allocates an independent pair
  (`take_native_context_cache`, `cache_native_context`).

- Reuse is guarded by explicit revisions for module mutation, shared tables,
  and the shared function registry, plus exact pointer/length validation for
  globals, memories, and tables. A changed revision refreshes the cached views;
  an unchanged Store reuses them without rebuilding (`cached_views_are_current`).

- A compiled module stores its maximum native frame size for O(1) cached-stack
  validation (`CompiledNativeModule::max_frame_bytes`).

## Facts

- 2026-07-22 measurement: a profile of regex-redux attributed 904 of 1,791
  samples inside `eval` to `NativeContext::refresh_cached_views`; caching the
  context/stack and adding exact invalidation reduced 30.393 us to 19.199 us,
  36.8% (sourced).

- 2026-07-22 measurement: after the cache landed, an empty exported native
  function measured about 22 ns per call and native-entry setup accounted for
  only 0.14% of a long regex profile, ruling root-call allocation/setup out as
  the remaining execution bottleneck (sourced).

## Moves

- 2026-07-22 replaced [[per-invocation-allocation]]: rebuilding invocation state
  on every exported call dominated short workloads; Store-owned
  revision-validated reuse removes fixed work while take/return ownership
  preserves re-entrancy (sourced).
