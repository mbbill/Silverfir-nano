- An arm64 direct local call emits a real `bl` (or `b` for tail calls) that the
  linker patches directly to the callee at module link time, keeping the hot
  path a single predicted branch; it falls back to a literal-loading veneer only
  when the target lies outside arm64's +/-128MiB branch range
  (`PendingDirectCall`, `patch_arm64_direct_branch`).

- A call whose callee already has native code resolves to a direct branch to the
  callee entry; a callee not yet compiled records a patch site back-patched when
  the module finishes precompiling; compiled-to-compiled calls never
  round-trip through the generic cold-helper call path.

## Facts

- 2026-05-13 (c37a36f6) rationale: the fallback veneer is interned per (callee,
  scratch_reg) inside each function's literal-pool area rather than one veneer
  per direct call site, so repeated direct calls to the same callee in one
  function share a single veneer; the hot call sites are unchanged (each still
  patches to a real bl/b) and only the rarely-taken out-of-range path is shared
  (code).

## Moves

- 2026-05-13 (d7d328e7) replaced [[deferred-literal-blr]]: loading the callee
  address from a deferred literal and doing an indirect BLR on every direct call
  pays the cost of an indirect branch even for the common in-range case;
  emitting a real bl (or b for tail calls) that the linker patches directly to
  the callee keeps the hot path a single predicted branch, falling back to a
  literal-loading veneer only when the target lies outside arm64's +/-128MiB
  branch range (code).
