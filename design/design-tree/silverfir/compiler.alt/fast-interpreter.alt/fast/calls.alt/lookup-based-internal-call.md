- The internal-call handler encodes only the callee func_idx and the stack-top
  offset; its handler resolves the callee FunctionInst by func_idx through the
  store on every execution, then reads frame layout (params, results, locals,
  temps) from the callee spec at run time.

- It differs from the generic call only in skipping the is_external check; it
  still builds the callee's fast IR on demand if absent and performs the same
  store-resolved frame setup.

## Moves

- 2025-12-11 (8654e952) replaced by [[calls]]: the internal-call handler still
  resolved its callee by func_idx through store.instance_at_module on every call;
  once functions are precompiled the callee's entry instruction pointer,
  FunctionInst pointer, and param/result/locals counts are known at build time and
  baked into the instruction, so the hot call path does no store lookup at all
  (diff).
