- Exactly one module instance runs per VM; no inter-module linking layer
  exists.
- Host functions are plain function pointers
  (`fn(&[Value], &mut [Value]) -> Result`): zero-alloc, no_std-friendly;
  imports resolve against these hooks.
