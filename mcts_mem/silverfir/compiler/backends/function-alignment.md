- When laying out compiled functions into the executable buffer, every backend
  rounds each function start up to a 64-byte cache line, and a function under
  one code page that would straddle a page boundary is padded forward to the
  next page (with architecture-specific NOP fill) before its base offset is
  fixed, but only when the required padding is small (`page_align_function`).

## Facts

- 2026-03-23 (c06c102a) rationale: functions are pushed onto the next 16 KB
  page only when the padding needed is at most 1 KB (code).

- 2026-03-23 (c06c102a) rationale: a function starting far from the boundary
  already has its hot early blocks on the current page so the tail crossing is
  cheap, and capping padding at 1 KB keeps overhead under 1% on large
  (~800-function) modules while still fixing roughly a third of page crossings;
  the goal is reduced iTLB pressure on hot entry blocks without inflating total
  code size (sourced).

- 2026-03-23 (6b4ba56e) rationale: the NOP/pad encoding is architecture-specific
  (ARM64 NOP 0xd503201f, x86_64 INT3 0xCC, ARMv7 MOV R0,R0 0xe1a00000), and the
  64-byte cache-line snap applies to every function start in addition to the
  conditional page-crossing push (sourced).

- 2026-07-16 (95fec85d) pitfall: 64-byte function alignment does not stabilize
  branch-predictor aliasing in large dispatch loops — lua fib38 wall time
  moved +4-6% when unrelated cold functions changed size, with the hot 45 KB
  dispatch function's MachineIR and (in the backend-reverted control build)
  its native bytes proven identical; the shift is 64-byte-granular base
  movement changing BTB index bits, the same phenomenon behind the af139e58
  regression. Before attributing a lua-fib delta to codegen quality, diff the
  hot function's MIR/bytes; deltas within ~5% with identical hot code are
  placement luck (code).
