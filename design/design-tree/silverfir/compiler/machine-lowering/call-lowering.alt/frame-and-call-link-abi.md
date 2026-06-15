- The MachineIR `Call` terminator carries `callee_frame_base` and `caller_result_base`
  GP registers (absolute pointers the middle-end has already computed) plus a
  `continuation` block id; the backend saves the caller frame pointer and
  `caller_result_base` in a backend-private host-stack call record, switches the frame
  pointer to `callee_frame_base`, issues the native call, and on the unified `Return`
  copies `return_results` from the callee frame to `*caller_result_base` and restores
  the frame pointer from the popped call record.

- Every local-call argument and result travels through frame slots; the dynamic GP/FP
  banks have no volatility split, their order is an allocation preference rather than
  an ABI class boundary, and the only machine state required to survive a local call is
  the four fixed MachineIR registers — any cached local or SSA value live across the
  call must be published to its canonical frame slot beforehand and reloaded after.

- The frame's `call_scratch` region carries only helper-scratch slots; it holds no
  MIR-visible call-link record — the call-link (caller frame pointer and result base)
  lives in a backend-private host-stack call record, not a frame slot.
  `MachineFunctionAbi` has no `param_locs`: with all parameters frame-passed, there is
  no register-passed-parameter contract.

## Facts

- 2026-03-13 (013fd297) rationale: prepared LIR already publishes all live results
  into canonical result slots before return, so the machine return need only perform
  call-link/frame restoration and must not carry register values — the register-value
  Return form was replaced by a frame-based Return here (author).

- 2026-03-15 (3fe904c6) rationale: the per-argument load/store copy into the fresh
  callee frame was redundant — pointing the callee frame base at the caller's argument
  span leaves the arguments already in place as the callee frame prefix, eliminating
  the copy loop and its scratch register (diff).

- 2026-04-01 (2a753247) rationale: the backend was recomputing the call-link
  continuation slot from callee frame base, call-scratch region and runtime layout and
  re-resolving indirect setup; moving the call-link base computation and logical-field
  writes up into MachineIR left the backend only to materialize and store the native
  continuation address, keeping ISA-specific code from rebuilding frame/call-link state
  (diff).

## Moves

- 2026-05-13 (adc74515) replaced by [[call-lowering]]: the old JIT-to-JIT local-call ABI treated dynamic-register order as an allocation preference with no semantic class boundary, so any value live across a local call had to be published to a frame slot before the call and reloaded after — registers could not carry values across a local call at all, and every call also reserved a backend-private host-stack call-link record holding caller frame pointer and result base; the new ABI splits each dynamic bank into a JIT-internal volatility contract (volatile caller-saved prefix, preserved callee-saved suffix, backend scratch), passes a trailing scalar parameter suffix in abstract GP/FP argument lanes described by `MachineFunctionAbi::param_locs` / `MachineCallArgs`, replaces the absolute base pointers and call-link record with a relative `frame_delta` the caller switches and restores around the native call, and fixes the callee fallback-result region at frame slots [0, result_count) so direct, indirect, and tail calls agree on result placement without any per-call link record (diff).
