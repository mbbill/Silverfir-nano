- Handlers receive a pointer to a 64-byte Regs bank holding the top-of-stack
  register window (t0/t1/t2 plus a u64 depth of live lanes), an sp spill pointer,
  heap base/limit, and locals base; the handler signature is (ctx, pc, regs*).

- Pushes and pops rotate the t0/t1/t2 register window and lazily spill the oldest
  lane to / refill it from the memory stack only when the window overflows or
  empties; on function exit the live lanes are flushed to the memory stack before
  results are read.

- Linear memory is addressed through heap_base and a heap_limit end pointer
  carried in the Regs bank.

## Facts

- 2025-08-13 (db80d0cd) rationale: the handler-carried register bank is sized to
  exactly one 64-byte cache line (a compile-time assert enforces it) — a few
  top-of-stack lanes plus depth, a spill-stack cursor, cached linear-memory
  base/limit, and a locals base pointer — so the whole hot window stays in one
  line across the tail-chained handlers (code).

- 2025-08-14 (01c8ee79) rationale: the hot operand window is a fixed 3-lane
  register set (t0..t2, t0 always TOS) plus a runtime depth count and lazy
  spill/fill to the memory stack, with depth held as u64 (code).

- 2025-08-14 (01c8ee79) rationale: the fixed 3-lane window keeps hot paths
  register-only with branchy code that predicts well and avoids indexed addressing
  that blocks SROA, and depth is held as u64 to avoid byte-width extends on some
  ABIs (sourced).

- 2025-08-16 (d748e227) pitfall: the window's spill/fill discipline was
  simplified from eager to lazy — the earlier pop_tos, on a pop while depth==3,
  eagerly read one slot back from the spill cursor into t2 and pinned depth at 3
  (window always full while spill is non-empty); pop now simply decrements depth
  and only the empty-window pop path reads directly from spill, so lanes fill
  lazily. The eager form left t1/t2 holding values refilled from already-consumed
  spill slots, which a subsequent flush could write back spuriously (code).

- 2025-08-16 (38b204bd) pitfall: on return from a callee the caller's live top
  must be reconstructed from where the callee's arguments began, not from the
  caller's own locals base — threading the caller's stack top into the callee as
  an incoming index but reconstructing on return from locals_base + results_len
  leaves sp wrong by the gap whenever the call args sit above intervening operand
  values; the fix computes the new top as incoming_index - callee_params +
  results, replacing the callee's args in place with its results (code).

## Moves

- 2025-08-16 (10c1c487) replaced [[value-stack-self-tracked-sp-offset]]: the stack
  kept its own sp_offset counter in parallel with the register window's Regs.sp
  pointer, so every register-window flush had to manually reconcile the two (flush
  bumped sp_offset by the spilled lane count); making Regs.sp the single source of
  truth for the live top and threading the logical frame size as an explicit
  stack_size parameter removes the dual-bookkeeping and its sync hazard (code).

- 2025-08-13 (240fb3d8) replaced [[by-value-tos-lanes]]: passing the four
  top-of-stack lanes plus depth as separate by-value arguments forced per-call
  stack argument traffic on Win64 (only four GP argument registers) and
  zero-extend shuffles for the byte-sized depth; a single by-pointer register
  bank removes both (code).

- 2025-08-17 (912cc440) replaced by [[pure-sp-memory-stack]]: the by-pointer Regs
  bank with a t0/t1/t2/depth top-of-stack register window and lazy spill/fill is
  dropped in favor of operands living directly on the memory stack addressed by
  an sp pointer, with (sp, mem0 base, mem0 size, locals base) passed as by-value
  scalar arguments threaded through each preserve_none tail call instead of
  carried in a heap-shaped register struct (code).
