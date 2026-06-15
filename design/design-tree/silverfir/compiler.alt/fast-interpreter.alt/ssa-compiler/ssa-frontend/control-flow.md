- Control flow leaves a basic block only through that block's terminator:
  block/loop/if and branch handlers set a Br/BrIf/BrTable terminator on the
  block being left rather than reassigning the current block directly.

- The builder tracks a stack of control frames for block/loop/if, each
  carrying its branch target, continuation, block signature, and the
  expression-stack height after its parameters were consumed; an unreachable
  flag suppresses non-structural opcodes after a branch.

- At a control-flow merge the builder reconciles both the result stack and the
  per-local value map across all incoming paths: a local holding the same SSA
  value on every path keeps it, a local that diverges gets a phi in the
  continuation, and a loop header gets a phi per live local and per block
  parameter fed by the entry edge and the back-edge.

- While unreachable, the decode loop processes only the structural opcodes
  (block/loop/if/else/end) needed to keep the control-frame stack balanced and
  ignores all other opcodes.

- A br_table whose targets (including the default) all resolve to the same
  block is canonicalized to an unconditional Br, dropping the index operand.

## Facts

- 2025-10-11 (3db2fb07) pitfall: loop-entry phi nodes must be created for
  parameters as well as locals — WebAssembly lets a loop body mutate a
  parameter via local.set, so treating parameters as loop-invariant and
  skipping them when inserting back-edge phis produces incorrect SSA for any
  function that writes a parameter inside a loop (diff).

- 2025-10-11 (ede7985e) pitfall: a then-block that already branched out (via
  br/return) must not be given an overwriting fall-through Br to the
  continuation; the builder checks the block's terminator before emitting the
  implicit branch, or it would clobber the real terminator and add a bogus
  predecessor edge (diff).

- 2025-10-12 (19a2c2d1) pitfall: an IF with no explicit ELSE must still feed
  its synthesized else block's value-map snapshot into the merge; omitting it
  drops the not-taken path's locals from the continuation's phi nodes, so
  locals unchanged on the else side resolve to the wrong value after the IF
  (diff).

- 2025-10-14 (0fe21aec) pitfall: an IF with no explicit ELSE whose block type
  produces results must also forward its block parameters down the implicit
  else edge — at END the unused else block is materialized from the frame's
  entry expr-stack snapshot and added as a value source — or the
  continuation's phi nodes merge only the then-branch values (diff).

- 2025-10-12 (c556b0a4) pitfall: a control frame's recorded entry stack depth
  must be the expression-stack height after consuming the block/loop/if
  parameters (and, for if, after popping the condition), not the raw height —
  the parameters stay on the stack inside the block but are replaced by results
  at the branch target, so using the unadjusted height mis-reconciles the stack
  at branches to that frame (diff).

- 2025-10-12 (023e0c86) pitfall: the then-branch's result values must be
  registered as a merge source for the continuation unconditionally, not only
  on the else-branch fall-through path — when the else branch is itself
  terminated the fall-through code never runs, and folding then-value handling
  into it drops the then edge from the continuation's phi nodes (diff).

- 2025-10-12 (cf160553) pitfall: on entering a continuation the unreachable
  flag may be reset only when that continuation actually has an incoming edge
  (a fall-through or a recorded branch source); when every path branched away —
  e.g. a br_table whose targets are all the function frame — clearing
  unreachable unconditionally lets the builder treat dead trailing code as a
  live fall-through (diff).

- 2025-10-12 (2cee86a4) statement: a loop's natural fall-through at its END
  goes to the loop's continuation (the exit), exactly like a block; only an
  explicit br to the loop label creates a back-edge to the loop header, so loop
  ends are handled by the same merge path as blocks rather than a separate
  back-edge case (diff).

- 2025-10-14 (7389c7c5) pitfall: a loop parameter's entry value must be
  materialized in the predecessor block before the loop header, not in the
  header itself; materializing it in the header would let the entry edge of the
  parameter's own phi reference the phi result, breaking the merge (diff).

- 2025-10-14 (f2eb3b4a) pitfall: a br_if whose label targets the function frame
  is split into a return block and a fallthrough block, and the shared
  return-emit helper sets the unreachable flag as a side effect, so the handler
  must explicitly reset unreachable for the still-reachable fallthrough path
  before continuing to decode (diff).

- 2025-10-14 (87b2834b) pitfall: a br_table with one or more arms targeting the
  function frame must still emit a real BrTable terminator — function-frame
  arms are redirected to a dedicated return block (with phi nodes receiving the
  branch values) wired in as a table target, rather than short-circuiting the
  whole table to a single recorded return source (diff).

- 2025-10-14 (0fe21aec) pitfall: a function body that ends with no return
  sources must not be given a synthesized dummy zero-valued Return when its
  current block already carries an Unreachable terminator (an always-trapping
  body); the Unreachable terminator is kept and lowering allocates placeholder
  result slots that are never read (diff).

- 2025-10-24 (35113cfc) pitfall: when finalizing a single-source function
  return, the source block's existing terminator must be inspected first — if
  it already carries a BrTable it must be preserved, since overwriting it with
  a Return would discard the multi-way branch (diff).

- 2025-10-30 (fbc5c966) pitfall: no instruction may be added after a block's
  terminator is set, and a value must be materialized in the block where it is
  computed — for a conditional branch the condition must be materialized into
  the source block before the BrIf is written and the current block must move
  to the fresh fallthrough block immediately after, or post-branch instructions
  leak into the already-terminated source block (diff).
