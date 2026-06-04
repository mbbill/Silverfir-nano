# The Design Tree

A representation for the *exploration* layer of a software project — the
choices, the paths tried, and the evidence they produced — modeled as a
**Monte Carlo Tree Search over the solution space**.

The premise: traditional software engineering performs very few searches,
because a single "rollout" — actually implementing the software — is enormously
expensive. So the solution space stays vastly underexplored, and the scarce,
valuable skill is generalizing well from a handful of expensive samples (what a
good architect does). When AI lowers the cost of trying an implementation, you
can afford far more rollouts, and — if you *record* them — the accumulated
evidence becomes a shared, queryable value function instead of tacit intuition
locked inside individuals.

This document specifies the model. It is general — not tied to this codebase —
but this codebase is its first intended instance.

## The framing: MCTS with expensive rollouts

The whole system is an MCTS:

- **Nodes are Options** — points in the solution space, candidate choices.
- **Choosing an option is a decision.** A complete design is a **traversal** —
  a *solution subtree*, not a single path (see "full AND/OR tree" below).
- **A rollout is building and measuring** — *faithfully*. You implement a path
  far enough to learn something, and a faithful attempt produces **Facts**
  whatever happens: benchmarks, bugs, "it compiles", "it regressed 8%", or "it
  cannot be implemented at all (violates Rust ownership/safety)". Only a faithful
  implementation is a valid rollout; unfaithful code is a harness defect, not
  data.
- **Backpropagation re-weights options** from those facts, which changes which
  traversal is best, which changes what you build next.

The one fact that defines everything: **in classic MCTS rollouts are cheap and
you do millions; here a rollout is building software and is catastrophically
expensive.** You can never afford enough rollouts to cover the space by brute
force. So the engine is **value estimation, not search volume** — predicting an
option's worth cheaply enough that you rarely pay for a full rollout. This is
the AlphaGo→AlphaZero move: replace random rollouts with a learned value
function. **The accumulated fact-base is that value function.**

Read that twice before extending this model. Anyone who treats it as cheap
board-game MCTS will design the wrong thing.

## The model

Two node types and a root. Nothing else is a node.

| Element | What it is |
| --- | --- |
| **Root** | The single special-cased "why" of the project. One node, not a type. |
| **Option** | A candidate choice in the solution space. Carries a weight derived from its facts (plus a taste/prior). Its status matures: `unexplored` → `explored-untested` → `tested` (with confidence and a narrative). An option with no facts is legal — it is simply `unknown`. |
| **Fact** | An immutable observation with provenance (impl / test / production / external) that **supports or undermines** an Option, or opens new Options. Produced by exploration and by running code. A fact is **not** an invariant and **not** an argument. |
| **Decision** | Not a node — the *act* of choosing an Option. Can be driven by facts or by taste. |
| **Traversal** | Not a node — the current chosen *solution subtree* (see "full AND/OR tree"). **It is the current design.** |

There are **no Goal, Question, or Invariant node types.** Earlier drafts had
them; they were falsified (see "How this model evolved"). Goals collapse into
the single root. "Invariant" does not exist — nothing is permanently invariant;
any choice is revisable given new facts.

### It is a full AND/OR tree

A design is not a single path — it is a **solution subtree**. Branches come in
two kinds:

- **OR-branch** — alternatives. Pick *one* child (static split vs. unified
  budget). This is the genuine search choice.
- **AND-branch** — independent sub-problems. Take *all* children, as an
  **unordered set**. Choosing "3-stage pipeline" opens an AND-branch
  `{design stage 1, design stage 2, design stage 3}` with no order among them;
  "implement wasm IR / machine IR / arch" are AND-siblings.

AND-branching is the compression. N independent decisions laid out in a path
have N! equivalent orderings and force you to carry the whole linear history to
reason about any one of them. As AND-branches they have one representation, no
order, and each is reasoned about **in isolation**. The tree becomes
shallow-and-wide instead of deep-and-linear. (Searching an AND/OR tree is the
AO\* family; a design's value aggregates over its AND-branches' chosen options.)

**Cut AND-branches at low-coupling boundaries — i.e., modular decomposition.**
This is what reconciles AND-factoring with path-dependence:

- Decisions that genuinely interact (how SSA-IR is defined constrains
  backend-IR options; the ABI contract both the middle and the arch backends
  depend on) are **coupled** — keep them in the *same* branch, ordered,
  path-dependence intact. The process is **non-Markovian** inside a coupled
  branch: a node's options depend on the whole prefix, and the uncompressed tree
  carries that history for free.
- Decisions that don't interact split into parallel AND-branches and never see
  each other.

So path-dependence is preserved exactly where coupling exists and dropped
exactly where it was spurious. AND-nodes *are* the module boundaries. This stays
a tree: AND-factoring is branching, never node-merging. A DAG /
graph-with-shared-nodes would merge nominally-identical nodes — lossy, and the
compression gets complex fast on long trajectories. Not worth it.

The decomposition itself is a decision: "split into these stages with these
interfaces" is a chosen option. A bad cut — factoring apart two things that were
actually coupled — surfaces later as a **Fact** ("these 'independent' modules
kept changing together") that undermines the decomposition option and drives a
re-cut. Wrong module boundaries are discoverable, not silent.

### Code is downstream of the tree

The tree is the design space. A traversal is the current design. **Code
implements the current traversal — that is all it is.** Consequences:

- Old code is just an abandoned traversal. It does not matter and the tree does
  not record it.
- The tree does not change *from* code. Code is an implementation *of* a
  traversal.
- The only alignment that matters: **is the code faithful to the current
  traversal?** Code's job is to honestly implement the chosen option, and
  faithfulness is the *validity condition for a rollout*. It splits two cases
  that look alike but are opposite:
  - **Unfaithful code** — the implementer/harness deviated from the design by
    error — is **not a Fact**. It is a *harness defect* (bad codegen or review),
    produces no valid signal about the option, and recording it would poison the
    value function with a measurement of the wrong thing. Fix the harness,
    regenerate faithfully; never backpropagate it.
  - A **faithful attempt** produces a **Fact** whatever its outcome: it runs and
    measures, it produces a wrong result, *or it cannot be completed at all* —
    the design is infeasible (it requires `unsafe` that is disallowed, violates
    Rust ownership/safety, etc.). "Infeasible to implement under these
    constraints" is a first-class Fact, as valid as a benchmark; a failed
    *faithful* attempt is real data that prunes the option.
  So faithfulness verification (review) is the **integrity gate for every Fact**:
  only a faithful implementation is a real test of the option. (A mismatch during
  an in-flight fork/rollout is neither — it is the expected transient until the
  rollout lands or fails.)
- Running the code is how you harvest facts (benchmarks, bugs), which re-weight
  options, which can re-select the traversal, which re-aligns the code. That
  loop is the whole system.

### Forking is expansion + rollout

A refactor is a **fork from a node**. Everything downstream of the fork becomes
a temporary fork that is updated until the software runs (or fails). Either
outcome harvests facts for the new path. Because each fork costs a real
rollout, the discipline is to **fork the most *informative* node** — highest
expected information gain — not the nearest one. "Improve the search itself"
means improving the value estimator so you fork better next time.

**A fork invalidates its downstream.** Because the tree is non-Markovian (a
node's options depend on its whole path), forking a node changes the path prefix
for everything below it. That node and all its descendants in the *coupled*
subtree must be marked **stale (revisit-pending)** — their exploration was
conditioned on the old upstream choice. Treating a stale descendant as
already-explored is a correctness bug: you would build on a decision that was
best under a path that no longer exists. Staleness does **not** delete the
subtree — the existing options and their facts are kept as **priors** (demoted
from confirmed-current to "worked under the old prefix; re-verify"), and a
revisit treats the node like a fresh one with a warm start: does the option
still fit, is a new option now enabled, or did something else change?

Three things keep this cheap and bounded:

- **AND/OR factoring bounds the blast radius.** Marking propagates down coupled
  descendants only, not independent AND-siblings — forking register allocation
  does not invalidate the independent wasm-decode subtree. Good low-coupling cuts
  keep the blast radius small; a bad cut surfaces as a coupling fact ("forking A
  also broke 'independent' B").
- **Provenance makes staleness computable.** A fact is stale iff an ancestor on
  its rollout's path was forked after it was recorded (its `tree-commit` predates
  the fork). Staleness is *derived* from provenance — lazily at revisit time or
  eagerly on fork — not hand-marked.
- **Stale ≠ re-run everything.** Rollouts are expensive; marking means "do not
  *trust* without revisit," and the revisit depth scales with the change's
  expected impact — a cheap confirm when the fork is plausibly irrelevant, a full
  faithful re-rollout when it is plausibly disruptive. A revisit re-stamps the
  re-validated fact (re-promote) or refutes it.

## Two artifacts, and only one is a tree

This separation is load-bearing. Conflating them defeats the entire purpose.

1. **The tree** — the search record. Full, uncompressed, forkable,
   history-intrinsic. Per-search structure.
2. **The fact-base** — the value function. A fact stapled to one tree node is
   useless for guiding *future* search, because the next search is a different
   tree (or a different project). Facts must live in a layer that
   **generalizes across branches and across trees**, indexed by the *context*
   in which they hold.

Every fact therefore carries a **validity scope** — the conditions under which
it is true ("unified budget is complex *given a 32-bit target, given these
constraints*"). Without scope, retrieval reproduces the senior-architect
failure mode: confidently over-applying a lesson that was only ever true in its
original context. The fact-base is the artifact that compounds; designing its
schema and its scoped retrieval is the hard, valuable, still-open problem.

## Why this can work now

Formal design-rationale systems (IBIS, gIBIS, QOC, DRL) had a similar shape in
the 1980s–90s and were abandoned. The documented cause was **capture cost
without immediate benefit** (Grudin's cost-bearer/beneficiary mismatch) and
**premature formalization** (Shipman & McCall). The fix that worked was a
trained real-time *facilitator* (Dialogue Mapping) who structured the record as
it happened, so capture bought immediate value.

An LLM is that facilitator, always available and free. It can propose options,
run rollouts, and record facts continuously. That removes the capture cost that
killed the field. It does not remove these rules:

- **Incremental formalization.** Every node is valid as prose first; structure
  is added only as it stabilizes. Never require a schema to be filled to save a
  thought.
- **Facts, not arguments.** A fact is a grounded observation with provenance. An
  argument dressed as a fact is the confirmation-bias trap; an option chosen on
  reasoning alone is an `unexplored`/taste choice, recorded as such — not a
  fact.

## Honest constraints

These bound what the system can do. State them to anyone who builds on it.

- **Rollouts are expensive.** The near-term reality is "few costly rollouts
  steered by heavy human/AI value-estimation," not millions of cheap ones. The
  trajectory improves as build-and-verify cost falls, but today value
  estimation dominates, not search volume.
- **The action space is generative and unbounded.** Unlike Go, the options at a
  node are not enumerable — you can always invent a new approach. The
  option-*proposer* is itself an AI component, and its quality is the ceiling on
  everything. This is LLM-guided program search, not board-game MCTS.
- **Facts go stale (non-stationarity).** The solution space shifts as tools,
  models, and hardware change. "Too slow to compile" becomes false when devices
  grow. Facts have a shelf life; scope and date them. Explicit, dated facts are
  better than tacit intuition — inspectable, not immune.
- **Abstraction is the bottleneck.** A mis-scoped or over-generalized fact is
  worse than no fact: it actively misleads every future search that retrieves
  it. Deciding *what a rollout actually teaches* is the highest-leverage step
  and is human-shaped.

## Human–AI division

- **AI may:** propose options (expand the tree), run rollouts, record facts with
  provenance, draft option weights, detect when code diverges from the current
  traversal, surface under-explored high-potential nodes.
- **Human must:** choose among high-level options (commit a traversal), and —
  the irreplaceable role — **curate and scope the facts**: decide what a rollout
  generalizes to, and what it does not. The machine gathers experience; the
  human decides what it *means*. Witnessed live during this model's own
  construction: an automated reviewer collected falsification data perfectly,
  then drew the wrong conclusion from the summary statistic. Correct data, wrong
  lesson. Closing that gap is the human's job and the system's center of
  gravity.

## The medium

Prose-first records in version control, tree-shaped. The graph lives in the
links. Incremental formalization throughout. This is the one part the 90s
systems got right by accident and the heavyweight ones got wrong: the medium
must be cheap to write, cheap to review, and diff cleanly — or it rots.

## How this model evolved

Recorded so no future reader re-proposes a dead end — the model's own failure
history, which is exactly the kind of evidence it exists to capture.

- **v1 — five typed nodes (Goal / Question / Decision / Invariant / Fact)** with
  "record contention, let code enforce completeness." Falsified against this
  codebase's own git history: 48 real design episodes were mined and tested. The
  recurring misfits were (a) **implicit invariants** — most binding rules were
  never *decided*, they were latent and *discovered when a bug violated them*,
  which v1 could only encode by fabricating a decision; and (b) **factless
  change** — taste, scope, complexity-judgment, and even reversal of *winning*
  work, which v1's "facts drive everything" axiom could not hold.
- **Collapse.** "The code was wrong" is a *fact lowering an option's
  confidence*, not a discovered invariant — so Invariant died. An option may be
  chosen with no evidence (`unexplored`/taste) — so Questions and the
  facts-only axiom relaxed into Options carrying status. Goals collapsed to the
  root.
- **Full tree, then AND/OR.** Path-dependent options (non-Markovian) killed the
  DAG and chose the uncompressed tree. Then linear traversals serialized
  independent decisions into arbitrary order; factoring into an AND/OR tree
  (OR = alternatives, AND = independent sub-problems cut at low-coupling
  boundaries) made the structure shallow-and-wide and reasoning local.
- **MCTS.** The pieces — options, decisions, expensive rollouts, facts as
  backpropagated value, the fact-base as a transferable value function —
  unified into Monte Carlo Tree Search over the solution space.
- **v2 pressure-tested, then declared done.** A second neutral falsification (24
  episodes, independent verify) returned zero confirmed strains — but the honest
  reading is that v2 removed v1's *hard claims* ("facts drive everything",
  "invariants are decided"), so episode-replay can no longer falsify it. A model
  this minimal absorbs any past episode; passing proves "not worse than the
  history," not "adds value." The ontology is at a hard-won local optimum and
  should not gain node types; the unsolved work moved entirely into the
  fact-base and capture discipline.
- **Fact scope → agentic memory.** Designing the fact-base's *scope* field and
  stress-testing it (15 real facts) showed scope is write-only and fails
  silently, and that a generative option space cannot be pre-scoped. Scope was
  dropped entirely: the fact-base became an agentic memory (store raw,
  HyDE-recall, read-time LLM judge, periodic reflection), grounded in 2023–2026
  memory/RAG research. See [FACT_BASE.md](FACT_BASE.md).

## Open problems

The pressure-tests confirmed the *ontology* is sound but exhausted
episode-replay as an instrument. All remaining value and risk live here, not in
the node types:

- **The fact-base — now an agentic memory (resolved; see
  [FACT_BASE.md](FACT_BASE.md)).** Authored per-fact *scope* was tried and
  dropped: it is write-only and fails silently (a scope pinned to a
  sibling-of-the-target exclusion buries a real transfer with zero error
  signal), and the generative option space means a capture-time scope cannot
  anticipate the future options a fact will bear on. Applicability moved to
  *retrieval time* — a wrongly-surfaced fact is read and discarded (visible),
  not wrongly-omitted (invisible). The architecture: store raw observations
  cheaply, HyDE-expand the query for cross-mechanism transfer, recall wide and
  let an LLM judge filter at read time, and a periodic reflection pass
  consolidates independent instances into class memories. The open part is now
  operational tuning of that loop, not a schema.
- **Capture discipline — at the decision point.** The reason a re-selection
  happened must be captured *when the fork/abandon happens*, not reconstructed
  weeks later. (Observed: early legalization was abandoned with a one-word
  commit message; the fact that justified its eventual revival was recorded
  weeks later in a side artifact.) Latency and locus of capture is the real
  failure surface in live use, and it is invisible to episode-replay.
- **Capture the qualitative facts, not just the benchmarks.** In the hardest
  decisions the chosen traversal *overrides* the measured facts (a winning,
  benchmarked CSE pass was reverted on architectural judgment). That divergence
  is not a flaw in the value-function premise — it is a *measurement gap*: the
  deciding fact (maintenance cost, architectural fit) was real but uncaptured.
  The value function only guides as well as its facts are complete, and the
  decisive facts are often qualitative.
- **An option's prior has provenance too.** A fact-free option may be chosen on
  arbitrary taste, on a rigorous derivation ("the backend can never need a third
  scratch register"), or on analogy to an external system. These are not equally
  trustworthy. Weighting should distinguish the *grounds* of a prior
  (taste / derivation / external-analogy) — a typed weight, not a new node type.
- **The option-proposer** — sample-efficient generation of candidate options at
  a node. The ceiling on search quality.
- **Cheap value estimation** — predicting an option's worth without a full
  rollout. The thing that makes the expensive-rollout economics survivable.
