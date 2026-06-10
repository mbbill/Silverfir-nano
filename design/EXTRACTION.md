# Fact extraction — recovering design history

**This is not daily work.** This workflow reconstructs the design tree for a
project that already has code: the current design, the deprecated tries, and
the facts that drove every move — recovered from git history and from the
author. It runs once per repository. Day-to-day maintenance and the tree
grammar live in [`README.md`](README.md); read it first — this document only
defines how to run the recovery.

**Why:** the current code is a single path through the solution space.
Rewriting or evolving a system needs the rest — why this path, what was
tried, what failed, what to avoid. Git holds a large share; the author holds
most of the remainder. This workflow drains both.

## The core loop

The unit of work is an **information piece**: a commit, an author statement,
a chat-log claim. Invariant being built: *for a perfect tree, every piece is
accounted for — the decision, move, or fact behind it can be found.* Each
piece is a **lookup against the tree, then a repair if the lookup fails**.

Classify every piece from its **content** (diffs — messages are often wrong
or fabricated), then:

- **add** — name the node whose items (+ ordinary engineering competence)
  generate this code. Found → HIT. Not found → a decision happened silently →
  REPAIR: amend an item, or create a node *only if a real alternative
  exists*. Pure adds are LOW density: at most one item; what code "is" is
  not a design decision and is already recorded in the code itself.
- **change / delete** — HIGH density: something happened; find it. A
  replacement of a working mechanism or representation is a re-decision by
  default → REPAIR: loser into `.alt/`, paired move lines (why verbatim on
  both sides, provenance tagged), fact filed if it has body. "Cosmetic"
  must clear a high bar: pure rename or mechanical restyle. A deletion with
  no successor → `dropped` (or `removed`) move line on the survivor.
- **bug fix** — does the fix change a holds-true item? yes → design flaw:
  correct the item and file the fact. No, but instructive → pitfall fact at
  the nearest node. No, pure slip → SKIP. When the fix **replaces the
  mechanism's form** — its algorithm or representation, not a predicate,
  constant, or arm within it — the replacement default governs instead of
  this branch: loser into `.alt/` with paired move lines, the failure as
  the why (a byte-scanning boundary-finder replaced by an opcode walk is a
  re-decision; a flipped skip-predicate inside the same loop is a pitfall
  fact).
- **conformance** — the spec/suite dictated it, no alternative → FORCED.
- **statement** (author answer, chat log) — file as a fact at the node it
  bears on; if it reveals an unrecorded design, REPAIR first.

Judgment calibration (learned from observed runs):

- Completing a skeleton ("now functional") changes no decision — the add is
  a HIT on the node that designed it.
- A spec-mandated component that does not yet embody a real choice gets no
  node; defer until it does, and note the deferral in the batch report.
- A real mechanism built ahead of its consumer is a current node.
- Never invent an alternative that this codebase did not contain or weigh.
- **Transient mid-refactor boundary.** A batch's last commit is an arbitrary
  cut and may land mid-refactor — a subsystem disabled or non-compiling as
  construction churn that adjacent commits restore. Faithfulness is then
  checked against the last in-batch commit where that subsystem was coherent;
  the transient disable is construction paperwork, not a tree event (no
  `dropped`/`removed`). Distinguish from a real removal: if the subsystem
  stays gone, it is a genuine delete; if the very next commits revive it, it
  was churn. Relatedly, when writing a superseded form into `.alt/`, diff
  its last *coherent* (compiling) blob, not a transiently-disabled or
  non-compiling one — the `.alt/` must record the design as it actually
  worked, so "last-true" means last-coherent, not last-written.
- **Bugs are not design.** Items describe the design the code implements, not
  its defects: a localized implementation bug (a wrong predicate, an
  off-by-one) does not change the design statement — write the item at the
  design the code structurally implements. A bug becomes a pitfall fact at
  the commit that *fixes* it (that is where the lesson crystallizes), never
  by describing buggy behavior in an item or by the extractor pre-emptively
  filing an unfixed slip it happened to notice. Pitfall admission line: a
  slip **inside the mechanism's own invariant logic** — its guards,
  boundary conditions, the rules that make it correct (an underflow guard
  that never fired, an error constructed at the decode boundary but never
  raised) — is instructive and is filed at its fix; a generic coding slip
  with no mechanism-specific lesson is a SKIP.
- **Settling vs re-decision**: settling applies only while the mechanism
  itself has never been exercised — never run by any binary, test, or pass
  over real input. A parser that parses real modules is consumed even if
  the wider product is incomplete; only a form that never executed is
  settling. And an expressivity delta always wins: if the prior form could
  not express something the new one can (see README), it is a re-decision
  no matter how briefly the prior form lived — **provided the prior form was
  a complete, working design.** Completing an incomplete stub or placeholder
  is settling, not a re-decision, even when the finished form expresses more
  than the stub: a stub that punts (a raw byte forwarded undecoded, a
  `todo!`, an arm that returns a default) was never a weighed alternative,
  just unfinished work — finishing it is progress, recorded as item edits,
  not an `.alt/`. The discriminator: was the old form a complete mechanism
  that *worked and was replaced*, or a placeholder that was *filled in*?
  Two tiebreakers, in order: a **type-level wall** — the old form's
  signature could not hold the new capability at all (a borrow-only reader
  cannot carry caller-owned input) — is a re-decision regardless of whether
  the form ever ran; an indirection or ergonomics wall ("possible but
  awkward") is not a wall. And when the repo cannot show whether a form was
  ever exercised (early construction, no tests yet), default to
  re-decision — a dropped transition is unrecoverable, while an
  over-recorded `.alt/` is visible and reviewable.
  One carve-out the other way: **adopting or dropping an external dependency
  is always a re-decision** — a dependency is a commitment the moment it is
  declared, whether or not the code that used it ever ran. This is about the
  *engine/product's own* dependencies. Test-harness, build, and tooling
  dependencies are paperwork (like the tests and scripts themselves):
  excluded from the tree entirely — adopting or dropping them is never a tree
  event.
- A bug fixed in a node being written this same batch: write the item at
  its corrected behavior, file the bug as a pitfall fact, and say so in the
  row's ref.

When a rejected alternative's code survives in history and its deciding
measurement does not, the measurement can be re-run: build the old commit in
a worktree, measure under the project's benchmarking rules, file the result
as a fact marked as a retroactive rollout.

## Questions to the author

Dig everything the content can show first; ask only what it cannot. Two
triggers:

1. A re-decision whose why the content cannot show.
2. A foundational commitment with no recorded alternative (from-scratch vs
   dependency, language/runtime, an architecture-shaping choice) — git never
   shows the road not taken. Reserved for high-commitment choices.

Extractors never interview; they append to `extraction/questions.md`:

```
## Q<n> (status: open)
context: <commit(s) + what the content shows>
question: <what the content cannot answer>
blocks: <node path the answer bears on>
```

Questions never block the walk. `questions.md` is the single index of
pending whys — nodes carry no pending markers; an `(inferred → Qn)`
provenance tag is the only in-tree reference. Observers record answers
inline; an **Injector** pass folds each answer into facts/whys, flips the
status to `folded`, and adds a `statement` ledger row.

## The ledger — claims must be auditable

All progress lives in `extraction/ledger.tsv`, one row per piece, appended
at processing time. **Narrative reports are not trusted; the ledger is.**

```
seq  piece-id  class  verdict  ref  depth  batch
```

- `seq`: run-global, strictly increasing. `batch`: the batch number.
- `piece-id`: full 40-char commit hash, or `author:<date>:<n>` /
  `log:<src>:<n>` per extractable claim.
- `class`: add | change | fix | forced | statement | note
- `verdict`:
  - HIT — the tree accounts for this piece (including implementations of an
    already-recorded design). **ref MUST name the covering node(s).**
  - REPAIR — the tree was fixed. **ref MUST name the delta** (items amended /
    node created / move lines / fact filed).
  - FORCED — conformance; no alternative existed.
  - SKIP — nothing for the tree to account for: licenses, lockfiles,
    formatting, pure file motion, implementation slips. One-word reason.
- `depth`: D = diff read; G = file-signature only; M = message only. D is
  required for any verdict except SKIP on trivial files. Downgrading a
  region to G needs observer approval *before* processing, recorded as a
  `note` row.
- `note` rows: observers (or agents, for deferrals) may append annotations
  or corrections; existing rows are never edited. Shape: seq `<n>b`,
  piece-id `note:<seq-referenced>`, class `note`, verdict `-`, ref = the
  annotation.
- **A row's ref narrates the tree as of that piece** — never from the
  future. Born-then-demoted within a batch: the birth row uses
  `<final-path> (as <birth-name>, then-current)`; the later row records the
  motion. An incumbent whose shape was itself established mid-batch is
  written to `.alt/` at its last-true state (diff the relevant blob), with
  the move tagged by the replacing commit's hash.

Hard rules (each exists because its violation has happened):

1. A verdict without its required ref is invalid — no bare assertions.
2. Moves sections, facts, and other batches' ledger rows are append-only.
3. Completion claims are ledger queries, nothing else: "done" means
   rows == pieces, seq contiguous, zero M rows. The words "analyzed /
   verified / complete" may only be used quoting such a query.
4. A batch with zero REPAIRs *and* zero questions must flag itself in its
   report — that pattern is what silent skimming produces.

## The agent workflow (sequential mode)

Subagents execute; observers (humans + coordinator) review between batches
and improve this document — the next extractor inherits the fixes.

**Extractor** (one per batch, sequential):
1. Read `README.md` and this document fully.
2. Resume from the ledger tail — never from a conversation summary.
3. Process the batch's pieces in order per the core loop, at depth D,
   repairing the tree per the grammar.
4. Append ledger rows as pieces are processed, questions as they arise.
4b. Run `python3 design/lint.py` after writing; fix every violation before
    reporting. A batch that does not lint clean is not done.
5. Report: verdict counts, repairs (node paths), questions, anomaly
   self-check (rule 4), and process feedback — anything ambiguous or that
   fought you is part of the deliverable.
Forbidden: verdicts beyond what was read; editing history; completion
language beyond ledger queries; silent depth downgrades; consulting any
`_bak*` directory.

**Auditor** (after every batch — the faithfulness gate the linter cannot
be). A FRESH agent with no extractor context: reads only the tree, the
batch's diffs, and these docs, and is adversarial (assumes defects exist).
Classifies from diffs, never commit messages. Detection only — never edits
the tree. Three checks:
1. **Faithfulness** — every item of every current (non-`.alt`) node is true
   of the code at the batch's last commit. Flag items describing code that
   does not exist there, a superseded approach stated as current, or wrong
   behavior. Read the source at that commit to confirm.
2. **Completeness** — every change/delete commit's design event is recorded
   faithfully and at the right structure: a replaced working mechanism is a
   re-decision (loser in `.alt/`, paired Moves); an expressivity-wall aspect
   is a promoted node+alt, not a fact, not buried on an unrelated node, not
   mislabeled (`dropped` vs `replaced`). Flag missing, mis-located,
   under-structured, or mislabeled events.
3. **Ledger spot-check** — re-derive a few HIT/REPAIR rows from their diffs.
Output: a numbered defect list, each with file/location, what is wrong, the
diff evidence, and the corrective action — or an explicit "clean" naming what
was verified. Every defect must be diff-backed; distinguish a defect (an item
is false / an event mis-recorded) from a note (true but worth flagging).
The observer relays the defect list to the parked extractor, which fixes and
re-lints; the same auditor then re-checks the fixed spots plus an adjacent
glance. Bound: two fix rounds, then escalate to the observer.

**Injector** (after author answers): fold answers into facts/whys, flip
question status, add `statement` rows.

**Observers** review batch report + ledger diff + tree diff (lint and
ledger checks are mechanical — review only judgment). At minimum,
re-read every `change`-class row whose ref claims no tree delta — that
reasoning is where dropped re-decisions hide. A reviewed batch is accepted
by **committing `design/` to git** — snapshots make batches replayable.

## Parallel mode — map/reduce over commit windows

Sequential mode threads tree state through every batch, so wall-clock is
the whole history. The dependency is narrower than it looks: of all the
judgments in the core loop, only two need global state — **naming** (is
this the same concept another window touched?) and **placement** (main
tree, or which `.alt/` generation). Everything else — what happened,
settling vs re-decision, the why, provenance, the loser's last-coherent
state — is commit-local, judged from the diff plus the code at the parent
commit, which git serves at any point in history. Parallel mode runs the
commit-local work concurrently and defers the two global judgments to one
reduce step. Every judgment rule in this document applies unchanged at map
time; only the orchestration differs. Throughout, **E** is the run's end
commit — repo HEAD for a full run, the last covered commit for a partial
one.

**Stage 1 — skeleton.** One pass over the code at E builds the main tree:
nodes and items only — no Facts, no Moves, no `.alt/`. Legitimate because
items are statements true of the current code, and the code at E is their
primary source; the faithfulness check applies as usual. Bias coarse: the
reduce deepens a too-coarse skeleton locally (aspect promotion), while a
too-fine one forces cross-window merges. The skeleton is **scaffolding,
not output**: it gives map agents shared coordinates and the reduce a
placement target, but its nodes are presumed to be module-map until
history earns them — the reduce prunes every node no timeline touches.
Tree size must track what history taught, never code size.

**Stage 2 — map.** History is cut into fixed windows of 10–20 commits in
`git log --reverse` order; `seq` is the commit's global ordinal, fixed by
the cut, so window outputs concatenate without coordination. One agent per
window reads every diff at depth D and writes exactly two files — it never
writes the tree:

- `extraction/win-<NN>.ledger.tsv` — same columns and hard rules as the
  ledger above; `batch` holds the window id. Map verdicts: **RECORD** —
  design information emitted (ref lists the record ids); **COVERED** — an
  implementation of a design already evident (ref names the covering
  concept: a skeleton node, an own-window record, or a mechanism present
  in the parent commit's code, with an anchor); FORCED and SKIP as above.
  **COVERED and SKIP are not fact-free**: before moving on, ask the diff
  one more question — does it demonstrate a rationale, a pitfall at its
  fix, or consumer evidence for an abstraction? If yes, emit the `fact`
  record even though no structure changed (observed failure: windows that
  treated COVERED as terminal lost every rationale fact).
- `extraction/win-<NN>.records.jsonl` — placement-free **evidence
  records**, one JSON object per line. Common fields: `id` (`W<NN>-r<k>`),
  `seq`, `hash` (8-char), `date`, `provenance`. A `concept` is always
  `{name: <one line>, anchors: [code identifiers / paths as of this
  commit]}` — anchors are the join keys the reduce resolves identity with.
  Types:

  - `transition` — a re-decision or deletion. `verb` (`replaced` /
    `dropped` / `removed`), `old` (a concept), `new` (a concept; null for
    dropped/removed), `why` (one sentence — the reduce writes it verbatim
    on both sides of the move pair), `frozen_items` (for replaced/removed:
    the loser's items at its last-coherent blob, drafted now — map time is
    the only moment the loser's code is already in hand).
  - `fact` — `kind` (open label as above), `concept`, `text`.
  - `birth` — a mechanism's first appearance: `concept` plus at most one
    item-worthy line. Births are identity evidence for the reduce, not
    tree content — pure adds stay LOW density. A why must never ride
    inside a birth's item text: reasoning demonstrable from the diff is
    its own `fact` record (kind `rationale`), or it is silently lost when
    the birth's node prunes (observed failure: a commit-once-on-success
    rationale discarded with its pruned instantiation birth).
  - `question` — the questions protocol, window-local id `W<NN>-q<k>`;
    in-record provenance reads `(inferred → W<NN>-q<k>)`.

  Exact shapes (`design/lint.py --window <NN>` enforces them; content here
  is from the fictional `acorn` project — copy the form, never the
  content):

  ```jsonl
  {"id":"W03-r1","seq":24,"hash":"ab12cd34","date":"2031-04-02","type":"transition","provenance":"diff","verb":"replaced","old":{"name":"write-through page cache","anchors":["WriteThrough","src/store/cache.rs"]},"new":{"name":"write-back page cache","anchors":["PageCache","src/store/cache.rs"]},"why":"write-through stalled every mutation on disk latency; batching at eviction removed the stall","frozen_items":["Every mutation writes its page to disk before returning."]}
  {"id":"W03-r2","seq":27,"hash":"cd34ef56","date":"2031-04-09","type":"fact","provenance":"diff","kind":"pitfall","concept":{"name":"write-back page cache","anchors":["PageCache"]},"text":"a crash between mutation and eviction loses the dirty page; a write-ahead log now sits in front of the cache"}
  {"id":"W03-r3","seq":21,"hash":"ef56ab78","date":"2031-03-30","type":"birth","provenance":"diff","concept":{"name":"page cache","anchors":["PageCache","src/store/cache.rs"]},"item":"Reads go through a fixed-size page cache; a page loads from disk only on a miss."}
  {"id":"W03-q1","seq":24,"hash":"ab12cd34","type":"question","context":"ab12cd34 replaces the allocator wholesale; the diff shows no failure of the old one","question":"what failed or was measured to motivate the replacement?","blocks":"page allocator"}
  ```

  For `dropped`/`removed`, `new` is null; `frozen_items` is required for
  `replaced`/`removed` (the superseded form needs its items) and absent
  for `dropped`. A `question` record carries no provenance — it is one.

A rename or representation change is itself a transition, so the old and
new anchors are linked exactly where name continuity breaks — identity
survives renames with no agent tracing lineages. A map agent may peek at
commits beyond its window to judge whether a deletion is transient churn
or real, but emits rows and records only for commits it owns. The
transient-boundary rule's faithfulness half has no analogue here —
faithfulness is checked only at E; its last-coherent-blob half lives on in
`frozen_items`.

**Stage 3 — reduce.** One agent; reads the skeleton and every record —
never the diffs. (If the records outgrow one context, fold per top-level
skeleton subtree first, then the root.) In order:

1. **Identity** — cluster records into concept timelines by anchors and
   names; transitions stitch timelines across their own rename boundaries.
2. **Fold, in seq order** — each timeline's transitions become the node
   and its generational `.alt/` chain: the final form must be a skeleton
   concept; predecessors nest by generation; move pairs are generated from
   each transition's single `why` (verbatim by construction);
   `frozen_items` become the alt's items; facts attach to the generation
   current at their seq; `dropped` lands on the surviving parent. Aspect
   promotion is decided here, with the node's whole transition set in
   view.
3. **Prune** — demote every node whose whole subtree gained no `.alt/`,
   no Facts, and no Moves from the fold: delete the node; its items
   collapse to the nearest surviving ancestor *iff they pass the
   regeneration test* (README) — what spec and competence regenerate
   anyway is deleted, what a rebuild could legitimately diverge on is
   kept. A commitment that passes the regeneration test but has no
   recorded why may instead earn its node outright with a `prior` fact
   (README) — marked, question opened, recovery lazy. This is the
   "node exists only if a real alternative exists" rule applied to the
   skeleton; `R-thin` enforces it mechanically.
4. **Questions** — renumber `W*-q*` globally, write `questions.md`,
   rewrite the `(inferred → ...)` tags accordingly.
5. **Placements** — write `extraction/placements.tsv`: every record id →
   the (post-prune) tree path(s) it landed at, or `discarded: <reason>`.
   The reduce never invents: every Facts/Moves line in the tree derives
   from a record.
6. Concatenate window ledgers into `ledger.tsv`; run lint; fix.

Closure — mechanical, and the reason parallel mode is safe (errors surface
as contradictions, not silent gaps):

- every transition chain ends at a skeleton concept or in a `removed` /
  `dropped` — an orphan chain is a missed transition or a misfiled dead
  lineage;
- every skeleton node either earned structure from the fold or was
  pruned — none survives on code shape alone;
- every record id appears in placements exactly once;
- concatenated ledger: rows == commits, seq contiguous, zero M rows.

**Stage 4 — audit.** Per window, the same adversarial fresh-agent contract
as sequential mode, retargeted at records: every change/delete commit's
design event has a faithful record (completeness); every record is
diff-backed (faithfulness); spot-check COVERED rows. One reduce auditor
checks tree-vs-records (no invention; placements honest) and skeleton
items against the code at E. Fix loops as above: defects relay to the
parked map/reduce agent, two rounds, then escalate. Acceptance = lint +
closure + all audits clean → commit `design/`.
