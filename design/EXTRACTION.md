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
  the nearest node. No, pure slip → SKIP.
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
- **Settling vs re-decision**: while a mechanism is still being built (no
  consumer has yet run against it), its internal iterations are settling —
  record the settled state, HIT the refinements. Once a form has been the
  working design, replacing it is a re-decision.
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

## The agent workflow

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

**Auditor** (era boundaries, or on observer demand):
1. Generative-sufficiency sweep at the boundary commit: every item true?
   any code subsystem with no generating node? Report discrepancies as
   proposed REPAIRs — do not apply.
2. Spot-audit K random HIT rows: re-derive each verdict from the diff;
   report rows whose evidence does not hold.

**Injector** (after author answers): fold answers into facts/whys, flip
question status, add `statement` rows.

**Observers** review batch report + ledger diff + tree diff (lint and
ledger checks are mechanical — review only judgment). At minimum,
re-read every `change`-class row whose ref claims no tree delta — that
reasoning is where dropped re-decisions hide. A reviewed batch is accepted
by **committing `design/` to git** — snapshots make batches replayable.
