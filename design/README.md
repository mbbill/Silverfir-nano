# design/ — the project's design tree

This directory externalizes design *reasoning* as a search over the solution
space: options are choices, facts are the evidence that weights them, and the
current design is the selected path. The code is one path through that space —
the path settled *today*; the tree records the explored region, the lessons,
and the known-bad territory. The model is in
[`docs/DESIGN_GRAPH.md`](docs/DESIGN_GRAPH.md); the memory / value-function is
in [`docs/FACT_BASE.md`](docs/FACT_BASE.md).

**Who reads this tree, and why.** Every line is read by someone — human or
agent — deciding what to build next: rewriting the system, extending it, or
re-opening a settled choice. A line pays rent only if it could change that
decision: something to keep, something to avoid, something now free to
reconsider. Calibration — pays rent: an expressivity wall ("the old shape
could not express X"), a kill-fact, a measured loss, a road not taken. Never
pays: allocation hygiene, idiom swaps, dependency bumps, progress milestones,
restatements of the code. This is the test for every item, move, and fact.

```
design/
├── design-tree/   the design — options, alternatives (.alt/), facts (.fact/)
├── docs/          DESIGN_GRAPH.md (the model) + FACT_BASE.md (the memory)
├── EXTRACTION.md  the one-time fact-extraction workflow (recovering history)
└── extraction/    working state of an extraction run (ledger, questions)
```

## Tree grammar (`design-tree/`)

The filesystem **is** the tree. Exactly one top-level node exists: the **root
option**, named after the system. The main tree IS the current design — ignore
every `.alt/` and you are reading exactly what the code implements.

**`<name>.md` — an option node.** Two sections:

- **Items** (top of file, no heading): statements that hold true of the
  implementation while this option is current. Each starts with `- `,
  separated by blank lines. Items are **concepts, not code**: each must hold
  for *any* faithful implementation of the decision. At most one code
  identifier per node, parenthesized, as a findability anchor; never
  enumerate functions, methods, or fields. Items are checkable, never
  argued — no "so that / because" tails (the why lives in facts and moves).
  Items are edited freely as the design evolves — git records when. When an
  option is superseded, its items freeze at their last-true state.
  **Items and facts describe the live code only** — never commented-out,
  aspirational, or planned code. A dead reference or a TODO is not a design
  statement. The distinction from a bug: a localized bug (the code runs but
  computes the wrong thing — a flipped predicate, an off-by-one) does not
  change the design, so the item states the design and the bug is a pitfall
  fact at its fix; but an *unimplemented* capability (a `TODO`, a stub, a
  missing branch — the code does not do it at all) must never be asserted by
  an item, even when it is the design's evident intent. State only what the
  code performs; the unimplemented half is recorded when it is implemented.

- **`## Facts`** (only if the node has any): an append-only log of what
  happened, broke, was measured, or was stated — plus recorded judgment and
  rationale. **Admission test, applied before writing any fact: name (a) the
  decision it bears on and (b) what a future implementor would do
  differently knowing it. If you cannot name both, it is not a fact — it is
  narration.** Observations about the repository or its history (commit
  shapes, squashed imports, development gaps, renames, file counts), about
  the development process, or about tooling are never facts: facts are about
  the design, not the project's paperwork. Evidence that an abstraction has
  more than one real consumer is a fact on the abstraction's own node, not an
  item elsewhere; runtime gating, logging, and similar plumbing are not
  items. **A re-decision's why lives in
  its Moves line, not here** — a `rationale` fact is allowed only for a why
  that has no move to carry it (the reasoning behind a current design that
  never displaced a predecessor). Never file a fact that restates a Moves
  line or an item. Entries start with `- `, blank-line separated:
  `- <date> [(<hash>)] <kind>: <text> (provenance)` — kind is a short open
  label (`pitfall`, `rationale`, `measurement`, `statement`, ...); the hash
  is required for facts demonstrable from a commit, omitted for author
  statements. A fact never re-describes the chosen design, and **the tree
  never references its own construction** (no batches, deferrals, or
  extraction bookkeeping — that lives in the ledger). A fact graduates to
  its own file in `<name>.fact/` only when it has *body* — measurement
  tables, a recovered document, a rollout report, a long diagnosis — and
  then the section carries a one-line entry linking it. Same principle as
  node-vs-item graduation.

- **`## Moves`** (only if the node has any): an append-only log written
  **only when something crosses the `.alt/` boundary or is dropped** — never
  for births, item edits, or progress. A node that never moved has no Moves
  section. Entries start with `- `, blank-line separated, 8-char hashes:

  - `- <date> (<hash>) replaced [[X]]: <why> (provenance)` — on the winner.
  - `- <date> (<hash>) replaced by [[X]]: <why> (provenance)` — on the loser,
    now in `.alt/`. **The why is copied verbatim on both sides** — same
    sentence, never paraphrased (verbatim means the sentence, not the
    line-wrapping).
  - `- <date> (<hash>) dropped: <what>: <why> (provenance)` — part of this
    design was deleted with **no successor**. One line here; no ghost node for
    "not doing it". Dropping an external dependency but reimplementing its
    capability in-tree is **not** a drop — the in-tree code is the successor,
    so it is a `replaced` re-decision (node + `.alt/` holding the dependency's
    rejected shape). `dropped` is only for a capability removed outright.
  - `- <date> (<hash>) removed: <why> (provenance)` — this whole node was
    deleted with no successor (the node itself now sits in `.alt/`).
  - `- <date> (<hash>) revived: <why> (provenance)` — rare.

  Every why carries provenance (see the provenance rule below).

**Provenance — applies to every Facts entry and every Moves entry.** Each
entry ends with its provenance: `(diff)` — demonstrable from the change
itself; `(author)` — stated by a human; `(inferred → Qn)` — a plausible
reading that MUST open question Qn. An unmarked entry is invalid. **The tag
covers every clause**: if any part of the entry is inference, split the
entry or downgrade the whole of it — a true observation must never smuggle
in an inferred cause.

**`<name>/`** — the option's parts: sub-options all in force at once.

**`<name>.alt/`** — its alternatives: predecessors and rivals, each a full
node (own subtree, own `.alt/`s). Walking in is walking back in time. An alt's
Moves end in `replaced by` or `removed`; revival is rare — `.alt/` is
effectively permanent. Record only alternatives that really existed in this
codebase or were really weighed — never invent rivals.

**`<name>.fact/`** — graduated facts only: fact files exist solely for
facts with body (tables, recovered documents, rollout reports, long
diagnoses); everything smaller lives in the node's `## Facts` section. Pure
prose, no headings; retroactive files carry `commit: <hash>` on the first
line; author statements are quoted with their date. Facts are immutable —
history does not change. Location is provenance (file the fact where it was
discovered), never exclusive bearing.

**Rules of structure:**

- A node exists only if a real alternative exists; otherwise the content is
  an item on the nearest real node, or nothing. The tree is **not a module
  map**. Representation budget follows information density: an unweighed path
  stays thin — never compensate missing facts with description. Fact density
  is the confidence signal: a node with no facts honestly says "nobody
  weighed this; reconsider freely."
- The tree must be *generatively sufficient*: tree + spec + ordinary
  engineering competence rebuilds the code. The faithfulness check reads
  items against the implementation.
- **A replacement is a re-decision by default.** When a working mechanism or
  representation is replaced — whether a whole node or its internal shape —
  move the old form into `.alt/` and write the paired move lines, unless the
  change is purely cosmetic (rename, restyle). If the old shape could not
  express something the new one can, that delta is the lesson; record it.
  **When the thing re-decided is an internal aspect of a node, not a node of
  its own, promote that aspect to a child node** so its rejected form has a
  home in the child's `.alt/`: `parent/aspect.md` (current) +
  `parent/aspect.alt/old-aspect.md`, with paired move lines. The test is
  objective: **if the chosen form exists because the old form hit an
  expressivity wall** — something it could not do, which the move why states —
  the rejected shape pays rent by definition; promote it. Only a pure-taste
  replacement with no wall (the old form worked, the new is merely nicer) may
  fold to a why-only rationale fact. Never put a half-node in `.alt/` that
  describes only one facet of its parent.
- A pivot is file motion: move the incumbent (`X.md` + `X/` + `X.alt/`) into
  the challenger's `.alt/`; borrowed components are copied so the alt keeps
  its complete design. The paired move lines land in the same change.
- Logical addressing: an `.alt/` member is logically a *sibling* of the
  option it is attached to; logical paths strip `.alt`/`.fact` segments and
  are pivot-proof.

A canonical node, for calibration — from a **fictional** project `acorn`, a
key-value store (root `acorn.md`; parts under `acorn/`; this is
`acorn/storage/page-cache.md`). Copy its **form**, never its names, paths, or
topology:

```markdown
- Reads go through a fixed-size page cache (`PageCache`); a page is only
  ever loaded from disk on a cache miss.

- Dirty pages are written back on eviction, not on every mutation.

## Facts

- 2031-04-02 (ab12cd34) measurement: under the ingest benchmark,
  write-through spent 71% of wall time blocked on synchronous page writes;
  batching at eviction cut ingest latency 3.4x — full run data in
  [[page-cache.fact/write-through-stall]] (diff).

- 2031-06-17 (ef56ab78) pitfall: a crash between mutation and eviction
  loses the dirty page; a write-ahead log now sits in front of the cache
  (diff).

## Moves

- 2031-04-02 (ab12cd34) replaced [[write-through-cache]]: write-through
  stalled every mutation on disk latency; batching at eviction removed the
  stall (diff).

- 2031-09-05 (cd34ef56) dropped: per-page checksums — the storage layer
  gained end-to-end checksums, making the cache's own redundant (diff).
```

`write-through-cache.md` sits in `page-cache.alt/`, items frozen, its Moves
ending with the same why verbatim: `replaced by [[page-cache]]: write-through
stalled every mutation on disk latency; batching at eviction removed the
stall (diff).` A canonical fact (`page-cache.fact/write-through-stall.md`):

```markdown
commit: ab12cd34

Under the ingest benchmark, write-through caching spent 71% of wall time
blocked on synchronous page writes; batching dirty pages at eviction cut
ingest latency 3.4x.
```

Lint — mechanically enforced by `python3 design/lint.py` (each check cites
the rule here): one top-level node; no orphan dirs (`X/`, `X.alt/`,
`X.fact/` need a sibling `X.md`); no empty `.alt/` or `.fact/`; no title
headings; Items first, then optional `## Facts`, then optional `## Moves`,
no other headings; every Facts/Moves entry dated, labeled, and
provenance-tagged; Moves only for boundary events, append-only (checked
against the last accepted commit), paired whys verbatim (wrap-insensitive);
every `[[link]]` resolves; an `.alt` member's Moves end in `replaced by` /
`removed`; fact files heading-free; no invented rivals; facts never
re-describe the design; the tree never references its own construction.

## Maintaining the tree during development

- A re-decision and its file motion land together: the change that alters the
  code moves the loser into `.alt/`, writes both move lines, and files the
  fact if it has body. Never let the tree lag the code.
- New mechanisms get nodes when they earn them (real alternative, properties
  beyond a line or two); otherwise items on the nearest node.
- Item edits are silent — git records when; a driving fact, if any, is filed
  in `.fact/`.
- Deletions without successor get a `dropped` (or `removed`) move line.
- Review reads the tree: a change contradicting a holds-true item either
  fixes the code or updates the tree — silent divergence is the failure mode
  the tree exists to prevent.
