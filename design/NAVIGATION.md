# Navigating the design tree (for planning a change)

The tree at `design-tree/` is a map of **design decisions and their reasons**,
not a module map. Read it *before* planning a change to the engine — it tells
you what is settled, what was already tried, and what is still open.

## Follow the tree — don't search it

The tree is a **map**, and you navigate it by *following its structure*, not by
grepping for keywords. The structure carries the meaning: walking from a node
into its `.alt/` chain shows you *what was tried, in what order, and why each
form lost* — context a keyword jump throws away. An agent that lands on a node
by grep sees the answer without the reasoning that makes it trustworthy.

- **Start at the root** (`silverfir.md`) and descend by subsystem. The **main
  tree** (ignore every `.alt/`) IS the current design: its Items are statements
  true of the code today.
- **Each node** has: Items (what holds true now), `## Facts` (dated evidence —
  measurements, pitfalls, rationale, each with a `(diff)`/`(author)` provenance
  tag), and `## Moves` (re-decisions: what was replaced and the why). Follow
  `[[links]]` to related nodes.
- **`.alt/` = roads not taken / superseded forms**, frozen with the reason they
  lost. **Walking into `.alt/` walks back in time.** When you're weighing an
  approach, read the relevant subsystem's whole `.alt/` chain — that *is* the
  record of what's been explored there.
- `grep` is a **last-resort locator** for when you genuinely can't place a topic
  by structure — not the primary method. If the tree is well-formed you rarely
  need it.

## Two layers: decisions you follow, evidence you consult on demand

- **The decision layer — Items, `## Facts`, `## Moves`, `.alt/`** — is complete
  for planning. Every decision and its why lives here. **Read this to decide.**
- **The archive layer — graduated `.fact/` documents** (recovered design papers,
  measurement tables, long diagnoses) — is the *evidence behind* the decision
  layer, kept for audit and for raw detail. Its decision-relevant content is
  already distilled into the node facts that cite it. **Read a `.fact/` document
  only when you need a specific measurement table or proof** — never just to
  "understand the design," which the nodes already give you. A 40 KB recovered
  paper is provenance, not required reading.

## The one rule that saves the most time

**Before you propose approach X, walk the subsystem's `.alt/` chain.** If X
already sits there, it was tried — read its `## Moves` and `## Facts` for the
measured reason it lost. Do not re-walk a fenced-off dead end — **unless** its
recorded rejection reason no longer applies under your new constraints, in which
case the alt may be *reclaimable*: say so explicitly, cite the reason that
lapsed, and build on it.

## Reading the confidence signal

- **Fact density = how hard a choice was weighed.** A node thick with
  measurements was fought over — reconsider it carefully, with evidence. A
  **thin node** (few/no facts) honestly says "nobody weighed this; reconsider
  freely."
- **`(inferred → Qn)` tags and the questions log** mark open unknowns or
  author-only knowledge the artifacts could not show.

## A planning checklist

1. Find the subsystem node; read its Items (current design), Facts (what is
   known / measured), Moves (what changed and why).
2. **Search `.alt/` for your intended approach.** If present → it was tried;
   read why it lost before proposing it again.
3. Check fact density: is this area settled-by-measurement or open?
4. Build your plan *on top of* what the tree already settled — extend the
   living design, do not re-derive it or re-open a measured-closed choice
   without new evidence that beats the recorded one.
