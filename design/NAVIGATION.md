# Navigating the design tree (for planning a change)

The tree at `design-tree/` is a map of **design decisions and their reasons**,
not a module map. Read it *before* planning a change to the engine — it tells
you what is settled, what was already tried, and what is still open.

## How to read it

- **Start at the root** (`silverfir.md`) and descend by subsystem. The **main
  tree** (ignore every `.alt/`) IS the current design: its Items are statements
  true of the code today.
- **Each node** has: Items (what holds true now), `## Facts` (dated evidence —
  measurements, pitfalls, rationale, each with a `(diff)`/`(author)` provenance
  tag), and `## Moves` (re-decisions: what was replaced and the why).
- **`.alt/` = roads not taken / superseded forms**, frozen with the reason they
  lost. **Walking into `.alt/` walks back in time.**
- **`.fact/` = graduated evidence** with body (measurement tables, recovered
  design docs) — the detail behind a one-line fact.

## The one rule that saves the most time

**Before you propose approach X, search the tree for X.** If X already sits in
an `.alt/`, it was tried — read its `## Moves` and `## Facts` for the measured
reason it lost. Do not re-walk a fenced-off dead end.

```
grep -rl "<your-approach-keyword>" design/design-tree --include=*.md
```

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
