# design/ — the project's design tree

This directory externalizes the design *reasoning* of Silverfir-nano as a Monte
Carlo Tree Search over the solution space: options are choices, facts are the
evidence that re-weights them, and the current design is the selected traversal.
The model is in [`docs/DESIGN_GRAPH.md`](docs/DESIGN_GRAPH.md); the memory /
value-function is in [`docs/FACT_BASE.md`](docs/FACT_BASE.md).

```
design/
├── design-tree/   the design — options, alternatives (.alt/), evidence (.evidence/)
└── docs/          DESIGN_GRAPH.md (the model) + FACT_BASE.md (the memory)
```

## Tree conventions (`design-tree/`)

The filesystem **is** the tree.

**The main tree IS the current design** — the live architecture laid down as
files. Ignore every `.alt/` and you are reading exactly what the code implements
(and what code review checks it against). Alternatives are recorded, never
deleted: they sit beside what beat them.

- `<name>.md` — an **option**: one element of the design. Body = *what* it is,
  plus an **"In practice" contract** (Must / Must not) — the source of truth for
  codegen and code review (the faithfulness gate's spec). **No justification
  narrative**: the *why* lives in `.evidence/` annexes and is read or retrieved,
  never narrated in the body.
- `<name>/` — its parts: the sub-options it is composed of, all in force —
  components living at the same time.
- `<name>.alt/` — its **alternatives**: predecessors and rivals, each a full
  option (with its own subtree and its own `.alt/`s). `.alt` nesting is the
  generational record — walking in is walking back in time.
- `<name>.evidence/` — everything recorded that **bears on it**: implementation
  discoveries, measurements and diagnosed causes, external references (papers,
  articles, experiments elsewhere — filed at the decision they informed), and
  judgment/taste records. Pure prose, no fields (retroactive entries carry
  `commit: <hash>`). The read-time judge *weighs the evidence*.
- Presence encodes selection: main tree = current; inside an `.alt/` =
  superseded or rejected. No `selected:` marker exists.

**Logical vs physical addressing.** `.alt/` is purely physical structuring. In
the logical graph an `.alt/` member is a *sibling* of the option it is attached
to — together they form the OR-group under the parent. Fact `origin:` paths are
**logical** (`.alt` segments stripped), which makes them pivot-proof: demoting an
option into an `.alt/` does not change its logical path.

**Refactoring is file motion.** A pivot = move the incumbent (`X.md` + `X/` +
`X.alt/`) into the challenger's `.alt/` and build the challenger in the main
tree — it is the working design from day one; a failed exploration is the
reverse move. Borrowed components are *copied*; the `.alt` keeps its complete
design.

Lint: no orphan dirs (`X/` and `X.alt/` require a sibling `X.md`), no empty
`.alt/`, no `selected:` anywhere, every fact origin resolves logically.

The only **nodes** are the `.md` files; directories are structure. The path
narrates the reasoning, and *is* the (non-Markovian) context. The current design
= follow the `selected: true` options from `root.md`.

## Evidence conventions (`X.evidence/`)

Evidence is **co-located**: each option's annex holds what bears on it, so
navigation is mechanical — browse the option, its evidence is beside it. No
retrieval needed to answer "why is this here / why was that abandoned".

- `X.evidence/<slug>.md` — pure prose, **no fields**: the observation, its
  conditions, and its nature (measurement, diagnosis, external reference,
  judgment/taste) all live in the statement. Names stay short — the location
  carries the context. One exception: evidence recorded *retroactively* carries
  `commit: <hash>` — the historical commit it belongs to.
- **Location is origin** (logical path = physical path with `.alt`/`.evidence`
  stripped); evidence moves with its option's subtree on pivots, nothing to
  remap. Placement is provenance and filing — never exclusive bearing: relevance
  to other decisions stays a **read-time judgment**, over a derived global index
  (`**/*.evidence/*.md`).
- Consolidations (lessons spanning a subtree) live at the subtree's root —
  **height = generality**; `root.evidence/` is the distilled-wisdom layer.
- End-to-end *unattributed* measurements (whole-system benchmark runs) have no
  home here by design; a chronological `runs/` log is the planned extension if
  needed — credit assignment is explicit work, never a filing default.
