# The Fact-Base: an Agentic Memory

The fact-base is the value function of the Design Tree (see
[DESIGN_GRAPH.md](DESIGN_GRAPH.md)): the accumulated, transferable evidence that
re-weights options and guides future search. It is an **agentic memory**, not a
structured rule base. Facts are raw observations; their *applicability* to a
given situation is judged at **retrieval time**, never authored at capture.

## Why no authored scope

An earlier design gave each fact a hand-written *scope* predicate ("applies to
32-bit split-pair backends; not 64-bit"). A stress-test against 15 real facts
killed it for two reasons:

1. **Scope is write-only and fails silently.** A too-narrow scope drops a fact
   with zero error signal — you never see what you didn't retrieve. A scope
   pinned to a sibling-of-the-target exclusion ("does not occur on SysV") buries
   the hazard on a new SysV target. The production memory already showed the rot
   (a fact scoped to a crate by name).
2. **The option space is generative.** A fact can bear on options that *do not
   exist yet* (a future ISA, a new lowering). A capture-time scope is a guess at
   a future you cannot enumerate.

Removing scope moves the applicability judgment to read time — where a
wrongly-*surfaced* fact is read and discarded (visible, correctable) instead of
wrongly-*omitted* (invisible). For a system whose whole failure mode was
invisibility, that is strictly better. The abstraction work does not vanish; it
relocates from a human-authored predicate to retrieval and reflection, both
LLM-driven.

## A fact (stored as evidence)

Minimal, prose-first, cheap to write — co-located with the option it bears on,
in that option's `.evidence/` annex:

```
<option>.evidence/<slug>.md
<the immutable observation, in prose: what was seen, under what conditions, of
 what nature — a benchmark, a bug, a judgment, an external reference,
 "infeasible".>
```

"Evidence" is the operational term because the annex holds every kind of input
that bears on a decision: implementation discoveries, measurements and diagnosed
causes, external references (papers, articles, experiments elsewhere — filed at
the decision they informed), and judgment/taste records. The read-time judge
*weighs the evidence* — one metaphor end to end.

**There are no fields at all** (one exception below). Everything a fact might
seem to need is location-derived, git-derived, or *part of the observation
itself*:

- **No `origin`.** Location *is* origin: the logical path (physical path with
  `.alt`/`.evidence` segments stripped) names the option it was observed under
  or informed. Evidence moves with its option's subtree on pivots — no field to
  maintain, nothing to remap.
- **No `kind`.** The file is in an `.evidence/` annex; that is what it is.
- **No `provenance` field.** Whether a fact is a hard measurement or a soft
  judgment, internal or external, is read from the *statement* (a benchmark reads
  as a measurement; "architecturally wrong" reads as a judgment), and the
  read-time judge weighs trust accordingly. It is content, not a tag.
- **No `env` field.** The conditions an observation holds under — host,
  toolchain, arch — are *part of the observation*; write them in the statement,
  where they can be exactly as specific as needed. A structured `env` would be
  premature and ill-defined (which dimensions? which versions?).
- **No `date`, `id`, or version stamps.** Git supplies the date (the fact's
  commit) and the identity (its path); the same commit pins the tree and code
  state when they share the repo. One exception: a fact recorded *retroactively*
  (backfilled from history-mining) carries `commit: <hash>` — the historical
  commit whose code produced the observation, since the fact's own commit is
  merely when it was transcribed. A hash strictly dominates a date: it pins the
  exact state, and the date is derivable from it.
- **No authored bearing.** A fact never declares which option it
  supports/undermines or how strongly — relevance and direction are a
  **read-time judgment**, the no-authored-scope rule applied to relevance.

A raw fact has one home; the same issue under a *different* traversal is another
observation, and generalizing across them is the reflection pass's job — a
consolidation lives at the lowest common ancestor of its sources, so **height =
generality**: deep evidence is specific, high evidence is distilled wisdom.

**Evidence is co-located, not pooled** (a reversal, made on evidence). An earlier
design kept facts in a separate flat `facts/` pool linked by an `origin` field.
The pool's costs materialized — long self-describing names, an unreadable flat
folder, and retrieval (RAG) as the *only* access path — while two later decisions
dissolved the original reasons for separation: logical addressing made links
pivot-proof, and `.alt`-relocation moves subtrees *intact*, carrying co-located
evidence for free. So evidence now lives beside its option, navigation is
mechanical (browse the annex), and the *logical* separation survives untouched:
"the evidence relevant to this decision" elsewhere in the tree is still
**retrieved**, never stored — placement is provenance and filing, not exclusive
bearing, and the retrieval index is a derived global view over
`**/*.evidence/*.md`. End-to-end *unattributed* measurements (whole-system
benchmark runs, where credit assignment hasn't happened) have no home here by
design; a chronological `runs/` log is the planned extension if the need
materializes — attribution is explicit work, never a filing default.

**Validity gate — faithfulness.** A fact is only valid if the code that produced
it *faithfully* implemented the option it is attributed to. Unfaithful code (the
harness deviated from the design) yields a plausible-looking **false fact** — a
benchmark for the wrong thing — which poisons the value function. Faithfulness
verification (review) gates every recorded fact; an unfaithful attempt is a
harness defect, discarded and re-run, never recorded. Conversely, a faithful
attempt that *cannot be completed* — the design is infeasible under the
constraints (e.g. it requires `unsafe` that is disallowed) — produces a valid
fact ("infeasible"), which prunes the option exactly like a measured regression.

## Provenance & traceability

Capture-time review is fallible — a false fact (unfaithful code that looked
right) can slip through. So every fact is stamped with the **lineage of the
rollout that produced it**, mechanically, not by hand:

```
the fact's own git commit  +  origin: <traversal path>
```

If the design tree, the facts, and the code share one repo, the fact's **own
commit** pins the entire state — tree, code, and *when* — so the only fields it
must carry are `origin` (which traversal was rolled out) and `env` (what git
can't infer). Everything temporal and version-related is `git log` of the fact
file, not authored formalism. (Separate repos for code, or a cross-project fact
pool, would re-introduce an explicit `code-sha`; don't pre-pay for it.) This
makes every fact auditable and correctable:

- **Re-audit.** Check out `code-sha`, re-review it for faithfulness to the option
  it was attributed to. Capture-time review stops being the only line of defense;
  faithfulness becomes a continuously checkable property, and a re-audited fact
  is more trustworthy than a fresh one.
- **Correct, don't delete.** A fact is an immutable record (the unfaithful code
  really did produce that number). When found false, you do not erase it — you
  invalidate its *attribution* and supersede it with a corrected fact (a faithful
  re-run). The correction is itself informative: it shows how wrong the earlier
  weighting was. Immutable facts, superseding corrections — the same discipline
  as the rest of the system.

Keep it light: the cheap core is the stamp plus *on-demand* re-run only when a
fact is suspect. A full reproducibility platform (pinned containers, automatic
re-execution) is overkill at this corpus size.

**Provenance is the factual replacement for the authored scope we rejected.**
Scope was a forward-looking *guess* at future applicability (write-only, cannot
anticipate future options). Provenance is a backward-looking *record* of the
actual conditions that produced the fact (where, when, on what toolchain/host).
It does not reintroduce scope — it grounds the read-time judge that replaced it:
the judge reads "measured on arm64/Apple-Silicon, toolchain X, date D" and
decides applicability from real conditions instead of a guessed predicate.

## Architecture

Grounded in the 2023–2026 agentic-memory / RAG literature (adversarially
verified; see Sources). Four mechanisms, with what is essential vs. overkill at
this corpus size (hundreds to low thousands of high-value facts).

### 1. Cheap write — store the raw observation

Capture is one prose statement plus provenance and date. Generative-agents and
Zep both store the complete raw episode stream and consolidate *later*; capture
must never be gated on structure (the cost that killed 1990s design-rationale
systems). Essential.

### 2. Recall-biased retrieval + read-time LLM judge

The core. Cast a wide net (high recall), then an LLM **judge** scores each
candidate's relevance to the current situation by *reasoning*, not cosine
distance, and filters. This is the visible-failure design: surface generously,
discard cheaply. The literature backs it directly — LLM-as-reranker /
JudgeRank, and MAIN-RAG's Predictor/Judge agents with an adaptive
score-threshold, are specifically aimed at reasoning-intensive retrieval where
surface similarity fails. Essential. A cross-encoder reranker is the cheap
version of the judge and a known low-cost fix for RAG misses.

### 3. Cross-mechanism transfer via HyDE-style query expansion

The hard requirement: an ARM32 register-aliasing fact (tokens `R2`,
`mandelbrot`, `soft-float`) must surface for a surface-dissimilar "new RISC-V32
backend" query. Naive dense cosine fails — the texts share almost no tokens.
**HyDE** (Hypothetical Document Embeddings) bridges it: an LLM first expands the
query into a hypothetical document describing the *mechanisms* the task involves
(register marshalling, ABI arg setup, aliasing hazards), then retrieves against
that — which is textually near the arm32 fact. Verified to work zero-shot with
no relevance labels and to beat the unsupervised dense retriever. Essential and
cheap (one LLM call per query). Optional add-on: **HippoRAG**-style associative
retrieval (single-step multi-hop over a lightweight graph) for "this connects to
that" chains — nice-to-have, not essential at this scale.

### 4. Periodic reflection / consolidation

A background pass reads accumulated observations and synthesizes higher-level
**class** memories — so several instances of one underlying class, recorded
independently and days apart, become one consolidated memory (the three
register-aliasing facts → one "source aliases destination during marshalling"
hazard class). This is the generative-agents *reflection* mechanism, productized
as Zep's episode→higher-level consolidation. It solves the class-homelessness
gap without fact→fact edges or capture-time coupling, and it is also where
contradiction and supersession get noticed. Essential — it is what makes the
memory *generalize* rather than just store.

### Recency, contradiction

Date every fact (cheap). LLMs exhibit a **causal** recency bias when dates are
visible — useful (newer facts preferred) but a hazard: guard against burying an
old-but-still-valid fact merely for being old. Contradiction and supersession
are handled by the reflection pass noticing two facts conflict; Graphiti's
bi-temporal auto-invalidation of superseded edges is the heavyweight productized
form of the same idea.

## What is overkill at this scale

- **Full GraphRAG community-summary indexing** — built for million-document
  global sensemaking; a documented cost cliff. (Note: the strong claim that
  vector RAG *structurally cannot* do global sensemaking was refuted — it is a
  cost/quality tradeoff, not an impossibility.)
- **Heavy temporal knowledge-graph infrastructure** (full Zep/Graphiti stack) —
  the *ideas* (bi-temporal, auto-contradiction) are worth borrowing; the infra
  is overkill for low thousands of facts.
- **ColBERT / late-interaction indexing** — unnecessary at this scale.

Essential and cheap: HyDE query expansion, recall + LLM-judge/rerank, periodic
reflection, dating. That is the whole system.

## The architecture in one line

**Store raw (cheap) → HyDE-expand the query + recall wide → LLM judge at read
time → periodic reflection consolidates classes; date for recency. No authored
scope.**

## Honest constraints

- Retrieval quality is bounded by the HyDE/judge LLM — the same ceiling as the
  option-proposer. The memory does not remove judgment; it relocates it to read
  time, where it is visible.
- Reflection can hallucinate generalizations (over-broad class memories). It
  needs the same human-curation/visibility discipline as everything else: a
  consolidated memory is a *proposed* generalization until reviewed.
- Recency bias is real and causal; recency must inform, not dominate.

## Sources

Adversarially verified (3-0) from the 2023–2026 literature: HyDE
([2212.10496 / ACL 2023](https://aclanthology.org/2023.acl-long.99/));
HippoRAG / HippoRAG 2 ([2405.14831](https://arxiv.org/abs/2405.14831),
[2502.14802](https://arxiv.org/pdf/2502.14802)); GraphRAG
([2404.16130](https://arxiv.org/abs/2404.16130)); LLM-as-judge retrieval —
JudgeRank / MAIN-RAG ([2411.00142](https://arxiv.org/abs/2411.00142),
[2501.00332](https://arxiv.org/pdf/2501.00332)); generative-agents reflection
([2304.03442](https://arxiv.org/abs/2304.03442)); Zep / Graphiti temporal memory
([2501.13956](https://arxiv.org/abs/2501.13956)); RAPTOR recursive summarization
([2401.18059](https://arxiv.org/html/2401.18059v1)). Two claims were refuted: the
GraphRAG "structurally cannot" overstatement, and a specific HippoRAG 2
mechanism/number — neither is relied on above.
