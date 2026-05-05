# sf-lens

Code-fact extractor + browser viewer for understanding ownership and architecture in Rust workspaces. Two parts:

- `src/` — Rust binary `sf-lens`. Walks crates with `syn`, emits a JSON document of per-type facts plus a global edge graph.
- `viewer/` — TypeScript + D3 + SVG browser viewer that consumes that JSON.

The user is the architect. The tool surfaces facts; the user derives meaning. The tool must not interpret.

## Working rules

1. **Keep the structure clean.** Every change must fit the existing design. If a new requirement doesn't fit cleanly — or the only path forward would be ugly, hacky, RefCell-shaped, threading-extra-params, or `#[allow]`-suppressed — **stop and discuss before writing code.** A short conversation usually reveals a cleaner path that the workaround would never reach.
2. **Step by step.** Build slices, not big bangs. Each step adds one capability and ships in good shape (typecheck + lint + tests + manual UI verification, all green) before the next step starts.
3. **Production grade in the viewer.** TS strict (`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, `allowImportingTsExtensions`). No `any`. Pure functions stay pure (no DOM, no D3, no mutated tree nodes). View state is keyed by stable IDs — never smeared onto domain objects. Tests cover pure analysis; rendering is verified by running the viewer.
4. **Validate UI by running it.** `npm run dev` and check the browser. Type-checking and unit tests don't verify feature correctness.

## Core discipline (do not violate)

These define what the tool *is*. Not negotiable for cosmetic or ergonomic reasons.

1. **Facts only, no labels.** Never put words like "data hub", "pipeline stage", "library type", "core entity", "API surface" into data or visualization. The user reads facts; the tool surfaces facts.
2. **LCA rule, no carveouts.** For any owned type T:
   - `T's correct module` = LCA of (modules of T's owners).
   - Classifications: `at_lca`, `within_budget` (≤ N levels below, default N=1), `drift_below`, `drift_above`, `drift_sideways`.
   - **No exceptions for "library types".** Top-level placement isn't free because a type is "library-shaped" — drift is computed from actual usage edges.
3. **Modules are API boundaries.** The module system is for API boundaries and only for API boundaries. Cross-module references that bypass a module's pub API are signal of design drift.
4. **`pub` is a backdoor.** Public visibility doesn't excuse misplacement. Drift is computed from usage edges, not visibility.
5. **Same-named types must disambiguate.** Display: shortest unique path suffix. Resolver: context-aware longest-module-prefix-match against the source location of the reference.
6. **Tests excluded.** Anything under a `tests` module path is excluded from analysis (extractor and viewer).

## Project layout

```
tools/sf-lens/
├── Cargo.toml              # crate "sf-lens", binary "sf-lens"
├── sf-lens.md              # this file
├── src/                    # Rust extractor
│   ├── main.rs             # CLI entry
│   ├── model.rs            # JSON schema — canonical source of truth
│   ├── extract.rs          # syn walk, edge emission
│   ├── resolve.rs          # type classifier, container_cardinality
│   ├── unified.rs          # drift analysis (LCA / classification)
│   ├── architecture.rs     # Tarjan SCC for module-cycle detection
│   ├── ownership.rs        # ownership-tree console reporter
│   └── survey.rs, print.rs # other console reporters
└── viewer/
    ├── package.json        # "sf-lens-viewer"
    ├── tsconfig.json       # strict + verbatimModuleSyntax + allowImportingTsExtensions
    ├── biome.json
    ├── vite.config.ts
    ├── vitest.config.ts
    ├── index.html
    ├── src/
    │   ├── main.ts
    │   ├── data/           # schema slice, JSON loader/validator
    │   ├── analysis/       # pure transforms — no DOM, no D3
    │   └── view/           # D3 rendering, encoding
    ├── tests/              # vitest, pure functions only
    └── data/facts.json     # cached extractor output (regenerable)
```

## Commands

```bash
# Extract facts (run from workspace root)
cargo run -p sf-lens -- extract --out tools/sf-lens/viewer/data/facts.json

# Viewer (run from tools/sf-lens/viewer/)
npm run dev          # vite dev server
npm run typecheck    # tsc --noEmit
npm run lint         # biome check (read-only)
npm run lint:fix     # biome check --write
npm test             # vitest
```

## Schema source of truth

`src/model.rs` is canonical. `viewer/src/data/schema.ts` is a hand-written slice of the fields the viewer actually consumes — keep them aligned by hand. Codegen (e.g., `ts-rs`) is on the table when the slice grows large enough to make hand-maintenance risky.
