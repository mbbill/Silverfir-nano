import { describe, expect, it } from 'vitest';
import { computeDrift } from '../src/analysis/drift.ts';
import { FIELD_ROW_H, ROW_H, buildLayout, buildOptimizedLayout } from '../src/analysis/layout.ts';
import { buildModuleTree } from '../src/analysis/module_tree.ts';
import { buildOwnershipIndex, computeOwnershipDepth } from '../src/analysis/ownership.ts';
import type { CrateFacts, Edge, Facts, ModuleFacts, TypeFacts } from '../src/data/schema.ts';
import { ViewState } from '../src/state/view_state.ts';

function ty(
  crate: string,
  modPath: string,
  name: string,
  fields: { name: string; ty_text: string }[] = [],
): TypeFacts {
  const full = modPath === '' ? `${crate}::${name}` : `${crate}::${modPath}::${name}`;
  return {
    name,
    full_path: full,
    kind: 'struct',
    fields: fields.map((f) => ({ ...f, ownership: 'owned' as const })),
  };
}

function mod(path: string, types: TypeFacts[] = []): ModuleFacts {
  const file = path === '' ? 'src/lib.rs' : `src/${path.replace(/::/g, '/')}.rs`;
  return { path, types, file };
}

function crateFacts(name: string, modules: ModuleFacts[]): CrateFacts {
  return { name, modules: Object.fromEntries(modules.map((m) => [m.path, m])) };
}

function edge(from: string, to: string, origin = 'field x'): Edge {
  return { from, to, kind: 'owns', via: 'struct_field', origin };
}

function facts(crate: CrateFacts, edges: Edge[]): Facts {
  return { crates: { [crate.name]: crate }, edges };
}

function setup(crate: CrateFacts, edges: Edge[], expandedIds: string[]) {
  const f = facts(crate, edges);
  const root = buildModuleTree(crate);
  const ownership = buildOwnershipIndex(f, crate.name);
  const typeModule = collectTypeModule(root);
  const drift = computeDrift(ownership, typeModule);
  const depth = computeOwnershipDepth(ownership, collectIds(root), drift);
  const state = new ViewState(expandedIds);
  return buildLayout({ staticRoot: root, ownership, depth, state, drift });
}

function collectTypeModule(root: ReturnType<typeof buildModuleTree>): Map<string, string> {
  const out = new Map<string, string>();
  type N = { kind: string; fullPath?: string; modulePath?: string; children?: readonly N[] };
  const walk = (n: N): void => {
    if (n.kind === 'type' && n.fullPath !== undefined && n.modulePath !== undefined) {
      out.set(n.fullPath, n.modulePath);
    } else {
      for (const c of n.children ?? []) walk(c);
    }
  };
  walk(root as never);
  return out;
}

function collectIds(node: {
  kind: string;
  children?: readonly { kind: string }[];
  fullPath?: string;
}): string[] {
  const out: string[] = [];
  const walk = (n: {
    kind: string;
    fullPath?: string;
    children?: readonly { kind: string }[];
  }): void => {
    if (n.kind === 'type' && n.fullPath) out.push(n.fullPath);
    else for (const c of n.children ?? []) walk(c as never);
  };
  walk(node as never);
  return out;
}

describe('buildLayout', () => {
  it('collapsed state: only the crate root row, all bands height = 1 row', () => {
    const c = crateFacts('c', [mod(''), mod('a', [ty('c', 'a', 'X')])]);
    const layout = setup(c, [], []); // crate root NOT expanded
    expect(layout.modules).toHaveLength(1);
    expect(layout.modules[0]?.bandHeight).toBe(ROW_H);
    expect(layout.types).toHaveLength(0);
  });

  it('expanded crate root reveals child modules; their bands are 1 row when collapsed', () => {
    const c = crateFacts('c', [mod(''), mod('a', [ty('c', 'a', 'X')]), mod('b')]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id]);
    const labels = layout.modules.map((m) => m.label);
    expect(labels).toEqual(['c', 'a', 'b']);
    expect(layout.types).toHaveLength(0);
  });

  it('expanding a module renders its types in the band', () => {
    const c = crateFacts('c', [mod(''), mod('a', [ty('c', 'a', 'Foo'), ty('c', 'a', 'Bar')])]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id, 'c::a']);
    expect(layout.types.map((t) => t.label).sort()).toEqual(['Bar', 'Foo']);
  });

  it('chain ownership A→B→C in one module packs into a single row', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'A'), ty('c', 'a', 'B'), ty('c', 'a', 'C')]),
    ]);
    const root = buildModuleTree(c);
    const layout = setup(
      c,
      [edge('c::a::A', 'c::a::B'), edge('c::a::B', 'c::a::C')],
      [root.id, 'c::a'],
    );
    // All three types share a y → packed into 1 row → band height = 1 × ROW_H
    const aBand = layout.modules.find((m) => m.label === 'a');
    expect(aBand?.bandHeight).toBe(ROW_H);
    const ys = new Set(layout.types.map((t) => t.y));
    expect(ys.size).toBe(1);
    // x increases by depth: A < B < C
    const byName = new Map(layout.types.map((t) => [t.label, t.x]));
    expect(byName.get('A')).toBeLessThan(byName.get('B') as number);
    expect(byName.get('B')).toBeLessThan(byName.get('C') as number);
  });

  it('two unrelated roots in one module take two rows', () => {
    const c = crateFacts('c', [mod(''), mod('a', [ty('c', 'a', 'X'), ty('c', 'a', 'Y')])]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id, 'c::a']);
    const aBand = layout.modules.find((m) => m.label === 'a');
    expect(aBand?.bandHeight).toBe(2 * ROW_H);
  });

  it('expanded type takes (1 + fieldCount) rows in its band', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [
        ty('c', 'a', 'A', [
          { name: 'x', ty_text: 'i32' },
          { name: 'y', ty_text: 'i32' },
          { name: 'z', ty_text: 'i32' },
        ]),
      ]),
    ]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id, 'c::a', 'c::a::A']);
    const aBand = layout.modules.find((m) => m.label === 'a');
    // Expanded type: 1 header (ROW_H) + 3 field rows (FIELD_ROW_H each).
    expect(aBand?.bandHeight).toBe(ROW_H + 3 * FIELD_ROW_H);
    const aType = layout.types.find((t) => t.label === 'A');
    expect(aType?.expanded).toBe(true);
    expect(aType?.fields).toHaveLength(3);
  });

  it('arrows are emitted from expanded type fields to in-layout target types', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'A', [{ name: 'b', ty_text: 'B' }]), ty('c', 'a', 'B')]),
    ]);
    const root = buildModuleTree(c);
    const layout = setup(c, [edge('c::a::A', 'c::a::B', 'field b')], [root.id, 'c::a', 'c::a::A']);
    expect(layout.arrows).toHaveLength(1);
    expect(layout.arrows[0]?.fromTypeId).toBe('c::a::A');
    expect(layout.arrows[0]?.toTypeId).toBe('c::a::B');
  });

  it('no arrow when target type module is collapsed (target not in layout)', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'A', [{ name: 'b', ty_text: 'B' }])]),
      mod('b', [ty('c', 'b', 'B')]),
    ]);
    const root = buildModuleTree(c);
    // expand A but leave c::b collapsed
    const layout = setup(c, [edge('c::a::A', 'c::b::B', 'field b')], [root.id, 'c::a', 'c::a::A']);
    expect(layout.arrows).toHaveLength(0);
  });

  it('barycenter sweep reorders types within a band to bring partners close', () => {
    // Setup: T1, T2 in `a`; X1, X2 in `a::sub` so they're within_budget (LCA
    // of X's owner = `a`, X.modulePath = `a::sub`, depth-diff = 1, default
    // budget = 1 → canonical). Drift'd types would be skipped by the
    // barycenter sweep, which is correct but not what this test exercises.
    // T1 owns X2; T2 owns X1. Alphabetical order: T1,T2 / X1,X2 → 1 crossing.
    // After barycenter: T1@top→X2@top, T2@bottom→X1@bottom → zero crossings.
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'T1'), ty('c', 'a', 'T2')]),
      mod('a::sub', [ty('c', 'a::sub', 'X1'), ty('c', 'a::sub', 'X2')]),
    ]);
    const root = buildModuleTree(c);
    const f = facts(c, [
      edge('c::a::T1', 'c::a::sub::X2', 'field x'),
      edge('c::a::T2', 'c::a::sub::X1', 'field x'),
    ]);
    const ownership = buildOwnershipIndex(f, c.name);
    const tm = collectTypeModule(root);
    const drift = computeDrift(ownership, tm);
    const dep = computeOwnershipDepth(ownership, collectIds(root), drift);
    const state = new ViewState([root.id, 'c::a', 'c::a::sub']);

    const naive = buildLayout({ staticRoot: root, ownership, depth: dep, state, drift });
    const optimized = buildOptimizedLayout({
      staticRoot: root,
      ownership,
      depth: dep,
      state,
      drift,
    });

    const yOf = (l: typeof naive, label: string) => l.types.find((t) => t.label === label)?.y ?? 0;

    // Naive: alphabetical → X1 above X2
    expect(yOf(naive, 'X1')).toBeLessThan(yOf(naive, 'X2'));
    // Optimized: barycenter pulls X2 above X1 to align with T1
    expect(yOf(optimized, 'X2')).toBeLessThan(yOf(optimized, 'X1'));
  });

  it('totalWidth covers the rightmost type box and module label', () => {
    const c = crateFacts('c', [mod(''), mod('a', [ty('c', 'a', 'X')])]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id, 'c::a']);
    // Must be at least as wide as every type's right edge.
    for (const t of layout.types) {
      expect(layout.totalWidth).toBeGreaterThanOrEqual(t.x + t.width);
    }
    // Must be positive when there's any content.
    expect(layout.totalWidth).toBeGreaterThan(0);
  });

  it('totalWidth grows or stays flat when a type expands', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'WideType', [{ name: 'aLongFieldNameHere', ty_text: 'u32' }])]),
    ]);
    const root = buildModuleTree(c);
    const collapsed = setup(c, [], [root.id, 'c::a']);
    const expanded = setup(c, [], [root.id, 'c::a', 'c::a::WideType']);
    // Expanding a type can only add field-name width; it must not shrink
    // the overall horizontal extent.
    expect(expanded.totalWidth).toBeGreaterThanOrEqual(collapsed.totalWidth);
  });

  it('global x-start is consistent across bands', () => {
    const c = crateFacts('c', [
      mod(''),
      mod('a', [ty('c', 'a', 'X')]),
      mod('a::b', [ty('c', 'a::b', 'Y')]),
    ]);
    const root = buildModuleTree(c);
    const layout = setup(c, [], [root.id, 'c::a', 'c::a::b']);
    const xs = layout.types.map((t) => t.x);
    // Both X and Y are roots (depth 0) → same x.
    expect(new Set(xs).size).toBe(1);
  });
});
