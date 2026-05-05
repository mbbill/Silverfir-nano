import { describe, expect, it } from 'vitest';
import { type ModuleNode, type TreeNode, buildModuleTree } from '../src/analysis/module_tree.ts';
import type { CrateFacts, ModuleFacts, TypeFacts, TypeKind } from '../src/data/schema.ts';

function ty(crate: string, modPath: string, name: string, kind: TypeKind = 'struct'): TypeFacts {
  const full = modPath === '' ? `${crate}::${name}` : `${crate}::${modPath}::${name}`;
  return { name, full_path: full, kind, fields: [] };
}

function mod(path: string, types: TypeFacts[] = [], file?: string): ModuleFacts {
  return { path, types, file: file ?? defaultFile(path) };
}

function defaultFile(path: string): string {
  return path === '' ? 'src/lib.rs' : `src/${path.replace(/::/g, '/')}.rs`;
}

function crateOf(name: string, modules: ModuleFacts[]): CrateFacts {
  return {
    name,
    modules: Object.fromEntries(modules.map((m) => [m.path, m])),
  };
}

function findModule(node: TreeNode, path: string): ModuleNode | undefined {
  if (node.kind !== 'module') return undefined;
  if (node.path === path) return node;
  for (const c of node.children) {
    const hit = findModule(c, path);
    if (hit) return hit;
  }
  return undefined;
}

function childLabels(node: ModuleNode): string[] {
  return node.children.map((c) => c.label);
}

describe('buildModuleTree', () => {
  it('returns a single root node labeled with the crate name', () => {
    const root = buildModuleTree(crateOf('c', [mod('')]));
    expect(root.kind).toBe('module');
    expect(root.label).toBe('c');
    expect(root.path).toBe('');
  });

  it('attaches types under the module that owns them', () => {
    const root = buildModuleTree(
      crateOf('c', [
        mod(''),
        mod('a', [ty('c', 'a', 'Foo', 'struct'), ty('c', 'a', 'Bar', 'enum')]),
      ]),
    );
    const a = findModule(root, 'a');
    expect(a).toBeDefined();
    expect(childLabels(a as ModuleNode)).toEqual(['Bar', 'Foo']);
  });

  it('builds parent module nodes for nested paths', () => {
    const root = buildModuleTree(
      crateOf('c', [mod(''), mod('a::b::c', [ty('c', 'a::b::c', 'X')])]),
    );
    // Every module — synthetic intermediate or real — uses its bare last
    // path segment as the label. We deliberately do NOT synthesize file-
    // shaped labels (e.g. `c.rs`) — the pane is a Rust module hierarchy.
    expect(findModule(root, 'a')?.label).toBe('a');
    expect(findModule(root, 'a::b')?.label).toBe('b');
    const abc = findModule(root, 'a::b::c');
    expect(abc?.label).toBe('c');
    expect(childLabels(abc as ModuleNode)).toEqual(['X']);
  });

  it('module label is the last path segment regardless of file shape', () => {
    // All three of these resolve to the same bare-name label even though
    // the underlying source layout differs (mod.rs-backed, leaf .rs, leaf
    // .rs with companion submodules). File shape isn't part of the label.
    const root = buildModuleTree(
      crateOf('c', [
        mod(''),
        mod('modrs_backed', [], 'src/modrs_backed/mod.rs'),
        mod('leaf', [], 'src/leaf.rs'),
        mod('split', [], 'src/split.rs'),
        mod('split::sub', [], 'src/split/sub.rs'),
      ]),
    );
    expect(findModule(root, 'modrs_backed')?.label).toBe('modrs_backed');
    expect(findModule(root, 'leaf')?.label).toBe('leaf');
    expect(findModule(root, 'split')?.label).toBe('split');
    expect(findModule(root, 'split::sub')?.label).toBe('sub');
  });

  it('places submodules before type leaves at the same level', () => {
    const root = buildModuleTree(crateOf('c', [mod('', [ty('c', '', 'TypeAtRoot')]), mod('sub')]));
    expect(childLabels(root)).toEqual(['sub', 'TypeAtRoot']);
  });

  it('excludes test modules by default', () => {
    const root = buildModuleTree(
      crateOf('c', [
        mod(''),
        mod('a', [ty('c', 'a', 'Keep')]),
        mod('a::tests', [ty('c', 'a::tests', 'Drop')]),
        mod('tests', [ty('c', 'tests', 'AlsoDrop')]),
      ]),
    );
    expect(findModule(root, 'a::tests')).toBeUndefined();
    expect(findModule(root, 'tests')).toBeUndefined();
    expect(childLabels(findModule(root, 'a') as ModuleNode)).toEqual(['Keep']);
  });

  it('preserves test modules when excludeTests is false', () => {
    const root = buildModuleTree(crateOf('c', [mod(''), mod('tests', [ty('c', 'tests', 'T')])]), {
      excludeTests: false,
    });
    expect(findModule(root, 'tests')).toBeDefined();
  });

  it('issues stable IDs derived from the crate name', () => {
    const root = buildModuleTree(
      crateOf('crate-x', [mod(''), mod('a', [ty('crate-x', 'a', 'Foo')])]),
    );
    expect(root.id).toBe('crate-x');
    const a = findModule(root, 'a');
    expect(a?.id).toBe('crate-x::a');
    expect(a?.children[0]?.id).toBe('crate-x::a::Foo');
  });
});
