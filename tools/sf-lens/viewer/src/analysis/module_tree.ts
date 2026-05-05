// Pure transform: a single crate's facts → a hierarchical tree of modules
// with types as leaves. No DOM, no D3, no view state — view-state machinery
// keys off the stable `id` field of each node.

import type { CrateFacts, FieldFacts, TypeKind } from '../data/schema.ts';

export type TreeNode = ModuleNode | TypeNode;

export interface ModuleNode {
  readonly kind: 'module';
  readonly id: string;
  readonly label: string;
  readonly path: string;
  readonly children: readonly TreeNode[];
}

export interface TypeNode {
  readonly kind: 'type';
  readonly id: string;
  readonly label: string;
  readonly typeKind: TypeKind;
  readonly fullPath: string;
  readonly modulePath: string;
  readonly fields: readonly FieldFacts[];
}

export interface BuildOptions {
  /** Drop any module whose path contains a `tests` segment. Default: true. */
  readonly excludeTests?: boolean;
}

export function buildModuleTree(crate: CrateFacts, options: BuildOptions = {}): ModuleNode {
  const excludeTests = options.excludeTests ?? true;

  const modules = Object.values(crate.modules).filter(
    (m) => !excludeTests || !hasTestsSegment(m.path),
  );

  type Scratch = {
    kind: 'module';
    id: string;
    label: string;
    path: string;
    children: TreeNode[];
  };

  const root: Scratch = {
    kind: 'module',
    id: idForModule(crate.name, ''),
    label: crate.name,
    path: '',
    children: [],
  };
  const byPath = new Map<string, Scratch>([['', root]]);

  // Build the prefix chain for every module path. Synthetic intermediates
  // (not in `crate.modules` themselves) default to folder-style labels since
  // their existence implies they hold submodules. Recursion is bounded by max
  // path depth, which is small in practice.
  const ensureChain = (path: string): Scratch => {
    const cached = byPath.get(path);
    if (cached) return cached;
    const segments = path.split('::');
    const parentPath = segments.slice(0, -1).join('::');
    const parent = ensureChain(parentPath);
    const lastSegment = segments[segments.length - 1] ?? path;
    const node: Scratch = {
      kind: 'module',
      id: idForModule(crate.name, path),
      label: lastSegment,
      path,
      children: [],
    };
    parent.children.push(node);
    byPath.set(path, node);
    return node;
  };

  // Labels are pure module names (last path segment). The crate root keeps
  // the bare crate name. We deliberately don't synthesize filesystem-style
  // labels (`name.rs`, etc.) — the pane is a Rust module hierarchy, not a
  // file tree, and rendering it as such avoids the "looks like files but
  // isn't" confusion. The renderer formats every row as `parent::path::leaf`
  // with the parent prefix dimmed/smaller to make module-ness explicit.
  for (const m of modules) {
    const node = m.path === '' ? root : ensureChain(m.path);
    for (const t of m.types) {
      node.children.push({
        kind: 'type',
        id: t.full_path,
        label: t.name,
        typeKind: t.kind,
        fullPath: t.full_path,
        modulePath: m.path,
        fields: t.fields,
      });
    }
  }

  for (const node of byPath.values()) {
    node.children.sort(compareTreeNodes);
  }

  return root as ModuleNode;
}

function hasTestsSegment(path: string): boolean {
  return path.split('::').includes('tests');
}

function idForModule(crateName: string, path: string): string {
  return path === '' ? crateName : `${crateName}::${path}`;
}

// Modules first (so structural levels stay above leaves at the same depth),
// then alphabetical by label.
function compareTreeNodes(a: TreeNode, b: TreeNode): number {
  if (a.kind !== b.kind) return a.kind === 'module' ? -1 : 1;
  return a.label.localeCompare(b.label);
}
