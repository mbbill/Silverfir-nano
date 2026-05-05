import type { TypeKind } from '../data/schema.ts';

const KIND_COLOR: Readonly<Record<TypeKind, string>> = {
  struct: '#3b82f6',
  enum: '#8b5cf6',
  union: '#06b6d4',
  trait: '#f59e0b',
  type_alias: '#9ca3af',
};

const FALLBACK = '#9ca3af';

export function colorForKind(kind: TypeKind): string {
  return KIND_COLOR[kind] ?? FALLBACK;
}
