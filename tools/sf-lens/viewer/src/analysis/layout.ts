// Composite layout: left-side module tree + per-module type bands on the
// right. Each module gets a row band whose height is determined by greedy
// 2-D packing of its types — a collapsed type takes one row, an expanded
// type takes 1 + fieldCount rows. Types that don't horizontally collide on
// the same row(s) share rows, so a chain `A→B→C` packs into one row.
//
// Type x = depth × COL_W past a global x-start (the rightmost module label
// end). Same global depth → same x across bands so owners stay left of
// owned types when LCA discipline holds.

import type { FieldFacts, Ownership, TypeKind } from '../data/schema.ts';
import type { ViewState } from '../state/view_state.ts';
import { type DriftClass, type DriftIndex, isCanonicalTarget } from './drift.ts';
import type { ModuleNode, TypeNode } from './module_tree.ts';
import type { OwnershipIndex } from './ownership.ts';

export const ROW_H = 26;
export const FIELD_ROW_H = 18;
export const INDENT_PX = 16;
export const LEFT_PAD = 8;
export const TOP_PAD = 8;

const COL_W = 200;
const TYPE_X_GAP = 16;
const MODULE_BAND_X_GAP = 24;
const CHAR_W = 7;
const TYPE_GLYPH_W = 32; // chevron room + circle + spacing before label
const MODULE_GLYPH_W = 18;
const FIELD_LABEL_INSET = 36;

export interface ModuleRow {
  readonly id: string;
  readonly label: string;
  readonly modDepth: number;
  readonly labelX: number;
  readonly y: number;
  readonly bandHeight: number;
  readonly expanded: boolean;
  readonly hasChildren: boolean;
}

export interface FieldRow {
  readonly name: string;
  readonly tyText: string;
  readonly ownership: Ownership;
  readonly x: number; // text x within the type box (start of name)
  readonly y: number; // absolute y (row center)
  /**
   * Arrow source x — end of the rendered field name plus a small gap. The
   * type-text portion overflows visually as semi-transparent grey but is not
   * counted toward the type box width or the arrow source.
   */
  readonly arrowSourceX: number;
  readonly targets: readonly string[]; // resolved owned target full_paths
}

export interface TypeBox {
  readonly id: string;
  readonly label: string;
  readonly typeKind: TypeKind;
  readonly fullPath: string;
  readonly modulePath: string;
  readonly x: number;
  readonly y: number; // row center of header
  readonly width: number;
  readonly height: number; // total rows × ROW_H (1 if collapsed)
  readonly hasFields: boolean;
  readonly expanded: boolean;
  readonly fields: readonly FieldRow[];
}

export interface ArrowWaypoint {
  readonly x: number;
  readonly y: number;
}

export interface Arrow {
  /**
   * Polyline waypoints for orthogonal (Manhattan) routing.
   * Always 4 points: [source, lane-entry, lane-exit, target]. Renderer draws
   * straight L segments between them. The marker on the final segment orients
   * along the horizontal entry tangent.
   */
  readonly waypoints: readonly ArrowWaypoint[];
  readonly fromTypeId: string;
  readonly fromFieldName: string;
  readonly toTypeId: string;
  readonly driftClass: DriftClass;
}

export interface Layout {
  readonly modules: readonly ModuleRow[];
  readonly types: readonly TypeBox[];
  readonly arrows: readonly Arrow[];
  readonly totalHeight: number;
  /** Rightmost data-space x coordinate used by the layout (across both the
   *  frozen module pane and the right type pane). Used by the zoom layer
   *  to compute a fit-to-view minimum scale. */
  readonly totalWidth: number;
}

export interface LayoutInputs {
  readonly staticRoot: ModuleNode;
  readonly ownership: OwnershipIndex;
  readonly depth: ReadonlyMap<string, number>;
  readonly state: ViewState;
  /**
   * Drift classification per type. Used to (a) filter non-canonical edges out
   * of the barycenter computation so drift'd types don't pull canonical types
   * around, and (b) tag each rendered arrow with its drift class for color.
   */
  readonly drift: DriftIndex;
  /**
   * Optional per-type ordering hint within each (band, depth) cell. Smaller
   * key → earlier in cell. Types without a key fall back to alphabetical.
   * Produced by `buildOptimizedLayout`'s barycenter sweeps.
   */
  readonly sortKey?: ReadonlyMap<string, number>;
  /**
   * Focus mode: when present, only modules whose id is in this set are
   * rendered — the rest of the tree is dropped entirely (no row, no name,
   * no children). Modules in the set are treated as effective-expanded by
   * the layout regardless of `state`. Caller must ensure the set is closed
   * under ancestors so the visible subtree stays connected.
   */
  readonly focusModules?: ReadonlySet<string>;
  /**
   * Optional precise text-width measurer for field names. Used to compute
   * `arrowSourceX` (the x where an arrow leaves a field) so the arrow's
   * tail starts exactly at the rendered text's right edge — no visible
   * gap from the proportional-font width mismatch. When omitted, falls
   * back to a flat `length * CHAR_W` approximation (fine for tests).
   */
  readonly measureText?: (text: string) => number;
}

export function buildLayout(inputs: LayoutInputs): Layout {
  const { staticRoot, ownership, depth, state, sortKey, drift, focusModules } = inputs;
  const measureText = inputs.measureText ?? ((s: string) => s.length * CHAR_W);

  const globalXStart = computeGlobalXStart(staticRoot);

  const modules: ModuleRow[] = [];
  const types: TypeBox[] = [];

  let cursorY = TOP_PAD;

  const visit = (m: ModuleNode, modDepth: number): void => {
    // Focus mode filter: drop any module whose id isn't in the focus set —
    // its row, its name, and its entire subtree are skipped.
    if (focusModules && !focusModules.has(m.id)) return;
    const labelX = LEFT_PAD + modDepth * INDENT_PX;
    // Module expansion is always driven by `state`, even in focus mode —
    // the caller is responsible for making sure relevance-set modules are
    // expanded in state before draw runs. This lets the user collapse a
    // module inside focus mode by clicking its row.
    const expanded = state.isExpanded(m.id);
    const directTypes = expanded ? (m.children.filter((c) => c.kind === 'type') as TypeNode[]) : [];

    const packed = packBand({
      types: directTypes,
      depth,
      globalXStart,
      ownership,
      state,
      bandTopY: cursorY,
      ...(sortKey !== undefined ? { sortKey } : {}),
    });
    const bandH = Math.max(ROW_H, packed.bandHeight);

    modules.push({
      id: m.id,
      label: m.label,
      modDepth,
      labelX,
      y: cursorY,
      bandHeight: bandH,
      expanded,
      hasChildren: m.children.length > 0,
    });

    for (const p of packed.boxes) {
      const headerY = cursorY + p.y + ROW_H / 2;
      const fieldRows: FieldRow[] = [];
      if (p.tExpanded) {
        for (let i = 0; i < p.t.fields.length; i++) {
          const f = p.t.fields[i] as FieldFacts;
          const nameStart = p.x + FIELD_LABEL_INSET;
          fieldRows.push({
            name: f.name,
            tyText: f.ty_text,
            ownership: f.ownership,
            x: nameStart,
            y: cursorY + p.y + ROW_H + (i + 0.5) * FIELD_ROW_H,
            arrowSourceX: nameStart + measureText(f.name) + 4,
            targets: ownership.fieldTargets.get(p.t.fullPath)?.get(f.name) ?? [],
          });
        }
      }
      types.push({
        id: p.t.id,
        label: p.t.label,
        typeKind: p.t.typeKind,
        fullPath: p.t.fullPath,
        modulePath: p.t.modulePath,
        x: p.x,
        y: headerY,
        width: p.width,
        height: p.pixelHeight,
        hasFields: p.t.fields.length > 0,
        expanded: p.tExpanded,
        fields: fieldRows,
      });
    }

    cursorY += bandH;

    if (expanded) {
      for (const c of m.children) {
        if (c.kind === 'module') visit(c, modDepth + 1);
      }
    }
  };

  visit(staticRoot, 0);

  const arrows = buildArrows(types, drift, depth);

  // Total horizontal extent: the rightmost edge across (a) the type pane
  // (t.x + t.width) and (b) the frozen module pane (estimated label end).
  // globalXStart is a lower bound — it's where the type pane begins.
  let totalWidth = globalXStart;
  for (const t of types) {
    const right = t.x + t.width;
    if (right > totalWidth) totalWidth = right;
  }
  for (const m of modules) {
    const right = m.labelX + MODULE_GLYPH_W + m.label.length * CHAR_W;
    if (right > totalWidth) totalWidth = right;
  }

  return { modules, types, arrows, totalHeight: cursorY + TOP_PAD, totalWidth };
}

interface PackedBox {
  readonly t: TypeNode;
  readonly tExpanded: boolean;
  readonly x: number;
  readonly y: number; // pixel offset from band top
  readonly width: number;
  readonly pixelHeight: number;
}

interface PlacedRect {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

function packBand(args: {
  readonly types: readonly TypeNode[];
  readonly depth: ReadonlyMap<string, number>;
  readonly globalXStart: number;
  readonly ownership: OwnershipIndex;
  readonly state: ViewState;
  /** Per-type barycenter (mean y of incoming-arrow source rows). When
   *  provided alongside `bandTopY`, the second pass uses it to place each
   *  type at the row nearest its desired y rather than top-of-band. */
  readonly sortKey?: ReadonlyMap<string, number>;
  /** Absolute y of the band's top edge in the layout. Needed to convert
   *  the absolute barycenter targets in `sortKey` into band-local y's. */
  readonly bandTopY?: number;
}): { boxes: readonly PackedBox[]; bandHeight: number } {
  const { types, depth, globalXStart, state, sortKey, bandTopY } = args;

  // Sort by depth, then sortKey (target y), then alphabetical. This drives
  // the column-y order in pass 2 — types with similar barycenters land
  // near each other.
  const sorted = [...types].sort((a, b) => {
    const da = depth.get(a.fullPath) ?? 0;
    const db = depth.get(b.fullPath) ?? 0;
    if (da !== db) return da - db;
    if (sortKey) {
      const ka = sortKey.get(a.fullPath);
      const kb = sortKey.get(b.fullPath);
      if (ka !== undefined && kb !== undefined && ka !== kb) return ka - kb;
      if (ka !== undefined && kb === undefined) return -1;
      if (kb !== undefined && ka === undefined) return 1;
    }
    return a.label.localeCompare(b.label);
  });

  // Per-type geometry, computed once.
  type Cell = { t: TypeNode; tExpanded: boolean; x: number; w: number; h: number };
  const cells: Cell[] = sorted.map((t) => {
    const d = depth.get(t.fullPath) ?? 0;
    const x = globalXStart + d * COL_W;
    const tExpanded = state.isExpanded(t.id);
    const w = computeTypeBoxWidth(t, tExpanded);
    const h = tExpanded ? ROW_H + t.fields.length * FIELD_ROW_H : ROW_H;
    return { t, tExpanded, x, w, h };
  });

  // Pass 1: greedy top-down packing to establish the band's natural height.
  // Used as a hard cap in pass 2 so types don't push the band taller just
  // to land near their barycenter.
  const placed1: PlacedRect[] = [];
  for (const c of cells) {
    const y = findFitY(placed1, c.x, c.w, c.h);
    placed1.push({ x: c.x, y, w: c.w, h: c.h });
  }
  let bandHeight = 0;
  for (const r of placed1) {
    if (r.y + r.h > bandHeight) bandHeight = r.y + r.h;
  }

  // Without barycenter info (first iteration of buildOptimizedLayout, or
  // direct buildLayout calls without it), pass 1 is the final placement.
  if (!sortKey || bandTopY === undefined) {
    const boxes = cells.map(
      (c, i): PackedBox => ({
        t: c.t,
        tExpanded: c.tExpanded,
        x: c.x,
        y: (placed1[i] as PlacedRect).y,
        pixelHeight: c.h,
        width: c.w,
      }),
    );
    return { boxes, bandHeight };
  }

  // Pass 2: place each type at the slot nearest its barycenter target,
  // clamped to the band height established by pass 1. Spreads "lonely"
  // types across spare rows instead of stacking them at the top.
  const placed2: PlacedRect[] = [];
  const boxes: PackedBox[] = [];
  for (const c of cells) {
    const abs = sortKey.get(c.t.fullPath);
    let targetY = 0;
    if (abs !== undefined) {
      // Center the box on the target y (target y is the row CENTER from
      // the previous pass, while box.y is the top edge).
      targetY = abs - bandTopY - c.h / 2;
    }
    targetY = Math.max(0, Math.min(bandHeight - c.h, targetY));
    const y = findFitYNearTarget(placed2, c.x, c.w, c.h, targetY, bandHeight);
    placed2.push({ x: c.x, y, w: c.w, h: c.h });
    boxes.push({ t: c.t, tExpanded: c.tExpanded, x: c.x, y, pixelHeight: c.h, width: c.w });
  }
  // Pass 2 might still grow the band slightly if all in-cap slots
  // conflict; recompute final bandHeight just in case.
  for (const r of placed2) {
    if (r.y + r.h > bandHeight) bandHeight = r.y + r.h;
  }
  return { boxes, bandHeight };
}

/** Find a y that fits the given rect closest to `targetY`. Candidates are:
 *  the target itself, top-of-band (y=0), and just-below / just-above each
 *  x-overlapping placed rect. If nothing fits within `cap`, falls back to
 *  the unbounded greedy `findFitY`. */
function findFitYNearTarget(
  placed: readonly PlacedRect[],
  x: number,
  w: number,
  h: number,
  targetY: number,
  cap: number,
): number {
  const candidates = new Set<number>();
  candidates.add(Math.max(0, targetY));
  candidates.add(0);
  for (const p of placed) {
    const xOverlap = !(p.x + p.w + TYPE_X_GAP <= x || p.x >= x + w + TYPE_X_GAP);
    if (!xOverlap) continue;
    candidates.add(p.y + p.h);
    if (p.y - h >= 0) candidates.add(p.y - h);
  }

  const fitting: number[] = [];
  for (const y of candidates) {
    if (y < 0 || y + h > cap) continue;
    let conflict = false;
    for (const p of placed) {
      const xOverlap = !(p.x + p.w + TYPE_X_GAP <= x || p.x >= x + w + TYPE_X_GAP);
      if (!xOverlap) continue;
      const yOverlap = !(p.y + p.h <= y || p.y >= y + h);
      if (yOverlap) {
        conflict = true;
        break;
      }
    }
    if (!conflict) fitting.push(y);
  }

  if (fitting.length === 0) return findFitY(placed, x, w, h);
  fitting.sort((a, b) => Math.abs(a - targetY) - Math.abs(b - targetY));
  return fitting[0] as number;
}

function findFitY(placed: readonly PlacedRect[], x: number, w: number, h: number): number {
  // Smallest y ≥ 0 such that the candidate rect doesn't overlap any placed
  // rect (with TYPE_X_GAP horizontal margin). Iterates by pushing y past each
  // conflicting rect; converges in O(N²) per insertion (fine at our scale).
  let y = 0;
  for (let safety = 0; safety < 1024; safety++) {
    let pushTo = y;
    let conflict = false;
    for (const p of placed) {
      const xOverlap = !(p.x + p.w + TYPE_X_GAP <= x || p.x >= x + w + TYPE_X_GAP);
      if (!xOverlap) continue;
      const yOverlap = !(p.y + p.h <= y || p.y >= y + h);
      if (!yOverlap) continue;
      conflict = true;
      if (p.y + p.h > pushTo) pushTo = p.y + p.h;
    }
    if (!conflict) return y;
    if (pushTo === y) return y; // safety: shouldn't happen
    y = pushTo;
  }
  return y;
}

function computeTypeBoxWidth(t: TypeNode, expanded: boolean): number {
  // Box width counts the header label and the longest field NAME only; the
  // ` : ty_text` suffix renders as a grey overflow and doesn't push the box
  // wider (so it doesn't shift downstream depth columns to the right).
  let w = TYPE_GLYPH_W + t.label.length * CHAR_W + 4;
  if (expanded) {
    for (const f of t.fields) {
      const rowW = FIELD_LABEL_INSET + f.name.length * CHAR_W + 4;
      if (rowW > w) w = rowW;
    }
  }
  return w;
}

interface RawArrow {
  readonly sourceTypeBox: TypeBox;
  readonly sourceX: number;
  readonly sourceY: number;
  readonly targetX: number;
  readonly targetY: number;
  readonly fromTypeId: string;
  readonly fromFieldName: string;
  readonly toTypeId: string;
  readonly driftClass: DriftClass;
  readonly isCanonical: boolean;
}

const LANE_BASE_GAP = 8;
const LANE_SLOT_W = 8;

function buildArrows(
  types: readonly TypeBox[],
  drift: DriftIndex,
  _depth: ReadonlyMap<string, number>,
): Arrow[] {
  const byFullPath = new Map(types.map((t) => [t.fullPath, t]));

  // Pass 1: collect raw arrow geometry without lane assignment.
  const raw: RawArrow[] = [];
  for (const t of types) {
    if (!t.expanded) continue;
    for (const f of t.fields) {
      for (const targetId of f.targets) {
        const target = byFullPath.get(targetId);
        if (!target) continue;
        if (target.fullPath === t.fullPath) continue;
        const driftClass = drift.typeClass.get(target.fullPath) ?? 'at_lca';
        const isCanonical = driftClass === 'at_lca' || driftClass === 'within_budget';
        const sourceX = isCanonical ? f.arrowSourceX : f.x - 4;
        raw.push({
          sourceTypeBox: t,
          sourceX,
          sourceY: f.y,
          targetX: target.x,
          targetY: target.y,
          fromTypeId: t.fullPath,
          fromFieldName: f.name,
          toTypeId: target.fullPath,
          driftClass,
          isCanonical,
        });
      }
    }
  }

  // Pass 2: split by direction. Canonical arrows (forward owner→owned) and
  // drift arrows (backward references) have disjoint lane regions, so they
  // can be colored independently — canonical lanes live BETWEEN source and
  // target columns; drift lanes live LEFT of the leftmost target.
  const canonical = raw.filter((r) => r.isCanonical);
  const driftArrows = raw.filter((r) => !r.isCanonical);

  // Pass 3: per-direction global lane assignment via interval-graph
  // coloring. The previous approach grouped by (sourceDepth, targetDepth,
  // direction) and distributed lanes evenly within each group — but arrows
  // from *different* (sd, td) channels could land at the same x with
  // overlapping y, stacking visually. Coloring against a global slot grid
  // guarantees that any two arrows with overlapping y-intervals end up in
  // distinct slots regardless of which channel they came from.
  const arrows: Arrow[] = [];
  for (const lane of assignLanes(canonical, true)) {
    arrows.push(makeArrow(lane.arrow, lane.laneX));
  }
  for (const lane of assignLanes(driftArrows, false)) {
    arrows.push(makeArrow(lane.arrow, lane.laneX));
  }
  return arrows;
}

function makeArrow(r: RawArrow, laneX: number): Arrow {
  return {
    waypoints: [
      { x: r.sourceX, y: r.sourceY },
      { x: laneX, y: r.sourceY },
      { x: laneX, y: r.targetY },
      { x: r.targetX, y: r.targetY },
    ],
    fromTypeId: r.fromTypeId,
    fromFieldName: r.fromFieldName,
    toTypeId: r.toTypeId,
    driftClass: r.driftClass,
  };
}

interface LaneAssignment {
  readonly arrow: RawArrow;
  readonly laneX: number;
}

function assignLanes(arrows: readonly RawArrow[], isCanonical: boolean): LaneAssignment[] {
  if (arrows.length === 0) return [];
  const N = arrows.length;

  // Compute the global x range that lanes can occupy across ALL arrows in
  // this direction. Canonical: from the leftmost source to the rightmost
  // target (each arrow's individual range is a sub-interval). Drift:
  // leftward of the leftmost target by a width that scales with N so even
  // in dense crates we have enough horizontal room for unique slots.
  let globalLeft: number;
  let globalRight: number;
  if (isCanonical) {
    globalLeft = Math.min(...arrows.map((a) => a.sourceX)) + LANE_BASE_GAP;
    globalRight = Math.max(...arrows.map((a) => a.targetX)) - LANE_BASE_GAP;
  } else {
    globalRight = Math.max(...arrows.map((a) => a.targetX)) - LANE_BASE_GAP;
    const driftWidth = Math.max(48, N * LANE_SLOT_W);
    globalLeft = Math.min(...arrows.map((a) => a.targetX)) - LANE_BASE_GAP - driftWidth;
  }
  if (globalRight < globalLeft) globalRight = globalLeft + 1;

  // Discretize into fixed-width slots. A slot is the unit of lane separation
  // — two arrows in the same slot share an x; in adjacent slots they're
  // LANE_SLOT_W apart. Slot count grows with the global range; we cap it so
  // pathological inputs don't allocate enormous arrays.
  const numSlots = Math.max(1, Math.min(2048, Math.ceil((globalRight - globalLeft) / LANE_SLOT_W)));
  const slotXs: number[] = [];
  for (let i = 0; i < numSlots; i++) slotXs.push(globalLeft + (i + 0.5) * LANE_SLOT_W);

  // Per-arrow constraints: which slots fall within this arrow's preferred
  // [chLeft, chRight] range. Canonical arrows are constrained on the left
  // (must clear their own source), drift arrows are constrained on the
  // right (must end before their target).
  interface Item {
    readonly arrow: RawArrow;
    readonly yMin: number;
    readonly yMax: number;
    readonly validSlots: number[];
  }
  const items: Item[] = arrows.map((a) => {
    const chLeft = isCanonical ? a.sourceX + LANE_BASE_GAP : globalLeft;
    const chRight = a.targetX - LANE_BASE_GAP;
    const midX = (chLeft + chRight) / 2;
    const valid: number[] = [];
    for (let i = 0; i < numSlots; i++) {
      const x = slotXs[i] as number;
      if (x >= chLeft && x <= chRight) valid.push(i);
    }
    if (valid.length === 0) {
      // Range collapsed (degenerate). Use the slot closest to the midpoint.
      let best = 0;
      for (let i = 1; i < numSlots; i++) {
        if (Math.abs((slotXs[i] as number) - midX) < Math.abs((slotXs[best] as number) - midX)) {
          best = i;
        }
      }
      valid.push(best);
    }
    // Sort valid slots by distance to the channel midpoint so the greedy
    // pass below prefers a lane near the middle of the source→target gap.
    // This keeps the arrow's vertical segment near the visual midpoint
    // instead of bunched right next to the source field.
    valid.sort(
      (i, j) => Math.abs((slotXs[i] as number) - midX) - Math.abs((slotXs[j] as number) - midX),
    );
    return {
      arrow: a,
      yMin: Math.min(a.sourceY, a.targetY),
      yMax: Math.max(a.sourceY, a.targetY),
      validSlots: valid,
    };
  });

  // Greedy interval coloring. Sort by yMin so we process top-to-bottom; for
  // each arrow, pick the lowest-index valid slot whose previously-placed
  // intervals don't overlap. If no conflict-free slot exists in the valid
  // range, fall back to the first valid one — degraded but still produces
  // a path. This case is rare in practice (only when an arrow's preferred
  // range is fully saturated by overlapping arrows).
  items.sort((p, q) => p.yMin - q.yMin || p.yMax - q.yMax);
  const slotIntervals: Array<Array<[number, number]>> = Array.from({ length: numSlots }, () => []);
  const out: LaneAssignment[] = [];
  for (const item of items) {
    let chosen = -1;
    for (const s of item.validSlots) {
      const intervals = slotIntervals[s] as Array<[number, number]>;
      let conflict = false;
      for (const [lo, hi] of intervals) {
        if (item.yMin < hi && lo < item.yMax) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        chosen = s;
        break;
      }
    }
    if (chosen === -1) chosen = item.validSlots[0] as number;
    (slotIntervals[chosen] as Array<[number, number]>).push([item.yMin, item.yMax]);
    out.push({ arrow: item.arrow, laneX: slotXs[chosen] as number });
  }
  return out;
}

function computeGlobalXStart(root: ModuleNode): number {
  let max = 0;
  const walk = (m: ModuleNode, modDepth: number): void => {
    const labelX = LEFT_PAD + modDepth * INDENT_PX;
    const labelEnd = labelX + estimateModuleLabelWidth(m.label);
    if (labelEnd > max) max = labelEnd;
    for (const c of m.children) {
      if (c.kind === 'module') walk(c, modDepth + 1);
    }
  };
  walk(root, 0);
  return max + MODULE_BAND_X_GAP;
}

function estimateModuleLabelWidth(label: string): number {
  return MODULE_GLYPH_W + label.length * CHAR_W;
}

/**
 * Build a layout, then run iterative barycenter sweeps to reorder types within
 * each (band, depth) cell so arrow crossings are reduced.
 *
 * Each pass: compute one sort-key per type from the mean y of its INCOMING
 * partners (its owners' field-source ys), rebuild the layout. After K passes
 * the ordering has propagated K layers downstream (depth 1 settles first,
 * then depth 2 picks up the new depth-1 positions, etc.).
 *
 * We use one direction (incoming) rather than alternating: the backward sweep
 * was using stale outgoing positions and pulling types back to alphabetical,
 * undoing the forward sweep.
 *
 * Stops when the y-signature stabilizes or `maxSweeps` iterations elapse.
 */
export function buildOptimizedLayout(inputs: LayoutInputs, maxSweeps = 8): Layout {
  let layout = buildLayout(inputs);
  let prevSig = ySignature(layout);
  for (let i = 0; i < maxSweeps; i++) {
    const sortKey = barycenterKeys(layout, inputs.ownership, 'incoming', inputs.drift);
    layout = buildLayout({ ...inputs, sortKey });
    const sig = ySignature(layout);
    if (sig === prevSig) break;
    prevSig = sig;
  }
  return layout;
}

function barycenterKeys(
  layout: Layout,
  ownership: OwnershipIndex,
  direction: 'incoming' | 'outgoing',
  drift: DriftIndex,
): Map<string, number> {
  // For incoming, use the y of the SOURCE field row (where the arrow actually
  // starts) inside each owner — not the owner's header. Drift'd targets (i.e.
  // non-canonical) get no incoming contribution: they fall back to current y
  // so anomalous edges don't pull them around.
  const typeByPath = new Map<string, Layout['types'][number]>();
  for (const t of layout.types) typeByPath.set(t.fullPath, t);

  const keys = new Map<string, number>();
  for (const t of layout.types) {
    const ys: number[] = [];
    const targetIsCanonical = isCanonicalTarget(t.fullPath, drift);
    if (direction === 'incoming' && targetIsCanonical) {
      for (const ownerId of ownership.ownedBy.get(t.fullPath) ?? []) {
        const owner = typeByPath.get(ownerId);
        if (!owner) continue;
        let pushedFieldY = false;
        if (owner.expanded) {
          for (const f of owner.fields) {
            if (f.targets.includes(t.fullPath)) {
              ys.push(f.y);
              pushedFieldY = true;
            }
          }
        }
        if (!pushedFieldY) ys.push(owner.y);
      }
    } else if (direction === 'outgoing') {
      for (const ownedId of ownership.owns.get(t.fullPath) ?? []) {
        if (!isCanonicalTarget(ownedId, drift)) continue;
        const owned = typeByPath.get(ownedId);
        if (owned) ys.push(owned.y);
      }
    }
    if (ys.length > 0) {
      keys.set(t.fullPath, ys.reduce((a, b) => a + b, 0) / ys.length);
    } else {
      keys.set(t.fullPath, t.y);
    }
  }
  return keys;
}

function ySignature(layout: Layout): string {
  // Stable identity of the current visual ordering — just per-type y values
  // serialized in a deterministic order.
  const parts: string[] = [];
  for (const t of layout.types) parts.push(`${t.fullPath}=${t.y}`);
  parts.sort();
  return parts.join('|');
}
