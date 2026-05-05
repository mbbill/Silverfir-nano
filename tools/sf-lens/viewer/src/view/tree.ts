// Two-area renderer: indented module tree on the left, per-module type bands
// on the right. Layout is precomputed in analysis/layout.ts; this module
// translates Layout objects into SVG and wires click handlers.
//
// Rendering uses a persistent DOM with d3 data-join and stable keys so that
// elements existing in both the previous and current render *tween* between
// their old and new positions instead of being wiped and rebuilt. This is
// what powers the smooth focus-mode toggle animation. Elements that appear
// fade in; elements that disappear fade out before removal.

import { type Selection, select } from 'd3';
import type { DriftClass } from '../analysis/drift.ts';
import { type Layout, ROW_H } from '../analysis/layout.ts';
import { colorForKind } from './encoding.ts';
import { ANIM_MS, type ZoomLayers } from './zoom.ts';

const TYPE_RADIUS = 4;
// Module rows still use a left chevron for expand/collapse.
const CHEVRON_X = 6;
// Type box layout: dot at x=6, label starts at x=14, expand arrow trails the
// name on the right (positioned dynamically after the name is measured).
const TYPE_CIRCLE_X = 6;
const TYPE_LABEL_X = 14;
const TYPE_ARROW_GAP = 6;
const TYPE_ARROW_HIT_PAD = 4;
const MODULE_LABEL_X = 18;
const HIT_PAD_RIGHT = 8;
const HIT_MIN_W = 40;

// Exported so other modules (e.g. the canvas-backed text measurer) can
// match the rendered font exactly. Keep these in sync with the SVG.
export const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
const FONT_SIZE = 12;
export const FONT_SIZE_FIELD = 12;
const FONT_SIZE_TYPE_ARROW = 22;
// Module leaf and type header bumped slightly above the base so the
// "main thing on this row" reads more prominently than ancillary text.
// Stays within the existing band height (ROW_H = 26) — at 14px the
// cap-height plus descender comfortably fits.
const FONT_SIZE_MODULE_LEAF = 14;
const FONT_SIZE_TYPE_LABEL = 14;
const FONT_SIZE_MODULE_PREFIX = 11; // smaller than the leaf, to keep the row tight
const FONT_SIZE_MODULE_CHEVRON = 14; // bumped above the base + bold so the
// "+/-" expand affordance reads clearly without changing direction-neutral
// semantics (modules expand both vertically and horizontally).

const COLOR_LABEL = '#1e293b';
const COLOR_MODULE_PREFIX = '#60a5fa'; // blue-400, dimmed parent path
const COLOR_CHEVRON = '#94a3b8';
const COLOR_FIELD_NAME = '#334155';
const COLOR_FIELD_TY = '#94a3b8'; // slate-400, grey for the on-hover type hint
const TY_HIDE_DELAY = 2000; // ms — type-hint persists this long after mouse-out
const COLOR_FOCUS_BG = '#bfdbfe'; // blue-200, soft fill behind selected fields
const FOCUS_BG_RADIUS = 5;
const FOCUS_BG_PAD_X = 4;
const FOCUS_BG_PAD_Y_FIELD = 3;
const FOCUS_FILTER_ID = 'sf-feather';
const EDGE_SHADOW_GRADIENT_ID = 'sf-edge-shadow';
const EDGE_SHADOW_W = 16; // data-units; scales with zoom (small but visible)
const COLOR_ARROW_CANONICAL = '#94a3b8'; // slate-400: at_lca / within_budget — neutral context
// Highlighted-canonical color (#3b82f6 blue) is applied via CSS in
// index.html (`.canonical.highlighted { stroke: ... }`) so the marker
// arrowhead can pick it up via context-stroke without per-state JS.
const COLOR_ARROW_SOFT = '#f59e0b'; // amber: drift_below
const COLOR_ARROW_HARD = '#ef4444'; // red:   drift_above / drift_sideways

const ARROW_MARKER_IDS: Readonly<Record<DriftClass, string>> = {
  at_lca: 'sf-arrow-canonical',
  within_budget: 'sf-arrow-canonical',
  drift_below: 'sf-arrow-soft',
  drift_above: 'sf-arrow-hard',
  drift_sideways: 'sf-arrow-hard',
};

function arrowColor(c: DriftClass): string {
  if (c === 'at_lca' || c === 'within_budget') return COLOR_ARROW_CANONICAL;
  if (c === 'drift_below') return COLOR_ARROW_SOFT;
  return COLOR_ARROW_HARD;
}

export interface TreeRenderOptions {
  /** Single-row click on a module or type → toggle expansion. Expansion is
   *  the only "focus" concept — there is no separate selected-types set. */
  readonly onToggle: (id: string) => void;
  /** Click on a field name → toggle its selection. */
  readonly onSelectField: (typePath: string, fieldName: string) => void;
  /** Set of "typePath::fieldName" keys currently selected. */
  readonly selectedFields: ReadonlySet<string>;
  /** Arrows in any selected field's chain — drawn highlighted by default. */
  readonly selectedArrows: ReadonlySet<Layout['arrows'][number]>;
  /** Hover on a type's dot → show the incoming-ownership popover. The
   *  callback receives the type's full path and a `getDotScreenPos`
   *  closure that returns the dot's current screen-space center. The
   *  overlay calls it on each pan/zoom while pinned, so the panel stays
   *  anchored to the moving dot. `onHideOwners` is fired when the cursor
   *  leaves the dot — the overlay handles the post-leave grace period. */
  readonly onShowOwners: (
    typePath: string,
    getDotScreenPos: () => { x: number; y: number },
  ) => void;
  readonly onHideOwners: () => void;
  /** Click on a type's dot → expand every type that owns it (and the
   *  modules containing those owners), so all incoming arrows render. */
  readonly onExpandAllOwners: (typePath: string) => void;
}

export function fieldKey(typePath: string, fieldName: string): string {
  return `${typePath}::${fieldName}`;
}

export function chainArrowsFromMany(
  layout: Layout,
  fields: ReadonlyArray<{ typePath: string; fieldName: string }>,
): Set<Layout['arrows'][number]> {
  const out = new Set<Layout['arrows'][number]>();
  for (const f of fields) {
    const c = chainArrowsFrom(layout, f.typePath, f.fieldName);
    for (const a of c) out.add(a);
  }
  return out;
}

export function renderTree(layers: ZoomLayers, layout: Layout, opts: TreeRenderOptions): void {
  const zoomLayer = select(layers.zoomLayer);
  const frozenLayer = select(layers.frozenLayer);

  zoomLayer.attr('font-family', FONT_FAMILY);
  frozenLayer.attr('font-family', FONT_FAMILY);
  ensureArrowMarker(zoomLayer);

  // Persistent parent groups — ensured once, then re-used across renders so
  // children with stable keys can tween rather than be wiped + rebuilt.
  const bandG = ensureGroup(zoomLayer, 'band-bg');
  const arrowG = ensureGroup(zoomLayer, 'arrows');
  arrowG.attr('fill', 'none').attr('stroke-width', 1);
  const typeG = ensureGroup(zoomLayer, 'types');
  const frozenBandG = ensureGroup(frozenLayer, 'frozen-band-bg');
  const moduleG = ensureGroup(frozenLayer, 'modules');

  renderBandBackgrounds(bandG, layout);
  renderFrozenBandBackgrounds(frozenBandG, layout);
  renderArrows(arrowG, layout, opts.selectedArrows);
  renderTypes(typeG, zoomLayer, layout, opts);
  renderModules(moduleG, layout, opts);
  sizeFrozenBackdrop(layers.backdrop, frozenLayer, layout);
  renderEdgeShadowsImpl(frozenLayer, layout);
}

/**
 * Per-band edge shadow on the frozen pane's right edge. Shows up only on
 * bands whose types have any portion hidden behind the frozen module
 * column. Caller invokes this from both `renderTree` (after layout) and
 * from the zoom callback (after each pan/zoom). Cheap: O(types) per call.
 */
export function renderEdgeShadows(
  layers: ZoomLayers,
  layout: Layout,
  transform: { x: number; k: number },
): void {
  renderEdgeShadowsImpl(select(layers.frozenLayer), layout, transform);
}

function renderEdgeShadowsImpl(
  frozenLayer: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
  transform?: { x: number; k: number },
): void {
  // Read the frozen pane's right-edge data-x from the separator line
  // (sizeFrozenBackdrop has set it just before this call). If no
  // separator exists yet (very first draw before sizeFrozenBackdrop ran)
  // bail — there are no shadows to render meaningfully.
  const sepLine = frozenLayer.select<SVGLineElement>('line.frozen-separator');
  if (sepLine.empty()) return;
  const rightEdge = Number(sepLine.attr('x1'));
  if (!Number.isFinite(rightEdge)) return;

  // If transform isn't supplied (called from renderTree's first pass
  // before zoom has fired), fall back to identity. The zoom callback
  // will refresh shadows shortly after with the real transform.
  const tx = transform?.x ?? 0;
  const tk = transform?.k ?? 1;
  if (tk <= 0) return;

  // Data-x at which the frozen pane's right edge sits (back-projected
  // from the screen). Anything to the LEFT of this is hidden.
  // Frozen layer transform is `translate(0, t.y) scale(t.k)` so the
  // frozen pane's right edge in screen space is at `rightEdge * tk`.
  // For the zoom layer (transform `translate(t.x, t.y) scale(t.k)`), a
  // type at data-x px renders at screen-x `px * tk + tx`. A type is fully
  // hidden if its right edge `(px + w) * tk + tx < rightEdge * tk`, which
  // simplifies to `(px + w) < rightEdge - tx/tk`.
  const viewLeftDataX = rightEdge - tx / tk;

  // Per-module: minimum type-right-edge. If the leftmost-finishing type
  // is still right of viewLeftDataX, no type is hidden in this band.
  const minRightByPath = new Map<string, number>();
  for (const t of layout.types) {
    const right = t.x + t.width;
    const cur = minRightByPath.get(t.modulePath);
    if (cur === undefined || right < cur) minRightByPath.set(t.modulePath, right);
  }

  // Module rows have id = `crate::path` (or just `crate` for the root);
  // types' modulePath = `path` (no crate prefix). Strip the crate prefix
  // off of the row id to join the two.
  const cratePrefixIdx = (id: string): string => {
    const idx = id.indexOf('::');
    return idx >= 0 ? id.slice(idx + 2) : '';
  };

  // d3 join keyed on the module id. Width is constant in data-units
  // (scales with zoom — fine, the cue stays proportional). Visibility
  // is toggled via `visibility` rather than DOM add/remove so transitions
  // don't fire on every zoom event.
  const shadowG = ensureGroup(frozenLayer, 'edge-shadows');
  const sel = shadowG
    .selectAll<SVGRectElement, Layout['modules'][number]>('rect.edge-shadow')
    .data(layout.modules, (m) => m.id);

  sel.exit().remove();

  const enter = sel
    .enter()
    .append('rect')
    .attr('class', 'edge-shadow')
    .attr('x', rightEdge)
    .attr('width', EDGE_SHADOW_W)
    .attr('fill', `url(#${EDGE_SHADOW_GRADIENT_ID})`)
    .attr('pointer-events', 'none');

  const merged = enter.merge(sel);
  merged
    .attr('x', rightEdge)
    .attr('width', EDGE_SHADOW_W)
    .attr('y', (m) => m.y)
    .attr('height', (m) => m.bandHeight)
    .style('visibility', (m) => {
      const path = cratePrefixIdx(m.id);
      const minRight = minRightByPath.get(path);
      if (minRight === undefined) return 'hidden';
      return minRight < viewLeftDataX ? 'visible' : 'hidden';
    });
}

function ensureGroup(
  parent: Selection<SVGGElement, unknown, null, undefined>,
  className: string,
): Selection<SVGGElement, unknown, null, undefined> {
  let g = parent.select<SVGGElement>(`g.${className}`);
  if (g.empty()) g = parent.append('g').attr('class', className);
  return g;
}

function renderFrozenBandBackgrounds(
  g: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
): void {
  // Mirror the zoom-layer alternating tint inside the frozen module column so
  // bands flow visually uninterrupted across the separator. Width is a
  // placeholder here; sizeFrozenBackdrop later trims it to the backdrop's
  // right edge.
  const tinted = layout.modules.filter((_m, i) => i % 2 === 1);
  const sel = g
    .selectAll<SVGRectElement, Layout['modules'][number]>('rect')
    .data(tinted, (m) => m.id);
  sel.exit().transition('exit').duration(ANIM_MS).style('opacity', 0).remove();
  const enter = sel
    .enter()
    .append('rect')
    .attr('x', -10000)
    .attr('y', (m) => m.y)
    .attr('width', 20000)
    .attr('height', (m) => m.bandHeight)
    .attr('fill', '#f1f5f9')
    .style('opacity', 0);
  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);
  sel
    .transition('move')
    .duration(ANIM_MS)
    .attr('y', (m) => m.y)
    .attr('height', (m) => m.bandHeight);
}

function renderBandBackgrounds(
  g: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
): void {
  // Subtle alternating tint per module band so the user can trace a horizontal
  // lane from any type back to its module label on the left frozen column.
  // Drawn first (so types and arrows render on top) and stretched far past
  // the visible viewport so panning never reveals an unfilled edge.
  const tinted = layout.modules.filter((_m, i) => i % 2 === 1);
  const sel = g
    .selectAll<SVGRectElement, Layout['modules'][number]>('rect')
    .data(tinted, (m) => m.id);
  sel.exit().transition('exit').duration(ANIM_MS).style('opacity', 0).remove();
  const enter = sel
    .enter()
    .append('rect')
    .attr('x', -10000)
    .attr('y', (m) => m.y)
    .attr('width', 20000)
    .attr('height', (m) => m.bandHeight)
    .attr('fill', '#f1f5f9')
    .style('opacity', 0);
  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);
  sel
    .transition('move')
    .duration(ANIM_MS)
    .attr('y', (m) => m.y)
    .attr('height', (m) => m.bandHeight);
}

function sizeFrozenBackdrop(
  backdrop: SVGRectElement,
  frozen: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
): void {
  // Compute the rightmost label edge per row by adding the row's translate-x
  // to the text/rect bbox.right inside that row group. Plain getBBox() on
  // descendants gives row-local coords; we have to add the row's transform-x.
  let maxX = 0;
  frozen.selectAll<SVGGElement, unknown>('g.module-row').each(function () {
    const baseVal = this.transform.baseVal;
    const tx = baseVal.length > 0 ? (baseVal.getItem(0).matrix.e ?? 0) : 0;
    const bbox = this.getBBox();
    const right = tx + bbox.x + bbox.width;
    if (right > maxX) maxX = right;
  });
  // Belt-and-suspenders: if the DOM measurement gave nothing (e.g. no rows),
  // fall back to a layout-derived estimate.
  if (maxX === 0) {
    for (const m of layout.modules) {
      const est = m.labelX + 24 + m.label.length * 7;
      if (est > maxX) maxX = est;
    }
  }
  const rightEdge = Math.max(maxX + 12, 80);
  backdrop.setAttribute('width', String(rightEdge + 10000));

  // Trim the alternating-tint rects to the same right edge so they end at
  // the separator line and don't bleed into the type area.
  frozen
    .selectAll<SVGRectElement, unknown>('g.frozen-band-bg rect')
    .attr('width', rightEdge + 10000);

  // Separator line at the right edge of the frozen pane. `non-scaling-stroke`
  // keeps the line 1px regardless of zoom level. Persistent: created once,
  // updated in place so tweens keep working across draws.
  let sep = frozen.select<SVGLineElement>('line.frozen-separator');
  if (sep.empty()) {
    sep = frozen
      .append('line')
      .attr('class', 'frozen-separator')
      .attr('y1', -10000)
      .attr('y2', 10000)
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', 1)
      .attr('vector-effect', 'non-scaling-stroke');
  }
  sep.attr('x1', rightEdge).attr('x2', rightEdge);
}

function ensureArrowMarker(layer: Selection<SVGGElement, unknown, null, undefined>): void {
  // Idempotent: skip if we've already set up <defs>. Markers and the focus
  // filter live there.
  if (!layer.select('defs').empty()) return;

  const defs = layer.append('defs');
  const define = (id: string): void => {
    // `context-stroke` makes the arrowhead's fill follow the path's
    // current stroke color. So when a canonical path's stroke is overridden
    // (grey by default → blue when .highlighted), the arrowhead changes
    // colour with it — no need to swap marker-end via JS.
    defs
      .append('marker')
      .attr('id', id)
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 7)
      .attr('refY', 0)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-4L8,0L0,4Z')
      .attr('fill', 'context-stroke');
  };
  define('sf-arrow-canonical');
  define('sf-arrow-soft');
  define('sf-arrow-hard');

  // Soft-edge filter applied to focus-background rects so the pill fades
  // smoothly into the band tint instead of having a hard rectangular edge.
  // The filter region is expanded so the Gaussian blur isn't clipped at the
  // rect's bounds.
  const filter = defs
    .append('filter')
    .attr('id', FOCUS_FILTER_ID)
    .attr('x', '-10%')
    .attr('y', '-40%')
    .attr('width', '120%')
    .attr('height', '180%');
  filter.append('feGaussianBlur').attr('stdDeviation', '1.2');

  // Edge-shadow gradient — used by per-band shadows on the frozen pane's
  // right edge to signal "this band has type content currently hidden
  // behind the column."
  const grad = defs
    .append('linearGradient')
    .attr('id', EDGE_SHADOW_GRADIENT_ID)
    .attr('x1', '0')
    .attr('y1', '0')
    .attr('x2', '1')
    .attr('y2', '0');
  grad.append('stop').attr('offset', '0').attr('stop-color', 'rgba(15,23,42,0.22)');
  grad.append('stop').attr('offset', '1').attr('stop-color', 'rgba(15,23,42,0)');
}

function arrowKey(a: Layout['arrows'][number]): string {
  // Identifies an arrow by endpoints + waypoint y-coords (capturing the
  // routing). Different routing under focus-toggle ⇒ different key, so the
  // old arrow exits (fade-out) and the new one enters (fade-in) — a true
  // path-tween would be nice but is brittle when path command counts differ.
  const ys = a.waypoints.map((w) => `${w.x},${w.y}`).join('|');
  return `${a.fromTypeId}::${a.fromFieldName}::${a.toTypeId}::${ys}`;
}

function renderArrows(
  g: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
  selectedArrows: ReadonlySet<Layout['arrows'][number]>,
): void {
  const sel = g
    .selectAll<SVGPathElement, Layout['arrows'][number]>('path')
    .data(layout.arrows, arrowKey);

  sel
    .exit()
    .classed('highlighted', false)
    .transition('exit')
    .duration(ANIM_MS)
    .style('opacity', 0)
    .remove();

  const enter = sel
    .enter()
    .append('path')
    .attr('d', (a) => polylinePath(a.waypoints))
    .attr('stroke', (a) => arrowColor(a.driftClass))
    .attr('marker-end', (a) => `url(#${ARROW_MARKER_IDS[a.driftClass]})`)
    .style('opacity', 0)
    .classed('canonical', (a) => a.driftClass === 'at_lca' || a.driftClass === 'within_budget')
    .classed('highlighted', (a) => selectedArrows.has(a));

  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);

  // Update existing arrows: only highlight class can change without their
  // identity changing (since waypoints are part of the key). Apply outside
  // the transition so it takes effect immediately.
  sel.classed('highlighted', (a) => selectedArrows.has(a));
}

function chainArrowsFrom(
  layout: Layout,
  fromTypeId: string,
  fieldName: string,
): Set<Layout['arrows'][number]> {
  const typesByPath = new Map(layout.types.map((t) => [t.fullPath, t]));
  const arrowsByFrom = new Map<string, Layout['arrows'][number][]>();
  for (const a of layout.arrows) {
    let list = arrowsByFrom.get(a.fromTypeId);
    if (!list) {
      list = [];
      arrowsByFrom.set(a.fromTypeId, list);
    }
    list.push(a);
  }

  const inChain = new Set<Layout['arrows'][number]>();
  const visitedTypes = new Set<string>();
  const queue: string[] = [];

  // Seed: arrows from the hovered (type, field).
  for (const a of arrowsByFrom.get(fromTypeId) ?? []) {
    if (a.fromFieldName !== fieldName) continue;
    inChain.add(a);
    const tgt = typesByPath.get(a.toTypeId);
    if (tgt?.expanded && !visitedTypes.has(a.toTypeId)) {
      visitedTypes.add(a.toTypeId);
      queue.push(a.toTypeId);
    }
  }

  // Walk the chain through any expanded targets.
  while (queue.length > 0) {
    const tid = queue.shift();
    if (tid === undefined) break;
    for (const a of arrowsByFrom.get(tid) ?? []) {
      if (inChain.has(a)) continue;
      inChain.add(a);
      const tgt = typesByPath.get(a.toTypeId);
      if (tgt?.expanded && !visitedTypes.has(a.toTypeId)) {
        visitedTypes.add(a.toTypeId);
        queue.push(a.toTypeId);
      }
    }
  }
  return inChain;
}

function applyChainHighlight(
  layer: Selection<SVGGElement, unknown, null, undefined>,
  inChain: ReadonlySet<Layout['arrows'][number]>,
): void {
  layer
    .selectAll<SVGPathElement, Layout['arrows'][number]>('g.arrows path')
    .classed('highlighted', (d) => inChain.has(d));
}

const CORNER_OFFSET = 8;

function polylinePath(waypoints: readonly { x: number; y: number }[]): string {
  if (waypoints.length < 2) return '';
  const head = waypoints[0];
  const tail = waypoints[waypoints.length - 1];
  if (!head || !tail) return '';

  // Round each interior corner with a quadratic bezier: trim back from the
  // corner along each adjacent segment by CORNER_OFFSET (or half-segment if
  // the segment is too short), then use the corner itself as the Q control
  // point. This smooths the bend without specifying an explicit radius.
  let d = `M${head.x},${head.y}`;
  for (let i = 1; i < waypoints.length - 1; i++) {
    const prev = waypoints[i - 1];
    const cur = waypoints[i];
    const next = waypoints[i + 1];
    if (!prev || !cur || !next) continue;

    const inLen = Math.hypot(cur.x - prev.x, cur.y - prev.y);
    const outLen = Math.hypot(next.x - cur.x, next.y - cur.y);
    const inOff = Math.min(CORNER_OFFSET, inLen / 2);
    const outOff = Math.min(CORNER_OFFSET, outLen / 2);

    if (inLen === 0 || outLen === 0) continue;
    const inUx = (cur.x - prev.x) / inLen;
    const inUy = (cur.y - prev.y) / inLen;
    const outUx = (next.x - cur.x) / outLen;
    const outUy = (next.y - cur.y) / outLen;

    const ax = cur.x - inUx * inOff;
    const ay = cur.y - inUy * inOff;
    const ex = cur.x + outUx * outOff;
    const ey = cur.y + outUy * outOff;

    d += `L${ax},${ay}Q${cur.x},${cur.y} ${ex},${ey}`;
  }
  d += `L${tail.x},${tail.y}`;
  return d;
}

function renderModules(
  g: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
  opts: TreeRenderOptions,
): void {
  const sel = g
    .selectAll<SVGGElement, Layout['modules'][number]>('g.module-row')
    .data(layout.modules, (d) => d.id);

  sel.exit().transition('exit').duration(ANIM_MS).style('opacity', 0).remove();

  const enter = sel
    .enter()
    .append('g')
    .attr('class', 'module-row')
    .attr('transform', (d) => `translate(${d.labelX},${d.y})`)
    .style('opacity', 0);

  enter
    .filter((d) => d.hasChildren)
    .append('text')
    .attr('class', 'chevron')
    .attr('x', CHEVRON_X)
    .attr('y', ROW_H / 2)
    .attr('dy', '0.32em')
    .attr('text-anchor', 'middle')
    .attr('font-size', FONT_SIZE_MODULE_CHEVRON)
    .attr('font-weight', 600)
    .attr('fill', COLOR_CHEVRON);

  // Module label is split into a dimmed/smaller "prefix" tspan (the
  // parent module path, e.g. "vm::wasm::") and a normal "leaf" tspan
  // (the module's own name). The prefix on every row makes it visually
  // unambiguous that this pane is a Rust module hierarchy — no file/dir
  // pretense — while staying scannable by leaf.
  const moduleText = enter
    .append('text')
    .attr('class', 'name')
    .attr('x', MODULE_LABEL_X)
    .attr('y', ROW_H / 2)
    .attr('dy', '0.32em')
    .attr('font-size', FONT_SIZE)
    .attr('fill', COLOR_LABEL);
  moduleText
    .append('tspan')
    .attr('class', 'prefix')
    .attr('font-size', FONT_SIZE_MODULE_PREFIX)
    .attr('fill', COLOR_MODULE_PREFIX);
  // The crate-root row gets a bolder leaf so the crate name stands out
  // as the top of the hierarchy. Submodules use the default weight.
  moduleText
    .append('tspan')
    .attr('class', 'leaf')
    .attr('font-size', FONT_SIZE_MODULE_LEAF)
    .attr('font-weight', (d) => (d.modDepth === 0 ? 700 : 400));

  enter
    .append('rect')
    .attr('class', 'expand-hit')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', HIT_MIN_W)
    .attr('height', ROW_H)
    .attr('fill', 'transparent');

  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);

  const merged = enter.merge(sel);

  // Tween position to new (labelX, y).
  merged
    .transition('move')
    .duration(ANIM_MS)
    .attr('transform', (d) => `translate(${d.labelX},${d.y})`);

  // Update chevron text (expansion state may have changed).
  merged
    .filter((d) => d.hasChildren)
    .select<SVGTextElement>('text.chevron')
    .text((d) => (d.expanded ? '-' : '+'));

  // Refresh the module label tspans each draw — content may shift when
  // crates are switched or filters are applied (focus mode collapses
  // some intermediate modules out of the visible tree).
  merged
    .select<SVGTSpanElement>('text.name tspan.prefix')
    .text((d) => splitModuleLabel(d.id).prefix);
  merged.select<SVGTSpanElement>('text.name tspan.leaf').text((d) => splitModuleLabel(d.id).leaf);

  // Refresh click handler with current closure each draw.
  merged
    .select<SVGRectElement>('rect.expand-hit')
    .attr('cursor', (d) => (d.hasChildren ? 'pointer' : 'default'))
    .on('click', (event: MouseEvent, d) => {
      event.stopPropagation();
      if (d.hasChildren) opts.onToggle(d.id);
    });

  sizeModuleExpandHit(merged);
}

// Split a module's full id (`crate::a::b::leaf`) into a parent prefix
// rendered dimmed and the leaf rendered normally. The crate root and
// top-level modules under the crate get an empty prefix — there's nothing
// useful to show above them.
function splitModuleLabel(id: string): { prefix: string; leaf: string } {
  const segs = id.split('::');
  const leaf = segs[segs.length - 1] ?? id;
  if (segs.length <= 2) return { prefix: '', leaf };
  return { prefix: `${segs.slice(1, -1).join('::')}::`, leaf };
}

function sizeModuleExpandHit<D>(sel: Selection<SVGGElement, D, SVGGElement, unknown>): void {
  // Width the expand-hit to cover chevron + gap + name + small trailing pad.
  sel.each(function () {
    const gg = select(this);
    const node = gg.select<SVGGraphicsElement>('text.name').node();
    const nameW = node ? node.getBBox().width : 0;
    const w = Math.max(MODULE_LABEL_X + nameW + HIT_PAD_RIGHT, HIT_MIN_W);
    gg.select<SVGRectElement>('rect.expand-hit').attr('width', w);
  });
}

function renderTypes(
  typeG: Selection<SVGGElement, unknown, null, undefined>,
  zoomLayer: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
  opts: TreeRenderOptions,
): void {
  const sel = typeG
    .selectAll<SVGGElement, Layout['types'][number]>('g.type-box')
    .data(layout.types, (d) => `${d.modulePath}::${d.id}`);

  sel.exit().transition('exit').duration(ANIM_MS).style('opacity', 0).remove();

  const enter = sel
    .enter()
    .append('g')
    .attr('class', 'type-box')
    .attr('transform', (d) => `translate(${d.x},${d.y - ROW_H / 2})`)
    .style('opacity', 0);

  enter
    .append('text')
    .attr('class', 'header-label name')
    .attr('x', TYPE_LABEL_X)
    .attr('y', ROW_H / 2)
    .attr('dy', '0.32em')
    .attr('font-size', FONT_SIZE_TYPE_LABEL)
    .attr('fill', COLOR_LABEL)
    .text((d) => d.label);

  enter
    .filter((d) => d.hasFields)
    .append('text')
    .attr('class', 'expand-arrow')
    .attr('y', ROW_H / 2)
    .attr('dy', '0.32em')
    .attr('font-size', FONT_SIZE_TYPE_ARROW)
    .attr('fill', COLOR_CHEVRON);

  enter
    .append('rect')
    .attr('class', 'expand-hit')
    .attr('x', 0)
    .attr('y', 0)
    .attr('width', HIT_MIN_W)
    .attr('height', ROW_H)
    .attr('fill', 'transparent');

  // Dot is appended LAST so it sits on top of the expand-hit rect for
  // pointer events — that lets hover on the dot fire owner-popover
  // handlers separately from row-level expand clicks. Click on the dot
  // still toggles expansion (delegated below) so it stays consistent
  // with the rest of the row.
  enter
    .append('circle')
    .attr('class', 'type-dot')
    .attr('cx', TYPE_CIRCLE_X)
    .attr('cy', ROW_H / 2)
    .attr('r', TYPE_RADIUS)
    .style('cursor', 'pointer')
    .attr('fill', (d) => colorForKind(d.typeKind));

  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);

  const merged = enter.merge(sel);

  // Tween group position (carries fields along inside).
  merged
    .transition('move')
    .duration(ANIM_MS)
    .attr('transform', (d) => `translate(${d.x},${d.y - ROW_H / 2})`);

  // Update expand-arrow text (expansion state may have changed).
  merged
    .filter((d) => d.hasFields)
    .select<SVGTextElement>('text.expand-arrow')
    .text((d) => (d.expanded ? '▾' : '▸'));

  // Refresh click handler each draw.
  merged
    .select<SVGRectElement>('rect.expand-hit')
    .attr('cursor', (d) => (d.hasFields ? 'pointer' : 'default'))
    .on('click', (event: MouseEvent, d) => {
      event.stopPropagation();
      if (d.hasFields) opts.onToggle(d.id);
    });

  // Refresh dot handlers each draw. Click on the dot expands every type
  // that owns this one — distinct from clicking the row, which toggles
  // the row's own expansion. Hover shows the owner-popover.
  merged
    .select<SVGCircleElement>('circle.type-dot')
    .on('click', (event: MouseEvent, d) => {
      event.stopPropagation();
      opts.onExpandAllOwners(d.fullPath);
    })
    .on('mouseenter', function (_event: MouseEvent, d) {
      const node = this as SVGCircleElement;
      opts.onShowOwners(d.fullPath, () => {
        const r = node.getBoundingClientRect();
        return { x: (r.left + r.right) / 2, y: (r.top + r.bottom) / 2 };
      });
    })
    .on('mouseleave', () => opts.onHideOwners());

  // Sub-data-join for field rows inside each type group.
  merged.each(function (d) {
    renderFieldsForType(select(this), zoomLayer, layout, d, opts);
  });

  sizeTypeHits(merged);
}

function renderFieldsForType(
  typeNode: Selection<SVGGElement, Layout['types'][number], null, undefined>,
  zoomLayer: Selection<SVGGElement, unknown, null, undefined>,
  layout: Layout,
  d: Layout['types'][number],
  opts: TreeRenderOptions,
): void {
  // Field rows are children of the type group so they pan with the parent's
  // transform. When the type is collapsed, fieldData is empty and the d3
  // join exits all field rows. Sub-key by field name (unique within a type).
  const fields = d.expanded ? d.fields : [];
  const groupTopY = d.y - ROW_H / 2;

  const sel = typeNode
    .selectAll<SVGGElement, Layout['types'][number]['fields'][number]>('g.field-row-g')
    .data(fields, (f) => f.name);

  sel.exit().transition('exit').duration(ANIM_MS).style('opacity', 0).remove();

  const enter = sel.enter().append('g').attr('class', 'field-row-g').style('opacity', 0);

  // Append the field name + the (hidden by default) type-text hint once on
  // enter. Position attrs are set on the merged selection so updates tween
  // correctly when the type's owner moves. The type-text sits past the
  // field name (at f.arrowSourceX) and toggles visibility on hover.
  enter
    .append('text')
    .attr('class', 'field-row')
    .attr('dy', '0.32em')
    .attr('font-size', FONT_SIZE_FIELD)
    .attr('fill', COLOR_FIELD_NAME);

  enter
    .append('text')
    .attr('class', 'field-ty')
    .attr('dy', '0.32em')
    .attr('font-size', FONT_SIZE_FIELD)
    .attr('fill', COLOR_FIELD_TY)
    .style('opacity', 0)
    .style('pointer-events', 'none');

  enter.transition('enter').duration(ANIM_MS).style('opacity', 1);

  const merged = enter.merge(sel);

  merged.each(function (f) {
    const fg = select(this);
    const localX = f.x - d.x;
    const localY = f.y - groupTopY;
    const isBorrow =
      f.ownership === 'borrow_immut' ||
      f.ownership === 'borrow_mut' ||
      f.ownership === 'indirection';
    const isSelected = opts.selectedFields.has(fieldKey(d.fullPath, f.name));

    const text = fg
      .select<SVGTextElement>('text.field-row')
      .attr('font-style', isBorrow ? 'italic' : 'normal')
      .attr('font-weight', isSelected ? 600 : 400)
      .text(f.name);

    text.transition('move').duration(ANIM_MS).attr('x', localX).attr('y', localY);

    const tyText = fg
      .select<SVGTextElement>('text.field-ty')
      .attr('x', f.arrowSourceX - d.x)
      .attr('y', localY)
      .text(f.tyText);

    text.on('click', (event: MouseEvent) => {
      event.stopPropagation();
      opts.onSelectField(d.fullPath, f.name);
    });

    // Type-hint stays for TY_HIDE_DELAY ms after mouse-out, so a glance
    // away doesn't immediately erase what the user just looked at. Re-entry
    // within the delay cancels the pending hide. Timer is stashed on the
    // DOM node so it survives re-renders (data-join keeps the same element).
    const node = text.node() as (SVGTextElement & { __sfTyTimer?: number | undefined }) | null;
    text.on('mouseenter', () => {
      const hover = chainArrowsFrom(layout, d.fullPath, f.name);
      const union = new Set<Layout['arrows'][number]>(opts.selectedArrows);
      for (const a of hover) union.add(a);
      applyChainHighlight(zoomLayer, union);
      if (node?.__sfTyTimer !== undefined) {
        clearTimeout(node.__sfTyTimer);
        node.__sfTyTimer = undefined;
      }
      tyText.transition('ty').duration(120).style('opacity', 1);
    });
    text.on('mouseleave', () => {
      applyChainHighlight(zoomLayer, opts.selectedArrows);
      if (!node) return;
      if (node.__sfTyTimer !== undefined) clearTimeout(node.__sfTyTimer);
      node.__sfTyTimer = window.setTimeout(() => {
        tyText.transition('ty').duration(200).style('opacity', 0);
        node.__sfTyTimer = undefined;
      }, TY_HIDE_DELAY);
    });

    // Selection pill — present only when isSelected. Insert before the text
    // so it draws beneath. Sized from the rendered text bbox.
    let bg = fg.select<SVGRectElement>('rect.focus-bg');
    if (isSelected) {
      if (bg.empty()) {
        bg = fg
          .insert('rect', 'text')
          .attr('class', 'focus-bg')
          .attr('rx', FOCUS_BG_RADIUS)
          .attr('ry', FOCUS_BG_RADIUS)
          .attr('fill', COLOR_FOCUS_BG)
          .attr('filter', `url(#${FOCUS_FILTER_ID})`);
      }
      const node = text.node();
      if (node) {
        const bbox = node.getBBox();
        bg.attr('x', bbox.x - FOCUS_BG_PAD_X)
          .attr('y', bbox.y - FOCUS_BG_PAD_Y_FIELD)
          .attr('width', bbox.width + 2 * FOCUS_BG_PAD_X)
          .attr('height', bbox.height + 2 * FOCUS_BG_PAD_Y_FIELD);
      }
    } else if (!bg.empty()) {
      bg.remove();
    }
  });
}

function sizeTypeHits<D>(sel: Selection<SVGGElement, D, SVGGElement, unknown>): void {
  // Place the trailing expand-arrow at end-of-name + gap; size the row's
  // single hit rect to span dot through past the arrow.
  sel.each(function () {
    const gg = select(this);
    const nameNode = gg.select<SVGGraphicsElement>('text.name').node();
    const nameW = nameNode ? nameNode.getBBox().width : 0;
    const nameEndX = TYPE_LABEL_X + nameW;
    const arrowX = nameEndX + TYPE_ARROW_GAP;

    gg.select<SVGTextElement>('text.expand-arrow').attr('x', arrowX);

    const arrowNode = gg.select<SVGGraphicsElement>('text.expand-arrow').node();
    const arrowW = arrowNode ? arrowNode.getBBox().width : 0;
    const right = arrowNode ? arrowX + arrowW + TYPE_ARROW_HIT_PAD : nameEndX + HIT_PAD_RIGHT;
    gg.select<SVGRectElement>('rect.expand-hit').attr('width', Math.max(right, HIT_MIN_W));
  });
}
