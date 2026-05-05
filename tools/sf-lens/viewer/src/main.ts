import { computeDrift } from './analysis/drift.ts';
import { type Layout, buildOptimizedLayout } from './analysis/layout.ts';
import { type TreeNode, buildModuleTree } from './analysis/module_tree.ts';
import {
  type OwnershipIndex,
  buildOwnershipIndex,
  computeOwnershipDepth,
} from './analysis/ownership.ts';
import { FactsLoadError, loadFacts } from './data/load.ts';
import type { Facts } from './data/schema.ts';
import { ViewState } from './state/view_state.ts';
import { createTextMeasurer } from './view/measure.ts';
import { createOwnersOverlay } from './view/owners_overlay.ts';
import {
  FONT_FAMILY,
  FONT_SIZE_FIELD,
  chainArrowsFromMany,
  fieldKey,
  renderEdgeShadows,
  renderTree,
} from './view/tree.ts';
import { attachZoom } from './view/zoom.ts';

const FACTS_URL = '/data/facts.json';
const PREFERRED_CRATE = 'sf-nano-core';
const SCALE_MAX = 1.5;
// Padding factor so content doesn't kiss the viewport edge at the fit scale.
const FIT_PADDING = 0.95;

interface TypeInfo {
  readonly label: string;
  readonly modulePath: string;
}

interface RenderCtx {
  readonly state: ViewState;
  readonly selectedFields: ReadonlySet<string>;
  readonly ownership: OwnershipIndex;
  readonly crateName: string;
  readonly draw: () => void;
  /** Set of ids that are types (vs modules). Used to filter ViewState's
   *  expanded ids when computing the type-level focus set. */
  readonly typeIdSet: ReadonlySet<string>;
  /** Lookup: type fullPath → label + module path. Used by the owner popover
   *  so it can render owner labels even when the owner isn't currently in
   *  the rendered layout. */
  readonly typeInfo: ReadonlyMap<string, TypeInfo>;
  /** Navigate the viewport to a type by id: expand the type (and its
   *  ancestor modules), redraw, then center the viewport on its new y. */
  readonly navigateToType: (typeId: string) => void;
  /** Reset to a pristine view: clear all field selections and expansions
   *  (keeping only the crate root open), exit focus mode, and reset
   *  the zoom transform to identity. */
  readonly resetAll: () => void;
  focusMode: boolean;
  /** Most recent layout. Used by the F-key handler to anchor the viewport
   *  on a focused item across the focus toggle (so content doesn't fly
   *  off-screen when the layout's y-extent changes). */
  lastLayout: Layout | null;
}

let currentCtx: RenderCtx | null = null;

void main();

async function main(): Promise<void> {
  const facts = await tryLoadFacts();
  if (!facts) return;

  const select = document.querySelector<HTMLSelectElement>('#crate');
  const svg = document.querySelector<SVGSVGElement>('#tree');
  if (!select || !svg) {
    showError('missing required DOM elements (#crate, #tree)');
    return;
  }

  const crateNames = Object.keys(facts.crates).sort();
  for (const name of crateNames) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    select.append(opt);
  }
  const initial = crateNames.includes(PREFERRED_CRATE) ? PREFERRED_CRATE : crateNames[0];
  if (!initial) {
    showError('facts.json contains no crates');
    return;
  }
  select.value = initial;

  // Single canvas-backed measurer for the whole session — one canvas
  // element, one font binding, results memoized per string. Field names
  // recur across crates and re-renders so the cache pays back many times.
  const measureText = createTextMeasurer(`${FONT_SIZE_FIELD}px ${FONT_FAMILY}`);

  const layers = attachZoom(svg, (t) => {
    updateScaleIndicator(t.k);
    // The owner popover is anchored in screen space to the dot's current
    // position. Notify it on every pan/zoom so it can either dismiss
    // (default) or, when pinned, follow the dot to its new spot.
    ownersOverlay.onCanvasMoved();
    // Per-band edge shadows on the frozen pane: which bands have hidden
    // type content depends on the current transform, so refresh every
    // pan/zoom event.
    if (currentCtx?.lastLayout) {
      renderEdgeShadows(layers, currentCtx.lastLayout, { x: t.x, k: t.k });
    }
  });
  updateScaleIndicator(1);

  // The owner popover is a single global instance reused across crates.
  // Click-to-navigate routes through currentCtx so it picks up the right
  // per-crate state.
  const ownersOverlay = createOwnersOverlay({
    onNavigate: (typeId) => currentCtx?.navigateToType(typeId),
  });

  // Toggle focus mode. Focus is a pure render-time filter: toggling it does
  // not mutate ViewState, so there's nothing to snapshot or restore.
  //
  // Anchor strategy across the toggle (always keeps something on screen):
  //   tier 1 — if the most recently selected item is in the viewport,
  //            anchor on it (keep it under the same screen y);
  //   tier 2 — otherwise, if any selected type is in the viewport, anchor
  //            on the topmost one;
  //   tier 3 — otherwise (nothing selected on screen), toggle, then
  //            center the viewport on the most recent selection's new y.
  const toggleFocus = (): void => {
    const ctx = currentCtx;
    if (!ctx) return;

    const range = layers.visibleYRange();
    const lastSelected = mostRecentSelection(ctx);

    let anchorId: string | null = null;
    let beforeY: number | null = null;

    // Tier 1
    if (lastSelected !== null) {
      const y = lookupY(ctx.lastLayout, lastSelected);
      if (y !== null && y >= range.min && y <= range.max) {
        anchorId = lastSelected;
        beforeY = y;
      }
    }
    // Tier 2: topmost on-screen expanded type.
    if (anchorId === null) {
      let bestY = Number.POSITIVE_INFINITY;
      for (const id of ctx.state.expandedIds()) {
        if (!ctx.typeIdSet.has(id)) continue;
        const y = lookupY(ctx.lastLayout, id);
        if (y === null || y < range.min || y > range.max) continue;
        if (y < bestY) {
          bestY = y;
          anchorId = id;
        }
      }
      if (anchorId !== null) beforeY = bestY;
    }

    ctx.focusMode = !ctx.focusMode;
    if (ctx.focusMode) {
      // Auto-expand the relevance modules in state so the focused subtree
      // is visible. Layout reads state directly, so this is what makes
      // module rows show their content. Expansions persist on exit.
      for (const id of computeFocus(ctx).modules) ctx.state.expand(id);
    }
    updateFocusModeIndicator(ctx.focusMode);
    ctx.draw();

    if (anchorId !== null && beforeY !== null) {
      const afterY = lookupY(ctx.lastLayout, anchorId);
      if (afterY !== null && afterY !== beforeY) {
        layers.translateBy(0, beforeY - afterY, true);
      }
      return;
    }
    // Tier 3: nothing visible. Bring the most-recent selection into view.
    if (lastSelected !== null) {
      const y = lookupY(ctx.lastLayout, lastSelected);
      if (y !== null) layers.centerOnY(y, true);
    }
  };

  // Keyboard bindings: F = toggle focus, S = reset scale to 100%,
  // R = reset everything (selections + expansion + focus + zoom/pan).
  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'f') toggleFocus();
    else if (e.key === 's') layers.resetScale(true);
    else if (e.key === 'r') currentCtx?.resetAll();
  });

  // Mouse bindings — clicking any keybinding chip triggers its action.
  const focusHint = document.querySelector<HTMLElement>('#hint-focus');
  if (focusHint) focusHint.addEventListener('click', () => toggleFocus());
  const scaleHint = document.querySelector<HTMLElement>('#hint-scale');
  if (scaleHint) scaleHint.addEventListener('click', () => layers.resetScale(true));
  const resetHint = document.querySelector<HTMLElement>('#hint-reset');
  if (resetHint) resetHint.addEventListener('click', () => currentCtx?.resetAll());

  // Window resize: viewport changes, so the fit-to-view minimum changes.
  // Debounce so a drag-resize doesn't fire dozens of times.
  let resizeTimer: number | null = null;
  window.addEventListener('resize', () => {
    if (resizeTimer !== null) clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      resizeTimer = null;
      const layout = currentCtx?.lastLayout;
      if (layout) applyFitScaleExtent(svg, layers, layout);
    }, 100);
  });

  const renderFor = (crateName: string): void => {
    const crate = facts.crates[crateName];
    if (!crate) {
      showError(`crate '${crateName}' not found in facts.json`);
      return;
    }
    const staticRoot = buildModuleTree(crate);
    const ownership = buildOwnershipIndex(facts, crateName);
    const allTypeIds = collectTypeIds(staticRoot);
    const typeModule = collectTypeModuleMap(staticRoot);
    const drift = computeDrift(ownership, typeModule);
    const depth = computeOwnershipDepth(ownership, allTypeIds, drift);
    const typeIdSet = new Set(allTypeIds);
    const typeInfo = collectTypeInfo(staticRoot);

    const state = new ViewState([staticRoot.id]);
    const selectedFields = new Set<string>();

    let lastLayout: Layout | null = null;
    const draw = (): void => {
      // In focus mode, derive a render-time filter from the current selection.
      // ViewState is left alone — focus is purely a layout-input override.
      const focus = currentCtx?.focusMode ? computeFocus(currentCtx) : null;
      lastLayout = buildOptimizedLayout({
        staticRoot,
        ownership,
        depth,
        state,
        drift,
        measureText,
        ...(focus ? { focusModules: focus.modules } : {}),
      });
      if (currentCtx) currentCtx.lastLayout = lastLayout;
      applyFitScaleExtent(svg, layers, lastLayout);
      // Compute the union of arrow chains for every selected field. These are
      // drawn highlighted by default; hover unions a transient chain on top.
      const selectedFieldRefs = [...selectedFields].map((k) => {
        const sep = k.lastIndexOf('::');
        return { typePath: k.slice(0, sep), fieldName: k.slice(sep + 2) };
      });
      const selectedArrows = chainArrowsFromMany(lastLayout, selectedFieldRefs);

      renderTree(layers, lastLayout, {
        selectedFields,
        selectedArrows,
        onToggle: (id) => {
          // Anchor the clicked item: capture its absolute y before the layout
          // changes, then after re-render compute the new y and shift the
          // viewport by the inverse so the click target stays under the cursor.
          const beforeY = lookupY(lastLayout, id);
          const wasExpanded = state.isExpanded(id);
          state.toggle(id);
          if (!wasExpanded && typeIdSet.has(id)) {
            for (const moduleId of targetModulesFor(id, ownership, crateName)) {
              state.expand(moduleId);
            }
          }
          draw();
          const afterY = lookupY(lastLayout, id);
          if (beforeY !== null && afterY !== null && beforeY !== afterY) {
            // Animated translate so the viewport tween runs in lockstep with
            // the renderer's transform tween, keeping the click target glued
            // under the cursor for the entire animation.
            layers.translateBy(0, beforeY - afterY, true);
          }
        },
        onSelectField: (typePath, fieldName) => {
          const key = fieldKey(typePath, fieldName);
          if (selectedFields.has(key)) selectedFields.delete(key);
          else selectedFields.add(key);
          draw();
        },
        onShowOwners: (typePath, getDotScreenPos) => {
          const ownerIds = ownership.ownedBy.get(typePath);
          if (!ownerIds || ownerIds.length === 0) return;
          const owners = ownerIds
            .map((id) => {
              const info = typeInfo.get(id);
              if (!info) return null;
              return { typeId: id, typeLabel: info.label, modulePath: info.modulePath };
            })
            .filter((x): x is NonNullable<typeof x> => x !== null);
          if (owners.length === 0) return;
          ownersOverlay.show({ owners, getDotScreenPos });
        },
        onHideOwners: () => ownersOverlay.hide(),
        onExpandAllOwners: (typePath) => {
          const ownerIds = ownership.ownedBy.get(typePath);
          if (!ownerIds || ownerIds.length === 0) return;
          for (const ownerId of ownerIds) {
            for (const m of ancestorModuleIds(ownerId, crateName)) state.expand(m);
            state.expand(ownerId);
          }
          draw();
        },
      });
    };

    const navigateToType = (typeId: string): void => {
      // Make the target visible: expand it (so its module is in focus
      // relevance and its row renders), expand its containing modules,
      // redraw, then center the viewport on its new y.
      for (const m of ancestorModuleIds(typeId, crateName)) state.expand(m);
      state.expand(typeId);
      draw();
      const y = lookupY(lastLayout, typeId);
      if (y !== null) layers.centerOnY(y, true);
    };

    const resetAll = (): void => {
      selectedFields.clear();
      state.clear();
      state.expand(staticRoot.id);
      if (currentCtx) currentCtx.focusMode = false;
      updateFocusModeIndicator(false);
      draw();
      layers.resetTransform(true);
    };

    // Each crate gets a fresh ctx; switching crates resets focus mode and
    // its keybinding indicator so old per-crate state never leaks.
    currentCtx = {
      state,
      selectedFields,
      ownership,
      crateName,
      draw,
      typeIdSet,
      typeInfo,
      navigateToType,
      resetAll,
      focusMode: false,
      lastLayout: null,
    };
    updateFocusModeIndicator(false);
    draw();
  };

  renderFor(initial);
  select.addEventListener('change', () => renderFor(select.value));
}

async function tryLoadFacts(): Promise<Facts | null> {
  try {
    return await loadFacts(FACTS_URL);
  } catch (err) {
    if (err instanceof FactsLoadError) {
      showError(err.message);
    } else {
      showError(`unexpected error: ${String(err)}`);
    }
    return null;
  }
}

// Compute the fit-to-view minimum scale and apply it as the zoom's lower
// bound. Content fits at scale = min(viewportW / contentW, viewportH /
// contentH). FIT_PADDING leaves a small margin around the edges.
//
// The fit value is the *floor* of the allowed range — it stops the user
// from shrinking below "everything visible". When content is small enough
// to already fit at normal size (fit > 1), we cap the floor at 1.0 so we
// don't force-zoom IN; otherwise the empty default view ends up rendered
// at SCALE_MAX, which feels jarring. The ceiling is always SCALE_MAX.
function applyFitScaleExtent(
  svgEl: SVGSVGElement,
  layers: ReturnType<typeof attachZoom>,
  layout: Layout,
): void {
  const vw = svgEl.clientWidth;
  const vh = svgEl.clientHeight;
  if (vw <= 0 || vh <= 0 || layout.totalWidth <= 0 || layout.totalHeight <= 0) return;
  const fit = Math.min(vw / layout.totalWidth, vh / layout.totalHeight) * FIT_PADDING;
  layers.setScaleExtent(Math.min(fit, 1), SCALE_MAX);
}

// The most recently expanded type — `state` iterates in insertion order,
// so the last entry is the latest add. Modules are filtered out via
// `typeIdSet`. Returns null if nothing is expanded.
function mostRecentSelection(ctx: RenderCtx): string | null {
  let last: string | null = null;
  for (const id of ctx.state.expandedIds()) {
    if (ctx.typeIdSet.has(id)) last = id;
  }
  return last;
}

// Render-time focus filter, derived from ViewState's expanded set. Returns
// the set of module ids that should render in focus mode — closed under
// ancestors so the visible subtree stays connected. Anything not in this
// set is dropped entirely (no row, no name, no children).
//
// Members of the focus set:
//   - the crate root,
//   - the containing module (and ancestors) of every expanded type,
//   - the containing module (and ancestors) of every one-step arrow
//     target of an expanded type's fields, so direct outgoing chains
//     stay visible (their type box renders collapsed in those modules),
//   - the containing module (and ancestors) of every arrow target of a
//     selected field. (The owner is necessarily expanded, so it's
//     already covered by the first bullet.)
function computeFocus(ctx: RenderCtx): { modules: ReadonlySet<string> } {
  const { state, selectedFields, ownership, crateName, typeIdSet } = ctx;
  const modules = new Set<string>([crateName]);

  const addAncestors = (typeFullPath: string): void => {
    for (const m of ancestorModuleIds(typeFullPath, crateName)) modules.add(m);
  };

  for (const id of state.expandedIds()) {
    if (!typeIdSet.has(id)) continue;
    addAncestors(id);
    const fieldsMap = ownership.fieldTargets.get(id);
    if (fieldsMap) {
      for (const targets of fieldsMap.values()) {
        for (const t of targets) addAncestors(t);
      }
    }
  }

  for (const k of selectedFields) {
    const sep = k.lastIndexOf('::');
    const typePath = k.slice(0, sep);
    const fieldName = k.slice(sep + 2);
    addAncestors(typePath);
    const targets = ownership.fieldTargets.get(typePath)?.get(fieldName);
    if (targets) {
      for (const t of targets) addAncestors(t);
    }
  }

  return { modules };
}

// All module ids on the path to (and including) every field target's owning
// module — these need to be expanded for arrow targets to appear in the layout.
function targetModulesFor(typeId: string, ownership: OwnershipIndex, crateName: string): string[] {
  const out = new Set<string>();
  const fields = ownership.fieldTargets.get(typeId);
  if (!fields) return [];
  for (const targets of fields.values()) {
    for (const targetFullPath of targets) {
      for (const id of ancestorModuleIds(targetFullPath, crateName)) {
        out.add(id);
      }
    }
  }
  return [...out];
}

// `crate::a::b::Type` → [`crate`, `crate::a`, `crate::a::b`].
function ancestorModuleIds(typeFullPath: string, crateName: string): string[] {
  const segments = typeFullPath.split('::');
  if (segments[0] !== crateName || segments.length < 2) return [];
  const ids = [crateName];
  let path = '';
  for (let i = 1; i < segments.length - 1; i++) {
    const seg = segments[i] ?? '';
    path = path === '' ? seg : `${path}::${seg}`;
    ids.push(`${crateName}::${path}`);
  }
  return ids;
}

function lookupY(
  layout: ReturnType<typeof buildOptimizedLayout> | null,
  id: string,
): number | null {
  if (!layout) return null;
  for (const t of layout.types) {
    if (t.id === id) return t.y;
  }
  for (const m of layout.modules) {
    if (m.id === id) return m.y;
  }
  return null;
}

function collectTypeIds(root: TreeNode): string[] {
  const out: string[] = [];
  const walk = (n: TreeNode): void => {
    if (n.kind === 'type') out.push(n.fullPath);
    else for (const c of n.children) walk(c);
  };
  walk(root);
  return out;
}

function collectTypeModuleMap(root: TreeNode): Map<string, string> {
  const out = new Map<string, string>();
  const walk = (n: TreeNode): void => {
    if (n.kind === 'type') out.set(n.fullPath, n.modulePath);
    else for (const c of n.children) walk(c);
  };
  walk(root);
  return out;
}

// Lookup table for the owner popover: every type's display label and
// containing module path, keyed by its full path. Built once per crate.
function collectTypeInfo(root: TreeNode): Map<string, TypeInfo> {
  const out = new Map<string, TypeInfo>();
  const walk = (n: TreeNode): void => {
    if (n.kind === 'type') out.set(n.fullPath, { label: n.label, modulePath: n.modulePath });
    else for (const c of n.children) walk(c);
  };
  walk(root);
  return out;
}

function updateFocusModeIndicator(active: boolean): void {
  const el = document.querySelector<HTMLElement>('#hint-focus');
  if (!el) return;
  el.classList.toggle('active', active);
}

function updateScaleIndicator(k: number): void {
  const el = document.querySelector<HTMLElement>('#scale-value');
  if (!el) return;
  el.textContent = `${Math.round(k * 100)}%`;
}

function showError(message: string): void {
  const el = document.querySelector<HTMLElement>('#error');
  if (!el) {
    console.error(message);
    return;
  }
  el.textContent = message;
  el.hidden = false;
}
