// Hover-triggered popover that lists incoming ownership relationships
// for a hovered type. Lives outside the SVG zoom layer so it doesn't
// scale with zoom; positioned in screen-space using the dot's current
// screen coordinates derived from the active zoom transform.
//
// Hover semantics: 300ms grace on dot mouse-leave so the cursor can
// travel to the panel; the panel itself cancels the hide on enter and
// re-arms on leave.

const PANEL_OFFSET_X = 18; // gap between dot and panel
const HIDE_DELAY_MS = 300;

export interface OwnerEntry {
  readonly modulePath: string;
  readonly typeLabel: string;
  readonly typeId: string;
}

export interface ShowArgs {
  readonly owners: ReadonlyArray<OwnerEntry>;
  /** Callback that returns the dot's current screen-space center. Captures
   *  the SVG circle element so the overlay can re-project on every zoom
   *  event when pinned, keeping the panel anchored to the moving dot. */
  readonly getDotScreenPos: () => { x: number; y: number };
}

export interface OwnersOverlay {
  show: (args: ShowArgs) => void;
  /** Schedule a delayed hide. No-op while pinned. Cancelled by panel hover. */
  hide: () => void;
  /** Hide right away — overrides the pinned state. Used by full reset
   *  flows (resetAll) and when a row navigation completes. */
  hideImmediately: () => void;
  /** Called by the zoom layer on every pan/zoom event. Behavior:
   *  - if not visible → no-op;
   *  - if not pinned → dismiss (the dot the popover points to has moved);
   *  - if pinned → re-project the dot's current screen position via the
   *    saved getDotScreenPos closure and reposition the panel so it stays
   *    anchored to the dot as the canvas moves under it. */
  onCanvasMoved: () => void;
}

export function createOwnersOverlay(opts: {
  onNavigate: (typeId: string) => void;
}): OwnersOverlay {
  // Root container: covers the viewport, transparent to clicks except where
  // a child element re-enables pointer events (the panel).
  const root = document.createElement('div');
  root.id = 'owners-overlay';
  root.style.position = 'fixed';
  root.style.inset = '0';
  root.style.pointerEvents = 'none';
  root.style.zIndex = '20';
  root.style.display = 'none';
  document.body.appendChild(root);

  const panel = document.createElement('div');
  panel.className = 'owners-panel';
  panel.style.pointerEvents = 'auto';
  panel.style.position = 'absolute';
  root.appendChild(panel);

  let hideTimer: number | null = null;
  let pinned = false;
  let getDotScreenPos: (() => { x: number; y: number }) | null = null;

  const cancelHide = (): void => {
    if (hideTimer !== null) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  };

  const scheduleHide = (): void => {
    if (pinned) return; // Pinned panels never auto-dismiss.
    cancelHide();
    hideTimer = window.setTimeout(() => {
      root.style.display = 'none';
      hideTimer = null;
    }, HIDE_DELAY_MS);
  };

  panel.addEventListener('mouseenter', cancelHide);
  panel.addEventListener('mouseleave', scheduleHide);

  // Position the panel near a given dot screen-position. Used both on the
  // initial show and on each repositionPinned call when the canvas moves.
  const placePanel = (dotScreenX: number, dotScreenY: number): void => {
    const rect = panel.getBoundingClientRect();
    const panelW = rect.width;
    const panelH = rect.height;
    let panelLeft = dotScreenX - PANEL_OFFSET_X - panelW;
    if (panelLeft < 8) panelLeft = dotScreenX + PANEL_OFFSET_X;
    let panelTop = dotScreenY - panelH / 2;
    panelTop = Math.max(8, Math.min(panelTop, window.innerHeight - panelH - 8));
    panel.style.left = `${panelLeft}px`;
    panel.style.top = `${panelTop}px`;
  };

  const show = (args: ShowArgs): void => {
    cancelHide();
    getDotScreenPos = args.getDotScreenPos;

    // Group owners by their module so e.g. "vm::wasm: Module, Builder"
    // collapses to one row instead of two.
    const byModule = new Map<string, { types: string[]; ids: string[] }>();
    for (const o of args.owners) {
      let g = byModule.get(o.modulePath);
      if (!g) {
        g = { types: [], ids: [] };
        byModule.set(o.modulePath, g);
      }
      g.types.push(o.typeLabel);
      g.ids.push(o.typeId);
    }
    const grouped = [...byModule.entries()].map(([modulePath, g]) => ({
      modulePath,
      typesText: g.types.join(', '),
      // Click navigates to the first type in the module (good enough — the
      // user can read all the names in the row and click again if needed).
      navigateToId: g.ids[0] ?? '',
    }));

    panel.innerHTML = '';

    // Reset pin state on every fresh show; pinning is per-popup-instance.
    pinned = false;

    const header = document.createElement('div');
    header.className = 'header';
    const title = document.createElement('div');
    title.className = 'title';
    const totalOwners = args.owners.length;
    title.textContent = `Owners (${totalOwners})`;
    header.appendChild(title);

    const pinBtn = document.createElement('button');
    pinBtn.className = 'pin-btn';
    pinBtn.type = 'button';
    pinBtn.title = 'Pin';
    pinBtn.textContent = '📌';
    pinBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      pinned = !pinned;
      pinBtn.classList.toggle('active', pinned);
      pinBtn.title = pinned ? 'Unpin (allows auto-dismiss)' : 'Pin (prevents auto-dismiss)';
      // Unpinning while the cursor is outside the panel should immediately
      // arm the dismiss timer; if cursor is inside the panel, scheduleHide
      // is a no-op because mouseleave hasn't fired.
      if (!pinned) scheduleHide();
      else cancelHide();
    });
    header.appendChild(pinBtn);
    panel.appendChild(header);

    const list = document.createElement('ul');
    list.className = 'owners-list';
    // No row cap — the panel scrolls when its content exceeds max-height.
    for (const g of grouped) {
      const li = document.createElement('li');
      li.className = 'owners-row';
      const m = document.createElement('span');
      m.className = 'module';
      m.textContent = g.modulePath || '(crate root)';
      const t = document.createElement('span');
      t.className = 'types';
      t.textContent = g.typesText;
      li.appendChild(m);
      li.appendChild(t);
      li.addEventListener('click', () => {
        if (g.navigateToId !== '') {
          opts.onNavigate(g.navigateToId);
          cancelHide();
          pinned = false;
          root.style.display = 'none';
        }
      });
      list.appendChild(li);
    }
    panel.appendChild(list);

    // Reveal so we can measure the panel's actual size.
    root.style.display = 'block';
    const pos = args.getDotScreenPos();
    placePanel(pos.x, pos.y);
  };

  return {
    show,
    hide: scheduleHide,
    hideImmediately: () => {
      cancelHide();
      pinned = false;
      getDotScreenPos = null;
      root.style.display = 'none';
    },
    onCanvasMoved: () => {
      if (root.style.display === 'none') return;
      if (!pinned) {
        cancelHide();
        getDotScreenPos = null;
        root.style.display = 'none';
        return;
      }
      // Pinned: re-anchor to the dot's new on-screen position.
      if (!getDotScreenPos) return;
      const pos = getDotScreenPos();
      placePanel(pos.x, pos.y);
    },
  };
}
