#!/usr/bin/env python3
"""Generate the benchmark charts from measured data.

Design notes, so the next person does not have to reverse-engineer them:

* Grouped horizontal bars. The job is magnitude comparison across named
  entities, and the entity names are long, so horizontal bars with direct
  labels beat any vertical or radial form.
* Bars are normalised to the best value IN THEIR ROW, and the absolute
  number is printed at the end of every bar. Cross-row comparison is
  meaningless anyway (different units), so each row gets its own scale.
* Two groups per row, split by engine class (compiled vs interpreted).
  A compiler is ~7x an interpreter here; mixing them on one scale would
  crush the interpreter bars into unreadable slivers, so they are grouped
  and separated by a rule.
* One exception: the README hero chart (see HERO_* below) puts the three
  interpreters on one shared scale, because one row of bars can carry the
  whole comparison directly.
* Colour is categorical (identity = runtime), assigned in a fixed order and
  never cycled. The palette is validated for the adjacent-pair CVD and
  normal-vision floors in BOTH light and dark mode -- see SERIES below.
  Three light-mode hues fall under 3:1 contrast on the light surface, which
  is why every bar carries a visible direct label and RESULTS.md carries the
  table view: that is the required relief, not an oversight.
* No animation. These are static comparison charts read in a README; a
  growing bar adds nothing and makes the file diff noisily.

Run after updating the DATA section below:  python3 gen_svg.py
"""

import os
import re

# ── geometry ────────────────────────────────────────────────────────────
W = 760          # canvas width
GUTTER = 168     # left column: runtime names, right-aligned to here
X0 = 176         # bar origin
BAR_MAX = 452    # longest bar
BAR_H = 18       # thin marks
BAR_PITCH = 20   # leaves a 2px surface gap between adjacent bars
GROUP_GAP = 10   # extra space between the two engine classes
ROW_HEAD = 24    # metric name + unit line
ROW_PAD = 12
LEGEND_Y0 = 62   # baseline of the first legend row
LEGEND_ROW = 18  # legend line height; the header grows by this per wrapped row

CHAR_W = 5.6     # ~advance of 11px UI sans, for layout estimation only

# Hero geometry. One benchmark, so there is room for taller marks, and no
# legend is needed — every bar already carries its runtime name.
HERO_BAR_H = 22
HERO_PITCH = 28
HERO_TIER_HEAD = 22   # tier caption line above each group of bars
HERO_TIER_GAP = 8     # after a tier's last bar
HERO_Y0 = 72          # first tier caption, clear of the title + subtitle

# ── palette ─────────────────────────────────────────────────────────────
# (key, label, light, dark). Order is fixed. Checked with the palette
# validator on the adjacent pairlist (the correct list for grouped bars),
# per chart, since adjacency only exists within a chart:
#   compiled light: worst adjacent CVD dE  9.2, normal-vision 27.6 -> all PASS
#   compiled dark:  worst adjacent CVD dE  9.0, normal-vision 26.5 -> all PASS
#   interp light:   worst adjacent CVD dE 25.0, normal-vision 29.9 -> all PASS
#   interp dark:    worst adjacent CVD dE 27.3, normal-vision 27.5 -> all PASS
# The hero reuses the three interpreter colours and labels every bar directly.
# Both Silverfir engines wear the same violet family on purpose, so "ours"
# reads at a glance; they are never adjacent in this order, and the hues
# beside them are deliberately far from violet (orange, teal, amber).
#
# The JIT violet was re-stepped 2026-07-25, from #7c3aed (OKLCH L 0.54,
# C 0.247 -- the most saturated mark in the file) to #7952d4 (L 0.55,
# C 0.191). Measured cause: the compiled group sat at mean L 0.603 against
# the interpreters' 0.689 and mean C 0.190 against 0.173, so it read heavier
# and hotter beside them; that neon violet was half of the gap. Dropping its
# chroma closes it without touching depth, so the two-step violet family
# survives -- JIT vs interp is still dE 9.7 apart in light (was 9.8). No
# worst pair moved: the binding pairs are orange-aqua and plum-violet and
# neither involves this slot.
#
# Winch's plum stays dark, which is load-bearing rather than an oversight.
# Lightening it in place walks it into the interpreter violet: a light
# magenta measures dE 14.2 against #a855f7, under the 15 normal-vision floor,
# a hard FAIL. Moving it warm collides with Cranelift instead -- a mid rust
# lands 5.5 from #eb6834, i.e. two bars of the same colour -- and a rust dark
# enough to separate (dE 20) renders as muddy brown. No teal step passes at
# all. The slot is boxed in by V8's aqua on the CVD gate and by the two
# violets on the normal-vision floor; its darkness is what buys the margin.
# Winch trails the three optimizing compilers rather than sitting beside
# Cranelift: it is a baseline compiler, a different tier, and keeping
# Silverfir/Cranelift/V8 adjacent is what makes that trio comparable at a
# glance. It is magenta rather than a second orange because the warm
# wasmtime-family steps only reach dE 7.7-7.9 against V8's dark green --
# inside the 6-8 band that is legal only with secondary encoding -- while
# magenta clears the dE 8 target at 9.0.
# 'column' is the header text this series has in the RESULTS.md tables.
SERIES = [
    ('sf', 'Silverfir (JIT)',    'SF (JIT)',    '#7952d4', '#8b5cf6'),
    ('cl', 'Cranelift 47.0.2',   'Cranelift',   '#eb6834', '#d95926'),
    ('v8', 'V8 / Node 25.9',     'V8',          '#1baf7a', '#199e70'),
    ('wn', 'Winch 47.0.2',       'Winch',       '#b02a8f', '#c13a9d'),
    ('si', 'Silverfir (interp)', 'SF (interp)', '#a855f7', '#9085e9'),
    ('w3', 'wasm3',              'wasm3',       '#eda100', '#c98500'),
    ('wi', 'wasmi 2.0.0-beta.7', 'wasmi',       '#4a9ede', '#3987e5'),
]
# Each entry produces its OWN chart file: comparing a JIT to an interpreter on
# one scale is ~7x, which crushes the interpreter bars into slivers. Split by
# class and every row renormalises within its class, so the comparisons that
# matter stay legible.
GROUPS = [
    ('',        'Compiled', ['sf', 'cl', 'v8', 'wn']),
    ('_interp', 'Interpreters',   ['si', 'w3', 'wi']),
]
LABEL = {k: n for k, n, _c, _l, _d in SERIES}
BY_COLUMN = {c: k for k, _n, c, _l, _d in SERIES}

# ── README hero ─────────────────────────────────────────────────────────
# The top-level README wants one direct interpreter comparison. A single
# CoreMark row fits all three runtimes on one shared scale, so every bar length
# is comparable and the current ordering can stay fastest-first.
#
HERO_FILE = 'coremark.svg'
HERO_BENCH = 'CoreMark'
HERO_TIERS = [
    ('Interpreters', ['si', 'wi', 'w3']),
]
COREMARK_LABEL = {
    'si': 'Silverfir-nano',
    'wi': 'Wasmi 2.0.0-beta.10',
    'w3': 'Wasm3 0.9.0',
}
LEGEND_LABEL = {'wi': 'wasmi 2.0'}


def fmt(n):
    if float(n).is_integer():        # scores are whole numbers; no ".0"
        return f'{n:,.0f}'
    if n >= 10000:
        return f'{n:,.0f}'
    if n >= 1000:
        return f'{n:,.1f}'
    if n >= 100:
        return f'{n:.1f}'
    return f'{n:.2f}'


def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def row_svg(y, bench, keys):
    """One benchmark: a heading line, then one bar per runtime in this chart."""
    out = []
    best = max(bench['data'][k] for k in keys)
    out.append(f'<text class="metric" x="{GUTTER}" y="{y + 14}" '
               f'text-anchor="end">{esc(bench["name"])}</text>')
    out.append(f'<text class="unit" x="{X0}" y="{y + 14}">{esc(bench["unit"])}</text>')
    if bench.get('note'):
        out.append(f'<text class="note" x="{W - 20}" y="{y + 14}" '
                   f'text-anchor="end">{esc(bench["note"])}</text>')

    by = y + ROW_HEAD
    row_keys = keys
    if bench['name'] == HERO_BENCH and set(keys) == set(COREMARK_LABEL):
        row_keys = sorted(keys, key=lambda k: bench['data'][k], reverse=True)
    for key in row_keys:
        val = bench['data'][key]
        w = max(2.0, BAR_MAX * val / best)
        label = (COREMARK_LABEL.get(key, LABEL[key])
                 if bench['name'] == HERO_BENCH else LABEL[key])
        out.append(f'<text class="name" x="{GUTTER}" y="{by + 13}" '
                   f'text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect class="s-{key}" x="{X0}" y="{by}" '
                   f'width="{w:.1f}" height="{BAR_H}" rx="4"/>')
        # Value labels always sit just past the bar end, in ink tokens.
        # Inside-the-bar white text was tried and dropped: some of the hues
        # are light (amber 2.1:1 against the surface), so white-on-fill fell
        # below text contrast for them.
        out.append(f'<text class="val" x="{X0 + w + 6:.1f}" '
                   f'y="{by + 13}">{fmt(val)}</text>')
        by += BAR_PITCH
    return '\n  '.join(out), by + ROW_PAD


def legend_layout(keys):
    """Spread the legend evenly across the bar band, wrapping only if needed.

    Item widths are estimated from label length -- an SVG carries no text
    metrics at build time, and the estimate only has to be good enough to
    decide how many fit per row. Returns (row_count, [(key, x, row), ...]).

    This replaced a fixed 3-per-row, 196px grid. That grid silently overlapped
    the first bar row as soon as a chart carried a fourth series: the second
    legend row landed at exactly the old constant header. Deriving the pitch
    from the item count keeps every chart's legend on one evenly spaced row
    for as long as the labels actually fit.
    """
    avail = W - 20 - X0
    widest = max(15 + len(LABEL[k]) * CHAR_W for k in keys)
    per_row = max(1, min(len(keys), int(avail // widest)))
    pitch = avail / per_row
    placed = [(k, X0 + (i % per_row) * pitch, i // per_row)
              for i, k in enumerate(keys)]
    return (len(keys) + per_row - 1) // per_row, placed


def header_height(keys):
    """Top band: title, subtitle, and however many rows the legend needs."""
    return LEGEND_Y0 + legend_layout(keys)[0] * LEGEND_ROW


def legend_svg(keys):
    out = []
    for key, x, row in legend_layout(keys)[1]:
        y = LEGEND_Y0 + row * LEGEND_ROW
        out.append(f'<rect class="s-{key}" x="{x}" y="{y - 9}" width="10" '
                   f'height="10" rx="2"/>')
        label = LEGEND_LABEL.get(key, LABEL[key])
        out.append(f'<text class="legend" x="{x + 15}" y="{y}">{esc(label)}</text>')
    return '\n  '.join(out)


def document(title, subtitle, h, inner):
    """The shared shell: surface, title, subtitle, styles, and `inner` body.

    Both chart kinds draw the same marks in the same tokens, so the palette,
    the dark-mode block and the type scale live here once.
    """
    light = '\n'.join(f'    .s-{k} {{ fill: {l}; }}' for k, _n, _c, l, _d in SERIES)
    dark = '\n'.join(f'      .s-{k} {{ fill: {d}; }}' for k, _n, _c, _l, d in SERIES)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 {W} {h}" width="{W}" height="{h}">
  <title>{esc(title)}</title>
  <style>
    text     {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .surface {{ fill: #fcfcfb; stroke: #d8d8d4; }}
    .title   {{ fill: #0b0b0b; font-size: 17px; font-weight: 600; }}
    .sub     {{ fill: #52514e; font-size: 11.5px; }}
    .metric  {{ fill: #0b0b0b; font-size: 12.5px; font-weight: 600; }}
    .unit    {{ fill: #77756e; font-size: 10.5px; }}
    .note    {{ fill: #77756e; font-size: 10px; font-style: italic; }}
    .name    {{ fill: #52514e; font-size: 11.5px; }}
    .legend  {{ fill: #52514e; font-size: 11px; }}
    .tier    {{ fill: #77756e; font-size: 10.5px; font-weight: 600; letter-spacing: 0.04em; }}
    .val     {{ fill: #0b0b0b; font-size: 11.5px; font-weight: 600; }}
    .rule    {{ stroke: #e4e4e0; stroke-width: 1; stroke-dasharray: 2 3; }}
    .axis    {{ stroke: #d8d8d4; stroke-width: 1; }}
{light}
    @media (prefers-color-scheme: dark) {{
      .surface {{ fill: #1a1a19; stroke: #3a3a37; }}
      .title, .metric {{ fill: #ffffff; }}
      .sub, .name, .legend {{ fill: #c3c2b7; }}
      .val     {{ fill: #ffffff; }}
      .unit, .note, .tier {{ fill: #96958c; }}
      .rule    {{ stroke: #3a3a37; }}
      .axis    {{ stroke: #3a3a37; }}
{dark}
    }}
  </style>
  <rect class="surface" width="{W}" height="{h}" rx="8" ry="8" stroke-width="1"/>
  <text class="title" x="{W // 2}" y="30" text-anchor="middle">{esc(title)}</text>
  <text class="sub" x="{W // 2}" y="47" text-anchor="middle">{esc(subtitle)}</text>
{inner}
</svg>
'''


def write(filename, svg, h):
    with open(filename, 'w') as f:
        f.write(svg)
    print(f'Generated: {filename}  ({h}px tall)')


def make_svg(title, subtitle, benches, filename, keys):
    header = header_height(keys)
    body, y = [], header
    for b in benches:
        chunk, y = row_svg(y, b, keys)
        body.append('  ' + chunk)
    h = y + 6
    nl = chr(10)

    inner = (f'  {legend_svg(keys)}\n'
             f'  <line class="axis" x1="{X0}" y1="{header - 4}" '
             f'x2="{X0}" y2="{h - 10}"/>\n\n'
             + nl.join(body))
    write(filename, document(title, subtitle, h, inner), h)


def make_hero(bench, filename, title, subtitle):
    """One benchmark, selected runtimes, one shared scale, fastest first.

    No legend and no per-row renormalisation: there is only one row, so the
    bar lengths are directly comparable end to end, which is the whole reason
    this chart exists.
    """
    missing = [k for _c, keys in HERO_TIERS for k in keys if k not in bench['data']]
    if missing:
        raise SystemExit(f'{RESULTS}: "{bench["name"]}" has no column for '
                         f'{[LABEL[k] for k in missing]}; the hero needs every runtime')

    best = max(bench['data'][k] for _c, keys in HERO_TIERS for k in keys)
    body, y = [], HERO_Y0
    for caption, keys in HERO_TIERS:
        body.append(f'<text class="tier" x="{GUTTER}" y="{y + 14}" '
                    f'text-anchor="end">{esc(caption)}</text>')
        body.append(f'<line class="rule" x1="{X0}" y1="{y + 10}" '
                    f'x2="{W - 20}" y2="{y + 10}"/>')
        y += HERO_TIER_HEAD
        for key in sorted(keys, key=lambda k: bench['data'][k], reverse=True):
            val = bench['data'][key]
            w = max(2.0, BAR_MAX * val / best)
            base = y + HERO_BAR_H / 2 + 4.5
            body.append(f'<text class="name" x="{GUTTER}" y="{base:.1f}" '
                        f'text-anchor="end">{esc(COREMARK_LABEL.get(key, LABEL[key]))}</text>')
            body.append(f'<rect class="s-{key}" x="{X0}" y="{y}" '
                        f'width="{w:.1f}" height="{HERO_BAR_H}" rx="5"/>')
            body.append(f'<text class="val" x="{X0 + w + 8:.1f}" '
                        f'y="{base:.1f}">{fmt(val)}</text>')
            y += HERO_PITCH
        y += HERO_TIER_GAP
    h = y - HERO_TIER_GAP + 12

    axis = (f'<line class="axis" x1="{X0}" y1="{HERO_Y0 - 6}" '
            f'x2="{X0}" y2="{h - 12}"/>')
    inner = '  ' + '\n  '.join([axis] + body)
    write(filename, document(title, subtitle, h, inner), h)


# ══════════════════════════════════════════════════════════════════════
#  DATA SOURCE — RESULTS.md is the single source of truth.
#
#  Numbers live in the RESULTS.md tables and nowhere else; this script reads
#  them so the charts and the tables can never disagree. Each chart is a
#  table wrapped in markers:
#
#    <!-- chart file="benchmark_x.svg" title="..." [note="Row Name=text"] -->
#    | Benchmark | SF (JIT) | ... |
#    |---|---:|...|
#    | CoreMark (score) | 44,958 | ... |
#    <!-- endchart -->
#
#  Column headers are matched to series via SERIES[].column, and a row label
#  is parsed as "Name (unit)". Bold (**best**) and footnote marks are
#  ignored. To add a chart, wrap another table -- no code change needed.
# ══════════════════════════════════════════════════════════════════════

RESULTS = 'RESULTS.md'
# Every image both READMEs embed lives in assets/ at the repo root — one place
# for rendered artifacts, so a reader following an image link never lands in
# the benchmark working directory. The script chdir's to its own directory,
# hence the relative hop. RESULTS.md still names charts by bare filename.
ASSETS = os.path.join('..', '..', 'assets')
CHART_RE = re.compile(r'<!--\s*chart\s+(?P<attrs>.*?)-->\n(?P<table>.*?)<!--\s*endchart\s*-->',
                      re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
SUB = 'Apple M4 · longer bar = better · each row scaled to its own best'


def parse_number(cell):
    return float(cell.replace('**', '').replace(',', '').strip())


def parse_charts(path):
    text = open(path, encoding='utf-8').read()
    charts = []
    for m in CHART_RE.finditer(text):
        attrs = dict(ATTR_RE.findall(m.group('attrs')))
        notes = {}
        if attrs.get('note'):
            row, _, msg = attrs['note'].partition('=')
            notes[row.strip()] = msg.strip()

        rows = [ln for ln in m.group('table').splitlines() if ln.strip().startswith('|')]
        header = [c.strip() for c in rows[0].strip().strip('|').split('|')]
        # header[0] is the benchmark-name column; the rest are runtimes
        keys = [BY_COLUMN.get(h) for h in header[1:]]
        missing = [h for h, k in zip(header[1:], keys) if k is None]
        if missing:
            raise SystemExit(f'{path}: unknown runtime column(s) {missing}; '
                             f'add them to SERIES')

        benches = []
        for line in rows[2:]:                      # skip the |---| separator
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            label = cells[0].replace('**', '').strip(' ¹²³*')
            name, _, unit = label.rpartition('(')
            name, unit = (name.strip(), unit.rstrip(')').strip()) if name else (label, '')
            data = {k: parse_number(c) for k, c in zip(keys, cells[1:])}
            bench = {'name': name, 'unit': unit, 'data': data}
            if name in notes:
                bench['note'] = notes[name]
            benches.append(bench)
        charts.append((attrs['title'], attrs['file'], benches))
    if not charts:
        raise SystemExit(f'{path}: no <!-- chart ... --> blocks found')
    return charts


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    charts = parse_charts(RESULTS)
    for title, filename, benches in charts:
        stem, ext = os.path.splitext(filename)
        for suffix, group_title, keys in GROUPS:
            make_svg(f'{title} — {group_title}', SUB, benches,
                     os.path.join(ASSETS, f'{stem}{suffix}{ext}'), keys)

    hero = next((b for _t, _f, benches in charts for b in benches
                 if b['name'] == HERO_BENCH), None)
    if hero is None:
        raise SystemExit(f'{RESULTS}: no "{HERO_BENCH}" row in any chart table; '
                         f'the README hero has nothing to draw')
    make_hero(hero, os.path.join(ASSETS, HERO_FILE),
              f'{HERO_BENCH} — Nano vs Wasmi vs Wasm3',
              f'Apple M4 MacBook Air · CoreMark {hero["unit"]} · higher is better')
