#!/usr/bin/env python3
"""Turn a cargo-criterion JSON stream from wasmi-benchmarks into an HTML report.

Usage:
    make_report.py <criterion-json> <environment-json> <out-html> [--standalone]

The criterion file is a stream of JSON objects (one per line-ish) as emitted by
`cargo criterion --message-format=json`.  Benchmark ids look like

    execute/<case>/<engine>/<param>
    startup/<case>/<engine>

Every timing is nanoseconds, lower is better.
"""

import json
import math
import sys
from collections import defaultdict

# ---------------------------------------------------------------- engine info
# "Kind" column of the wasmi-benchmarks README.
KIND = {
    "silverfir-nano.jit": "Optimizing JIT",
    "wasmtime.cranelift": "Optimizing JIT",
    "wasmer.cranelift": "Optimizing JIT",
    "wasmtime.winch": "Baseline JIT",
    "wasmer.singlepass": "Baseline JIT",
    "v8": "Multi-Tier JIT",
}
KIND_LABEL = {
    "Optimizing JIT": "optimizing JIT",
    "Baseline JIT": "baseline JIT",
    "Multi-Tier JIT": "multi-tier JIT",
    "Interpreter": "interpreter",
}
NANO = ("silverfir-nano.jit", "silverfir-nano.interpreter")
# Landmarks a reader already has a feel for: the leading production JIT and the
# browser engine, and the two interpreters Silverfir-nano is usually measured
# against.  Each gets its own hue, held across every chart it appears in —
# colour follows the engine, never its position in the ranking.
#
# Five hues in play, but never more than three in one chart: a JIT chart draws
# blue + a + b, an interpreter chart blue + c + d.  Both of those triples clear
# the all-pairs colour-vision gates in light and dark mode, which a single
# five-hue set cannot.
REFERENCE = {
    "wasmtime.cranelift": "a",
    "v8": "b",
    "wasm3.eager": "c",
    "wasmi-v2.eager.checked": "d",
}


def kind_of(engine):
    return KIND.get(engine, "Interpreter")


def is_jit(engine):
    return kind_of(engine) != "Interpreter"


# ---------------------------------------------------------------- data loading
def load_stream(path):
    """cargo-criterion emits concatenated JSON objects; parse them all."""
    text = open(path, encoding="utf-8").read()
    dec = json.JSONDecoder()
    idx, out = 0, []
    while idx < len(text):
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        obj, idx = dec.raw_decode(text, idx)
        out.append(obj)
    return out


def collect(records):
    """-> {category: {case_key: {engine: ns}}}, preserving first-seen order."""
    data = {"execute": defaultdict(dict), "startup": defaultdict(dict)}
    order = {"execute": [], "startup": []}
    engines = []
    for rec in records:
        if rec.get("reason") != "benchmark-complete":
            continue
        parts = rec["id"].split("/")
        if len(parts) < 3 or parts[0] not in data:
            continue
        cat, case, engine = parts[0], parts[1], parts[2]
        param = parts[3] if len(parts) > 3 else ""
        key = f"{case}/{param}" if param else case
        est = rec.get("typical") or rec.get("mean")
        if not est:
            continue
        data[cat][key][engine] = {
            "ns": est["estimate"],
            "lo": est.get("lower_bound", est["estimate"]),
            "hi": est.get("upper_bound", est["estimate"]),
            "case": case,
            "param": param,
        }
        if key not in order[cat]:
            order[cat].append(key)
        if engine not in engines:
            engines.append(engine)
    return data, order, engines


def geomean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def ranking(cases, order, engine_set):
    """Relative-to-best geomean over the cases every listed engine ran.

    Returns (rows, common_case_keys).  rows: [{engine, score, wins, rel{}}]
    sorted fastest first.
    """
    common = [k for k in order if all(e in cases[k] for e in engine_set)]
    rel = defaultdict(dict)
    wins = defaultdict(int)
    for key in common:
        best = min(cases[key][e]["ns"] for e in engine_set)
        for e in engine_set:
            rel[e][key] = cases[key][e]["ns"] / best
            if cases[key][e]["ns"] == best:
                wins[e] += 1
    rows = [
        {
            "engine": e,
            "score": geomean([rel[e][k] for k in common]),
            "wins": wins[e],
            "rel": rel[e],
        }
        for e in engine_set
    ]
    rows.sort(key=lambda r: r["score"])
    return rows, common


# ---------------------------------------------------------------- SVG helpers
def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_x(v):
    if v < 9.95:
        return f"{v:.2f}×"
    if v < 99.5:
        return f"{v:.1f}×"
    return f"{v:.0f}×"


def fmt_ns(ns):
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:,.1f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:,.2f} ms"
    return f"{ns / 1e9:,.2f} s"


def nice_ticks(vmax, count=4):
    """Round tick values covering [0, vmax]."""
    raw = vmax / count
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if step >= raw:
            break
    ticks, t = [], 0.0
    while t <= vmax * 1.0001:
        ticks.append(round(t, 6))
        t += step
    if not ticks or ticks[-1] < vmax:
        ticks.append(round(ticks[-1] + step if ticks else step, 6))
    return ticks



# ---------------------------------------------------------------- charts
def _legend(engines):
    keys = ['<span class="key"><i class="sw sw-nano"></i>Silverfir-nano</span>']
    for e in engines:
        if e in REFERENCE:
            keys.append(
                f'<span class="key"><i class="sw sw-ref-{REFERENCE[e]}"></i>'
                f"{esc(e)}</span>"
            )
    keys.append('<span class="key"><i class="sw sw-other"></i>Other engines</span>')
    return '<div class="legend">' + "".join(keys) + "</div>"


def _bars(rows, value_of, label_of, title, subtitle, note, chart_id, axis_title,
          tick_fmt, row_h=26, bar_h=15, pad_l=250, pad_r=132, plot_w=520,
          tag_of=None, ref=None, log=False, legend=True):
    """One horizontal bar per engine, linear axis, every bar labelled.

    Every chart on this page is this shape: all engines that ran the workload,
    in one plot, sorted fastest first.  That is the convention the upstream
    suite's own plots use, and it is the only layout that answers "how does
    Silverfir-nano compare" without the reader doing arithmetic.
    """
    pad_t = 40
    height = pad_t + row_h * len(rows) + 36
    width = pad_l + plot_w + pad_r
    vmax = max(value_of(r) for r in rows)
    bottom = pad_t + row_h * len(rows)
    if log:
        # A single workload can spread two decades or more between the fastest
        # JIT and the slowest interpreter.  On a linear axis every fast engine
        # collapses into the same 2px stub, so the decade axis is what makes the
        # top of the field readable at all; the value labels stay absolute.
        vmin = min(value_of(r) for r in rows)
        lo = 10 ** math.floor(math.log10(vmin))
        hi = 10 ** math.ceil(math.log10(vmax))
        span = math.log10(hi) - math.log10(lo)
        pos = lambda v: plot_w * (math.log10(v) - math.log10(lo)) / span
        ticks = [10 ** k for k in range(int(math.log10(lo)), int(math.log10(hi)) + 1)]
    else:
        ticks = nice_ticks(vmax)
        scale = plot_w / ticks[-1]
        pos = lambda v: v * scale

    out = [
        f'<figure class="chart" id="{chart_id}">',
        f'<figcaption><h3>{esc(title)}</h3><p class="sub">{esc(subtitle)}</p></figcaption>',
        (_legend([r["engine"] for r in rows]) if legend else ""),
        '<div class="svg-wrap">',
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{esc(title)}">',
    ]
    for t in ticks:
        x = pad_l + pos(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t - 10}" x2="{x:.1f}" y2="{bottom}"/>')
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">'
            f"{tick_fmt(t)}</text>"
        )
    if ref is not None and ref <= ticks[-1]:
        x = pad_l + pos(ref)
        out.append(f'<line class="axis-mid" x1="{x:.1f}" y1="{pad_t - 22}" x2="{x:.1f}" y2="{bottom}"/>')
    out.append(
        f'<text class="tick" x="{pad_l}" y="{pad_t - 20}" text-anchor="start">{esc(axis_title)}</text>'
    )
    for i, r in enumerate(rows):
        y = pad_t + i * row_h
        cy = y + (row_h - bar_h) / 2
        v = value_of(r)
        w = max(pos(v), 2.0)
        nano = r["engine"] in NANO
        ref = REFERENCE.get(r["engine"])
        cls = ("bar bar-nano" if nano
               else f"bar bar-ref-{ref}" if ref else "bar bar-other")
        name = esc(r["engine"])
        if tag_of:
            name += f'<tspan class="tag"> {esc(tag_of(r["engine"]))}</tspan>'
        lab_cls = "lab lab-nano" if nano else ("lab lab-ref" if ref else "lab")
        out.append(
            f'<text class="{lab_cls}" x="{pad_l - 12}" '
            f'y="{y + row_h / 2 + 4}" text-anchor="end">{name}</text>'
        )
        out.append(
            f'<rect class="{cls}" x="{pad_l}" y="{cy:.1f}" width="{w:.1f}" '
            f'height="{bar_h}" rx="4"/>'
            f'<rect class="{cls}" x="{pad_l}" y="{cy:.1f}" width="{min(4.0, w):.1f}" '
            f'height="{bar_h}"/>'
        )
        out.append(
            f'<text class="{"val val-nano" if nano or ref else "val"}" '
            f'x="{pad_l + w + 8:.1f}" y="{y + row_h / 2 + 4}">{esc(label_of(r))}</text>'
        )
        out.append(
            f'<rect class="hit" x="0" y="{y}" width="{width}" height="{row_h}">'
            f'<title>{esc(r.get("tip", ""))}</title></rect>'
        )
    out.append("</svg></div>")
    if note:
        out.append(f'<p class="note">{esc(note)}</p>')
    out.append("</figure>")
    return "\n".join(out)


def summary_chart(rows, title, subtitle, note, chart_id, n_cases, tags=True,
                  cap=100):
    """One class on one axis: geomean time relative to the class leader.

    Scoring each engine against the winner *of each case* is the right way to
    average across workloads whose absolute times differ by decades, but plotted
    raw it has no anchor: unless one engine wins every case, nothing reads
    1.00×.  Dividing by the leader's score fixes that and costs nothing —
    the per-case baseline cancels exactly, so each bar is the pairwise
    geometric mean against the leader over the same cases.

    `tags` prints each engine's kind after its name.  That earns its place when
    the chart mixes kinds (the JIT chart holds optimizing, baseline and
    multi-tier compilers; the startup chart holds everything); inside a chart
    that is one kind throughout it is the same word on every row.
    """
    leader = rows[0]
    for r in rows:
        r["rel"] = r["score"] / leader["score"]
        r["tip"] = (
            f'{r["engine"]} — {KIND_LABEL[kind_of(r["engine"])]}\n'
            + (f'class leader, fastest on {r["wins"]} of {n_cases} cases'
               if r is leader else
               f'{fmt_x(r["rel"])} the time of {leader["engine"]}, geomean over '
               f'{n_cases} cases\nfastest on {r["wins"]} of them')
        )
    # A single runaway engine flattens everyone else on a linear axis
    # (wasmtime.pulley translates through Cranelift, so its startup sits with
    # the compilers even though it executes as an interpreter).  Move anything
    # past `cap` times the leader into the note rather than switching the axis:
    # a log axis keeps the outlier on screen but stops the remaining bars from
    # showing their real gaps, which is the comparison the chart is for.
    shown = [r for r in rows if r["rel"] <= cap]
    dropped = [r for r in rows if r not in shown]
    if dropped:
        note = (note + " " if note else "") + "Off the scale, not charted: " + ", ".join(
            f'{d["engine"]} ({fmt_x(d["rel"])})' for d in dropped
        ) + ". Measured times are in the table below."
    return _bars(
        shown,
        value_of=lambda r: r["rel"],
        label_of=lambda r: fmt_x(r["rel"]),
        title=title, subtitle=subtitle, note=note, chart_id=chart_id,
        axis_title=f"× {leader['engine']} — lower is better",
        tick_fmt=lambda t: f"{t:g}×",
        tag_of=(lambda e: KIND_LABEL[kind_of(e)]) if tags else None,
        ref=1.0,
        # the tag rides after the engine name, and the longest of those
        # ("wasmi-v2.lazy-translation.checked" plus "interpreter") needs the room
        pad_l=350 if tags else 290, plot_w=440 if tags else 500,
    )


def case_chart(cases, key, title, chart_id, engines=None, note=""):
    """One workload, one engine class, measured time."""
    entries = sorted(
        ((e, d) for e, d in cases[key].items() if engines is None or e in engines),
        key=lambda kv: kv[1]["ns"],
    )
    if not entries:
        return ""
    best = entries[0][1]["ns"]
    rows = [
        {
            "engine": e,
            "ns": d["ns"],
            "tip": f'{e} — {KIND_LABEL[kind_of(e)]}\n{fmt_ns(d["ns"])}\n'
                   f'{fmt_x(d["ns"] / best)} the fastest engine here',
        }
        for e, d in entries
    ]
    unit = "ms" if rows[-1]["ns"] >= 1e6 else "µs"
    div = 1e6 if unit == "ms" else 1e3
    return _bars(
        rows,
        value_of=lambda r: r["ns"] / div,
        label_of=lambda r: fmt_ns(r["ns"]),
        title=title, subtitle=f"{len(rows)} engines", note=note, chart_id=chart_id,
        axis_title=f"time in {unit} — lower is better",
        tick_fmt=lambda t: f"{t:g}",
        # "silverfir-nano.interpreter" is 26 mono characters; the gutter has to
        # clear it or the name loses its first letters to the viewBox edge
        row_h=24, bar_h=13, pad_l=212, plot_w=236, pad_r=112, legend=False,
    )
def table(cases, order, engine_set, title, chart_id):
    common = [k for k in order if any(e in cases[k] for e in engine_set)]
    head = "".join(f"<th>{esc(k.split('/')[0])}</th>" for k in common)
    body = []
    for e in engine_set:
        best_marks = []
        for k in common:
            if e not in cases[k]:
                best_marks.append("<td class='na'>—</td>")
                continue
            ns = cases[k][e]["ns"]
            best = min(cases[k][x]["ns"] for x in engine_set if x in cases[k])
            cls = " class='best'" if ns == best else ""
            best_marks.append(f"<td{cls}>{fmt_ns(ns)}</td>")
        name_cls = " class='nano'" if e in NANO else ""
        body.append(
            f"<tr><th scope='row'{name_cls}>{esc(e)}</th>" + "".join(best_marks) + "</tr>"
        )
    return (
        f'<details class="tbl" id="{chart_id}"><summary>{esc(title)} — full table of measured times</summary>'
        f'<div class="tbl-wrap"><table><thead><tr><th scope="col">Engine</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></details>"
    )


CSS = """
:root { color-scheme: light dark; }
.viz-root {
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --accent:#2a78d6; --ref-a:#eb6834; --ref-b:#1baf7a; --ref-c:#eda100;
  --ref-d:#e87ba4; --other:#c3c2b7;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --accent:#3987e5; --ref-a:#d95926; --ref-b:#199e70; --ref-c:#c98500;
  --ref-d:#d55181; --other:#4a4a46;
  }
}
:root[data-theme="dark"] .viz-root {
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --accent:#3987e5; --ref-a:#d95926; --ref-b:#199e70; --ref-c:#c98500;
  --ref-d:#d55181; --other:#4a4a46;
}
.viz-root {
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  background:var(--plane); color:var(--text-primary);
  font-family:var(--sans);
  line-height:1.6; padding:40px 24px 72px; margin:0;
}
.wrap { max-width:1080px; margin:0 auto; }
header.top { margin-bottom:32px; }
h1 { font-size:30px; line-height:1.25; margin:0 0 8px; letter-spacing:-0.01em;
  text-wrap:balance; }
.dek { color:var(--text-secondary); font-size:16px; margin:0 0 20px; max-width:62ch; }
.meta { display:flex; flex-wrap:wrap; gap:8px 10px; margin:0; padding:0; list-style:none; }
.meta li { font-family:var(--mono); font-size:11.5px; color:var(--text-secondary);
  background:var(--surface-1); border:1px solid var(--border); border-radius:999px;
  padding:5px 11px; }
.meta b { font-family:var(--sans); font-weight:600; color:var(--text-primary);
  letter-spacing:0.02em; }
h2 { font-size:20px; margin:44px 0 6px; letter-spacing:-0.01em; text-wrap:balance; }
.lede { color:var(--text-secondary); margin:0 0 18px; max-width:70ch; font-size:14.5px; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0 8px; }
.kpi { background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px 18px; }
.kpi .k { font-size:12.5px; color:var(--text-secondary); margin:0 0 6px; }
.kpi .v { font-size:32px; font-weight:600; margin:0; letter-spacing:-0.02em; }
.kpi .v small { font-size:15px; font-weight:500; color:var(--text-secondary); margin-left:4px; }
.kpi .d { font-size:12.5px; color:var(--muted); margin:4px 0 0; }
.standings { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px;
  margin:22px 0 8px; }
.standing { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  padding:20px 22px; }
.standing .k { font-size:12.5px; color:var(--text-secondary); margin:0 0 2px;
  text-transform:uppercase; letter-spacing:0.06em; }
.standing .who { font-family:var(--mono); font-size:13px; font-weight:500;
  margin:0 0 10px; color:var(--text-primary); }
.standing .v { font-size:50px; font-weight:600; letter-spacing:-0.03em; margin:0; line-height:1; }
.standing .v small { font-size:17px; font-weight:500; color:var(--text-secondary); margin-left:6px;
  letter-spacing:0; }
.standing .d { color:var(--text-secondary); margin:10px 0 0; font-size:13.5px; }
.standing .d b { color:var(--text-primary); font-weight:600; }
.chart { background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
  margin:18px 0 0; padding:18px 20px 14px; }
figcaption h3 { font-size:15.5px; margin:0 0 2px; }
figcaption .sub { font-size:13px; color:var(--text-secondary); margin:0 0 10px; }
.legend { display:flex; gap:16px; align-items:center; margin:0 0 6px; font-size:12.5px;
  color:var(--text-secondary); flex-wrap:wrap; }
.key { display:inline-flex; align-items:center; gap:6px; }
.sw { width:11px; height:11px; border-radius:3px; display:inline-block; }
.sw-nano { background:var(--accent); } .sw-other { background:var(--other); }
.sw-ref-a { background:var(--ref-a); } .sw-ref-b { background:var(--ref-b); }
.sw-ref-c { background:var(--ref-c); } .sw-ref-d { background:var(--ref-d); }
.axis-mid { stroke:var(--axis); stroke-width:1; }
.svg-wrap { overflow-x:auto; }
svg { display:block; max-width:100%; height:auto; }
.grid { stroke:var(--grid); stroke-width:1; }
.tick { fill:var(--muted); font-family:var(--mono); font-size:10.5px; }
.lab { fill:var(--text-secondary); font-family:var(--mono); font-size:11.5px; }
.lab-nano { fill:var(--text-primary); font-weight:650; }
.lab-ref { fill:var(--text-primary); }
.val { fill:var(--text-secondary); font-family:var(--mono); font-size:11.5px;
  /* value labels sit past the bar end, where they cross gridlines; a halo in
     the surface colour keeps them legible without hiding the grid */
  paint-order:stroke; stroke:var(--surface-1); stroke-width:3px;
  stroke-linejoin:round; }
.val-nano { fill:var(--text-primary); font-weight:650; }
.bar-nano { fill:var(--accent); } .bar-other { fill:var(--other); }
.bar-ref-a { fill:var(--ref-a); } .bar-ref-b { fill:var(--ref-b); }
.bar-ref-c { fill:var(--ref-c); } .bar-ref-d { fill:var(--ref-d); }
.hit { fill:transparent; }
.chart .note, .note { font-size:12px; color:var(--muted); margin:10px 0 0; }
.case { margin:22px 0 0; }
.case-h { font-size:14px; margin:0 0 8px; font-family:var(--mono); font-weight:600;
  display:flex; align-items:baseline; gap:10px; }
.case-in { font-family:var(--sans); font-size:12px; font-weight:400; color:var(--muted); }
.case-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
  gap:12px; align-items:start; }
.case-grid .chart { margin:0; }
.toc { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 6px; }
.toc a { font-family:var(--mono); font-size:11.5px; color:var(--text-secondary);
  text-decoration:none;
  background:var(--surface-1); border:1px solid var(--border); border-radius:999px;
  padding:3px 10px; font-variant-numeric:tabular-nums; }
.toc a:hover { color:var(--text-primary); border-color:var(--axis); }
.toc a:focus-visible, .tbl summary:focus-visible {
  outline:2px solid var(--accent); outline-offset:2px; }
.tag { fill:var(--muted); font-family:var(--sans); font-size:10.5px; }
.tbl { margin:14px 0 0; border:1px solid var(--border); border-radius:12px;
  background:var(--surface-1); padding:10px 16px; }
.tbl summary { cursor:pointer; font-size:13.5px; color:var(--text-secondary); }
.tbl-wrap { overflow-x:auto; margin-top:12px; }
table { border-collapse:collapse; font-family:var(--mono); font-size:11.5px;
  font-variant-numeric:tabular-nums; }
th, td { padding:5px 10px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--grid); }
thead th { color:var(--muted); font-weight:500; text-align:right; }
tbody th { text-align:left; font-weight:500; color:var(--text-secondary); }
tbody th.nano { color:var(--text-primary); font-weight:650; }
td.best { color:var(--text-primary); font-weight:650; }
td.na { color:var(--muted); }
.foot { margin-top:44px; font-size:12.5px; color:var(--muted); border-top:1px solid var(--border);
  padding-top:16px; }
.foot code { font-size:12px; background:var(--surface-1); border:1px solid var(--border);
  border-radius:4px; padding:1px 5px; }
"""




def main():
    crit_path, meta_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # A standalone page (repo copy, opened straight from disk) needs its own
    # doctype/charset; the artifact publisher supplies that skeleton itself.
    standalone = "--standalone" in sys.argv[4:]
    meta = json.load(open(meta_path, encoding="utf-8"))
    data, order, _ = collect(load_stream(crit_path))

    ex, st = data["execute"], data["startup"]
    ex_order, st_order = order["execute"], order["startup"]
    ex_engines = sorted({e for k in ex for e in ex[k]})
    st_engines = sorted({e for k in st for e in st[k]})

    jit_ex = [e for e in ex_engines if is_jit(e)]
    int_ex = [e for e in ex_engines if not is_jit(e)]
    jit_st = [e for e in st_engines if is_jit(e)]
    int_st = [e for e in st_engines if not is_jit(e)]

    jit_rows, jit_common = ranking(ex, ex_order, jit_ex)
    int_rows, int_common = ranking(ex, ex_order, int_ex)
    st_jit_rows, st_jit_common = ranking(st, st_order, jit_st)
    st_int_rows, st_int_common = ranking(st, st_order, int_st)

    def rank_of(rows, engine):
        for i, r in enumerate(rows):
            if r["engine"] == engine:
                return i + 1, r
        return None, None

    jit_rank, jit_nano = rank_of(jit_rows, "silverfir-nano.jit")
    int_rank, int_nano = rank_of(int_rows, "silverfir-nano.interpreter")
    st_jit_rank, st_jit_nano = rank_of(st_jit_rows, "silverfir-nano.jit")
    st_int_rank, st_int_nano = rank_of(st_int_rows, "silverfir-nano.interpreter")

    P = []
    if standalone:
        P.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
        P.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    P.append(f"<title>{esc(meta['title'])}</title>")
    P.append(f"<style>{CSS}</style>")
    if standalone:
        P.append("</head><body>")
    P.append('<div class="viz-root"><div class="wrap">')
    P.append('<header class="top">')
    P.append(f"<h1>{esc(meta['title'])}</h1>")
    P.append(f"<p class=\"dek\">{esc(meta['dek'])}</p>")
    P.append("<ul class='meta'>")
    for k, v in meta["facts"].items():
        P.append(f"<li><b>{esc(k)}</b> {esc(v)}</li>")
    P.append("</ul></header>")

    # ---- where Silverfir-nano lands in each class
    def standing(kind, engine, rank, rows, common):
        """Rank card. Gaps are stated against the neighbour the rank implies:
        the runner-up if Silverfir-nano leads, the leader if it does not."""
        total = len(rows)
        row = rows[rank - 1]
        if rank == 1:
            other = rows[1]
            gap = (
                f'The next engine, {esc(other["engine"])}, averages '
                f'<b>{fmt_x(other["score"] / row["score"])}</b> its time.'
            )
        else:
            other = rows[0]
            gap = (
                f'{esc(other["engine"])} leads the class, averaging '
                f'<b>{fmt_x(row["score"] / other["score"])}</b> faster.'
            )
        return (
            f'<div class="standing"><p class="k">{esc(kind)}</p>'
            f'<p class="who">{esc(engine)}</p>'
            f'<p class="v">#{rank}<small>of {total}</small></p>'
            f'<p class="d">Fastest on <b>{row["wins"]}</b> of {common} execution '
            f'cases. {gap}</p></div>'
        )

    P.append('<div class="standings">')
    P.append(standing("Optimizing JIT class", "silverfir-nano.jit", jit_rank,
                      jit_rows, len(jit_common)))
    P.append(standing("Interpreter class", "silverfir-nano.interpreter", int_rank,
                      int_rows, len(int_common)))
    P.append("</div>")

    kpis = [
        ("Startup rank, JIT class", f"{st_jit_rank}<small>/{len(jit_st)}</small>",
         f"{fmt_x(st_jit_nano['score'] / st_jit_rows[0]['score'])} the time of "
         f"{st_jit_rows[0]['engine']}, the fastest JIT to start"),
        ("Startup rank, interpreter class", f"{st_int_rank}<small>/{len(int_st)}</small>",
         f"{fmt_x(st_int_nano['score'] / st_int_rows[0]['score'])} the time of "
         f"{st_int_rows[0]['engine']}"),
        ("Engines compared", f"{len(ex_engines)}<small> engines</small>",
         f"{len(ex_order)} execution cases, {len(st_order)} startup cases"),
    ]
    P.append('<div class="kpis">')
    for k, v, d in kpis:
        P.append(f'<div class="kpi"><p class="k">{esc(k)}</p><p class="v">{v}</p>'
                 f'<p class="d">{esc(d)}</p></div>')
    P.append("</div>")

    # ---- 01 execution, everything on one axis
    P.append('<h2>Execution — by engine class</h2>')
    P.append(
        '<p class="lede">The execution benchmarks time calls into an already '
        'instantiated module; compilation and instantiation stay outside the timed '
        'loop. Absolute times differ by orders of magnitude between workloads, so a '
        'cross-workload summary has to be relative: each engine is scored against '
        'whichever engine won that case, those per-case ratios are combined with a '
        'geometric mean, and the result is divided by the fastest engine in the '
        'class — so the leader reads 1.00× and every other bar is how many times '
        'longer that engine takes on average. JITs and interpreters get their own '
        'chart and their own leader: the two classes are two orders of magnitude '
        'apart, and on one axis every JIT collapses into the same stub. The '
        'per-case charts below carry the measured times.</p>'
    )
    P.append(
        summary_chart(
            jit_rows,
            "JIT execution speed",
            f"Geometric mean over the {len(jit_common)} cases every JIT ran",
            "Each bar is the geometric mean of the per-case time ratio against "
              "the class leader, over the cases both engines ran.",
            "chart-exec-jit",
            len(jit_common),
        )
    )
    P.append(
        summary_chart(
            int_rows,
            "Interpreter execution speed",
            f"Geometric mean over the {len(int_common)} cases every interpreter ran",
            "Same construction as the JIT chart, inside the interpreter class — the "
              "two baselines are different engines.",
            "chart-exec-int",
            len(int_common),
            tags=False,
        )
    )

    # ---- 02 execution, case by case
    P.append('<h2>Execution — case by case</h2>')
    P.append(
        '<p class="lede">Measured time per workload, JITs on the left and '
        'interpreters on the right, each on its own linear axis. This is where a '
        'summary can mislead: engines trade places from one workload to the next. '
        'The colours are fixed across every chart on this page: Silverfir-nano is '
        'blue, and each of the four reference engines keeps a hue of its own — '
        'wasmtime.cranelift orange and v8 green among the JITs, wasm3.eager amber '
        'and wasmi-v2.eager.checked pink among the interpreters.</p>'
    )
    P.append('<nav class="toc">' + " ".join(
        f'<a href="#case-{esc(k.split("/")[0])}">{esc(k.split("/")[0])}</a>'
        for k in ex_order
    ) + "</nav>")
    for k in ex_order:
        case, _, param = k.partition("/")
        P.append(f'<section class="case" id="case-{esc(case)}">')
        P.append(
            f'<h3 class="case-h">execute/{esc(case)}'
            + (f'<span class="case-in">input {esc(param)}</span>' if param else "")
            + "</h3>"
        )
        P.append('<div class="case-grid">')
        P.append(case_chart(ex, k, "JIT", f"case-{case}-jit", engines=set(jit_ex)))
        P.append(case_chart(ex, k, "Interpreter", f"case-{case}-int", engines=set(int_ex)))
        P.append("</div></section>")
    P.append(table(ex, ex_order, ex_engines, "Execution benchmarks", "tbl-exec"))

    # ---- 03 startup
    P.append('<h2>Startup — by engine class</h2>')
    P.append(
        '<p class="lede">The startup benchmarks time '
        '<code>rt.instantiate(&amp;wasm)</code>: parsing, validation, compilation, '
        'linking and instantiation all count. Silverfir-nano runs with parallel '
        'compilation disabled, so these are serial numbers. Interpreters have a '
        'structural advantage here — they emit little or no machine code — so each '
        'class gets its own chart and its own baseline, as above.</p>'
    )
    P.append(
        summary_chart(
            st_jit_rows,
            "JIT startup speed",
            f"Geometric mean over the {len(st_jit_common)} startup cases every JIT ran",
            "V8 leads because it compiles lazily, not because it compiles the same "
              "functions faster than the others.",
            "chart-startup-jit",
            len(st_jit_common),
        )
    )
    P.append(
        summary_chart(
            st_int_rows,
            "Interpreter startup speed",
            f"Geometric mean over the {len(st_int_common)} startup cases every "
            "interpreter ran",
            "The .lazy* configurations defer function-body translation to first "
            "call, so the top of this chart is not doing the same work as the "
            "eager engines below it.",
            "chart-startup-int",
            len(st_int_common),
            tags=False,
        )
    )
    P.append(table(st, st_order, st_engines, "Startup benchmarks", "tbl-startup"))

    P.append('<h2>How to read this</h2>')
    P.append(f'<p class="lede">{meta["method"]}</p>')
    P.append(f'<p class="foot">{meta["foot"]}</p>')
    P.append("</div></div>")
    if standalone:
        P.append("</body></html>")
    open(out_path, "w", encoding="utf-8").write("\n".join(P))
    print(f"wrote {out_path}")
    print(f"engines: {len(ex_engines)} execute / {len(st_engines)} startup")
    print(f"cases: {len(ex_order)} execute / {len(st_order)} startup")
    print(f"nano.jit exec rank {jit_rank}/{len(jit_ex)} geomean {jit_nano['score']:.3f}")
    print(f"nano.interp exec rank {int_rank}/{len(int_ex)} geomean {int_nano['score']:.3f}")


if __name__ == "__main__":
    main()
