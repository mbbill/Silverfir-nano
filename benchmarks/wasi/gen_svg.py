#!/usr/bin/env python3
"""Generate benchmark SVG charts from data. Run after updating DATA below."""

import os

MW = 490  # max bar width
X0 = 175  # bar start x

def fmt(n):
    if n != int(n):
        s = f"{n:.3f}" if n < 100 else f"{n:.2f}"
    else:
        s = str(int(n))
    parts = s.split('.')
    p = parts[0]
    r = ''
    for i, c in enumerate(reversed(p)):
        if i and i % 3 == 0: r = ',' + r
        r = c + r
    parts[0] = r
    return '.'.join(parts)

def bar_w(val, ref, inv):
    return MW * ref / val if inv else MW * val / ref

def make_benchmark_section(idx, name, unit, d, inv, is_first):
    """Generate SVG snippet for one benchmark."""
    ref = min(d['sf'], d['cl'], d['v8']) if inv else max(d['sf'], d['cl'], d['v8'])
    ws = bar_w(d['sf'], ref, inv)
    wc = bar_w(d['cl'], ref, inv)
    wv = bar_w(d['v8'], ref, inv)
    y_base = 84 + idx * 130
    dur_base = 0.35 + idx * 0.04
    lines = []
    if not is_first:
        lines.append(f'  <line x1="20" y1="{y_base}" x2="680" y2="{y_base}" stroke="#eaeef2" stroke-width="1"/>')
    # label
    lines.append(f'  <text x="168" y="{y_base+20.0}" text-anchor="end" fill="#656d76" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="12" font-weight="600">{name}</text>')
    lines.append(f'  <text x="182" y="{y_base+20.0}" fill="#8b949e" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="10">{unit}</text>')
    # Silverfir row
    lines.append(f'  <rect x="4" y="{y_base+28}" width="692" height="32" rx="4" fill="#f3e8ff" opacity="0.5"/>')
    lines.append(f'  <text x="168" y="{y_base+48.5}" text-anchor="end" fill="#581c87" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="700">Silverfir</text>')
    lines.append(f'  <rect x="175" y="{y_base+30}" width="{ws:.1f}" height="28" rx="4" fill="url(#hlGrad)" filter="url(#glow)"><animate attributeName="width" from="0" to="{ws:.1f}" dur="{dur_base:.2f}s" fill="freeze"/></rect>')
    lines.append(f'  <text x="{X0+ws-8:.1f}" y="{y_base+49.0}" text-anchor="end" fill="#ffffff" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="700">{fmt(d["sf"])}</text>')
    # Cranelift row
    lines.append(f'  <text x="168" y="{y_base+79.5}" text-anchor="end" fill="#1f2328" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="400">Cranelift</text>')
    lines.append(f'  <rect x="175" y="{y_base+61}" width="{wc:.1f}" height="28" rx="4" fill="#e8873d" opacity="0.85"><animate attributeName="width" from="0" to="{wc:.1f}" dur="{dur_base+0.04:.2f}s" fill="freeze"/></rect>')
    lines.append(f'  <text x="{X0+wc-8:.1f}" y="{y_base+80.0}" text-anchor="end" fill="#ffffff" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="700">{fmt(d["cl"])}</text>')
    # V8 row
    lines.append(f'  <text x="168" y="{y_base+110.5}" text-anchor="end" fill="#1f2328" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="400">V8</text>')
    lines.append(f'  <rect x="175" y="{y_base+92}" width="{wv:.1f}" height="28" rx="4" fill="#4a9ede" opacity="0.85"><animate attributeName="width" from="0" to="{wv:.1f}" dur="{dur_base+0.08:.2f}s" fill="freeze"/></rect>')
    lines.append(f'  <text x="{X0+wv-8:.1f}" y="{y_base+111.0}" text-anchor="end" fill="#ffffff" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="13" font-weight="700">{fmt(d["v8"])}</text>')
    return '\n'.join(lines)

def make_svg(title, subtitle, benchmarks, filename):
    """Generate a complete SVG file."""
    n = len(benchmarks)
    h = 84 + n * 130 + 10
    body_parts = []
    for i, b in enumerate(benchmarks):
        body_parts.append(make_benchmark_section(i, b['name'], b['unit'], b['data'], b.get('inv', False), i == 0))
        body_parts.append('')

    body = '\n'.join(body_parts)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" role="img" viewBox="0 0 700 {h}" width="700" height="{h}">
  <title>{title}</title>
  <defs>
    <linearGradient id="hlGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#a855f7"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="700" height="{h}" rx="8" ry="8" fill="#ffffff" stroke="#d0d7de" stroke-width="1"/>
  <text x="350.0" y="32" text-anchor="middle" fill="#1f2328" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="18" font-weight="600">{title}</text>
  <text x="350.0" y="51" text-anchor="middle" fill="#656d76" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="12">{subtitle}</text>
  <rect x="150.0" y="64" width="12" height="12" rx="2" fill="url(#hlGrad)"/>
  <text x="167.0" y="74" fill="#581c87" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="11">Silverfir (micro-JIT)</text>
  <rect x="305.0" y="64" width="12" height="12" rx="2" fill="#e8873d" opacity="0.85"/>
  <text x="322.0" y="74" fill="#bf6a1f" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="11">Cranelift (wasmtime)</text>
  <rect x="460.0" y="64" width="12" height="12" rx="2" fill="#4a9ede" opacity="0.85"/>
  <text x="477.0" y="74" fill="#2b7cbf" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="11">V8 (Node.js 25.4)</text>
  <line x1="175" y1="84" x2="175" y2="{h-12}" stroke="#d0d7de" stroke-width="1"/>

{body}</svg>
'''
    with open(filename, 'w') as f:
        f.write(svg)
    print(f"Generated: {filename}")


# ══════════════════════════════════════════════════
#  UPDATE THESE VALUES
# ══════════════════════════════════════════════════

INTEGER_DATA = [
    {'name': 'CoreMark',       'unit': 'score', 'data': {'sf': 33770,   'cl': 14669,   'v8': 37869  }},
    {'name': 'SHA-256',        'unit': 'MB/s',  'data': {'sf': 269.10,  'cl': 249.26,  'v8': 201.20 }},
    {'name': 'bzip2',          'unit': 'MB/s',  'data': {'sf': 16.84,   'cl': 19.41,   'v8': 19.88  }},
    {'name': 'LZ4 compress',   'unit': 'MB/s',  'data': {'sf': 783.14,  'cl': 736.45,  'v8': 704.01 }},
    {'name': 'LZ4 decompress', 'unit': 'MB/s',  'data': {'sf': 3136.42, 'cl': 3455.15, 'v8': 2908.07}},
]

LUA_DATA = [
    {'name': 'lua / fib38',     'unit': 's',     'data': {'sf': 2.543, 'cl': 12.18, 'v8': 3.14}, 'inv': True},
    {'name': 'lua / sunfish',   'unit': 'score', 'data': {'sf': 5372,  'cl': 2896,  'v8': 11101}},
    {'name': 'lua / json_bench','unit': 'score', 'data': {'sf': 15249, 'cl': 9616,  'v8': 30179}},
]

FP_DATA = [
    {'name': 'mandelbrot', 'unit': 'ms', 'data': {'sf': 808, 'cl': 855,  'v8': 2035}, 'inv': True},
    {'name': 'c-ray (4K)', 'unit': 'ms', 'data': {'sf': 3629, 'cl': 2055, 'v8': 1947}, 'inv': True},
]

MEMORY_DATA = [
    {'name': 'STREAM Copy',  'unit': 'MB/s', 'data': {'sf': 44113, 'cl': 44124, 'v8': 39714}},
    {'name': 'STREAM Scale', 'unit': 'MB/s', 'data': {'sf': 49644, 'cl': 49692, 'v8': 18332}},
    {'name': 'STREAM Add',   'unit': 'MB/s', 'data': {'sf': 64223, 'cl': 48398, 'v8': 29989}},
    {'name': 'STREAM Triad', 'unit': 'MB/s', 'data': {'sf': 48408, 'cl': 47864, 'v8': 30869}},
]

# ══════════════════════════════════════════════════

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_svg('Integer / Control Flow — WebAssembly Runtimes',
             'Apple M4  \u00b7  longer bar = better',
             INTEGER_DATA, 'benchmark_integer.svg')
    make_svg('Lua Interpreter Benchmarks — WebAssembly Runtimes',
             'Apple M4  \u00b7  longer bar = better  \u00b7  time metrics inverted',
             LUA_DATA, 'benchmark_lua.svg')
    make_svg('Floating-Point Benchmarks — WebAssembly Runtimes',
             'Apple M4  \u00b7  longer bar = better (lower time)  \u00b7  bars inverted',
             FP_DATA, 'benchmark_fp.svg')
    make_svg('Memory-Bound Benchmarks (STREAM) — WebAssembly Runtimes',
             'Apple M4  \u00b7  longer bar = better',
             MEMORY_DATA, 'benchmark_memory.svg')
