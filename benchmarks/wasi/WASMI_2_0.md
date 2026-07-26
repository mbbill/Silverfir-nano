# wasmi 2.0.0-beta.7 vs wasmi 1.1.0

Measured 2026-07-26 on a **4-core Intel Xeon (Skylake-SP, 2.80 GHz) KVM
guest running Linux**, with the same `run_tests.py` harness that produces
[RESULTS.md](RESULTS.md).

**These numbers share no baseline with RESULTS.md.** That file is an Apple M4;
this one is a virtualised Xeon roughly 3× slower on this suite. Nothing here
may be pasted into the M4 tables, and no row here should be compared against a
row there. The only comparisons this file makes are between columns measured in
the same session on the same machine.

| Runtime | Version | How it was built |
|---|---|---|
| wasmi | 1.1.0 | `cargo install wasmi_cli --version 1.1.0 --locked` |
| wasmi | 2.0.0-beta.7 | `cargo install wasmi_cli --version 2.0.0-beta.7 --locked` |
| wasmi (control) | 2.0.0-beta.7 | same, `--no-default-features` + `portable-dispatch` |
| Silverfir (interpreter) | this tree @ `c062ff3`, `--interp` | `cargo build --release` |

There is **no stable wasmi 2.0**. As of 2026-07-26 the newest published version
is `2.0.0-beta.7` (2026-07-09); every 2.x release on crates.io is a beta. Treat
these as prerelease numbers.

## What changed in the CLI

wasmi 2.0 renames the binary from `wasmi_cli` to `wasmi` and moves execution
under a `run` sub-command, so the harness invocation changes:

```sh
# 1.1
python3 run_tests.py --exec "$HOME/.cargo/bin/wasmi_cli"     --cli-args "--dir . --"
# 2.0
python3 run_tests.py --exec "/path/to/bin/wasmi run"         --cli-args "--dir . --"
```

The `--` separator is still required, for the reason already noted in
RESULTS.md: without it wasmi's argument parser eats guest flags such as
`--memdb`. Both versions default to `--compilation-mode lazy-translation`, and
both were left on that default.

## Results

Mean of two interleaved rounds per runtime, 2 s target per benchmark, machine
otherwise idle. Every metric is a rate — higher is better.

| Benchmark | wasmi 1.1.0 | wasmi 2.0.0-beta.7 | 2.0 / 1.1 | SF interp | SF / 2.0 |
|---|---:|---:|---:|---:|---:|
| CoreMark (score) | 1,317 | 2,023 | **1.54×** | 3,768 | 1.86× |
| SHA-256 (MB/s) | 5.75 | 13.88 | **2.41×** | 16.26 | 1.17× |
| bzip2 (MB/s) | 0.93 | 1.67 | **1.80×** | 2.48 | 1.48× |
| LZ4 compress (MB/s) | 55.28 | 107.8 | **1.95×** | 150.2 | 1.39× |
| LZ4 decompress (MB/s) | 95.36 | 148.5 | **1.56×** | 263.2 | 1.77× |
| sqlite speedtest1 (size/s) | 1.95 | 2.67 | **1.37×** | 3.63 | 1.36× |
| lua / fib (fib20/s) | 71.00 | 139.5 | **1.96×** | 198.6 | 1.42× |
| lua / sunfish (score) | 266 | 399 | **1.50×** | 375 | 0.94× |
| lua / json (score) | 582 | 775 | **1.33×** | 878.5 | 1.13× |
| mandelbrot (Kpixel/s) | 42.02 | 66.55 | **1.58×** | 107.8 | 1.62× |
| c-ray (Kpixel/s) | 248.3 | 457.3 | **1.84×** | 525.0 | 1.15× |
| STREAM Copy (MB/s)¹ | 18,251 | 19,580 | 1.07× | 10,914 | 0.56× |
| STREAM Scale (MB/s) | 1,128 | 3,263 | **2.89×** | 3,368 | 1.03× |
| STREAM Add (MB/s) | 1,118 | 2,982 | **2.67×** | 3,537 | 1.19× |
| STREAM Triad (MB/s) | 992.2 | 2,726 | **2.75×** | 3,193 | 1.17× |

¹ The `memory.copy` row — see the footnote in
[RESULTS.md](RESULTS.md#memory-bound). It measures each engine's bulk-copy
path, not its dispatch, which is why it is the one row where 2.0 barely moves.

**wasmi 2.0-beta.7 is faster than 1.1.0 on all 15 metrics: median 1.80×, range
1.33× (lua/json) to 2.89× (STREAM Scale).** Dropping the `memory.copy` row, the
range is 1.33–2.89× and the median is 1.82×. This is the largest single-release
move any engine in this suite has made.

## Where the speedup comes from

wasmi 2.0's default features enable `auto-dispatch`, which its `build.rs`
resolves to **tail-call dispatch** whenever the target architecture is known to
lower LLVM sibling calls *and* the build is optimizing — x86_64 at `opt-level`
3 satisfies both, so a plain `cargo install` gets it. wasmi 1.1.0 has no
`build.rs` and no such backend.

To size that effect, a third binary was built from the identical source with
`portable-dispatch` (the loop-based backend) forced on. The only difference
between the two 2.0 columns is that one feature — `wasmi_cli` pins
`wasmi = { features = ["std"], default-features = false }`, so nothing else
moved.

| Benchmark | 2.0 auto (tail-call) | 2.0 portable | auto / portable | portable / 1.1 |
|---|---:|---:|---:|---:|
| CoreMark (score) | 2,023 | 1,028 | 1.97× | 0.78× |
| SHA-256 (MB/s) | 13.88 | 4.58 | 3.03× | 0.80× |
| bzip2 (MB/s) | 1.67 | 0.77 | 2.18× | 0.83× |
| LZ4 compress (MB/s) | 107.8 | 45.31 | 2.38× | 0.82× |
| LZ4 decompress (MB/s) | 148.5 | 67.69 | 2.19× | 0.71× |
| sqlite speedtest1 (size/s) | 2.67 | 1.62 | 1.65× | 0.83× |
| lua / fib (fib20/s) | 139.5 | 61.50 | 2.27× | 0.87× |
| lua / sunfish (score) | 399 | 206 | 1.94× | 0.77× |
| lua / json (score) | 775 | 485 | 1.60× | 0.83× |
| mandelbrot (Kpixel/s) | 66.55 | 26.56 | 2.51× | 0.63× |
| c-ray (Kpixel/s) | 457.3 | 152.0 | 3.01× | 0.61× |
| STREAM Copy (MB/s) | 19,580 | 18,378 | 1.07× | 1.01× |
| STREAM Scale (MB/s) | 3,263 | 1,058 | 3.08× | 0.94× |
| STREAM Add (MB/s) | 2,982 | 960.7 | 3.10× | 0.86× |
| STREAM Triad (MB/s) | 2,726 | 879.7 | 3.10× | 0.89× |

Two things fall out of that control, and only the first is a claim about
wasmi 2.0 as shipped:

1. **On this machine the whole 1.80× median comes from the dispatch backend.**
   Tail-call dispatch is a median 2.27× over portable dispatch on the same
   2.0 source.
2. **2.0 with portable dispatch is *slower* than 1.1 on 14 of 15 metrics**
   (median 0.83×, worst 0.61× on c-ray). Why the rest of the 2.0 work measures
   as a regression when the tail-call backend is taken away is not something
   this run answers — it needs a per-release bisect, not one control binary.
   It does mean the 2.0 speedup is not portable to targets `build.rs` excludes:
   powerpc, powerpc64, and wasm32/wasm64 without the tail-call proposal all
   fall back to the portable backend.

## What this means for Silverfir's interpreter numbers

The published claim — "the interpreter beats the best-in-class Wasm
interpreters by 1.16–1.73×", and RESULTS.md's "1.31–2.89× over wasmi" — is
against **wasmi 1.1.0**, on an M4. On this Xeon, against 1.1.0, the same shape
holds (1.41–3.22× excluding the `memory.copy` row). Against 2.0.0-beta.7 it
narrows sharply:

- Silverfir's interpreter still leads on 13 of 15 metrics, but the median
  margin is **1.19×**, not ~2×, and the range is 0.94–1.86×.
- It **loses two rows**: lua/sunfish (0.94×) and STREAM Copy (0.56×).
- The float pair that carried the biggest margin against wasmi 1.1 on M4
  (mandelbrot 2.89×, c-ray 2.88×) reads 1.62× and 1.15× here against 2.0.

The STREAM Copy row is worth a separate look: at 0.56× of wasmi 2.0 and 0.60×
of wasmi 1.1, this box does not reproduce the "joined the pack" reading
RESULTS.md records on M4 (0.97× of wasmi 1.1). The interpreter's handler is a
plain `copy_within` (`sf-nano-core/src/vm/interpreter/exec.rs:3075`), so it is
not an arch-specific handler split; beyond that, these runs do not say what the
difference is.

**None of this updates RESULTS.md.** Confirming the effect on the published
numbers needs a re-measure of the wasmi column on the M4, with 2.0.0-beta.7
alongside 1.1.0 — and a decision about whether a beta belongs in a table of
released runtimes at all.

## Reproducing

```sh
cargo install wasmi_cli --version 1.1.0         --root /tmp/wasmi-1.1 --locked
cargo install wasmi_cli --version 2.0.0-beta.7  --root /tmp/wasmi-2.0 --locked
cargo install wasmi_cli --version 2.0.0-beta.7  --root /tmp/wasmi-2.0-portable --locked \
  --no-default-features \
  --features "stable,run,wast,wasi,wat,validate,memory64,portable-dispatch"
cargo build --release --bin sf-nano-cli

cd benchmarks/wasi
python3 run_tests.py --exec "/tmp/wasmi-1.1/bin/wasmi_cli"          --cli-args "--dir . --"
python3 run_tests.py --exec "/tmp/wasmi-2.0/bin/wasmi run"          --cli-args "--dir . --"
python3 run_tests.py --exec "/tmp/wasmi-2.0-portable/bin/wasmi run" --cli-args "--dir . --"
python3 run_tests.py --interp
```

Toolchain: rustc 1.97.0 (the pinned toolchain), `cargo install` release profile
(`opt-level` 3).

## Raw rounds

Two rounds per runtime, run in the order 1.1 → 2.0 → 1.1 → 2.0, then the
interpreter twice, then the portable control twice.

| Benchmark | 1.1 r1 | 1.1 r2 | 2.0 r1 | 2.0 r2 | port r1 | port r2 | SF r1 | SF r2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoreMark | 1,303.8 | 1,330.8 | 2,014.8 | 2,030.6 | 1,052.9 | 1,002.5 | 3,773.9 | 3,762.9 |
| SHA-256 | 5.78 | 5.72 | 13.93 | 13.82 | 4.69 | 4.46 | 16.09 | 16.43 |
| bzip2 | 0.93 | 0.93 | 1.67 | 1.68 | 0.76 | 0.78 | 2.43 | 2.53 |
| LZ4 compress | 54.95 | 55.62 | 107.87 | 107.76 | 43.70 | 46.92 | 152.11 | 148.20 |
| LZ4 decompress | 94.85 | 95.88 | 148.48 | 148.57 | 64.86 | 70.52 | 266.18 | 260.13 |
| sqlite | 1.88 | 2.02 | 2.78 | 2.56 | 1.61 | 1.62 | 3.53 | 3.74 |
| lua / fib | 72.50 | 69.50 | 139.50 | 139.50 | 61.50 | 61.50 | 203.26 | 193.94 |
| lua / sunfish | 266 | 266 | 399 | 399 | 206 | 206 | 388 | 362 |
| lua / json | 582 | 582 | 775 | 775 | 485 | 485 | 881 | 876 |
| mandelbrot | 41.86 | 42.17 | 67.37 | 65.73 | 26.24 | 26.88 | 106.48 | 109.07 |
| c-ray | 250.38 | 246.23 | 463.78 | 450.73 | 151.02 | 152.93 | 516.91 | 533.02 |
| STREAM Copy | 17,143 | 19,359 | 19,432 | 19,729 | 18,764 | 17,992 | 10,924 | 10,903 |
| STREAM Scale | 1,091 | 1,166 | 3,391 | 3,135 | 1,056 | 1,061 | 3,349 | 3,387 |
| STREAM Add | 1,068 | 1,168 | 3,032 | 2,933 | 957.2 | 964.1 | 3,501 | 3,574 |
| STREAM Triad | 929.4 | 1,055 | 2,741 | 2,711 | 870.6 | 888.7 | 3,200 | 3,187 |

Round-to-round spread was ≤3% on 41 of the 60 pairs. The exceptions cluster in
two places, and both are known properties of the harness rather than of the
engines:

- **STREAM, sqlite and LZ4 on the slower columns** (up to 12.7% — wasmi 1.1's
  Triad). These are the rows whose batch sizing lands coarsest when one unit is
  expensive relative to the target.
- **The three Lua rows repeat exactly** for every wasmi column (266/266,
  399/399, 775/775, 206/206). Both wasmi versions return `0.0` from Lua's
  `os.clock()`, so `bench.lua` drops to its aligned `os.time()` path — the same
  quantisation RESULTS.md documents for wasmtime. Those rows therefore cannot
  estimate their own noise. Silverfir's interpreter does have a working
  `os.clock()` (4.4e-08 s) and its Lua rows do vary (sunfish 6.9%), so read its
  sunfish loss as "roughly a tie", not as a measured 0.94×.
