# Cache Pressure Analysis

This document records the pressure characteristics of the old (fixed-bank) and
new (unified per-block) register allocation models, measured across 9 WASI
benchmarks.  The goal is to establish the facts that any future algorithm must
satisfy before we design that algorithm.

## Register budgets

### Old model (commit 40cfc0e)

Fixed two-bank split per register class.  The split is decided at compile time
and never changes within a function.

ARM64:

| Bank             | Registers                           | Count |
|------------------|-------------------------------------|-------|
| GP local cache   | x23-x28 (callee), x9-x15 (caller)  | 13    |
| GP transient     | x3-x8, x0-x2                       | 9     |
| FP local cache   | d8-d15 (callee), d21-d31 (caller)   | 19    |
| FP transient     | d3-d7, d16-d20                      | 10    |
| GP scratch       | x16, x17                            | 2     |
| Fixed            | x19 (ctx), x20 (fp), x21/x22 (mem) | 4     |

The function-level planner picks up to 13 GP locals to cache for the entire
function.  The 9 GP transient registers hold the top-of-stack operand values.
The split is static: even if a block only uses 2 transient values, the
remaining 7 sit idle — they cannot be used for local caching.

### New model (current microjit branch)

Unified dynamic bank per register class.

ARM64:

| Bank        | Count |
|-------------|-------|
| GP dynamic  | 22    |
| FP dynamic  | 29    |

All 22 GP registers are shared between cached locals and transient stack
values.  The per-block joint planner decides how many go to each purpose at
every block boundary.

## How the old transient bank actually works

The transient bank is a **sliding window** over the operand stack.  The top N
values live in registers; everything below is in the frame.

Critically, a block does not need transient registers for *all* live stack
values — only for the ones it actually touches.  Consider:

- A block enters with 12 values on the operand stack.
- The block only pops the top 2, computes, and pushes 1 result.
- The transient bank only needs 3 registers (2 consumed + 1 produced).
- The bottom 10 values sit untouched in the frame.  No spill, no fill.

Spill and fill only happen when a block's computation reaches deeper into the
stack than the transient bank can hold — and that is rare.

## Measured data

### SSA-IR op counts: old vs new

All numbers are from ARM64 release builds of the old (40cfc0e) and new
(microjit branch, with sink_plan) systems.  Each benchmark was compiled once;
the table shows total SSA-IR ops across all functions in the module.

```
benchmark     old_code  old_s  old_f  old_sf  new_code  new_e  new_d  new_ed  new_s  new_f  new_sf   delta
-------------------------------------------------------------------------------------------------------------------
coremark         83056    447     96     543    143212   2843   3965    6808    453    102     555   +6820
bzip2           174144    802    242    1044    246516   5338   7138   12476    834    274    1108  +12540
lz4             102220    470    116     586    143460   2750   3660    6410    507    153     660   +6484
sha256           90788    454    114     568    121484   2169   3067    5236    487    147     634   +5302
c-ray           168452    855    242    1097    262260   5555   7148   12703    869    256    1125  +12731
mandelbrot      141228    641    140     781    232076   5052   6891   11943    636    135     771  +11933
stream           90452    444     67     511    144092   3127   4327    7454    452     75     527   +7470
lua            1500800  11225   2486   13711   1950344  32801  62635   95436  11733   2994   14727  +96452
sqlite         4616904  31978   6684   38662   6150104 121780 193673  315453  32757   7463   40220 +317011
```

Column definitions:

- `old_code` / `new_code`: native binary size in bytes.
- `old_s` / `old_f` / `old_sf`: spill / fill / total in old system (transient
  register overflow cost).
- `new_e` / `new_d` / `new_ed`: ensure_cache / drop_cache / total in new
  system (block boundary fixup cost).
- `new_s` / `new_f` / `new_sf`: spill / fill / total in new system.
- `delta`: `(new_ed + new_sf) - old_sf`.  Net extra SSA ops the new system
  introduces.

### Key observations

1. **Old spill+fill is small.**  Even sqlite (the largest module at 4.6 MB of
   code) only has 38,662 spill+fill ops.  The 9-register transient bank is
   rarely a bottleneck.

2. **New spill+fill is almost identical to old.**  The unified budget did not
   reduce spill/fill.  The transient bank was never the problem.

3. **The entire cost gap is ensure+drop.**  The `delta` column is almost
   exactly `new_ed` in every case.  The new system adds thousands to hundreds
   of thousands of boundary fixup ops while gaining nothing on the spill/fill
   side.

4. **Code size blowup is 1.3x-1.7x** across all benchmarks, entirely from the
   boundary fixup code.

### Per-function pressure profile (CoreMark)

```
  func locals  gp  fp  blks    code ensure  drop fix%  c_avg c_max  t_avg t_max  pk_avg pk_max  old_c old_t
    15     66  65   1   776   54472   1053  1492  34%   11.2    21    1.9     6    13.1     26    13!    6
    29     55  55   0   343   23272    591   825  48%   12.1    21    1.6     7    13.7     26    13!    7
    35     58  56   2   240   17412    505   735  45%    9.3    21    2.0     9    11.3     25    13!    9
     8     23  23   0   125    8844    245   360  32%   11.1    16    2.0     7    13.0     20    13!    7
     6     19  19   0   138    7012    177   238  42%    5.9    11    1.5     8     7.4     19    13!    8
    23     10  10   0    55    3876     59    51  27%    5.5     8    1.8     5     7.2     12    10     5
     9     10  10   0    25    3112     24    66  32%    4.4     6    1.6     4     6.0      9    10     4
```

- `c_avg` / `c_max`: cached-local count per block (average / maximum).
- `t_avg` / `t_max`: peak live transient SSA values per block.
- `pk_avg` / `pk_max`: combined (cache + transient).
- `old_c`: locals the old system would cache (capped at 13). `!` = exceeds.
- `old_t`: peak transient. `!` = would exceed old 9-register transient bank.

Even the function with 66 locals (func 15) only needs 6 peak transient.  The
old system's 9-register transient bank was never stressed.

### Old-system pressure check across benchmarks

```
benchmark    funcs  gp>13   t>9  both max_gp  max_t  max_pk headroom
----------------------------------------------------------------------------------------------------
coremark        33      5     0     0     65      9      26       +0
bzip2           72      6     1     1    194     21      40      -12
lz4             49      5     0     0     59      7      26       +2
sha256          47      3     0     0     59      7      26       +2
c-ray           79      8     3     0    176     12      31       -3
mandelbrot      57     11     1     0    174     11      29       -2
stream          31      1     1     0    174     12      28       -3
lua            770    144    10     2    199     14      30       -5
sqlite        1412    263    18    13    281     13      33       -4
```

- `gp>13`: functions with more GP locals than the old cache budget.
- `t>9`: functions with peak transient exceeding old transient budget.
- `both`: functions exceeding both budgets simultaneously.
- `headroom`: `9 - max_t`.  Positive = wasted transient registers in old model.

The benchmarks split into two groups:

1. **Low transient pressure** (coremark, lz4, sha256): the old 9-register
   transient bank is never exceeded (`headroom >= 0`).  The old system's static
   split is optimal for these.

2. **Moderate transient pressure** (bzip2, c-ray, mandelbrot, stream, lua,
   sqlite): a few functions exceed 9 transient (`headroom < 0`).  The old
   system spills on those functions.  A unified budget could theoretically
   help — but only if the boundary fixup cost is lower than the saved
   spills, and current data shows it isn't.

### Block composition (new system)

```
benchmark     blocks  pure_logic  boundary_only  mixed
------------------------------------------------------
coremark       1,966         752            555    659
bzip2          3,262       1,103            778  1,381
lz4            2,055         737            417    901
sha256         1,613         586            320    707
c-ray          3,706       1,279            936  1,491
mandelbrot     3,008       1,004            726  1,278
stream         1,588         565            485    538
lua           26,396       9,988          6,958  9,450
sqlite        77,172      29,195         21,484 26,493
```

- **Boundary-only blocks** have no computation — they exist solely to run
  ensure/drop ops for cache set transitions.  They are 20-31% of all blocks
  across benchmarks.

### Cache stability (new system, CoreMark)

```
  func locals  blks unique  ratio  union  isect  empty
    15     66   776    214  0.28x     66      0      8
    29     55   343     67  0.20x     55      0      1
    35     58   240     86  0.36x     58      0      1
     8     23   125     39  0.31x     23      0      1
     6     19   138     44  0.32x     19      0      1
```

- `unique`: distinct cached-slot sets across blocks.
- `ratio`: `unique / blocks` (lower = more sharing, higher = more churn).
- `union`: total distinct locals cached anywhere in the function.
- `isect`: locals cached in every non-empty block (the stable core).

`isect = 0` for all large functions means no local is cached in every block.
The cache set is completely unstable — it churns at almost every block
boundary.  `union = local_count` means every local gets cached somewhere, just
never all at once.

## The core problem

The old system's advantage is simplicity: pick up to 13 locals at function
entry, never change the set, zero boundary cost.  The 9-register transient
bank handles stack values with minimal spill/fill because most blocks only
touch the top 2-4 stack values.

The new system's advantage is flexibility: the unified 22-register budget can
allocate registers dynamically between locals and stack based on per-block
needs.  This should win when local pressure is high and stack pressure is low
(give more registers to locals) or when stack pressure is high and local
pressure is low (give more to transient).

But the current per-block algorithm creates massive boundary fixup overhead:
ensure/drop ops, boundary-only blocks, and code size blowup.  Even when the
unified budget has enough room, the cache set churns at every block boundary.

The data shows that **the new system is strictly worse than the old system on
every benchmark** because the boundary fixup cost overwhelms any theoretical
benefit of flexible allocation.  The spill/fill cost (which is what the
unified budget is supposed to reduce) is unchanged.

## Requirements for any future algorithm

**Requirement 1 (baseline):** On low-pressure workloads where both locals and
stack fit within the old system's fixed banks (locals <= 13, stack swing <= 9),
the new algorithm must produce code that is no worse than the old fixed-bank
model.  This means near-zero boundary fixup ops — the cache set must be
stable.

**Requirement 2 (pressure):** On high-pressure workloads where the old system
would spill (locals > 13 or stack swing > 9), the new algorithm should do
better by dynamically reassigning the freed registers.

**Requirement 3 (code size):** The new algorithm must not produce more blocks
or more boundary-only blocks than the old system.  The current 20-31%
boundary-only block overhead is unacceptable.

## Measurement infrastructure

The pressure data in this document was collected using:

1. `SF_NATIVE_DUMP_DIR` to emit `native_index.txt` and `native_code.bin`.
2. The enriched SSA-IR dump format that now includes:
   - `local_types=[i32, i32, ...]` per function.
   - `cached=[fp[0], fp[3], ...]` per block (from `block_entry_cached_slots`).
3. `scripts/postprocess_native_dump.py` which writes `pressure_report.txt`
   alongside the per-function output.

To regenerate:

```bash
# Build both old and new CLIs
cd /tmp/sf-nano-old-40cfc0 && cargo build --release --bin sf-nano-cli
cd /path/to/repo && cargo build --release --bin sf-nano-cli

# Generate dumps
SF_NATIVE_DUMP_DIR=/tmp/dump-old ./old-cli --backend native path/to/module.wasm
SF_NATIVE_DUMP_DIR=/tmp/dump-new ./new-cli --backend native path/to/module.wasm

# Postprocess
python3 scripts/postprocess_native_dump.py \
  --wasm path/to/module.wasm \
  --dump-dir /tmp/dump-new \
  --out-dir /tmp/postprocessed
cat /tmp/postprocessed/pressure_report.txt
```
