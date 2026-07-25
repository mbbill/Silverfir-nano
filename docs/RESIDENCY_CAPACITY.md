# Residency capacity: the region-max rule, three refuted fixes, and the static gate

Status: investigation 2026-07-24. **Nothing landed.** The shipped behaviour is
unchanged; the defect described in §1 is still present. Three candidate fixes
were implemented and refuted (§3–§4). The reusable output is the measurement
method in §5, which decided all three without a single timing run.

Design tree: `mcts_mem/silverfir/compiler/ssa-prepare/cache-residency.alt/pressure-tiered-regions.md`,
with a pointer fact and a method fact in the parent `cache-residency.md`.

Claims below are tagged `(code)` read from source, `(dump)` observed in real
pipeline output, `(measured)` produced by the §5 gate, `(model)` from an
external analysis not reproduced here, `(hypothesis)` unverified reasoning.

---

## 1. The defect

Residency is decided per **region**, where
`Regions = { root } ∪ { one per Wasm loop instruction }` — `block` and `if`
create no region (`ALGORITHM4.md` §3.1; `region_solver.rs:691 build_region_tree`)
**(code)**. Each region's capacity is:

```rust
for region in &mut regions.nodes {
    let mut gp_headroom = 0usize;
    for &block_index in &region.owned_blocks {
        gp_headroom = gp_headroom.max(peak_gp[block_index]);   // max over the region
    }
    region.gp_capacity = usize::from(gp_dynamic_budget).saturating_sub(gp_headroom);
}
```
`region_solver.rs:237-246` **(code)**

`peak_gp[b]` is the **interior** peak of block `b` — the worst live-window
pressure at any point inside it — computed by `compute_lightweight_plan`
(`build.rs:649`) before residency is known, with `0, // no cache slots` and no
alias tags, and deliberately conservative **(code)**.

So a region's capacity is `budget − (worst op of the worst block)`, and that
subtraction applies to **every block in the region, for the whole region**. One
outlier block starves all of its siblings.

**The mechanism, observed.** bzip2 function 19 on arm64 (budget 23) **(dump)**:

- `b154`, 1,137 ops, interior peak **24** — one unit over budget;
- `cap(R) = 23.saturating_sub(24) = 0` for the entire 25-block region;
- `b168` sits in that region at loop depth 2 with its own interior peak of 4,
  holds no residency at all, and reads `c6` from its frame home **99 times**,
  `c4` 52 times and `c5` 50 times — with ~19 lanes idle.

**Scale.** Regions with `cap = 0` hold 15.5% of local accesses on x86_64 and
38.7% on arm32 **(model)**. On arm64 exactly 1 region of 2,849 is affected
**(model)** — see §2 for why that matters more than it looks.

---

## 2. Real budgets, and why arm64 cannot judge this

`allocatable_gp_dynamic_budget() = gp_volatile + gp_preserved`
(`backend.rs:142`); the internal-scratch tail is not spendable by the middle
end. This is one **shared bank** for cache residency *and* transients, not a
dedicated cache-register count **(code)**.

| target | volatile | preserved | scratch | **spendable GP** | i64 units | FP |
|---|---|---|---|---|---|---|
| arm64 | 16 | 7 | 1 | **23** | 1 | 30 |
| armv7a / thumbm | 4 | 2 | 2 | **6** | **2** | 13 |
| x86_64 | 7 | 0 | 1 | **7** | 1 | 14 |
| riscv64 | 20 | 0 | 1 | **20** | 1 | 30 |
| riscv32 | 18 | 0 | 2 | **18** | **2** | 0 or 30 |

Sources: `arch/arm64/abi.rs:328-342`, `arch/arm32/abi.rs:165-190`,
`arch/x86_64/abi.rs:111-118`, `arch/riscv/abi.rs`, `backend.rs:90-121` **(code)**.

Two documents disagree with this and the code wins: `ALGORITHM4.md` §3.2/§3.7/§4.7
say "x86_64 (budget = 9) or ARMv7a (budget = 8)"; the working note in session
memory says "2 GP cache registers on x86_64 (13 on ARM64)", which is stale in
kind as well as number.

**arm64 is not a valid judge of any capacity change.** The most permissive rule
that can be stated (capacity from block *entry* pressure rather than interior
peak — strictly more admission than any design in §3) changes arm64 code size by
at most **0.022%** across the 7 modules that survive it, and 4 are byte-identical
**(measured)**. That bounds the entire capacity-model line of work on arm64.
Worse, arm64 still fails 3 of 10 modules under that rule, so it hides the whole
benefit while exposing only part of the hazard. Gate on armv7 or x86_64.

---

## 3. What was tried, and the numbers

All three were implemented in a worktree behind `SF_CACHE_POLICY` knobs so one
binary could A/B itself. Baselines reproduce committed behaviour exactly.

### 3.1 Entry-pressure capacity (unpriced)

`cap(R) = budget − max(entry pressure over R's blocks)`, leaving interior
overflow to the mid-block `ensure_capacity` clamp. Motivated by the observation
that 98.8% of blocks are entered with zero live transients **(model)**, so the
constraint is stated at the wrong point.

**Result: infeasible.** Fails during native lowering on 10/10 modules at armv7
and x86_64, 3/10 at arm64 **(measured)**. The clamp does not absorb
over-admission: it prefers spilling transients to dropping residents
(`discipline.rs:544`), and its alias-discounted accounting is not the machine's
**(code)**.

### 3.2 Pressure-tiered sub-regions

Split each region's maximal runs of consecutive blocks whose peak exceeds the
region's floor into child pseudo-regions at the same loop depth, with boundary
frequency once per parent-body execution. The unchanged DP then chooses, per
cell: resident through the spike, *sheltered* around it, or not resident.
Capacity keeps its shape, so every block still individually fits its own peak.

### 3.3 Tiering plus an economic gate

As 3.2, but split only regions whose accessed-cell demand exceeds the capacity
today's rule gives them — where residency already fits, a tier admits nothing
and is pure boundary cost.

**Results** (native code size, `tier=1` vs `tier=0`) **(measured)**:

| target | 3.2 tiering | 3.3 + economic gate | failures |
|---|---|---|---|
| arm64 | +0.33 … +3.85% | +0.14 … **+2.47%** | 0 / 10 |
| armv7a | +1.12 … +2.02% | +1.10 … **+2.01%** | **6 / 10** |
| x86_64 | +1.90 … +4.90% | +1.84 … **+4.91%** | 1 / 10 |

No module improved on any target, in any variant.

The economic gate behaved exactly as designed — it suppressed the pointless
arm64 tiers (lz4 +3.85% → +0.95%, c-ray +1.17% → +0.48%) and moved the tight
targets by under 0.1pp, **because there the gate correctly fires**. That is the
strongest form of the refutation: where the gate confirms the prize exists,
collecting it still costs code.

### 3.4 The hot-path check that closes it

Whole-module size weights a cold prologue like an inner-loop body, so a
regression could in principle be cost-on-cold-paths with a hot-path win hiding
inside. Per-block bytes weighted by loop depth (§5.2) refutes that on x86_64
**(measured)**:

| module | module total | **in-loop** | depth ≥ 2 | 8^depth (cap 3) |
|---|---|---|---|---|
| coremark | +2.79% | **+6.47%** | +3.89% | +0.78% |
| sha256 | +3.64% | **+6.41%** | +6.62% | +4.55% |
| lz4 | +3.63% | **+5.64%** | +8.17% | +18.29% |
| bzip2 | +4.05% | **+5.48%** | +5.61% | +5.39% |
| stream | +2.39% | **+5.14%** | +3.22% | +3.96% |
| fib | +4.06% | **+5.83%** | +5.97% | +4.18% |

The loop bodies grew *more* than the modules did, on every one. The added code
lands where it executes most.

---

## 4. The four findings worth keeping

These outlived the designs and are the reason the investigation was worth
running.

**4.1 Unit-count feasibility is not lane-assignment feasibility.** Under §3.2
every block still individually fit its own interior peak, and armv7 still failed
6/10 in `allocate_cache_binding` (`lower_context.rs:1146`) because an i64 cached
cell needs a register **pair** on a 32-bit target. x86_64 — equally tight at 7
lanes, no pairs — failed 1/10. Modelling i64-pair adjacency plan-side is a
precondition for widening capacity on any 32-bit backend, which the cell-refactor
notes had already anticipated ("Phase C must model i64-pair adjacency plan-side,
not resurrect silent recovery").

**4.2 The region-max rule has a second, undocumented job.** `compute_joint_plan`
lifts the entry block's peak to cover incoming register params
(`build.rs:53-69`), and that lift constrains residency *only* by way of the
region maximum. Splitting the lifted entry block into a tier over-admitted
immediately — a third distinct error, "middle cache demand exceeded available
dynamic lanes after canonical register params were frame-published". Pinning the
entry block out of tiering was necessary and not sufficient.

**4.3 Middle-end counts can move opposite to native code.** x86_64 coremark MIR
ops **−1.1%** while native code **+4.6%** **(measured)**. The middle end really
does remove frame operations and the machine layer still emits more bytes.
Rank residency changes on native code within one fixed build config, never on
middle-end counts. (The 2026-06-24 policy sweep recorded the same trap.)

**4.4 The load-bearing hypothesis, still untested.** ALGORITHM4's benefit term
prices a resident cell's *accesses* but not the code residency itself costs —
establishment loads, boundary publishes, cached-cell block params, and reduced
lane headroom for transients inside loops. So any change that collects the
capacity residual by admitting more residents pays more than it saves on a
scarce budget **(hypothesis)**. Same shape as the `algorithm4:call=0` probe,
which cut SSA frame ops 82% on coremark while raising machine-level cell-home
traffic 30% and code size 5.9% **(model)**.

The visible mechanism, from MachineIR: blocks carry their resident set as
explicit params (`params=[r4:cache:gp, r5:cache:gp, …]`) with `move.cache.gp` on
edges, so more residents means wider param lists and more lane shuffling on
every intra-loop edge **(dump)**.

---

## 5. The static gate

The method that decided all of the above. Deterministic, cross-target, no timing,
about two minutes per iteration — cheap enough to falsify several hypotheses in
one sitting.

### 5.1 Whole-module size and feasibility

`--compile-only` prints one line per module:

```
[armv7a] (func:35, ssa:11477, mir:16442, code:153192)
```

`code:` is emitted native bytes. Bit-stable on armv7 and x86_64 across repeats.
On arm64 it carries a ~4-byte per-module ASLR jitter from address-dependent
constant materialization (sha256 read 53448 once and 53452 five times over six
runs), so treat sub-0.01% arm64 deltas as noise **(measured)**.

**Run the feasibility check first.** Three of four iterations in this
investigation were killed by a module failing to compile rather than by a size
delta, and each failure named a different invariant. Those messages were the
most informative output of the session and cost nothing.

### 5.2 Loop-weighted per-block size

The whole-module total cannot answer "did the hot path improve". The dump's
`[regions]` table carries per-block emitted bytes:

```
symbol=jit::main::func6::b114	func=6	region=b114	file_off=0x00000634	file_end=0x00000680	code_size=76
```

and the MachineIR terminators give the CFG, so loop depth follows from natural
loops over back edges. Summing block size weighted by loop depth separates cost
paid on cold boundary paths from cost paid inside loops. Script in Appendix B.

Requires the `ir-dump` feature; it is auto-on in dev builds, and for a release
cross build add `--features jit,sf-nano-core/ir-dump`.

### 5.3 The boundary of the method

It decides changes that alter **how much code runs and where**. It says nothing
about changes that alter **where values live at equal instruction count**. The
interpreter's l0 class was +16% CoreMark through a store→load dependency chain
and would have measured as noise on this gate — gated statically, l0 would have
been rejected.

For that class the analogue is a deterministic *counter* (the interpreter's
`COUNT_MODE` in `dispatch_arm64.rs`, which counts only handlers whose variant
involves L0 or L1). The general rule both domains share:

> Deterministic metrics falsify cheaply and confirm that a mechanism *engages*.
> Only a clock confirms that engagement *pays*.

The l1 class is the worked example: 81.5% dynamic engagement measured exactly by
counter, and only ~+4% payoff measured by clock.

---

## 6. Where to resume

The defect in §1 is real and unfixed. What changed is our belief about its
value: three ways of collecting it all cost more than they save, and §4.4 says
why that may be structural rather than incidental.

Do **not** start by trying a fourth capacity-widening variant. In order:

1. **Test §4.4 directly.** Take one function where tiering admitted extra
   residents and attribute the added in-loop bytes: are they cache-establishment
   loads, boundary publishes, `move.cache.gp` on intra-loop edges, or transient
   spills displaced by residents taking lanes? This is a dump-and-count job on
   one function, not a redesign. It either confirms the objective is missing a
   cost term or points somewhere not yet looked.
2. **If confirmed, the fix is in the objective, not the region tree** — price
   residency's own code in `benefit`/`edge_cost`. That is a change to
   ALGORITHM4's cost model with corpus-wide blast radius and should be argued
   before it is written.
3. **Prerequisite for any 32-bit work**: model i64-pair adjacency plan-side
   (§4.1). Without it, armv7 cannot even compile a widened plan, so it cannot be
   measured there — and armv7 is the target that matters (6 lanes, and the
   thumbm profile for the RP2350 port).

Complementary and independent: preserved-class Phase C for x86_64 and riscv,
which the tree calls the biggest unclaimed win for the non-arm64 backends. It
attacks the call-tax residual (x86_64 has `gp_preserved_dynamic = 0`, so every
call kills the whole cache) rather than the capacity residual, and does not
overlap with anything above.

---

## Appendix A — harness

The host is arm64 macOS; armv7 and x86_64 run under qemu inside colima. Only
`/Users/$USER` is mounted into the VM, so binaries are staged to VM-local `/tmp`.

```bash
# VM (once). qemu-user-static provides qemu-arm-static and qemu-x86_64-static.
colima start
colima ssh -- sudo apt-get install -y -qq qemu-user-static

# armv7: rust-lld is configured in .cargo/config.toml, no external toolchain.
cargo build --release --target armv7-unknown-linux-musleabihf \
  -p sf-nano-cli --no-default-features --features jit

# x86_64: musl needs __clear_cache, which rust-lld cannot supply. Use zig as the
# linker, mirroring scripts/zig-riscv32-linux-musl-cc.sh.
printf '#!/usr/bin/env sh\nset -eu\nexec zig cc -target x86_64-linux-musl "$@"\n' \
  > /tmp/zig-x86_64-linux-musl-cc.sh && chmod +x /tmp/zig-x86_64-linux-musl-cc.sh
RUSTFLAGS="-C linker=/tmp/zig-x86_64-linux-musl-cc.sh \
  -C target-feature=+crt-static,+ssse3,+sse4.1 -C link-self-contained=no" \
cargo build --release --target x86_64-unknown-linux-musl \
  -p sf-nano-cli --no-default-features --features jit

# stage and sweep
cat target/<triple>/release/sf-nano-cli | colima ssh -- tee /tmp/bench/cli > /dev/null
colima ssh -- chmod +x /tmp/bench/cli
colima ssh -- sh -c 'cd /tmp/bench && qemu-arm-static -cpu cortex-a15 ./cli \
  --compile-only coremark.wasm'
```

**Harness validation.** The same module compiled by an `x86_64-unknown-linux-musl`
build under qemu and an `x86_64-apple-darwin` build under Rosetta produces
byte-identical `ssa:` and `mir:` counts on all ten corpus modules, with `code:`
differing by a consistent +0.48…+0.74% (native emission reacting to build
config, not to the runner). The pipeline through MachineIR is reproducible
across OS, libc, linker, and emulator; the runner is irrelevant to the gate
**(measured)**.

**Wall clock in this VM is not usable.** An A/A test — the identical binary run
as two interleaved pseudo-variants, 12 measured runs — reported an apparent
**+7.9%** effect between them, CV 20.3%, 1.97× spread, on a contended host.
Before trusting any timing number on any machine, run that A/A first.

## Appendix B — loop-weighted size script

Save as `scripts/loop_weighted_size.py` (or run from anywhere; it reads
`d0-<module>/native_index.txt` and `d1-<module>/native_index.txt` in the
working directory).

```python
#!/usr/bin/env python3
"""Per-block native code size weighted by loop depth.

Whole-module code size treats an inner-loop instruction and a cold prologue
instruction alike. Residency changes pay their cost at region boundaries
(outside loops) and collect their benefit inside loop bodies, so the module
total can grow while the loops shrink. This separates the two.

CFG comes from the MachineIR terminators, per-block emitted bytes from the
[regions] table. Loop depth is the number of natural loops containing a block.
"""
import re
import sys
from collections import defaultdict


def parse(path):
    sizes = {}                      # (func, block) -> emitted bytes
    cfg = defaultdict(dict)         # func -> block -> set(succ)
    entry = {}                      # func -> entry block
    func = None
    in_mir = False
    block = None
    for line in open(path, errors="replace"):
        m = re.match(r"symbol=\S+\tfunc=(\d+)\tregion=(b\d+)\t.*code_size=(\d+)", line)
        if m:
            sizes[(int(m.group(1)), m.group(2))] = int(m.group(3))
            continue
        m = re.match(r"^\[function (\d+)\]", line)
        if m:
            func, in_mir, block = int(m.group(1)), False, None
            cfg[func]
            continue
        if func is None:
            continue
        if line.startswith("machine_ir:"):
            in_mir, block = True, None
            continue
        if line.startswith("ssa_ir:"):
            in_mir = False
            continue
        m = re.match(r"\s*entry=(b\d+)", line)
        if m and func not in entry:
            entry[func] = m.group(1)
            continue
        if not in_mir:
            continue
        m = re.match(r"^  block (b\d+)", line)
        if m:
            block = m.group(1)
            cfg[func].setdefault(block, set())
            continue
        if block and line.lstrip().startswith("term:"):
            # register operands are rN; only bN tokens are block labels
            cfg[func][block].update(re.findall(r"\bb\d+\b", line))
    return sizes, cfg, entry


def back_edges(graph, root):
    """Iterative DFS; an edge to a node on the current path is a back edge."""
    edges = []
    on_path, done = set(), set()
    stack = [(root, iter(sorted(graph.get(root, ()))))]
    on_path.add(root)
    while stack:
        node, it = stack[-1]
        advanced = False
        for succ in it:
            if succ not in graph:
                continue
            if succ in on_path:
                edges.append((node, succ))
            elif succ not in done:
                stack.append((succ, iter(sorted(graph.get(succ, ())))))
                on_path.add(succ)
                advanced = True
                break
        if not advanced:
            stack.pop()
            on_path.discard(node)
            done.add(node)
    return edges


def loop_depths(graph, root):
    """depth[b] = number of natural loops whose body contains b."""
    depth = defaultdict(int)
    preds = defaultdict(set)
    for node, succs in graph.items():
        for succ in succs:
            if succ in graph:
                preds[succ].add(node)
    for tail, head in back_edges(graph, root):
        body = {head}
        work = [tail]
        while work:
            node = work.pop()
            if node in body:
                continue
            body.add(node)
            work.extend(preds.get(node, ()))
        for node in body:
            depth[node] += 1
    return depth


def measure(path):
    sizes, cfg, entry = parse(path)
    total = loop = deep = weighted = 0
    for func, graph in cfg.items():
        root = entry.get(func)
        if root is None or root not in graph:
            root = next(iter(graph), None)
        if root is None:
            continue
        depth = loop_depths(graph, root)
        for block in graph:
            size = sizes.get((func, block), 0)
            d = depth.get(block, 0)
            total += size
            if d >= 1:
                loop += size
            if d >= 2:
                deep += size
            weighted += size * (8 ** min(d, 3))
    return total, loop, deep, weighted


def pct(a, b):
    return "n/a" if a == 0 else f"{(b - a) * 100.0 / a:+.2f}%"


print(f"{'module':<14}{'blocks-total':>13}{'in-loop':>12}{'depth>=2':>11}{'wt(8^d,cap3)':>15}")
for module in sys.argv[1:]:
    a = measure(f"d0-{module}/native_index.txt")
    b = measure(f"d1-{module}/native_index.txt")
    print(
        f"{module:<14}{pct(a[0], b[0]):>13}{pct(a[1], b[1]):>12}"
        f"{pct(a[2], b[2]):>11}{pct(a[3], b[3]):>15}"
    )
```

Produce the two dump sets with the policy knob under test, then run it:

```bash
for t in 0 1; do for m in coremark sha256 lz4 bzip2 stream fib_stripped; do
  SF_CACHE_POLICY=algorithm4:tier=$t SF_NATIVE_DUMP_DIR=./d$t-$m \
    ./cli --compile-only $m.wasm >/dev/null 2>&1
done; done
python3 loop_weighted_size.py coremark sha256 lz4 bzip2 stream fib_stripped
```

## Appendix C — the probe code

The three variants were implemented behind `SF_CACHE_POLICY` extensions in a
scratch worktree (+325 lines across `joint_plan/build.rs` and
`joint_plan/region_solver.rs`), never committed:

- `algorithm4:head=max|mean|min|zero|entry` — the capacity statistic (§3.1);
  `max` is the shipped rule and the default.
- `algorithm4:tier=0|1` — pressure-tiered sub-regions with the economic gate
  (§3.2, §3.3); off by default.

Since all three variants were refuted, the code is disposable; what is worth
keeping is in §5 and Appendix B. If the §6.1 attribution job needs it, the knobs
are ~40 lines of parsing plus `split_pressure_tiers` and `region_headroom` in
`region_solver.rs`, straightforward to rebuild from this document.
