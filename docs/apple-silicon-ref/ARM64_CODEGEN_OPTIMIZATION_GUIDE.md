# ARM64 Codegen Optimization Guide for Apple Silicon

General-purpose reference for optimizing JIT-generated ARM64 code on Apple
Silicon M-series processors. Distilled from Apple Silicon microarchitecture
reverse-engineering (Dougall Johnson), Apple's CPU Optimization Guide,
ETH Zurich M-series benchmarking thesis (Sidler, Spring 2025),
KTH HPC M-series evaluation (Hübner et al., 2025),
ICCS 2024 Apple Silicon study (Struniawski et al.),
and academic branch prediction analysis.

This document describes **what the hardware likes and dislikes**, what
codegen patterns are cheap or expensive, and what optimization opportunities
exist in our ARM64 backend. It is not tied to any specific benchmark.

---

## 1. Firestorm/Avalanche/Everest Microarchitecture

All M-series P-cores (Firestorm in M1, Avalanche in M2, Everest in M3,
unnamed in M4) share the same fundamental execution model with incremental
improvements per generation.

### 1.1 Pipeline Overview

| Resource | Capacity | Notes |
|---|---|---|
| Decode/Rename width | 8 uops/cycle | |
| Coalesced retire queue | ~330 groups | Up to 7 uops per group |
| Rename retire queue | ~623 entries | Tracks architectural register writes |
| Physical GP registers | ~380-394 | |
| Physical FP/SIMD registers | ~432 | |
| Physical flags registers | ~128 | |
| In-flight branches | ~144 | |
| In-flight loads | ~130 | |
| In-flight stores | ~60 | **Tight constraint** |
| Retirement rate | 8 groups/cycle or 16 renames/cycle | |

**Retire groups**: Firestorm coalesces uops into retire groups that all retire
together. A group may contain up to 7 uops. Uops that can fail before
retiring (memory accesses) must appear at the start of a group; uops that
can fail after retiring (conditional branches) appear at the end. Issuing
uops are limited to roughly 4 per group. This means the effective
out-of-order window is >1000 issuing instructions, or >2200 NOPs.

### 1.2 Execution Units (14 Ports)

```
Integer (6 units):
  u1: ALU, bitfield, flags, branch, address gen, MSR/MRS
  u2: ALU, bitfield, flags, branch, indirect branch, pointer auth
  u3: ALU, bitfield, flags, mov-from-SIMD/FP
  u4: ALU, bitfield, mov-from-SIMD/FP
  u5: ALU, bitfield, multiply, divide
  u6: ALU, bitfield, multiply, MADD, CRC, extraction

Load/Store (4 units):
  u7:  Store only, AMX
  u8:  Load/Store, AMX
  u9:  Load only
  u10: Load only

FP/SIMD (4 units):
  u11: FP/SIMD general
  u12: FP/SIMD general
  u13: FP/SIMD, FCSEL, int-to-FP conversion, to-GPR
  u14: FP/SIMD, FCSEL, to-GPR, FCMP, FDIV, FRECPE, FRSQRTE, SHA
```

**Key port constraints**:
- Stores go through only u7 and u8 (2 ports). Combined with only 60
  in-flight stores, store-heavy code is easily bottlenecked.
- FDIV, FSQRT, FCMP, FRECPE, FRSQRTE only run on u14 (1 port). Multiple
  dependent FDIVs serialize completely.
- Integer multiply runs on u5-u6 only (2 ports).
- Branches run on u1-u2 only, with indirect branches only on u2.
- GP-to-SIMD/FP transfers go through u3-u4 on the integer side and u13-u14
  on the FP side. This two-unit requirement explains the high domain-crossing
  latency.

**Sustained throughput**: In contrived cases, Firestorm can sustain 11 issues
per cycle because ALU operations with optional shift/extend generate an
extra issue independently.

### 1.3 Instruction Latency and Throughput Table

TP = reciprocal throughput (cycles per instruction; lower = better).

#### Integer

| Instruction | Latency | TP | Units | Notes |
|---|---|---|---|---|
| ADD/SUB (imm or reg) | 1 | 0.17 | u1-u6 | 6/cycle |
| ADD/SUB (shifted reg) | 1+1 | 0.17 | u1-u6 | Extra issue for shift |
| MOV Xd, Xm | **0** | — | — | Eliminated via renaming (max 2 per 8-insn group) |
| MOV Xd, #imm | **0** | — | — | Eliminated via renaming (max 2 per 8-insn group) |
| MOV Wd, Wm | 1 | 0.17 | u1-u6 | **NOT eliminated** (32-bit form) |
| MOV Xd, XZR | 1 | 0.17 | u1-u6 | **NOT eliminated** (use `movz xd, #0`) |
| MUL | 3 | 0.5 | u5-u6 | |
| MADD | 3 | 0.5 | u5-u6 | 1c addend→output shortcut |
| SDIV (32-bit) | 7-10 | varies | u5 only | Data-dependent |
| UDIV (32-bit) | 7-10 | varies | u5 only | Data-dependent |
| LSL/LSR/ASR (variable) | 1 | 0.17 | u1-u6 | |
| CLZ | 1 | 0.17 | u1-u6 | |
| RBIT | 1 | 0.17 | u1-u6 | |
| CMP/CMN/TST | 1 | 0.17 | u1-u6 | Fuses with B.cond |
| CSEL/CSINC/CSET | 1 | 0.17 | u1-u6 | |

#### Load/Store

| Instruction | Latency | TP | Units | Notes |
|---|---|---|---|---|
| LDR Xt, [Xn, #imm] | 4 | 0.33 | u8-u10 | 3c for load→load-address |
| LDR Xt, [Xn, Xm] | 4 | 0.33 | u8-u10 | +1c if index from shift-like op |
| LDR (literal/PC-rel) | 4 | 0.33 | u8-u10 | 1 uop; useful for const pools |
| LDP Xt1, Xt2, [Xn] | 4+1 | 0.33 | u8-u10 | 2 uops, 1 LS issue; 2nd reg +1c |
| LDR (post-index) | 4 | 0.37 | u8-u10 + u1-u6 | 2 uops: load + addr update |
| STR Xt, [Xn, #imm] | 1 | 0.5 | u7-u8 | 2/cycle max |
| STP Xt1, Xt2, [Xn] | 1 | 0.5 | u7-u8 | Same port as STR |

#### Floating-Point Scalar

| Instruction | Latency | TP | Units | Notes |
|---|---|---|---|---|
| FADD/FSUB (D) | 3 | 0.25 | u11-u14 | 4/cycle |
| FMUL (D) | 3 | 0.25 | u11-u14 | 4/cycle |
| FMADD/FMSUB (D) | 4 | 0.25 | u11-u14 | Fused multiply-add, 4/cycle |
| FDIV (D) | **10** | **1.0** | **u14 only** | Pipelined: 1/cycle; S: 8c/1.0 TP |
| FSQRT (D) | **13** | **2.0** | **u14 only** | 0.5/cycle; S: 10c/2.0 TP |
| FABS/FNEG | 2 | 0.25 | u11-u14 | |
| FMIN/FMAX | 2 | 0.25 | u11-u14 | |
| FMOV Dd, Dm | **2** | 0.25 | u11-u14 | **NOT eliminated** |
| FMOV Dd, #imm8 | 2 | 0.25 | u11-u14 | Direct FP constant |
| FCMP Dn, Dm | 2 | 1.0 | u14 only | |
| FCMP Dn, #0.0 | 2 | 1.0 | u14 only | |
| FCSEL | 2 | 0.5 | u13-u14 | |
| FCVT (S↔D) | 3 | 0.25 | u11-u14 | Precision conversion |
| FRINT{P,M,Z,N} | 3 | 0.25 | u11-u14 | Round to integral |
| FRECPE | 3 | 1.0 | u14 only | Reciprocal estimate |
| FRSQRTE | 3 | 1.0 | u14 only | Reciprocal sqrt estimate |
| FRECPS | 4 | 0.25 | u11-u14 | Reciprocal step |
| FRSQRTS | 4 | 0.25 | u11-u14 | Reciprocal sqrt step |

#### GP ↔ FP Domain Crossings

| Instruction | Latency | TP | Notes |
|---|---|---|---|
| FMOV Xn → Dn | **5-7** | 0.5 | GP to FP (2 uops: 1 int + 1 FP) |
| FMOV Dn → Xn | **5-7** | 0.5 | FP to GP (2 uops: 1 int + 1 FP) |
| SCVTF Dn, Xn | **7** | 0.5 | Int to float |
| UCVTF Dn, Xn | **7** | 0.5 | Unsigned int to float |
| FCVTZS Xn, Dn | **≤13** | 0.5 | Float to int (2 uops) |
| FCVTZU Xn, Dn | **≤13** | 0.5 | Unsigned float to int (2 uops) |
| MOVI Vd.2D, #0 | **0** | — | SIMD zero, eliminated at rename |
| MOV Vd.16B, Vm.16B | **0** | — | SIMD full-width copy, eliminated |

**Note on FCVT to GPR**: The measured latency of ≤13 cycles (from Dougall
Johnson's data) is higher than the 7 cycles listed in some references.
The 2-uop decomposition (one FP unit + one integer unit) adds overhead
beyond a simple domain crossing.

#### Atomic and Barrier Instructions

| Instruction | TP | Uops | Notes |
|---|---|---|---|
| DMB (any scope) | 17 | 1 | Data memory barrier |
| DSB (any scope) | 17 | 1 | Data synchronization barrier |
| ISB | 28 | 4 | Instruction synchronization barrier; **very expensive** |
| CAS (32/64) | 3 | 4 | Compare-and-swap (3 LS uops) |
| CASA (acquire) | 3 | 4 | |
| CASAL (acq+rel) | 7 | 4 | Release semantics adds ~4c |
| CASP (pair) | 14 | 6 | Compare-and-swap pair (3 LS uops) |
| CASPAL (pair acq+rel) | 18 | 6 | |
| SWP (32/64) | 3 | 2 | Atomic swap (2 LS uops) |
| SWPAL (acq+rel) | 7 | 2 | |

**Key takeaway for atomics**: Release semantics (the `L` suffix) adds ~4
cycles of throughput cost. Acquire-only (`A` suffix) is cheaper. Use
relaxed atomics where ordering permits. Prefer `LDAPR` (load-acquire with
limited ordering, ARMv8.3+) over `LDAR` for acquire semantics when full
sequential consistency is not required.

### 1.4 Instruction Elimination (Zero-Latency, No Port)

Handled at rename, consuming no execution port:
- `MOV Xd, Xm` — 64-bit GP register copy (max 2 per 8-instruction group)
- `MOV Xd, #imm` — small immediate (max 2 per 8-instruction group, includes
  MOVZ/MOVN aliases). Both 32-bit and 64-bit immediate moves are eliminated.
- `MOVI Vd.2D, #0` — SIMD zero (any type)
- `MOV Vd.16B, Vm.16B` — SIMD full-width copy (also `Vd.8H`, etc., but NOT `Vd.8B`)
- `NOP` — never issues
- `B` — unconditional forward branch (within decode group)

**NOT eliminated** (common pitfalls):
- `MOV Wd, Wm` — 32-bit GP copy, goes through ALU (1 cycle)
- `MOV Xd, XZR` — goes through ALU; use `MOVZ Xd, #0` instead
- `FMOV Dd, Dm` — goes through FP unit (2 cycles)
- `ADR/ADRP` — goes through ALU
- `MOV Vd.8B, Vm.8B` — half-width SIMD copy, NOT eliminated

### 1.5 Instruction Fusion

Consecutive instruction pairs fused into a single uop:
- `CMP/CMN/TST/ADDS/SUBS` + `B.cond` → fused compare-and-branch
  - Does NOT work with shifted/extended register forms
  - Does NOT work with flag-reading instructions like ADC
  - Complete fusion requires that fused instructions read no more than
    4 registers per 6 instructions
- `ADD/SUB/AND/ORR/EOR/BIC` + `CBZ/CBNZ` → fused ALU-and-branch
  - Only when ALU destination matches CBZ/CBNZ operand
  - Works with flag-setting variants too
- `AESE` + `AESMC` / `AESD` + `AESIMC` → fused AES
  (operands must match pattern `A, B ; A, A`)
- `AESE/AESD` + `EOR` → fused (operands: `A, B ; A, A, C` or `A, B ; A, C, A`)
- `PMULL` + `EOR` → fused (operands: `A, B, C ; A, A, D` or `A, B, C ; A, D, A`)
- `AMX` + `AMX` → fused (excluding loads and stores)

**NOT fused**: `ADRP+ADD`, `MOV+MOVK`, `MUL+UMULH`, `UDIV+MSUB`.

### 1.6 Complex Latencies

- **MADD output → addend input**: 1 cycle (vs 3 cycles for other chains).
  Useful for integer accumulation loops.
- **Load → load address**: 3 cycles (vs 4 cycles to ALU). Pointer chasing
  through linked lists benefits from this fast path.
- **Load with shifted index**: +1 cycle if index register comes from a
  shift-like instruction. ADD/AND outputs do not have this penalty.
- **LDP second register**: Always +1 cycle latency over the first.
- **GP↔FP roundtrip**: Minimum 7 cycles (e.g., flags-chain crossing).
  Float-to-int conversions (FCVTZS to GPR) measured at ≤13 cycles.

---

## 2. Cache Hierarchy and Memory Subsystem

### 2.1 Cache Parameters

| Level | P-core | E-core | Line Size | Notes |
|---|---|---|---|---|
| L1 I-cache | 192 KB | 64-128 KB | 64B | |
| L1 D-cache | 128 KB | 64 KB | **128 bytes** | |
| L2 cache | 12-16 MB shared | 4 MB shared | 128 bytes | 16 MB on M2+ |
| SLC (L3/System) | ~8 MB | shared | — | System-level cache |

**Critical: 128-byte cache lines.** Apple Silicon uses 128-byte cache lines,
double the 64-byte lines of most x86 processors and many other ARM
implementations. This has significant implications:

- **Spatial prefetching is more aggressive**: Each cache miss fetches 128
  bytes, benefiting sequential access patterns.
- **False sharing radius is wider**: Two threads writing to addresses within
  the same 128-byte line will cause coherence traffic, even if they are
  64+ bytes apart.
- **Alignment matters more**: Data structures that straddle 128-byte
  boundaries incur two cache line fetches. Align hot structures to 128 bytes
  for optimal access.
- **STP/LDP benefit amplified**: Paired operations naturally exploit the
  wider cache line.

### 2.2 Cache Latencies

| Access | Latency | Notes |
|---|---|---|
| L1 D-cache (scalar) | 3-4 cycles | 3c for load→load-addr, 4c for load→ALU |
| L1 D-cache (FP/SIMD) | ~5 cycles | Domain crossing adds overhead |
| L2 cache | ~12-15 cycles | Shared among P-cores |
| SLC (L3) | ~18 + 10-15ns | System-level cache |
| DRAM | ~100-120ns | Unified memory (no NUMA) |

### 2.3 TLB Structure

| Level | Entries | Miss Penalty |
|---|---|---|
| Data TLB L1 | ~160 entries | 6 cycles |
| Data TLB L2 | ~3072 entries | 26 cycles |

Large working sets (>160 × 16KB pages = 2.5 MB with 16KB pages) will
start hitting L1 DTLB misses. macOS uses 16KB pages by default (vs 4KB
on Linux x86), which means each TLB entry covers more memory.

### 2.4 Data Memory-Dependent Prefetcher (DMP)

Apple Silicon includes a data memory-dependent prefetcher (DMP) that
inspects loaded data values and treats pointer-like values as prefetch
targets. This means:

- **Array-of-pointers traversal** is hardware-accelerated: after the CPU
  observes dereferences of `arr[0]`, `arr[1]`, `arr[2]`, it prefetches
  `*arr[3]` and beyond.
- **Security implication**: This is the mechanism behind the "GoFetch" and
  "Augury" vulnerabilities. For cryptographic code, be aware that data
  values may trigger speculative memory accesses.
- **Optimization implication**: Pointer-chasing code through arrays of
  pointers gets automatic prefetching. Linked list traversal with random
  node placement does not benefit (use software prefetch hints instead).

### 2.5 Memory Bandwidth

Measured peak bandwidth across generations (from KTH STREAM benchmarks):

| Chip | CPU Peak | GPU Peak | Theoretical |
|---|---|---|---|
| M1 | 59 GB/s | 60 GB/s | 67 GB/s |
| M2 | 78 GB/s | 91 GB/s | 100 GB/s |
| M3 | 92 GB/s | 92 GB/s | 100 GB/s |
| M4 | 103 GB/s | 100 GB/s | 120 GB/s |

All chips achieve ~85% of theoretical peak bandwidth. The unified memory
architecture means CPU and GPU access the same physical memory without
explicit data transfers.

---

## 3. Branch Prediction

### 3.1 Branch Predictor Architecture

Apple Silicon uses a **TAGE-based** (TAgged GEometric length) branch
predictor, confirmed through academic reverse-engineering:

- **6 pattern history tables** of varying geometric lengths
- **BTB**: ~1024 entries in first level
- **Misprediction penalty**: **13-14 cycles**
- Prediction quality is comparable to AMD Zen 2 and Intel Sunny Cove

### 3.2 Implications for Codegen

- **Misprediction penalty is moderate** (13-14c), but still dominates
  tight loops. A mispredicted branch in a loop body that otherwise runs
  at 1 iteration/cycle costs 13-14× the iteration cost.
- **Indirect branch prediction** is supported (via u2 only), but uses a
  separate indirect target predictor. Indirect branches with many targets
  (e.g., interpreter dispatch) will suffer higher misprediction rates.
- **Prefer conditional execution over branches** for simple
  predicated operations: `CSEL` is 1 cycle, always predicted correctly
  (no branch), while a mispredicted branch costs 13-14 cycles.
- **Loop alignment** is less critical than on x86 due to the wide decode
  (8 instructions/cycle) and large L1 I-cache (192KB).
- **Taken branches**: Maximum 1 per cycle. Tiny basic blocks with taken
  branches waste decode bandwidth.

---

## 4. Codegen Rules Derived from Microarchitecture

### Rule 1: FMOV Dd,Dm Is Expensive — Minimize FP Register Copies

Every `fmov d,d` costs 2 cycles of latency and occupies one of only 4 FP
execution ports. Integer `mov x,x` is free (eliminated at rename). This
asymmetry means:

- **Prefer defining FP results directly into their final destination register**
  rather than producing into a temporary and copying.
- **Copy propagation for FP registers has higher payoff** than for GP registers.
- When choosing between an extra GP `mov` and an extra FP `fmov`, always
  prefer the GP `mov` (0 vs 2 cycles).

### Rule 2: Avoid GP↔FP Domain Crossings

Each FMOV between GP and FP registers costs 5-7 cycles and consumes 2 uops
(one from integer unit u3/u4, one from FP unit u13/u14). Float-to-int
conversions (FCVTZS to GPR) are even worse at ≤13 cycles with 2 uops.

- **Float constant materialization**: Building a float constant via
  `movz+movk+fmov` always pays the 5-7 cycle domain-crossing penalty for the
  final transfer. Alternatives:
  - `FMOV Dd, #imm8` for encodable values (no crossing, 2c)
  - `MOVI Vd.2D, #0` for zero (eliminated at rename, 0c)
  - `LDR Dd, [PC, #offset]` from literal pool (4c load, no crossing)
- **Reinterpret casts**: `I64ReinterpretF64` / `F64ReinterpretI64` require
  an FMOV GP↔FP. These are unavoidable for the semantics but should not be
  routed through scratch registers (adds an extra GP `mov` on top of the
  5-7c crossing).
- **Int-to-float conversions**: SCVTF/UCVTF cross domains (7c). No way to
  avoid, but minimize redundant conversions.

### Rule 3: Use STP/LDP for Paired Memory Operations

LDP uses 2 uops but only **1 load-store unit issue** for 2 registers; two
LDR instructions use 2 port issues. STP uses 1 store port for 2 registers.
With only 2 store ports and 60 in-flight stores, pairing is important.

Applicable patterns:
- Prologue/epilogue callee-save/restore (FP regs especially)
- Cached-local spill/reload sequences around calls
- Frame zero-initialization sequences
- Any consecutive stores/loads to adjacent frame slots

**LDP scheduling tip**: The second destination register has +1 cycle latency.
Place the less-critical value as the second operand.

**LDP throughput note**: LDP has TP 0.33 (3/cycle), same as LDR, because it
uses only 1 load-store issue slot. This means LDP gives 2× the data per
issue slot compared to LDR.

### Rule 4: Separate Loads from Their Consumers

LDR has 4-cycle latency. If the next instruction uses the loaded value, the
CPU stalls for up to 3 cycles. Reorder to put independent work between
load and first use.

**Anti-pattern**:
```asm
ldr   d3, [x21, x3]    ; 4c latency
fmul  d4, d3, d5        ; stalls waiting for d3
```

**Better**:
```asm
ldr   d3, [x21, x3]    ; 4c latency starts
add   x6, x21, w7, uxtw ; independent address computation (hides 1c)
ldr   d7, [x6, #8]      ; independent load (hides 1c)
fadd  d8, d9, d10       ; independent FP work (hides 1c)
fmul  d4, d3, d5        ; d3 ready by now (4c elapsed)
```

### Rule 5: Schedule Around FDIV, FSQRT, and u14-Only Operations

FDIV(D) has 10-cycle latency and FSQRT(D) has **13-cycle latency**, both
running only on u14 (shared with FCMP, FRECPE, FRSQRTE). FDIV is pipelined
(TP=1.0, so 1 per cycle throughput), but FSQRT is not (TP=2.0, so 1 every
2 cycles). Multiple dependent FSQRTs are especially costly. Always emit as
much independent work as possible between FDIV/FSQRT and their result use
— 10-13 cycles is enough to hide many independent FADD/FMUL operations.

**Newton-Raphson alternative**: For approximate reciprocal or reciprocal
square root, consider FRECPE+FRECPS or FRSQRTE+FRSQRTS sequences:
- `FRECPE` (3c, u14-only) + `FRECPS` (4c, u11-u14) gives ~12-bit accuracy
- One Newton-Raphson refinement step achieves ~24-bit accuracy
- Total: ~10c vs FDIV's 10c latency, but the FRECPS steps can run on
  u11-u14 (4 ports) instead of being stuck on u14 alone. Better for
  throughput when multiple divisions are needed.
- For FSQRT replacement, FRSQRTE+FRSQRTS at ~10c total beats FSQRT's
  13c latency while also having better throughput characteristics.

### Rule 6: Enable Hardware Fusion

To benefit from CMP+B.cond fusion (saves 1 uop):
- Keep the comparison and conditional branch **adjacent** in the instruction
  stream.
- Don't insert CSET/CSEL between them unless the boolean value is needed
  independently.
- The comparison must NOT use shifted/extended register forms.
- The fused pair must read no more than 4 registers per 6 instructions.

Similarly, ALU+CBZ/CBNZ fuses when the ALU destination matches the CBZ
operand. Keep them adjacent.

### Rule 7: Prefer Immediate Operands

ARM64 supports 12-bit unsigned immediates (optionally shifted by 12) in
ADD/SUB/CMP. Using immediate form saves materializing a constant into a
register:

**Anti-pattern**:
```asm
mov   x16, #42
cmp   x3, x16
```

**Better**:
```asm
cmp   x3, #42           ; immediate form, no register needed
```

This applies to ADD, SUB, CMP, CMN, and AND/ORR/EOR (with bitmask
immediates). The ADDS/SUBS forms also support immediates and fuse with
B.cond.

### Rule 8: Minimize Taken Branches

Only 1 taken branch per cycle. Tiny basic blocks that always branch force
the CPU to waste decode bandwidth. Prefer fall-through for the common
path. Unconditional forward branches within a decode group are eliminated,
but backward branches and taken conditional branches are not.

**Branch misprediction costs 13-14 cycles.** For simple two-way decisions
where both paths are short, prefer branchless code using CSEL/CSINC/CSINV
(1 cycle, always correct) over a conditional branch.

### Rule 9: Be Aware of Store Pressure

With only 2 store ports and 60 in-flight stores, store-heavy sequences
(function prologues, call-site spills, frame initialization) can become
bottlenecks. Optimizations:
- STP instead of 2× STR (halves store port usage)
- `STR XZR` for zero stores (avoids materializing zero first)
- Minimize the number of cached locals that need spilling
- STP XZR, XZR for paired zero-initialization (1 port instead of 2)

### Rule 10: Exploit MOV Elimination Budget

Up to 2 integer MOV instructions per 8-instruction decode group are
eliminated (zero latency, no port). Beyond that, additional MOVs go
through ALU. This means:
- A few GP register copies per block are essentially free.
- Excessive GP copies (>2 per 8-instruction window) start costing 1 cycle each.
- FP copies (`fmov d,d`) are NEVER eliminated and always cost 2 cycles.
- SIMD full-width copies (`mov v0.16b, v1.16b`) ARE eliminated.

### Rule 11: Minimize Barrier and Atomic Costs

Memory barriers are expensive on Apple Silicon:
- `DMB` and `DSB`: ~17 cycles throughput
- `ISB`: ~28 cycles, 4 uops — extremely expensive, avoid in hot paths

For atomic operations:
- Plain CAS/SWP: 3 cycles throughput
- With release semantics (CASAL/SWPAL): 7 cycles — ~2× slower
- Paired CASPAL: 18 cycles

**Prefer `LDAPR` over `LDAR`** for acquire semantics (ARMv8.3+, supported on
all M-series). LDAPR allows reordering before STLR to different locations,
providing better performance while still guaranteeing acquire semantics.

### Rule 12: Align Data for 128-Byte Cache Lines

Apple Silicon's 128-byte cache lines mean:
- **Structure alignment**: Hot structures crossing a 128-byte boundary fetch
  two cache lines. Pad or align to 128 bytes.
- **Array element size**: If array elements are not powers of 2, every Nth
  element may straddle a cache line boundary. Prefer power-of-2 sizes.
- **Stack frame alignment**: Frame slots accessed together should be in the
  same 128-byte region. Group hot locals together.
- **False sharing**: In multi-threaded code, ensure independently-written
  variables are at least 128 bytes apart.

### Rule 13: Use Conditional Select Over Branches for Simple Predicates

`CSEL/CSINC/CSINV/CSNEG` are 1-cycle, 6-per-cycle (u1-u6) instructions
with zero misprediction risk. A branch misprediction costs 13-14 cycles.

**Anti-pattern** (unpredictable condition):
```asm
cmp   x0, #0
b.eq  .use_default
mov   x1, x2
b     .done
.use_default:
mov   x1, x3
.done:
```

**Better**:
```asm
cmp   x0, #0
csel  x1, x3, x2, eq   ; 1 cycle, no branch
```

Also consider `CCMP` for chaining multiple conditions:
```asm
cmp   x0, #10
ccmp  x1, #20, #0, ge   ; if x0 >= 10, also compare x1 with 20
b.eq  target             ; branch if x0 >= 10 AND x1 == 20
```

---

## 5. NEON/SIMD Considerations

### 5.1 NEON Architecture on Apple Silicon

All M-series chips implement NEON with 128-bit vector registers (V0-V31).
The 4 FP/SIMD execution units (u11-u14) handle both scalar FP and NEON
vector operations with identical throughput for most instructions.

**Key property**: Vector NEON instructions have the **same latency and
throughput** as their scalar equivalents on the same units. A vector
`FADD V0.4S, V1.4S, V2.4S` has the same 3-cycle latency and 0.25 TP as
a scalar `FADD S0, S1, S2`. This means NEON vectorization gives a direct
4× (for 4S) or 2× (for 2D) throughput improvement with no latency penalty.

### 5.2 NEON-Specific Optimization Notes

- **SIMD zero is free**: `MOVI Vd.2D, #0` is eliminated at rename (0 cycles).
- **Full-width SIMD copy is free**: `MOV Vd.16B, Vm.16B` is eliminated.
  Half-width (`Vd.8B`) is NOT eliminated.
- **SIMD loads**: `LDR Qn` (128-bit) has the same 4c latency as scalar LDR.
  `LDP Qn, Qm` loads 256 bits in one issue.
- **Cross-lane operations** (TBL, TBX, ZIP, UZP, TRN): Typically 2-3
  cycles, 0.25-0.5 TP. More expensive than same-lane operations.
- **Horizontal reductions** (FADDP, ADDV, UMAXV, etc.): Require
  cross-lane operations. Prefer tree-structured reductions when possible.
- **Element insert/extract** (INS, UMOV, SMOV): Moving individual elements
  between GP and SIMD registers incurs domain-crossing penalties similar
  to FMOV GP↔FP.

### 5.3 No SVE/SVE2 on Apple Silicon (Except M4 Streaming SVE)

Apple M1/M2/M3 do **not** implement SVE or SVE2. M4 implements ARMv9.2-A
with SME (Scalable Matrix Extension) and **Streaming SVE** (SSVE) — but
only within an SME streaming mode context. Standalone SVE is not available
on any current Apple Silicon chip.

For JIT compilers, this means:
- **Do not generate SVE instructions** for general computation.
- NEON (128-bit fixed) is the only SIMD ISA available across all generations.
- M4's SME is relevant for matrix multiplication kernels (512-bit tiles,
  16×16 elements) but requires explicit streaming mode entry/exit.
- LLVM pragmatically flags M4 as ARMv8.7a due to lack of full SVE support.

---

## 6. Efficiency Core (E-Core) Differences

The E-cores (Icestorm M1, Blizzard M2, Sawtooth M3/M4) have a narrower
pipeline than P-cores:

| Resource | P-core | E-core (approx.) |
|---|---|---|
| Decode width | 8 uops/cycle | ~4 uops/cycle |
| Execution units | 14 (6+4+4) | ~8-9 (fewer of each) |
| L1 D-cache | 128 KB | 64 KB |
| L1 I-cache | 192 KB | 64-128 KB |
| L2 cache | 12-16 MB shared | 4 MB shared |
| Frequency | 3.2-4.4 GHz | 2.06-2.85 GHz |

**Key implications**:
- E-cores have roughly **50% of P-core throughput** but at much lower power.
- The same instruction latencies generally apply, but throughput bottlenecks
  hit sooner due to fewer ports.
- Store pressure is even more critical on E-cores with fewer store ports.
- macOS schedules threads to E-cores or P-cores based on QoS level, not
  explicit CPU affinity. Background/utility tasks often run on E-cores.
- **Optimization targeting P-cores generally helps E-cores too**, since both
  share the same fundamental microarchitecture principles.

**E-core evolution**:
- Icestorm (M1/A14): ~2.06 GHz, base E-core design
- Blizzard (M2/A15): ~2.42 GHz, +16% integer, +8% FP over Icestorm
- Sawtooth (M3/A16): ~2.75 GHz, further incremental improvements
- M4 E-core: ~2.85 GHz, ARMv9.2-A

---

## 7. Our Backend: Current State and Optimization Opportunities

This section maps the hardware rules above to specific patterns in our
ARM64 codegen pipeline.

### 7.1 FP Register Copy Elimination

**Rule violated**: Rule 1 (FMOV Dd,Dm is expensive)

**Current pattern**: The lowering produces `fmov d,d` copies when:
- A transient FP result is stored to a cached-local FP register
- Block-entry reloads restore cached locals that are immediately overwritten
- Intermediate results pass through transients before reaching final destination

**Optimization approaches**:
1. Store-side coalescing: Rewrite producer to define directly into cached-local
2. Extended coalescing window: Look back N instructions for coalescing opportunity
3. Direct allocation: Allocate results into destination register from the start
4. Full copy propagation: Track aliases, rewrite uses (requires fixing
   bounds-check split implicit register dependencies first)

### 7.2 STP/LDP Usage

**Rule violated**: Rule 3 (Use STP/LDP for paired operations)

**Currently missing**:
- Prologue FP callee-save: 8 individual STR → should be 4 STP
- Epilogue FP callee-restore: 8 individual LDR → should be 4 LDP
- Cached-local spill before call: individual STR → should pair adjacent slots
- Block-entry cached-local reload: individual LDR → should pair adjacent slots
- Callee frame zero-init: individual STR XZR → should pair as STP XZR, XZR

**Estimated savings**: Each STR→STP conversion saves 1 store port issue.
For a function saving 8 FP callee-saved registers: 8 STR = 8 port issues
→ 4 STP = 4 port issues (50% reduction in store port pressure).

### 7.3 Float Constant Materialization

**Rule violated**: Rule 2 (Avoid GP↔FP domain crossings)

**Current**: All f64 constants go through GP: `movz+movk+...+fmov Dd, Xn`
(5-7 cycle domain crossing on the final transfer).

**Available alternatives**:
- `FMOV Dd, #imm8`: Single instruction, no domain crossing, 2c latency.
  Encodes `±(1 + n/16) × 2^r` where n ∈ [0,15], r ∈ [-3,4].
  Covers: 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0,
  and their negatives, plus many more values.
- `MOVI Vd.2D, #0`: Zero constant, eliminated at rename (0 cycles).
- `LDR Dd, [PC, #offset]`: Literal pool, 4c load latency, no domain crossing.
  Uses a load port but avoids GP register and FP port usage. Best for
  constants used multiple times or in hot loops.

**Priority implementation**: Check for FMOV #imm8 encodability first (cheapest),
then MOVI for zero, then literal pool, and only fall back to movz+movk+fmov
as a last resort.

### 7.4 Address Computation

**Rule violated**: Rule 7 (Prefer immediate operands)

**Current 3-instruction pattern for non-zero-offset memory access**:
```asm
mov   w3, w24             ; I64ExtendI32U
add   x3, x3, #offset    ; add field offset
ldr   d11, [x21, x3]     ; base + computed address
```

**Optimal 2-instruction form**:
```asm
add   x3, x21, w24, uxtw ; base + zero-extended pointer
ldr   d11, [x3, #offset] ; immediate offset load
```

Zero-offset loads already fuse to `ldr d, [x21, w24, uxtw]`.

**Caveat on shifted index latency**: When using `ldr d, [base, index, lsl #3]`,
if the index register comes from a shift-like instruction, there is a +1
cycle penalty. Prefer computing the address with ADD first if the index
was recently shifted.

### 7.5 Instruction Scheduling

**Rule violated**: Rules 4 and 5 (Separate loads from consumers, schedule
around FDIV/FSQRT)

**Current**: Instructions emitted in program order without latency-awareness.
No reordering between load and first use. No interleaving of independent
computation around long-latency operations.

**Opportunity**: Block-level list scheduling after MachineIR generation.
Priority by critical path length. ARM64's 3-operand encoding makes this
straightforward (no destructive overwrite constraints).

**Scheduling priorities**:
1. FSQRT (13c latency, u14-only, TP=2): Schedule as early as possible
2. FDIV (10c latency, u14-only, TP=1): Schedule early; pipelined so
   consecutive independent FDIVs can overlap
3. Loads (4c latency): Insert 3+ independent instructions before first use
4. Domain crossings (5-13c): Schedule early, fill with same-domain work
5. FCMP (u14-only, 1c TP): Avoid clustering with FDIV/FSQRT

### 7.6 Callee-Saved Register Strategy

**Rule violated**: Rule 9 (Be aware of store pressure)

**Current**: All functions save/restore callee-saved registers (x19-x28,
d8-d15) unconditionally in prologue/epilogue.

**Optimizations**:
- Leaf functions (no calls): Use only caller-saved registers, eliminating
  prologue/epilogue entirely.
- Liveness-based save: Only save registers actually written in the function body.
- Reversed tier preference for leaf functions: Use caller-saved FP regs
  (d3-d7, d16-d31) before callee-saved (d8-d15).
- Context registers (mem0_base, mem0_size): Don't load/save if function never
  accesses Wasm linear memory.

### 7.7 Compare-and-Branch Patterns

**Rule violated**: Rule 6 (Enable hardware fusion)

**Current anti-pattern** (breaks CMP+B.cond fusion):
```asm
cmp   x3, x4
cset  w5, eq       ; intermediate result breaks fusion
cbnz  x5, target
```

**Optimal fused form**:
```asm
cmp   x3, x4
b.eq  target       ; fused with CMP into 1 uop
```

When the comparison result is used only as a branch condition, emit
compare and branch adjacently. Also consider ARM64's `CCMP` for chaining
multiple conditions without intermediate boolean materialization.

### 7.8 Bounds-Check Split Implicit Dependencies

**Current issue**: Continuation blocks after bounds-check splits carry
transient register values implicitly (not as block params). This prevents
safe copy propagation across block boundaries.

**Fix**: Thread live transient registers through continuation edges as
explicit block params. This is a correctness prerequisite for enabling
copy propagation.

### 7.9 Redundant Zero-Extension After Shift

**Current pattern**:
```asm
lsr   x3, x23, #52    ; result guaranteed < 2^12
mov   w26, w3          ; redundant zero-extension
```

**Peephole opportunity**: After `LSR Xd, Xn, #(≥32)`, the result already
fits in 32 bits. The following `MOV Wd, Wd` zero-extension is redundant
and can be eliminated.

Additional peephole: `AND Xd, Xn, #mask` where mask < 2^32 also guarantees
zero-extension. A subsequent `MOV Wd, Wd` is redundant.

### 7.10 FMADD/FMSUB Recognition

**Current**: FMUL followed by FADD/FSUB emitted as separate instructions
(6 cycles serial).

**Opportunity**: Fuse to FMADD/FMSUB (4 cycles, 1 instruction) when:
- FMUL result is transient and single-use
- FADD/FSUB uses it as one operand
- Precision semantics allow (single vs double rounding)

Note: Wasm MVP does not have FMA; this is a non-conformant optimization
that most programs won't notice but may affect floating-point reproducibility.

**Throughput benefit**: FMADD has TP 0.25 (4/cycle on u11-u14), same as
separate FADD or FMUL. But fusing saves 1 instruction slot in the 8-wide
decode window and reduces total uop count.

### 7.11 Duplicate Constant Elimination

**Current**: Same float constant can be materialized multiple times within
a block (5 instructions per occurrence).

**Peephole opportunity**: Track previously-materialized FP constants (as
the full movz+movk+fmov sequence). When the same bit pattern reappears,
replace with `fmov Dd, Dprev` (2c) instead of re-materializing (5 insns + 5-7c).

**Even better**: If using literal pool loading (Section 7.3), the constant is
in a register after one LDR. Subsequent uses are just register references
(0c if the register is still live).

### 7.12 FRECPE/FRSQRTE for Approximate Division

**Opportunity**: When full FDIV precision is not required (e.g., graphics,
physics simulation), replace FDIV with a FRECPE+FRECPS Newton-Raphson
sequence:

```asm
; Approximate 1/x with ~24-bit precision (vs FDIV's full precision)
frecpe d1, d0           ; 3c, u14-only: ~12-bit estimate
frecps d2, d0, d1       ; 4c, u11-u14: refinement step
fmul   d1, d1, d2       ; 3c, u11-u14: ~24-bit result
; Total: ~10c but better throughput than FDIV when many are needed
```

For reciprocal square root (replacing FSQRT(D) at 13c/2.0 TP):
```asm
frsqrte d1, d0          ; 3c, u14-only
frsqrts d2, d0, d1      ; 4c, u11-u14
fmul    d1, d1, d2      ; 3c, u11-u14
; ~10c total, significantly better than FSQRT's 13c latency + 2.0 TP
```

---

## 8. Cost Reference Tables

### Common Pattern Costs

| Pattern | Instructions | Latency (est.) | Better Alternative | Savings |
|---|---|---|---|---|
| FP copy (transient→cached) | `ldr d3; fmov d11, d3` (2) | 4+2=6c | `ldr d11, [...]` (1) | 1 insn, 2c |
| GP→FP float const | `movz+movk×3+fmov` (5) | 5-7c for fmov | `fmov d, #imm8` (1) | 4 insns, 3-5c |
| Literal pool const | `ldr d, [pc, #off]` (1) | 4c | — | vs GP: 4 insns, 1-3c |
| 3-insn address | `mov w,w; add; ldr` (3) | 1+1+4=6c | `add x,base,w,uxtw; ldr` (2) | 1 insn |
| Zero-init N slots | N × `str xzr` | N × 1c, N ports | N/2 × `stp xzr,xzr` | N/2 insns + ports |
| FP prologue (8 regs) | 8 × `str d` | 8c, 8 ports | 4 × `stp d,d` | 4 insns, 4 ports |
| fmul+fadd serial | 2 insns | 3+3=6c | `fmadd` (1) | 1 insn, 2c |
| CMP+CSET+CBNZ | 3 insns | 1+1+1=3c | `CMP+B.cond` (2, fused) | 1 insn + fusion |
| Conditional branch | 1 insn | 1c (predict) / 14c (mispredict) | `CSEL` (1) | 0-13c |
| DMB fence | 1 insn | 17c TP | LDAPR (if applicable) | ~14c |

### Port Pressure Quick Reference

| Port(s) | Operations | Bottleneck Risk |
|---|---|---|
| u1-u6 (6 INT) | ADD, SUB, CMP, MOV, shifts, branches (u1-u2) | LOW |
| u5-u6 (2 MUL) | MUL, MADD, CRC | MEDIUM |
| u7-u8 (2 STORE) | STR, STP | **HIGH — only 2 ports, max 60 in-flight** |
| u8-u10 (3 LOAD) | LDR, LDP | MEDIUM — 3 ports, 130 in-flight |
| u11-u14 (4 FP) | FADD, FMUL, FMOV d,d, NEON vector | MEDIUM — copies steal FP slots |
| u14 only (1 FP-special) | FDIV, FSQRT, FCMP, FRECPE, FRSQRTE | **HIGH — serialization risk** |

---

## 9. Apple Silicon Generational Reference

| Feature | M1 (Firestorm) | M2 (Avalanche) | M3 (Everest) | M4 |
|---|---|---|---|---|
| P-core frequency | 3.2 GHz | 3.5 GHz | 4.05 GHz | 4.4 GHz |
| E-core frequency | 2.06 GHz | 2.42 GHz | 2.75 GHz | 2.85 GHz |
| E-core name | Icestorm | Blizzard | Sawtooth | — |
| A-series equivalent | A14 | A15 | A16 | A17/A18 |
| ISA version | ARMv8.5-A | ARMv8.6-A | ARMv8.6-A | ARMv9.2-A |
| P/E core count | 4/4 | 4/4 | 4/4 | 4/6 |
| L1 I-cache (P) | 192 KB | 192 KB | 192 KB | 192 KB |
| L1 D-cache (P) | 128 KB | 128 KB | 128 KB | 128 KB |
| L2 cache (P) | 12 MB | 16 MB | 16 MB | 16 MB |
| L1 cache (E) | 64 KB | 64 KB | 64 KB | 64 KB |
| L2 cache (E) | 4 MB | 4 MB | 4 MB | 4 MB |
| Cache line size | 128 bytes | 128 bytes | 128 bytes | 128 bytes |
| NEON | 128-bit | 128-bit | 128-bit | 128-bit |
| SVE/SVE2 | No | No | No | SSVE only (in SME) |
| SME | No | No | No | Yes (SME2) |
| Memory technology | LPDDR4X | LPDDR5 | LPDDR5 | LPDDR5X |
| Max memory (base) | 8-16 GB | 8-24 GB | 8-24 GB | 16-32 GB |
| Memory bandwidth | 67 GB/s | 100 GB/s | 100 GB/s | 120 GB/s |
| AMX | FP16,32,64 | FP16,32,64,BF16 | FP16,32,64,BF16 | FP16,32,64,BF16 |
| Process node | 5nm | 5/4nm | 3nm | 3nm |
| GPU cores | 7-8 | 8-10 | 8-10 | 8-10 |
| GPU FP32 TFLOPS | 2.29-2.61 | 2.86-3.57 | 2.82-3.53 | 4.26 |
| CPU FP32 TFLOPS (vDSP) | 0.90 | 1.09 | 1.38 | 1.49 |

**Key takeaways for JIT optimization**:
- All generations share the same 6+4+4 execution port layout. Port-pressure
  optimizations apply universally across M1-M4.
- M4 (ARMv9.2-A) adds SME support but this doesn't affect scalar code.
  No standalone SVE is available — do not generate SVE instructions.
- Memory bandwidth increases generationally (67→120 GB/s), meaning
  memory-bound code improves automatically on newer chips.
- L1 cache sizes and cache line size (128B) are identical across all
  generations. Code/data locality optimizations have the same benefit
  everywhere.
- Each generation improves branch prediction accuracy and IPC incrementally,
  but the fundamental instruction costs and fusion rules remain the same.
- Power efficiency is exceptional: M-series CPUs achieve ~200 GFLOPS/W for
  optimized workloads (via Accelerate/AMX), competitive with dedicated HPC
  hardware at 10-20W total power draw.

---

## 10. References and Further Reading

- Dougall Johnson, "Apple Microarchitecture Research" — Firestorm/Icestorm
  reverse-engineered instruction tables:
  https://dougallj.github.io/applecpu/firestorm.html
- Apple, "CPU Optimization Guide" (Version 4):
  https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide
- Fabian Sidler, "Benchmarking M-series Apple CPUs" (ETH Zurich, Spring 2025):
  M1 Pro and M3 Max instruction-level benchmarking with PMU access
- Paul Hübner et al., "Apple vs. Oranges: Evaluating the Apple Silicon M-Series
  SoCs for HPC Performance and Efficiency" (KTH, arXiv:2502.05317, 2025):
  STREAM and GEMM benchmarks across M1-M4 with power measurements
- Karol Struniawski et al., "Exploring Apple Silicon's Potential from Simulation
  and Optimization Perspective" (ICCS 2024):
  ML classifier benchmarks on M1/M2 vs x86 and GPU
- Daniel Lemire, "Counting cycles and instructions on the Apple M1 processor":
  https://lemire.me/blog/2021/03/24/counting-cycles-and-instructions-on-the-apple-m1-processor/
- Dissecting Conditional Branch Predictors of Apple Firestorm and Qualcomm Oryon
  (arXiv:2411.13900, 2024)
- GoFetch: Data Memory-Dependent Prefetcher vulnerability analysis:
  https://gofetch.fail/
- ocxtal/insn_bench_aarch64: Independent M1 instruction benchmarks with
  optimization notes:
  https://github.com/ocxtal/insn_bench_aarch64
