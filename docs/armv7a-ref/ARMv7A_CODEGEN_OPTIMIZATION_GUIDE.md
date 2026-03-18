# ARMv7-A Codegen Optimization Guide

General-purpose reference for optimizing JIT-generated ARMv7-A (32-bit ARM)
code targeting Cortex-A class application processors. Distilled from ARM
Technical Reference Manuals (DDI0344, DDI0388, DDI0438, DDI0464), the ARM
Cortex-A Series Programmer's Guide (DEN0013D), the NEON Programmer's Guide
(DEN0018), WikiChip microarchitecture analyses, hardwarebug.org empirical
measurements, AnandTech architecture deep-dives, and Qualcomm Krait
documentation.

This document describes **what the hardware likes and dislikes**, what
codegen patterns are cheap or expensive, and what optimization opportunities
exist in an ARMv7-A backend. It is not tied to any specific benchmark.

Unlike the ARM64/Apple Silicon guide which targets a single vendor's
microarchitecture family, ARMv7-A encompasses multiple distinct
microarchitectures with significantly different characteristics. This guide
covers the four major ARM-designed Cortex-A cores (A7, A8, A9, A15) and
notes Qualcomm Krait where relevant. Optimizations are presented in order
of broadest applicability.

---

## 1. Microarchitecture Overview

### 1.1 Core Comparison Summary

| Feature | Cortex-A7 | Cortex-A8 | Cortex-A9 | Cortex-A15 | Krait (Qualcomm) |
|---|---|---|---|---|---|
| Pipeline depth (INT) | 8 stages | 13 stages | 8 stages | 15 stages | 11 stages |
| Pipeline depth (FP/NEON) | 8+ stages | 10 stages (NEON) | ~10 stages | 17-25 stages | 13+ stages |
| Issue width | Partial dual | Dual (in-order) | Dual (partial OoO) | 3-way (full OoO) | 4-way (full OoO) |
| Decode width | 1-2 | 2 | 2 | 3 | 3 |
| Execution model | In-order | In-order | Partial OoO | Full OoO | Full OoO |
| ROB entries | — | — | 32-40 | 128 | ~40+ |
| DMIPS/MHz | ~1.9 | ~2.0 | ~2.5 | ~3.5 | ~3.3 |
| VFP | VFPv4 | VFPLite (non-pipelined) | VFPv3 (pipelined) | VFPv4 (pipelined, OoO) | VFPv4 (pipelined) |
| NEON issue | Single-issue | Separate pipeline | In-order in INT pipeline | Out-of-order | 3-way |
| Integer divide | SDIV/UDIV | Not supported | Not supported | SDIV/UDIV | SDIV/UDIV |
| ISA version | ARMv7-A | ARMv7-A | ARMv7-A | ARMv7-A | ARMv7-A |
| Thumb-2 | Yes | Yes | Yes | Yes | Yes |
| big.LITTLE pair | With A15/A17 | — | — | With A7 | — |

### 1.2 Cortex-A8 Pipeline (In-Order Dual-Issue)

The Cortex-A8 is a 13-stage dual-issue in-order superscalar. The pipeline
is divided into:

```
Fetch:   F0 → F1 → F2           (3 stages, fetch + branch predict)
Decode:  D0 → D1 → D2 → D3 → D4 (5 stages, decode + register rename)
Execute: E0 → E1 → E2 → E3 → E4 (5 stages, execute + writeback)
```

**Dual-issue rules**: Two instructions issue together if they go to
different pipelines (ALU pipe 0 + ALU pipe 1, or ALU + MUL, or ALU + LS).
Two instructions to the same pipeline serialize.

**Execution units**:
- 2 symmetric ALU pipelines (ADD, SUB, MOV, logical, shift)
- 1 multiply pipeline (MUL, MLA, SMULL, etc.)
- 1 load/store pipeline
- 1 NEON/VFP pipeline (separate 10-stage NEON, non-pipelined VFPLite)

**Key constraint**: Only 1 load/store per cycle. Load-heavy code serializes.

### 1.3 Cortex-A9 Pipeline (Partial Out-of-Order)

The Cortex-A9 is an 8-stage partially out-of-order dual-issue superscalar.
It can dispatch up to 2 instructions per cycle to 4 execution pipelines,
with limited reordering (ROB: 32-40 entries).

```
Fetch:    F0 → F1 → F2    (3 stages)
Decode:   D0 → D1          (2 stages, decode + rename)
Execute:  E0 → E1 → E2    (3 stages, variable per unit)
```

**Execution units**:
- 2 ALU pipelines (ALU0 handles shifts + MUL; ALU1 handles branches)
- 1 load/store pipeline (with 4-entry merging store buffer)
- 1 FP/NEON pipeline (pipelined VFPv3, in-order NEON execution)

**Key advantage over A8**: Out-of-order execution hides some load latency
automatically. Pipelined VFP (not VFPLite) makes scalar FP much faster.

### 1.4 Cortex-A15 Pipeline (Full Out-of-Order)

The Cortex-A15 is a 15-stage 3-wide out-of-order superscalar — the most
aggressive ARMv7-A microarchitecture ARM designed.

```
Fetch:    5 stages (4 instructions/cycle)
Decode:   3 stages (3 instructions/cycle, micro-op expansion)
Rename:   2 stages
Dispatch: 1 stage  (up to 8 micro-ops/cycle to issue queues)
Execute:  variable (2+ stages per unit)
Retire:   2 stages (128-entry ROB)
```

**Execution units**:
- 2 integer ALU pipelines
- 1 integer multiply/divide pipeline
- 1 branch pipeline
- 1 load pipeline
- 1 store pipeline (simultaneous load + store per cycle)
- 2 FP/NEON pipelines (out-of-order execution for both FP and NEON)

**Key advantages**: 128-entry ROB provides substantial out-of-order window.
Both integer and NEON/FP execute out-of-order (A9 NEON is in-order).
Simultaneous load + store per cycle (A8/A9: only 1 LS operation per cycle).

### 1.5 Cortex-A7 Pipeline (In-Order, Low Power)

The Cortex-A7 is an 8-stage partial dual-issue in-order core designed for
power efficiency. It is the little core in big.LITTLE configurations with
A15 or A17.

**Dual-issue rules**: Can dual-issue two integer instructions. Multiply,
divide, NEON, load/store, and branch are single-issue only.

**Key property**: Architecturally compatible with A15 (same ISA features
including integer divide) but with much simpler microarchitecture.

---

## 2. Instruction Latency and Throughput Tables

TP = throughput (instructions per cycle; higher = better). Latency =
cycles from issue to result available.

### 2.1 Integer Instructions

| Instruction | A7 Lat | A8 Lat | A9 Lat | A15 Lat | Notes |
|---|---|---|---|---|---|
| ADD/SUB (reg/imm) | 1 | 1 | 1 | 1 | All cores: 1 cycle |
| ADD/SUB (shifted reg) | 2 | 1-2 | 1-2 | 1-2 | A7: extra cycle for shift; others: depends on shift type |
| MOV (reg) | 1 | 1 | 1 | 1 | No elimination on any ARMv7 core |
| MOV (imm) | 1 | 1 | 1 | 1 | MOVW/MOVT for 32-bit constants |
| AND/ORR/EOR/BIC | 1 | 1 | 1 | 1 | |
| LSL/LSR/ASR (reg) | 2 | 1 | 1 | 1 | A7: 2c for variable shift |
| CLZ | 1 | 1 | 1 | 1 | |
| RBIT/REV | 2 | 1 | 1 | 1 | A7: 2 cycles |
| MUL | 3 | 2 | 2-4 | 3 | A8: 32×8 early termination; A9: data-dependent |
| MLA | 3 | 2 | 2-4 | 3 | Multiply-accumulate |
| SMULL/UMULL | 3 | 3 | 3-5 | 3 | 64-bit result |
| SMLAL/UMLAL | 3 | 3 | 3-5 | 3 | 64-bit accumulate |
| SDIV (A7/A15 only) | 6-22 | — | — | 4-20 | Data-dependent; NOT on A8/A9 |
| UDIV (A7/A15 only) | 5-21 | — | — | 4-20 | Data-dependent; NOT on A8/A9 |
| CMP/CMN/TST/TEQ | 1 | 1 | 1 | 1 | Sets flags only |

**Notes on multiply**:
- A8 uses early termination based on leading zeros/ones in the multiplier.
  Small multipliers (8-bit range) complete in fewer cycles.
- A9 multiply latency is data-dependent (2-4 cycles for 32-bit result).
- MLA accumulator forwarding: On A15, the accumulator input has a shorter
  dependency path than the multiplicands (1 cycle shorter).

### 2.2 Load/Store Instructions

| Instruction | A7 Lat | A8 Lat | A9 Lat | A15 Lat | Notes |
|---|---|---|---|---|---|
| LDR [Rn, #imm] | 3 | 2 | 1-4 | 4 | A9: 1c if forwarded from store buffer |
| LDR [Rn, Rm] | 3 | 2 | 1-4 | 4 | Register offset |
| LDR [Rn, Rm, LSL #n] | 3 | 3 | 2-4 | 4-5 | Scaled register: +1c on A8, A15 |
| LDRB/LDRH | 3 | 2 | 1-4 | 4 | Same as LDR |
| LDRD | 3-4 | 3 | 2-4 | 4-5 | Double-word load; 2 uops on A15 |
| LDM (N regs) | 3+N-1 | 1+N | varies | varies | Microcode; blocks pipeline |
| STR [Rn, #imm] | 2 | 1 | 1 | 1 | Store: pipelined, non-blocking |
| STRD | 2 | 1 | 1 | 1 | Double-word store |
| STM (N regs) | 2+N-1 | N | varies | varies | Microcode; blocks pipeline |
| LDR (PC-rel) | 3 | 2 | 1-4 | 4 | Literal pool load |
| PLD [addr] | — | — | — | — | Prefetch hint; no result register |

**Critical differences**:
- **A8**: Load latency is 2 cycles (L1 hit). This is short — interleaving
  1 independent instruction between load and use is often sufficient.
- **A9**: Load latency varies; store-to-load forwarding can give 1-cycle
  effective latency. Out-of-order execution hides some stalls.
- **A15**: Load latency is 4 cycles (similar to Apple Silicon). The large
  ROB and OoO execution help, but explicit scheduling still helps.
- **A7**: Load latency is 3 cycles. In-order execution means every stall
  is fully exposed.

### 2.3 VFP (Scalar Floating-Point)

| Instruction | A7 Lat | A8 Lat | A9 Lat | A15 Lat | Notes |
|---|---|---|---|---|---|
| VADD.F32 | 4 | 8-10 | 4 | 4-6 | A8: non-pipelined VFPLite! |
| VADD.F64 | 4 | 8-10 | 5 | 4-6 | A8: even worse for double |
| VMUL.F32 | 4 | 9-12 | 5 | 5-6 | |
| VMUL.F64 | 7 | 9-12 | 6 | 5-6 | A7: F64 much slower |
| VMLA.F32 | 7 | 18-21 | 8 | 9-10 | A8: catastrophically slow |
| VMLA.F64 | 11 | 18-21 | 9 | 9-10 | Fused multiply-accumulate |
| VFMA.F32 (A7/A15) | 7 | — | — | 9-10 | VFPv4 FMA; not on A8/A9 |
| VDIV.F32 | 18 | 20-37 | 15 | 10-14 | All cores: very expensive |
| VDIV.F64 | 32 | 29-65 | 25 | 15-29 | Double: much worse |
| VSQRT.F32 | 17 | 19-33 | 14 | 10-14 | Similar to VDIV |
| VSQRT.F64 | 31 | 29-60 | 24 | 15-29 | |
| VABS/VNEG | 4 | 1-2 | 1 | 3-4 | |
| VCMP | 4 | 1-2 | 1 | 3-4 | Compare, sets FPSCR |
| VMRS APSR_nzcv | 1 | 1 | 1 | 1 | Move FPSCR flags → APSR |
| VCVT (int↔float) | 4 | 6-8 | 4-5 | 4-6 | Conversion |
| VMOV (Rd, Sn) | 1 | 1 | 1 | 2 | FP → GP transfer |
| VMOV (Sn, Rd) | 1 | 1 | 1 | 2 | GP → FP transfer |

**Critical warning about Cortex-A8 VFP**: The A8 uses a non-pipelined
VFPLite coprocessor for scalar VFP operations. A single-precision
multiply-accumulate takes 18-21 cycles — roughly 10× slower than using
NEON for the same operation. **On A8, always prefer NEON instructions
(VMUL.F32, VADD.F32 on Q/D registers) over scalar VFP.**

### 2.4 NEON (SIMD) Instructions

| Instruction | A7 Lat | A8 Lat | A9 Lat | A15 Lat | Notes |
|---|---|---|---|---|---|
| VADD.I{8,16,32} | 4 | 1-2 | 3-4 | 3-4 | Integer SIMD add |
| VADD.F32 (D/Q) | 4 | 5 | 5 | 4-6 | NEON FP add |
| VMUL.I{8,16,32} | 4 | 2-4 | 5-6 | 4-6 | Integer SIMD multiply |
| VMUL.F32 (D/Q) | 4 | 5 | 5-6 | 5-6 | NEON FP multiply |
| VMLA.F32 (D/Q) | 7 | 8-9 | 8-9 | 9-10 | NEON FP multiply-accumulate |
| VFMA.F32 (A7/A15) | 7 | — | — | 9-10 | Fused; VFPv4/NEONv2 only |
| VABS/VNEG (int) | 4 | 1-2 | 3-4 | 3-4 | |
| VORR/VAND/VEOR | 4 | 1-2 | 1-3 | 3-4 | Bitwise logical |
| VDUP (from GP) | 4 | 2 | 3-4 | 3-4 | Broadcast scalar to lanes |
| VTBL/VTBX | 4 | 2-3 | 3-4 | 3-4 | Table lookup (permute) |
| VZIP/VUZP/VTRN | 4 | 2 | 2-4 | 3-4 | Interleave/deinterleave |
| VEXT | 4 | 1-2 | 2-3 | 3-4 | Extract (byte-level concat + shift) |
| VLD1 (1 reg, aligned) | 3 | 1-2 | 2-4 | 4 | NEON load |
| VLD1 (4 regs, aligned) | 3+ | 3-5 | varies | 4+ | Multi-register load |
| VST1 (1 reg, aligned) | 2 | 1-2 | 1-3 | 1 | NEON store |

**NEON pipeline differences are enormous**:
- **A8**: NEON has its own dedicated 10-stage pipeline, decoupled from the
  integer pipeline. NEON→ARM register transfers incur ~20 cycle penalty.
  NEON operates in-order with limited forwarding.
- **A9**: NEON executes in the integer pipeline's issue stage, still
  in-order. Pipelined VFP means NEON FP and scalar VFP have similar
  throughput (unlike A8).
- **A15**: NEON executes out-of-order alongside integer instructions. Two
  NEON/FP pipelines allow 2 NEON instructions per cycle.
- **A7**: NEON is single-issue. No dual-issue with NEON instructions.

### 2.5 GP ↔ NEON/VFP Domain Crossings

| Instruction | A7 | A8 | A9 | A15 | Notes |
|---|---|---|---|---|---|
| VMOV Rd, Sn (FP→GP) | 1 | 1 | 1 | 2 | Single-precision to GP |
| VMOV Sn, Rd (GP→FP) | 1 | 1 | 1 | 2 | GP to single-precision |
| VMOV Rd, Rd, Dn (FP→GP pair) | 1-2 | 2 | 2 | 4 | 64-bit: 2 GP regs |
| VMOV Dn, Rd, Rd (GP pair→FP) | 1-2 | 2 | 2 | 4 | 2 GP regs to D-reg |
| VMOV.32 Dd[i], Rd (lane insert) | 4 | ~7-20 | 3-4 | 3-4 | A8: extreme penalty |

**A8 NEON↔ARM penalty**: On Cortex-A8, transferring data from NEON
registers to ARM registers (VMOV Rd, Sn or VMRS) incurs a **~20 cycle
pipeline drain** because the NEON pipeline runs ahead and must synchronize
with the integer pipeline. This is the single most important hazard to
avoid on A8. On A9/A15, this penalty is much smaller (1-4 cycles).

### 2.6 Branch Instructions

| Instruction | A7 | A8 | A9 | A15 | Notes |
|---|---|---|---|---|---|
| B (unconditional) | 0 (predicted) | 0 | 0 | 0 | Predicted taken |
| B.cond (taken, predicted) | 1 | 1 | 1 | 1 | |
| B.cond (mispredicted) | ~8 | **13** | **11-13** | **~14** | Full pipeline flush |
| BL (call) | 1 | 1 | 1 | 1 | Link register saved |
| BX Rm (indirect) | 1 | 1-13 | 1-13 | 1-14 | Misprediction varies |
| BLX Rm (indirect call) | 1 | 1-13 | 1-13 | 1-14 | |
| IT block (Thumb-2) | 0-1 | 0-1 | 0-1 | 0-1 | Predicated execution |

**Misprediction penalty summary**:
- **A7**: ~8 cycles (short pipeline helps)
- **A8**: 13 cycles (deep pipeline, in-order — full stall)
- **A9**: 11-13 cycles
- **A15**: ~14 cycles (deepest pipeline)
- **Krait**: ~11 cycles

---

## 3. Cache Hierarchy and Memory Subsystem

### 3.1 Cache Parameters

| Parameter | Cortex-A7 | Cortex-A8 | Cortex-A9 | Cortex-A15 |
|---|---|---|---|---|
| L1 I-cache | 4-64 KB | 16-32 KB | 32 KB | 32 KB |
| L1 D-cache | 4-64 KB | 16-32 KB | 32 KB | 32 KB |
| L1 D-cache assoc. | 2-way | 4-way | 4-way | 2-way |
| L1 line size | 32-64 B | 64 B | 32 B (D) / 64 B (I) | 64 B |
| L2 cache | 0-1 MB | 0-1 MB | 0-4 MB (ext.) | 512 KB-4 MB |
| L2 assoc. | varies | 8-way | varies (L2C-310) | 16-way |
| L2 line size | 64 B | 64 B | varies | 64 B |
| Store buffer | small | 2 entries | 4-entry, 64-bit | 6 linefill buffers |
| Write buffer | — | 1 entry | 1 eviction buffer | — |

**Key differences from Apple Silicon (ARM64)**:
- Cache lines are **32-64 bytes** (vs Apple Silicon's 128 bytes).
- L1 D-cache is typically **32 KB** (vs Apple Silicon's 128 KB).
- L2 caches are significantly smaller and higher latency.
- No data memory-dependent prefetcher (DMP). Software prefetch (PLD)
  is the primary mechanism.

### 3.2 Cache Latencies

| Access | A7 | A8 | A9 | A15 | Notes |
|---|---|---|---|---|---|
| L1 D-cache hit | 3c | 1-2c | 1-4c | 4c | A8 is fastest |
| L2 cache hit | ~10-15c | ~8-9c | ~7-12c | ~11-20c | Highly variable by config |
| Main memory | varies | varies | varies | varies | 50-200 ns typical |
| L1 I-cache miss | ~10c | ~8c | ~8c | ~12c | Fetch stall |

### 3.3 TLB Structure

| Parameter | Cortex-A7 | Cortex-A8 | Cortex-A9 | Cortex-A15 |
|---|---|---|---|---|
| ITLB | varies | 32 entries FA | 32 entries FA | 32 entries FA |
| DTLB | varies | 32 entries FA | 32 entries FA | 2×32 entries FA |
| Main TLB | varies | — | 64-512, 2-way | 512, 4-way |
| Page sizes | 4K/64K/1M/16M | 4K/64K/1M/16M | 4K/64K/1M/16M | 4K/64K/1M/16M + LPAE |
| DTLB miss | ~10c | ~10c | ~10c | ~10-30c |

With 4 KB pages and 32 DTLB entries, the directly-mapped window is only
128 KB. Working sets larger than this will see frequent TLB misses. Use
PLD (prefetch) and organize data for spatial locality.

### 3.4 Prefetch

ARMv7-A provides the PLD (Preload Data) instruction as a software prefetch
hint. Hardware prefetchers vary:

- **A8**: No hardware stride prefetcher. PLD is essential for streaming
  access patterns. PLD issues a cache linefill to L2 (or L1 on some
  configurations). Optimal PLD distance: 3-5 cache lines ahead.
- **A9**: Basic hardware prefetcher in some configurations. PLD still
  beneficial.
- **A15**: Automatic hardware prefetcher with stride detection. PLD less
  critical but still useful for non-stride patterns.
- **A7**: Minimal hardware prefetching. PLD important.

**PLD guidelines**:
- Issue PLD 3-5 cache lines (192-320 bytes for 64B lines) ahead of use
- PLD is a hint; it cannot cause faults
- Excessive PLD wastes memory bandwidth and can evict useful data
- PLD to the same cache line as a pending load is a NOP (no harm, no benefit)

---

## 4. Branch Prediction

### 4.1 Predictor Architectures

**Cortex-A8**:
- Two-level global history predictor
- 512-entry BTB, 2-way set associative
- 4096-entry Global History Buffer (2-bit saturating counters)
- 8-entry return stack (for BL/BX lr patterns)
- Prediction accuracy: >95% on typical code
- Misprediction penalty: **13 cycles**

**Cortex-A9**:
- Hybrid predictor (2-level dynamic)
- Configurable GHB: 1024-16384 entries
- Branch Target Address Cache (BTAC)
- Return stack for subroutine returns
- Misprediction penalty: **11-13 cycles**

**Cortex-A15**:
- Bi-mode predictor (hybrid): 2 Pattern History Tables (8192 entries each)
  with a choice predictor to select between them
- 64-entry BTB (fully associative, taken branches only)
- 256-entry indirect predictor (XOR of branch history and address)
- Misprediction penalty: **~14 cycles**
- Fetches 4 instructions/cycle (twice A8/A9)

### 4.2 Implications for Codegen

- **Misprediction is the #1 performance hazard** on in-order cores (A7, A8).
  A misprediction on A8 costs 13 cycles — potentially 26 instructions of
  dual-issue work.
- **IT blocks provide predicated execution** (Thumb-2): For simple 1-4
  instruction sequences, `IT` blocks avoid branches entirely. On A8/A9,
  predicated instructions in IT blocks execute in 1 cycle regardless of
  condition.
- **Prefer conditional execution over branches** for small bodies:
  `MOVEQ`/`MOVNE` or IT-block predication eliminates misprediction risk.
- **Indirect branch prediction** is limited: A8/A9 have small indirect
  target buffers. Interpreter dispatch tables with many targets will
  suffer high misprediction rates.
- **Return stack**: All cores have return stacks (8 entries on A8).
  BL/BLX for calls and BX LR for returns enables the predictor. Avoid
  non-standard return patterns (e.g., POP {PC} instead of BX LR is fine
  on most cores, but BL to a function that returns with a computed branch
  defeats the return stack).

---

## 5. Codegen Rules Derived from Microarchitecture

### Rule 1: Avoid NEON↔ARM Register Transfers (Especially on A8)

On Cortex-A8, moving data from a NEON/VFP register to an ARM register
(VMOV Rd, Sn or VMRS APSR_nzcv) causes the integer pipeline to stall
for **~20 cycles** while the decoupled NEON pipeline drains. This is
because the A8 NEON pipeline runs ahead of the integer pipeline and has
no fast forwarding path back.

- **Never use VMOV/VMRS in a hot loop on A8.** If you need to branch on
  a floating-point comparison result, structure the code to do all NEON
  work first, then transfer and branch.
- **A9/A15**: The penalty is 1-4 cycles (much less severe), but still
  avoid unnecessary transfers.
- **Alternative for A8**: Keep all computation in NEON registers if
  possible. Use NEON comparisons and bit-select (VBSL) to avoid
  transferring condition flags to ARM.

### Rule 2: Use NEON Instead of VFP on Cortex-A8

The A8 VFPLite is non-pipelined. A scalar VMLA.F32 takes 18-21 cycles.
The equivalent NEON operation (VMLA.F32 Dd, Dn, Dm) takes ~8-9 cycles
and processes 2 floats simultaneously.

- **Always vectorize scalar FP to NEON on A8.** Even for scalar work,
  using NEON Dd (64-bit) registers for single FP operations is faster
  than using VFP Sn registers.
- **A9/A15**: VFP is fully pipelined; scalar VFP and NEON have comparable
  throughput. The choice is less critical.
- **Exception**: VFP supports double-precision; NEON on ARMv7 does **not**
  support F64 operations. For double-precision on A8, VFPLite is the only
  option (and it's painfully slow).

### Rule 3: Avoid LDM/STM in Performance-Critical Code

LDM (Load Multiple) and STM (Store Multiple) are microcode sequences
that block the pipeline for N cycles (where N is the number of registers).
During this time, no other instructions can issue.

- **Prefer sequences of LDR/STR or LDRD/STRD** for small register counts
  (≤4 registers). These can interleave with other work on A9/A15.
- **LDRD/STRD** (double-word) loads/stores 2 registers in ~1 extra cycle
  over a single LDR/STR, and is much more efficient than 2 separate
  LDR/STR instructions.
- **Exception**: For function prologue/epilogue with many registers
  (PUSH/POP of 8+ callee-saved registers), LDM/STM is still compact and
  acceptable because the pipeline drain happens at a non-critical point.
- **A15**: LDM/STM is decomposed into micro-ops and can execute somewhat
  out-of-order. The penalty is less severe but still worse than individual
  loads/stores for small counts.

### Rule 4: Schedule Loads Away from Consumers

Load latency ranges from 2 cycles (A8 L1) to 4 cycles (A15 L1). On
in-order cores (A7, A8), every cycle between load and first use that
lacks independent work is a full stall.

**Anti-pattern** (A8, 2-cycle stall):
```asm
ldr   r3, [r0, #8]     @ 2c latency
add   r4, r3, r5       @ stalls 1c waiting for r3
```

**Better** (A8):
```asm
ldr   r3, [r0, #8]     @ 2c latency starts
add   r6, r7, r8       @ independent work (fills 1c)
add   r4, r3, r5       @ r3 ready
```

**For A15** (4c latency), insert 3 independent instructions between load
and first use. The out-of-order engine helps, but explicit scheduling
still improves performance because the ROB is finite (128 entries).

### Rule 5: Use Conditional Execution and IT Blocks

ARMv7-A provides two mechanisms for branchless conditional execution:
1. **ARM mode**: Condition codes on any instruction (ADDEQ, MOVNE, etc.)
2. **Thumb-2 mode**: IT (If-Then) blocks, up to 4 predicated instructions

Both eliminate branch misprediction risk entirely.

**Anti-pattern** (unpredictable branch):
```asm
cmp   r0, #0
beq   .use_default    @ 13c misprediction on A8
mov   r1, r2
b     .done
.use_default:
mov   r1, r3
.done:
```

**Better** (ARM mode):
```asm
cmp   r0, #0
movne r1, r2
moveq r1, r3           @ always 1 cycle, no branch
```

**Better** (Thumb-2 mode):
```asm
cmp   r0, #0
ite   ne               @ If-Then-Else
movne r1, r2
moveq r1, r3
```

**Guidelines**:
- IT blocks of 1-4 instructions: always prefer over short branches
- Both paths of IT must execute (wasted work on the "wrong" path)
- For longer sequences (>4 instructions), branches are better because
  the wasted execution exceeds the misprediction cost on average
- A15's branch predictor is excellent; the break-even point is lower

### Rule 6: Align NEON Loads and Stores

NEON load/store alignment has dramatic performance impact, especially
on Cortex-A8:

- **Aligned VLD1/VST1**: 1-2 cycles per register
- **Unaligned VLD1/VST1 on A8**: Up to **9 extra cycles** per operation
  (number of Q registers + 1). This is a 5-10× slowdown.
- **A9/A15**: Unaligned penalty is smaller (~1-2 extra cycles) but
  still measurable.

**Always specify alignment hints**: `VLD1.32 {d0, d1}, [r0:128]` tells
the hardware the address is 128-bit aligned. Even on cores where
unaligned access works, the alignment hint enables faster paths.

**Data structure alignment**: Ensure NEON-accessed arrays are aligned to
at least 8 bytes (64-bit), preferably 16 bytes (128-bit). Use
`__attribute__((aligned(16)))` or manual alignment in the allocator.

### Rule 7: Minimize Store Buffer Pressure

ARMv7-A cores have very small store buffers:
- **A8**: 2 write buffer entries
- **A9**: 4-entry merging store buffer + 1 eviction buffer
- **A15**: 6 linefill buffers

When the store buffer is full, the pipeline stalls. This is especially
problematic in function prologues (PUSH) and stack frame initialization.

**Optimizations**:
- **STRD instead of 2× STR**: Saves 1 store buffer slot for 2 registers
- **Interleave stores with computation**: Don't emit consecutive stores;
  put independent ALU work between them
- **Minimize callee-saved registers**: Use fewer registers to reduce
  prologue/epilogue store/load count
- **STR + writeback addressing**: `STR Rd, [Rn, #offset]!` combines
  address update with store, saving an ADD instruction

### Rule 8: Use MOVW/MOVT for 32-Bit Constants

ARMv7-A introduced MOVW (move wide) and MOVT (move top) for loading
arbitrary 32-bit constants in 2 instructions:

```asm
movw  r0, #0x1234      @ lower 16 bits
movt  r0, #0x5678      @ upper 16 bits
@ r0 = 0x56781234
```

**Alternatives and tradeoffs**:
- **LDR Rd, [PC, #offset]** (literal pool): 1 instruction but incurs
  load latency (2-4 cycles) and uses a load port. Consumes D-cache space.
- **MOVW+MOVT**: 2 instructions, 2 cycles, no memory access, no cache
  pressure. On A8, these can dual-issue with other ALU work.
- **MOV+ORR sequence**: Older technique, more instructions. MOVW/MOVT
  is strictly better on ARMv7.

**For float constants**: Load from literal pool (VLDR Sd, [PC, #offset])
is usually best. NEON/VFP has no MOVW equivalent.

### Rule 9: Exploit LDRD/STRD for Adjacent Memory Operations

LDRD and STRD load/store a pair of 32-bit registers from adjacent memory
addresses in a single instruction. This is the ARMv7-A equivalent of
ARM64's LDP/STP.

**Benefits**:
- 1 instruction instead of 2 LDR/STR
- Fewer instruction cache bytes
- On A15: decomposes to 2 micro-ops but uses only 1 instruction slot
- On A8: issues as a single operation (slightly cheaper than 2 LDR)

**Constraints**:
- Destination registers must be consecutive even-odd pair (R0-R1, R2-R3,
  etc.) on ARMv7-A (relaxed in Thumb-2 encoding for some cores)
- Address must be word-aligned (4-byte)
- LDRD latency: first register same as LDR; second register +1 cycle

**Apply to**: Prologue/epilogue callee-save/restore, struct field access
for adjacent fields, stack spill/reload of pairs.

### Rule 10: Avoid Integer Division Where Possible

Integer division (SDIV/UDIV) is only available on Cortex-A7, A15, and
Krait — **NOT on Cortex-A8 or A9**. On cores that support it, division
is data-dependent and expensive:

- **A7**: SDIV 6-22 cycles, UDIV 5-21 cycles
- **A15**: SDIV/UDIV 4-20 cycles

**Alternatives**:
- **Multiply by reciprocal**: For division by a compile-time constant,
  use the magic number multiplication technique: `x / d ≈ (x * M) >> S`
  where M and S are precomputed. This takes 2-3 instructions (MUL + shift).
- **Shift for power-of-2**: `x / 8` → `ASR r0, r0, #3` (1 cycle)
- **A8/A9 fallback**: Must use a software division routine (libgcc's
  `__aeabi_idiv`). This is ~20-50 cycles depending on operand values.
  Avoid division in hot paths on these cores.

### Rule 11: Prefer Thumb-2 for Code Density

Thumb-2 instructions are 16 or 32 bits (vs ARM's fixed 32 bits). For
the same logic, Thumb-2 code is typically 25-35% smaller.

**Why this matters for performance**:
- Smaller code → better I-cache utilization (critical with 16-32 KB L1 I-cache)
- Fewer I-cache misses → fewer stalls
- All Cortex-A cores have full Thumb-2 support with no performance penalty
  for most instructions
- **A15**: The wider fetch (4 instructions/cycle) benefits from compact code

**Exceptions where ARM mode may be better**:
- Predicated instruction sequences (ARM supports condition on every
  instruction; Thumb-2 needs IT blocks with 4-instruction limit)
- Very hot inner loops where alignment matters more than density

### Rule 12: Use PLD for Streaming Access Patterns

On cores without hardware prefetchers (A7, A8) or with weak prefetchers
(A9), software prefetch (PLD) is essential for streaming data access.

```asm
@ Prefetch 4 cache lines ahead in a processing loop
pld   [r0, #256]        @ prefetch 256 bytes ahead (4 × 64B lines)
vld1.32 {q0}, [r0]!     @ process current data
@ ... processing ...
```

**PLD guidelines per core**:
- **A8**: PLD to L2 takes ~8 cycles. Prefetch 3-5 lines ahead. PLD is
  a NOP if the line is already in cache.
- **A9**: PLD triggers linefill. 2-4 lines ahead is typical.
- **A15**: Hardware prefetcher handles regular strides. PLD useful for
  irregular patterns. Don't over-prefetch (wastes bandwidth).
- **A7**: PLD important due to limited hardware prefetching.

### Rule 13: Be Aware of Dual-Issue Pairing Rules (A8)

On Cortex-A8, dual-issue only works when consecutive instructions go to
different execution units. Understanding the pairing rules is critical
for in-order performance:

**Can dual-issue** (Pipeline 0 + Pipeline 1):
```asm
add   r0, r1, r2       @ ALU pipe 0
sub   r3, r4, r5       @ ALU pipe 1
```

**Can dual-issue** (ALU + Load/Store):
```asm
add   r0, r1, r2       @ ALU
ldr   r3, [r4]         @ Load/Store
```

**Cannot dual-issue** (same pipeline):
```asm
mul   r0, r1, r2       @ Multiply pipe
mla   r3, r4, r5, r6   @ Multiply pipe (same unit → serialize)
```

**Cannot dual-issue** (data dependency):
```asm
add   r0, r1, r2       @ Writes r0
sub   r3, r0, r5       @ Reads r0 (RAW hazard)
```

**Scheduling strategy for A8**: Alternate between different execution
unit types. Interleave ALU, load/store, and multiply instructions.

### Rule 14: Minimize ARM↔Thumb Interworking

Switching between ARM and Thumb instruction sets requires BX/BLX
instructions and may cause pipeline flushes on some cores.

- **Stay in one mode** (preferably Thumb-2) for an entire hot path
- Indirect calls (BLX Rm) handle mode switching automatically
- Direct calls to known-mode targets don't need interworking stubs
- A15's wide decode naturally handles Thumb-2 efficiently

---

## 6. NEON Optimization Details

### 6.1 NEON Architecture on ARMv7-A

NEON provides 128-bit SIMD with 32 × 64-bit registers (D0-D31) that can
be viewed as 16 × 128-bit registers (Q0-Q15). Each Q register overlaps
two D registers (Q0 = D0:D1).

**Data types supported**: I8, I16, I32, I64, F32 (no F64 in NEON).

**Key ISA features**:
- 64-bit (D-register) and 128-bit (Q-register) operations
- Multiply-accumulate (VMLA, VMLAL with widening)
- Pairwise operations (VPADD, VPMAX, VPMIN)
- Table lookup (VTBL, VTBX) for arbitrary permutations
- Structure load/store (VLD1-VLD4, VST1-VST4) with interleave/deinterleave
- Saturation arithmetic (VQADD, VQSUB, VQMUL)

### 6.2 NEON Structure Loads (VLDn/VSTn)

NEON provides specialized multi-element structure loads that interleave
or deinterleave data during the load/store:

| Instruction | Effect | Typical Use |
|---|---|---|
| VLD1 | Load 1-4 registers, no interleave | Contiguous array access |
| VLD2 | Load 2 registers, deinterleave by 2 | Stereo audio, complex numbers |
| VLD3 | Load 3 registers, deinterleave by 3 | RGB pixel data |
| VLD4 | Load 4 registers, deinterleave by 4 | RGBA pixel data |

**Performance**: VLD2-VLD4 have higher latency than VLD1 (extra cycles
for the permutation) but save explicit VZIP/VUZP instructions. On A8,
VLD3/VLD4 are significantly slower (5-8 cycles). On A15, the cost is
lower due to OoO execution hiding latency.

### 6.3 NEON Accumulator Hazards

On Cortex-A8 and A9, NEON multiply-accumulate (VMLA) instructions have
a hazard when the accumulator dependency chain is tight:

```asm
@ A8 hazard: VMLA result → VMLA accumulator
vmla.f32  q0, q1, q2    @ writes q0 in cycle N+8
vmla.f32  q0, q3, q4    @ reads q0: stalls until cycle N+8
```

**Mitigation**: Unroll and use multiple accumulators:
```asm
vmla.f32  q0, q1, q2    @ accumulator 0
vmla.f32  q8, q3, q4    @ accumulator 1 (independent)
vmla.f32  q0, q5, q6    @ accumulator 0 (q0 ready by now)
vmla.f32  q8, q7, q9    @ accumulator 1
@ Final: vadd.f32 q0, q0, q8
```

Using 2-4 independent accumulators hides the VMLA latency and keeps the
NEON pipeline full.

### 6.4 NEON Reduction Patterns

Horizontal reductions (summing all lanes) require cross-lane operations:

```asm
@ Sum all 4 floats in q0
vpadd.f32 d0, d0, d1    @ pairwise add: d0 = {s0+s1, s2+s3}
vpadd.f32 d0, d0, d0    @ pairwise add: d0 = {sum, sum}
@ Result in s0
```

**Cost**: 2 pairwise adds × ~5 cycles each. Minimize reductions in inner
loops; accumulate in vectors and reduce once after the loop.

### 6.5 No Double-Precision in NEON

ARMv7-A NEON does **not** support F64 (double-precision float). Only
the VFP unit handles F64. This means:
- F64 code cannot be vectorized using NEON
- On A8, F64 is limited to the non-pipelined VFPLite (18-65 cycles per op)
- On A9/A15, scalar VFP handles F64 with reasonable performance
- If possible, use F32 for NEON-amenable code paths

---

## 7. Memory Ordering and Synchronization

### 7.1 Barrier Instructions

| Instruction | Effect | Typical Cost |
|---|---|---|
| DMB | Data Memory Barrier: orders all data memory accesses | 10-40 cycles (core-dependent) |
| DMB ST | DMB for stores only (lighter) | ~10-20 cycles |
| DMB ISH | Inner-shareable DMB (multicore) | 10-40 cycles |
| DSB | Data Synchronization Barrier: like DMB + waits for completion | 20-50+ cycles |
| ISB | Instruction Synchronization Barrier: pipeline flush | **30-60+ cycles** |

**A8**: Barriers are relatively cheap because it's a simple in-order core
with limited memory reordering. DMB is still ~10-15 cycles.

**A15**: Barriers are expensive because the OoO engine must drain. DMB can
cost 20-40+ cycles depending on outstanding memory operations. DSB is
worse. ISB causes a full pipeline flush (~50+ cycles).

### 7.2 Exclusive Access (LDREX/STREX)

ARMv7-A uses LDREX/STREX for atomic operations (no CAS instruction):

```asm
@ Atomic increment
.retry:
  ldrex r1, [r0]         @ load exclusive
  add   r1, r1, #1
  strex r2, r1, [r0]     @ store exclusive; r2 = 0 if success
  cmp   r2, #0
  bne   .retry            @ retry if exclusive lost
```

**Performance**:
- LDREX: Same latency as LDR + exclusive monitor overhead (~2-4 cycles)
- STREX: ~2-4 cycles if exclusive succeeds; full retry loop if failed
- The retry loop is expensive on contended locations
- DMB before/after STREX is often needed for acquire/release semantics

**Optimization**:
- Minimize the LDREX→STREX window (fewer instructions between them
  reduces the chance of losing exclusivity)
- Use DMB only where required by the memory model (not "just in case")
- On A15, STREX can be speculated; the hardware handles rollback

### 7.3 Memory Access Ordering

ARMv7-A has a **weakly-ordered** memory model. The processor may reorder:
- Load-Load (reads can pass other reads)
- Load-Store (reads can pass writes to different addresses)
- Store-Store (writes can pass other writes, except with DMB)
- Store-Load (writes can pass reads)

This is weaker than x86's TSO model. Any code relying on memory ordering
between cores must use explicit barriers (DMB/DSB) or LDREX/STREX.

---

## 8. Backend Optimization Opportunities

This section maps the hardware rules above to specific patterns in an
ARMv7-A codegen pipeline (analogous to Section 7 in the ARM64 guide).

### 8.1 Float Constant Materialization

**Current typical pattern**:
```asm
movw  r0, #0xABCD        @ lower 16 bits of IEEE 754 encoding
movt  r0, #0x4049        @ upper 16 bits
vmov  s0, r0             @ GP → FP transfer (1-20c penalty on A8!)
```

**Better alternatives**:
- **VLDR Sd, [PC, #offset]**: Load from literal pool. 1 instruction,
  load latency only (2-4c), no GP→FP transfer.
- **VMOV.F32 Sd, #imm8**: Immediate float constant (VFPv3+). Encodes
  `±(1 + n/16) × 2^r` where n ∈ [0,15], r ∈ [-3,4]. Covers common
  values: 0.5, 1.0, 2.0, 10.0, etc. Single instruction, ~4c latency.
- **VMOV.I32 Dd, #0**: NEON immediate zero.

**Priority**: VMOV.F32 #imm8 > VLDR literal pool > MOVW+MOVT+VMOV.

### 8.2 Multiply-Accumulate Optimization

ARMv7-A has rich multiply-accumulate support that many backends
underutilize:

**Integer**:
- `MLA Rd, Rn, Rm, Ra` — Rd = Rn × Rm + Ra (1 instruction, 2-3c)
- `MLS Rd, Rn, Rm, Ra` — Rd = Ra - Rn × Rm (1 instruction)
- `SMLAL/UMLAL` — 64-bit accumulate

**NEON**:
- `VMLA.F32` — fused on A7/A15 with VFPv4; not fused on A8/A9
- `VMLAL` — widening multiply-accumulate (e.g., I16×I16 → I32)

**Pattern to recognize**:
```asm
@ Anti-pattern
mul   r2, r0, r1
add   r3, r2, r4
@ Better
mla   r3, r0, r1, r4    @ 1 instruction, same latency as MUL
```

### 8.3 Address Computation

ARMv7-A's flexible second operand (barrel shifter) allows complex
address computations in a single instruction:

**Anti-pattern** (3 instructions):
```asm
lsl   r2, r1, #2        @ index * 4
add   r2, r0, r2        @ base + offset
ldr   r3, [r2]          @ load
```

**Better** (1 instruction):
```asm
ldr   r3, [r0, r1, lsl #2]  @ load from base + index*4
```

**Available addressing modes**:
- `[Rn, #±imm12]` — 12-bit immediate offset (±4095)
- `[Rn, ±Rm]` — register offset
- `[Rn, ±Rm, shift #n]` — shifted register (LSL, LSR, ASR, ROR)
- Pre-index: `[Rn, #offset]!` — updates Rn
- Post-index: `[Rn], #offset` — uses Rn then updates

These are **free** — the barrel shifter runs in parallel with the address
generation unit. Use them aggressively to fold address arithmetic into
load/store instructions.

### 8.4 Conditional Compare Chains

For multi-condition tests, avoid materializing boolean intermediates:

**Anti-pattern**:
```asm
cmp   r0, #10
movge r2, #1
movlt r2, #0
cmp   r1, #20
moveq r3, r2
```

**Better** (ARM mode):
```asm
cmp   r0, #10
cmpge r1, #20          @ conditional compare: only if r0 >= 10
beq   target            @ branch if r0 >= 10 AND r1 == 20
```

The conditional compare (`CMPGE`) reads flags from the first CMP and
only executes the second comparison if the condition holds. This saves
instructions and avoids intermediate register allocation.

### 8.5 Barrel Shifter Exploitation

Nearly all ARMv7-A data-processing instructions accept a shifted second
operand at no extra cost. The barrel shifter operates in parallel.

**Foldable patterns**:
```asm
@ x * 5 = x + x * 4
add   r0, r1, r1, lsl #2   @ r0 = r1 + (r1 << 2) = r1 * 5

@ x * 7 = x * 8 - x
rsb   r0, r1, r1, lsl #3   @ r0 = (r1 << 3) - r1 = r1 * 7

@ (x >> 4) & 0xFF — extract byte field
ubfx  r0, r1, #4, #8        @ bit field extract (ARMv7)

@ Clear bits [7:4]
bfc   r0, #4, #4             @ bit field clear (ARMv7)
```

**Cost**: All barrel-shifted operations are **1 cycle** on all cores
(except A7 where variable shifts add 1 cycle). This makes multiply-by-
constant optimization very effective.

### 8.6 Prologue/Epilogue Optimization

**Standard prologue** (PUSH/POP = STM/LDM):
```asm
push  {r4-r11, lr}      @ STM: 9 registers, 9 cycles blocking
@ ... function body ...
pop   {r4-r11, pc}      @ LDM: 9 registers, returns via PC
```

**Optimized for small functions**:
```asm
@ Only save what's actually used
push  {r4-r7, lr}       @ 5 registers instead of 9
@ ... function body using r4-r7 only ...
pop   {r4-r7, pc}
```

**For leaf functions** (no calls):
```asm
@ No prologue needed! Use only r0-r3, r12 (caller-saved)
@ ... function body ...
bx    lr                 @ return, no restore needed
```

**Reducing callee-saved registers** is one of the highest-impact
optimizations on ARMv7-A because PUSH/POP blocks the pipeline for
N cycles per register.

### 8.7 NEON Register Allocation for Spill Reduction

The NEON register file is large (32 × D-registers or 16 × Q-registers)
and callee-saved registers are only D8-D15 (8 registers). D0-D7 and
D16-D31 are caller-saved.

**Strategy**: For leaf functions with heavy FP/SIMD work, prefer
D0-D7 and D16-D31 (no save/restore needed). Reserve D8-D15 for values
that must survive calls.

### 8.8 Redundant Flag Setting

Many ARMv7-A instructions have S-suffix variants that set flags (ADDS,
SUBS, etc.). Using the flag-setting form when the flags are immediately
consumed by a branch or conditional instruction saves a separate CMP:

**Anti-pattern**:
```asm
sub   r0, r0, #1
cmp   r0, #0
beq   .done
```

**Better**:
```asm
subs  r0, r0, #1        @ sets Z flag when result is 0
beq   .done              @ uses Z flag directly
```

This saves 1 instruction and 1 cycle, and the flag-setting form has
the same latency as the non-flag-setting form on all cores.

---

## 9. Cost Reference Tables

### Common Pattern Costs

| Pattern | Instructions | Latency (A8/A15) | Better Alternative | Savings |
|---|---|---|---|---|
| Float const (GP→FP) | movw+movt+vmov (3) | 2+1+20c (A8) / 2+1+2c (A15) | VLDR [PC,#off] (1) | 2 insns, 18c (A8) |
| VMOV.F32 #imm8 | 1 | 4c | — | vs GP→FP: 2 insns, 16c (A8) |
| 3-insn address | lsl+add+ldr (3) | 1+1+2=4c (A8) | ldr [Rn,Rm,LSL#n] (1) | 2 insns, 2c |
| MUL+ADD | 2 insns | 2+1=3c (A8) | MLA (1) | 1 insn, 0c |
| Prologue 9 regs | PUSH {r4-r11,lr} (1) | 9c blocking | PUSH {r4-r7,lr} (1) | 4c blocking |
| VFP VMLA.F32 (A8) | 1 | **18-21c** | NEON VMLA.F32 Dd (1) | 9-12c |
| CMP+branch (mispredict) | 2 | 1+13=14c (A8) | MOVEQ/MOVNE (2) | 12c worst case |
| SUB+CMP+BEQ | 3 | 1+1+1=3c | SUBS+BEQ (2) | 1 insn |
| 2×LDR adjacent | 2 | 2+2=4c (A8) | LDRD (1) | 1 insn |
| Unaligned VLD1 (A8) | 1 | **9+ extra c** | Aligned VLD1 :128 (1) | 9c |

### Port/Pipeline Pressure Quick Reference

| Resource | A8 | A9 | A15 | Bottleneck Risk |
|---|---|---|---|---|
| ALU pipelines | 2 | 2 | 2 | LOW |
| Multiply | 1 | 1 (in ALU0) | 1 | MEDIUM (blocks ALU0 on A9) |
| Load/Store | 1 | 1 | 1+1 (L+S) | **HIGH on A8/A9** (1/cycle) |
| NEON | 1 (separate) | 1 (in INT) | 2 | MEDIUM |
| VFP | 1 (non-pipelined A8) | 1 | 2 | **HIGH on A8** |
| Branch | 1 | 1 | 1 | LOW |
| Store buffer | 2 entries | 4 entries | 6 entries | **HIGH on A8** |

---

## 10. Cortex-A Core Generational Reference

| Feature | Cortex-A7 | Cortex-A8 | Cortex-A9 | Cortex-A15 | Cortex-A17 |
|---|---|---|---|---|---|
| Year | 2011 | 2005 | 2007 | 2010 | 2014 |
| Pipeline | 8-stage | 13-stage | 8-stage | 15-stage | 11-stage |
| Issue | Partial dual | Dual in-order | Dual partial-OoO | 3-way OoO | 2-way OoO |
| ROB | — | — | 32-40 | 128 | ~64 |
| DMIPS/MHz | ~1.9 | ~2.0 | ~2.5 | ~3.5 | ~2.8 |
| ISA | ARMv7-A | ARMv7-A | ARMv7-A | ARMv7-A | ARMv7-A |
| VFP | VFPv4 | VFPLite | VFPv3 | VFPv4 | VFPv4 |
| NEON | NEONv2 | NEON | NEON | NEONv2 | NEONv2 |
| Hardware divide | Yes | No | No | Yes | Yes |
| LPAE (>4GB) | Yes | No | Optional | Yes | Yes |
| Virtualization | No | No | No | Yes | Yes |
| L1 D-cache | 4-64 KB | 16-32 KB | 32 KB | 32 KB | 32 KB |
| L1 I-cache | 4-64 KB | 16-32 KB | 32 KB | 32 KB | 32 KB |
| L2 cache | 0-1 MB | 0-1 MB | Ext. 0-4 MB | 512K-4 MB | 256K-8 MB |
| Cache line (D) | 32-64 B | 64 B | 32 B | 64 B | 64 B |
| Max frequency | ~1.5 GHz | ~1.3 GHz | ~2.0 GHz | ~2.5 GHz | ~2.0 GHz |
| Process node | 28nm | 65nm | 40nm | 28nm | 28nm |
| Typical SoCs | Snapdragon 4xx, Exynos 5 LITTLE | OMAP3, Tegra 2 | OMAP4, Tegra 3, Exynos 4 | Exynos 5, OMAP5, Tegra 4 | MT6795 |
| big.LITTLE pair | With A15/A17 | — | — | With A7 | With A7 |

### Qualcomm Custom Cores

| Feature | Scorpion | Krait 200 | Krait 300/400 |
|---|---|---|---|
| Year | 2008 | 2012 | 2013 |
| Pipeline (INT) | 10-13 stage | 11 stage | 11 stage |
| Issue | Dual in-order | 4-way OoO | 4-way OoO |
| VFP | Pipelined (!) | VFPv4 | VFPv4 |
| NEON | 2-way | 3-way | 3-way |
| DMIPS/MHz | ~2.1 | ~3.3 | ~3.3+ |
| L0 cache | — | 4+4 KB | 4+4 KB |
| L1 cache | 32+32 KB | 16+16 KB | 16+16 KB |
| L2 cache | 256-512 KB | 1 MB shared | 2 MB shared |
| DTLB L1 | — | 32, 5c miss | 32, 5c miss |
| DTLB L2 | — | 128, 65c miss | 128, 65c miss |
| Max frequency | 1.0-1.5 GHz | 1.5 GHz | 2.3 GHz |
| Typical SoCs | Snapdragon S1/S2 | Snapdragon S4, 600 | Snapdragon 800/801 |

**Key takeaways for codegen**:
- Scorpion has a pipelined VFP (unlike A8's VFPLite), making scalar FP
  much faster than on A8.
- Krait's 4-way OoO is more aggressive than A15's 3-way; scheduling is
  less critical but alignment and cache effects still matter.
- Krait's L2 TLB miss penalty (65 cycles) is severe. Keep working sets
  within L1 TLB coverage (32 × 4KB = 128 KB) for hot data.

---

## 11. ARMv7-A vs ARM64 Key Differences for Codegen

For teams maintaining both ARMv7-A and ARM64 backends, these are the
most important architectural differences:

| Feature | ARMv7-A | ARM64 (AArch64) |
|---|---|---|
| GP registers | 16 (r0-r15, r13=SP, r14=LR, r15=PC) | 31 + SP + ZR |
| FP/SIMD registers | 32 D-regs / 16 Q-regs | 32 × 128-bit V-regs |
| Instruction width | 32-bit (ARM) / 16-32-bit (Thumb-2) | Fixed 32-bit |
| Condition codes | On every instruction (ARM) / IT blocks (Thumb-2) | Separate CSEL/CCMP |
| Barrel shifter | Free shifted 2nd operand on ALU ops | Limited shift/extend on some ops |
| Load pair | LDRD (even-odd pair only) | LDP (any 2 registers) |
| Literal pool | LDR [PC, #offset] (relative to PC) | LDR (literal), ADRP+LDR |
| Atomic ops | LDREX/STREX loop | LDXR/STXR + CAS (ARMv8.1) |
| NEON F64 | Not supported | Full F64 SIMD |
| MOV elimination | Never | Some implementations (Apple) |
| Integer divide | A7/A15 only | All cores |
| Instruction fusion | Minimal | CMP+B.cond, etc. |
| PC-relative addressing | LDR [PC,#off], ADR | ADRP+ADD |

**Register pressure is much higher on ARMv7-A**: Only 13 usable GP
registers (r0-r12; r13-r15 are SP/LR/PC). ARM64 has 31 GP registers.
This makes register allocation and spill optimization far more critical
on ARMv7-A.

---

## 12. Optimization Priority Summary

Ranked by impact across all ARMv7-A cores:

1. **Avoid A8 VFPLite**: Use NEON for all float work on A8 (10-15× speedup)
2. **Avoid NEON→ARM transfers on A8**: ~20 cycle penalty per transfer
3. **Schedule loads away from consumers**: 2-4 cycles wasted per stall
4. **Use conditional execution over branches**: 8-14 cycle misprediction
5. **Align NEON memory access**: 9+ cycle penalty for unaligned on A8
6. **Minimize LDM/STM in hot code**: Use LDRD/STRD or individual loads
7. **Use barrel shifter**: Fold shifts into ALU/address operations for free
8. **Use MLA/MLS**: Fold multiply-add into single instruction
9. **Use MOVW/MOVT or literal pool for constants**: Avoid GP→FP transfers
10. **Minimize callee-saved registers**: Reduce prologue/epilogue cost
11. **Use PLD for streaming data**: Essential on A7/A8 (no HW prefetch)
12. **Prefer Thumb-2**: 25-35% code size reduction → better I-cache

---

## 13. References and Further Reading

- ARM, "Cortex-A8 Technical Reference Manual" (DDI0344):
  https://developer.arm.com/documentation/ddi0344/latest/
- ARM, "Cortex-A9 Technical Reference Manual" (DDI0388):
  https://developer.arm.com/documentation/ddi0388/latest/
- ARM, "Cortex-A15 MPCore Technical Reference Manual" (DDI0438):
  https://developer.arm.com/documentation/ddi0438/latest/
- ARM, "Cortex-A7 MPCore Technical Reference Manual" (DDI0464):
  https://developer.arm.com/documentation/ddi0464/f/
- ARM, "Cortex-A Series Programmer's Guide" (DEN0013D):
  https://developer.arm.com/documentation/den0013/latest/
- ARM, "NEON Programmer's Guide" (DEN0018):
  https://developer.arm.com/documentation/den0018/latest/
- ARM, "Architecture Reference Manual ARMv7-A/R" (DDI0406):
  https://developer.arm.com/documentation/ddi0406/latest/
- Cortex-A7 Instruction Cycle Timings (empirical):
  https://hardwarebug.org/2014/05/15/cortex-a7-instruction-cycle-timings/
- ARM NEON Memory Hazards (empirical):
  https://hardwarebug.org/2008/12/31/arm-neon-memory-hazards/
- ARM Instruction Scheduling Secrets (empirical):
  https://ssvb.github.io/2011/08/03/discovering-instructions-scheduling-secrets.html
- WikiChip Cortex-A8 Microarchitecture:
  https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a8
- WikiChip Cortex-A9 Microarchitecture:
  https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a9
- WikiChip Cortex-A15 Microarchitecture:
  https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a15
- Qualcomm Krait Architecture (AnandTech):
  https://www.anandtech.com/show/4940/qualcomm-new-snapdragon-s4-msm8960-krait-architecture
- ARM Cortex-A Series Processors (UTK/ICL Report):
  https://icl.utk.edu/~luszczek/teaching/courses/fall2013/cosc530/Cosc530Report_ARM_Cortex-A.pdf
- Exploring FP Performance of Modern ARM Processors (AnandTech):
  https://www.anandtech.com/show/6971/exploring-the-floating-point-performance-of-modern-arm-processors
