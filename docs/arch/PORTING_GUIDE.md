# Architecture Backend Porting Guide

This document covers the rules and design decisions that each architecture
backend must follow when emitting MachineIR instructions. It captures
platform-specific insights that affect performance but are not obvious from
the MachineIR representation alone.

---

## 1. Indexed Memory Operations

### MachineIR representation

The shared peephole (`peephole.rs::fuse_indexed_memory`) fuses address
computation sequences into `IndexedLoad` / `IndexedStore`:

```
IndexedLoad {
    dst,              // destination register
    base,             // memory base register (typically Wasm linear memory base)
    index,            // address/index register
    index_extend,     // None | ZeroExtend32
    offset,           // i32 immediate (Wasm load/store offset field)
    width,            // U8 / U16 / U32 / U64
    extension,        // None / ZeroExtend / SignExtend
}
```

Semantics: `dst <- mem[base + extend(index) + offset]`

Two source patterns are recognized:

**Pattern A** (with zero-extend, common for Wasm memory access):
```
cvt.I64ExtendI32U  r <- addr
[i64.Add           r <- r OFFSET]     // optional Wasm offset
i64.Add            r <- base r
load/store         [r + 0]
```

**Pattern B** (no extend):
```
i64.Add            r <- base index
load/store         [r + 0]
```

### Rule: Stable base register for store-to-load forwarding

Modern CPUs use the load instruction's **base register** as the primary tag
for speculative store-to-load forwarding. When the same stable base register
appears in both a store and a subsequent load to the same address, the address
predictor matches confidently and forwards the data without waiting for full
address resolution.

If the base register is a freshly-computed scratch register, the predictor
loses confidence, stalling the load for 5+ extra cycles. This was measured as
a **17% regression on SHA-256** on Apple Silicon when using the scratch-base
form.

**Rule**: When the ISA has a real base+index addressing form and
`offset != 0`, fold the offset into the **index** register and keep the
original `base` as the load/store base operand. Do NOT compute
`base + extend(index)` into a scratch register and use `[scratch, #offset]`.

ISAs without base+index memory operands, such as RISC-V, cannot preserve that
exact physical base register. They should still avoid extra scratch pressure:
compute the effective address once, use the load/store immediate field when the
static offset fits, and fold larger offsets into the address before acquiring
other long-lived scratch operands.

### Per-architecture emit strategy

#### ARM64

```
offset == 0:
  LDR  Wt, [Xbase, Windex, UXTW]           // 1 instruction

offset != 0 (stable-base form):
  MOV  Wscratch, Windex                      // zero-extend into scratch
  ADD  Xscratch, Xscratch, #offset           // fold offset
  LDR  Wt, [Xbase, Xscratch]                // stable base (3 instructions)
```

**DO NOT** use:
```
ADD  Xscratch, Xbase, Windex, UXTW          // scratch = base + zext(index)
LDR  Wt, [Xscratch, #offset]                // BREAKS store-forwarding
```

#### x86_64

x86_64 has native `[base + index + displacement]` addressing. The offset goes
into the displacement field:

```
offset == 0 or != 0:
  MOVZX  r_tmp, index_32                     // zero-extend
  MOV    dst, [base + r_tmp + offset]         // 1 instruction, base stable
```

x86_64 does NOT have the store-forwarding issue with displacements because the
displacement is part of the instruction encoding, not a computed register.

#### ARMv7

Similar to ARM64 but 32-bit (no UXTW needed). Same stable-base rule applies.

#### RISC-V 64

RISC-V loads and stores have only `[base + signed-12-bit-offset]` addressing.
There is no base+index memory form, so the backend computes
`addr = base + extend(index)` into a scratch register:

```
offset fits signed 12 bits:
  ADD   tmp, base, index
  LW/LD dst, offset(tmp)

offset does not fit:
  ADD   tmp, base, index
  LI    off_tmp, offset
  ADD   tmp, tmp, off_tmp
  LW/LD dst, 0(tmp)
```

For indexed stores, fold an out-of-range offset into `tmp` before loading or
materializing the source value. Otherwise the address scratch, source scratch,
and large-offset scratch can overlap live ranges and exhaust a small scratch
pool.

### Summary table

| Scenario | ARM64 | x86_64 | ARMv7 | RV64 |
|----------|-------|--------|-------|------|
| offset=0, no extend | `LDR [base, index]` (1) | `MOV [base+index]` (1) | `LDR [base, index]` (1) | `ADD+LD [tmp]` (2) |
| offset=0, UXTW | `LDR [base, W, UXTW]` (1) | `MOVZX+MOV [base+idx]` (2) | N/A (32-bit) | `ZEXT.W+ADD+LD [tmp]` (3) |
| offset!=0, UXTW | `MOV+ADD+LDR [base,tmp]` (3) | `MOVZX+MOV [base+idx+off]` (2) | `ADD+LDR [base+idx,off]` (2) | fits: `ZEXT.W+ADD+LD off(tmp)`; large: add offset first |
| offset!=0, no extend | `MOV+ADD+LDR [base,tmp]` (3) | `MOV [base+idx+off]` (1) | `ADD+LDR [base+idx,off]` (2) | fits: `ADD+LD off(tmp)`; large: add offset first |

---

## 2. Compare-and-Branch Fusion

The shared peephole (`fuse_compare_branch`) rewrites:
```
IntCompare { dst } ; Branch { Value(dst) }
```
into:
```
Branch { IntCompare { ... } }
```

Each backend maps this to its most efficient form:

- **ARM64**: `CMP + B.cond` (hardware macro-fused). Also `CBZ`/`CBNZ`/`TBZ`/`TBNZ` for zero-compare patterns.
- **x86_64**: `CMP + Jcc` (hardware macro-fused on Intel/AMD).
- **ARMv7**: `CMP + B.cond` or IT-block conditional execution.
- **RV64**: `SLT`/`SLTU`/`SUB`-style condition setup plus `BEQ`/`BNE`/`BLT`/`BGE` forms where directly encodable.

Float compares are NOT fused in the shared peephole because x86_64 requires
multi-instruction NaN handling that cannot be expressed as a single branch.
ARM64 fuses `FCMP + B.cond` in its backend-specific codegen.

---

## 3. Store Pair Fusion (ARM64 only)

ARM64's `STP` stores two registers with a single store-port issue. The ARM64
backend detects consecutive `Store { src: Imm64(0), width: U64 }` pairs with
adjacent offsets and fuses them into `STP XZR, XZR, [base, #offset]`.

This is ARM64-specific (x86_64 does not benefit; ARMv7 lacks STP).

---

## 4. Page-Boundary-Aware Function Alignment

All backends use `page_align_function()` to reduce iTLB pressure:

1. **Cache-line alignment**: Function starts are rounded up to 64 bytes.
2. **Page-boundary avoidance**: If a function (< 16 KB) would straddle a
   16 KB page boundary, it is bumped to the next page start — provided the
   required NOP padding is <= 1 KB.

The NOP encoding is architecture-specific:
- ARM64: `0xd503201f` (NOP)
- x86_64: `0xCC` (INT3)
- ARMv7: `0xe1a00000` (MOV R0, R0)
- RV64: `0x00000013` (ADDI X0, X0, 0)
