# ARMv7-A Instruction Encoding Reference for Code Emitter Development

Quick-reference for implementing an ARMv7-A (32-bit ARM mode) code emitter.
Covers instruction encoding formats, operand encoding rules, condition codes,
immediate encoding constraints, and register conventions. Derived from the
ARM Architecture Reference Manual (DDI0406) and the Cortex-A Series
Programmer's Guide (DEN0013D).

For Thumb-2 encoding, see Section 10. For NEON/VFP encoding, see Section 8.

---

## 1. Instruction Format Overview

All ARM mode instructions are 32 bits wide with a 4-bit condition code
in bits [31:28].

### 1.1 Base Encoding Classes

```
31  28 27 26 25 24       20 19    16 15    12 11                    0
[cond] [op1 ] [  opcode  ] [  Rn  ] [  Rd  ] [     operand2       ]
```

| Bits | Field | Description |
|---|---|---|
| [31:28] | cond | Condition code (0x0-0xE = conditional, 0xF = unconditional) |
| [27:25] | op1 | Instruction class |
| [24:20] | opcode/flags | Operation + S bit + additional encoding |
| [19:16] | Rn | First operand register (or additional opcode) |
| [15:12] | Rd | Destination register |
| [11:0] | operand2 | Flexible second operand (immediate or register) |

### 1.2 Instruction Classes (bits [27:25])

| op1 [27:25] | Class | Description |
|---|---|---|
| 000 | Data processing (register) | ALU ops with register operand2 |
| 001 | Data processing (immediate) | ALU ops with rotated immediate |
| 010 | Load/Store (immediate offset) | LDR/STR with 12-bit immediate |
| 011 | Load/Store (register offset) | LDR/STR with shifted register |
| 100 | Load/Store Multiple | LDM/STM |
| 101 | Branch | B/BL with 24-bit offset |
| 110 | Coprocessor load/store | VFP/NEON memory ops (VLDR/VSTR/VLDM/VSTM) |
| 111 | Coprocessor data/SVC | VFP/NEON data ops, SVC |

---

## 2. Condition Codes

Every ARM instruction can be conditionally executed based on the APSR
flags (N, Z, C, V).

| Code | Suffix | Meaning | Flags |
|---|---|---|---|
| 0000 | EQ | Equal / Zero | Z=1 |
| 0001 | NE | Not equal / Non-zero | Z=0 |
| 0010 | CS/HS | Carry set / Unsigned higher or same | C=1 |
| 0011 | CC/LO | Carry clear / Unsigned lower | C=0 |
| 0100 | MI | Minus / Negative | N=1 |
| 0101 | PL | Plus / Positive or zero | N=0 |
| 0110 | VS | Overflow set | V=1 |
| 0111 | VC | Overflow clear | V=0 |
| 1000 | HI | Unsigned higher | C=1 AND Z=0 |
| 1001 | LS | Unsigned lower or same | C=0 OR Z=1 |
| 1010 | GE | Signed greater or equal | N=V |
| 1011 | LT | Signed less than | N≠V |
| 1100 | GT | Signed greater than | Z=0 AND N=V |
| 1101 | LE | Signed less or equal | Z=1 OR N≠V |
| 1110 | AL | Always (unconditional) | — |
| 1111 | — | Unconditional (special instructions) | — |

**Emitter note**: Always set cond=0xE (AL) for unconditional instructions.
Cond=0xF is reserved for unconditional instruction space (PLD, DSB, etc.).

---

## 3. Data Processing Instructions

### 3.1 Encoding Format

```
31  28 27 26 25 24 23 22 21 20 19    16 15    12 11                0
[cond] 0  0  I  [  opcode ] S [  Rn  ] [  Rd  ] [   operand2     ]
```

- **I** (bit 25): 0 = register operand2, 1 = immediate operand2
- **S** (bit 20): 1 = update condition flags (ADDS vs ADD)
- **opcode** (bits [24:21]):

| Opcode | Mnemonic | Operation | Notes |
|---|---|---|---|
| 0000 | AND | Rd = Rn AND operand2 | |
| 0001 | EOR | Rd = Rn XOR operand2 | |
| 0010 | SUB | Rd = Rn - operand2 | |
| 0011 | RSB | Rd = operand2 - Rn | Reverse subtract |
| 0100 | ADD | Rd = Rn + operand2 | |
| 0101 | ADC | Rd = Rn + operand2 + C | Add with carry |
| 0110 | SBC | Rd = Rn - operand2 - !C | Subtract with carry |
| 0111 | RSC | Rd = operand2 - Rn - !C | Reverse subtract with carry |
| 1000 | TST | flags = Rn AND operand2 | Rd must be 0000 (r0) |
| 1001 | TEQ | flags = Rn XOR operand2 | Rd must be 0000 |
| 1010 | CMP | flags = Rn - operand2 | Rd must be 0000, S=1 implicit |
| 1011 | CMN | flags = Rn + operand2 | Rd must be 0000, S=1 implicit |
| 1100 | ORR | Rd = Rn OR operand2 | |
| 1101 | MOV | Rd = operand2 | Rn must be 0000 |
| 1110 | BIC | Rd = Rn AND NOT operand2 | Bit clear |
| 1111 | MVN | Rd = NOT operand2 | Rn must be 0000 |

### 3.2 Operand2: Immediate (I=1)

```
11        8  7              0
[ rotate  ] [    imm8      ]
```

The immediate value is: `imm8 ROR (rotate × 2)`

This encodes a subset of 32-bit constants. The effective value is an
8-bit value rotated right by an even number of positions (0-30).

**Encodable values include**:
- 0x00-0xFF (any byte, no rotation)
- 0x00-0xFF shifted left by any even amount
- Examples: 0xFF, 0xFF00, 0xFF0000, 0xFF000000, 0x3FC, 0x1FE

**NOT encodable**: Values like 0x101 (two non-adjacent bits set), 0x1FF
(9 consecutive bits), etc.

**Emitter helper**: To check if a value `v` is encodable as a modified
immediate:
```
for rot in range(0, 32, 2):
    candidate = ror32(v, rot)
    if candidate <= 0xFF:
        return (rot // 2, candidate)  # rotate field, imm8
return None  # not encodable
```

**MVN trick**: If `v` is not encodable but `~v` is, use MVN instead of
MOV. Example: MOV Rd, #0xFFFFFFF0 → MVN Rd, #0x0F.

**Negation trick**: If SUB with #imm is not encodable but ADD with
#(-imm) is (or vice versa), the assembler may substitute. The emitter
should try both.

### 3.3 Operand2: Register (I=0)

```
11  7   6  5  4  3        0
[shamt] [type] 0 [   Rm   ]    (immediate shift)
[ Rs  ] 0 [type] 1 [  Rm  ]    (register shift)
```

**Shift types** (bits [6:5]):
| Code | Type | Operation |
|---|---|---|
| 00 | LSL | Logical shift left |
| 01 | LSR | Logical shift right |
| 10 | ASR | Arithmetic shift right |
| 11 | ROR | Rotate right (or RRX if shamt=0) |

**Immediate shift** (bit 4 = 0):
- `shamt` (bits [11:7]): shift amount 0-31
- Special: LSR #0 encodes LSR #32, ASR #0 encodes ASR #32,
  ROR #0 encodes RRX (rotate right extend through carry)

**Register shift** (bit 4 = 1):
- Rs (bits [11:8]): shift amount register (only bottom 8 bits used)
- Cannot use PC (r15) as Rs
- Adds 1 extra cycle on some cores (A7)

---

## 4. Load/Store Instructions

### 4.1 Word/Byte Load/Store (LDR/STR/LDRB/STRB)

```
31  28 27 26 25 24 23 22 21 20 19    16 15    12 11                0
[cond]  0  1  I  P  U  B  W  L [  Rn  ] [  Rd  ] [   offset      ]
```

| Bit | Field | Meaning |
|---|---|---|
| 25 | I | 0 = immediate offset, 1 = register offset |
| 24 | P | Pre/Post indexing: 1 = pre (offset before), 0 = post (offset after) |
| 23 | U | Up/Down: 1 = add offset, 0 = subtract offset |
| 22 | B | Byte: 1 = LDRB/STRB, 0 = LDR/STR |
| 21 | W | Write-back: 1 = write address back to Rn (pre-index: Rn!; post: always) |
| 20 | L | Load/Store: 1 = LDR, 0 = STR |

**Addressing mode matrix**:

| P | W | Mode | Syntax | Behavior |
|---|---|---|---|---|
| 1 | 0 | Offset | `[Rn, #off]` | Address = Rn ± off; Rn unchanged |
| 1 | 1 | Pre-indexed | `[Rn, #off]!` | Address = Rn ± off; Rn = Address |
| 0 | 0 | Post-indexed | `[Rn], #off` | Address = Rn; Rn = Rn ± off |
| 0 | 1 | (privileged) | — | LDRT/STRT user-mode access |

**Immediate offset** (I=0): 12-bit unsigned immediate (0-4095) in
bits [11:0]. Sign determined by U bit.

**Register offset** (I=1): Same encoding as data processing operand2
register form (Rm with optional shift).
```
[Rn, Rm]            @ register offset
[Rn, Rm, LSL #n]    @ scaled register offset
[Rn, -Rm]           @ negative register offset
```

### 4.2 Halfword/Signed Load/Store (LDRH/STRH/LDRSB/LDRSH)

Different encoding from word/byte. Uses bits [7:4] = 1011 (LDRH/STRH),
1101 (LDRSB), 1111 (LDRSH):

```
31  28 27    24 23 22 21 20 19    16 15    12 11  8  7  6 5  4  3     0
[cond] 0 0 0 P  U  1  W  L [  Rn  ] [  Rd  ] [imm4H] 1 S H  1 [imm4L]
```

- Immediate offset: `imm4H:imm4L` (8-bit, 0-255)
- Register offset: Rm in bits [3:0], bits [11:8] = 0000

| S | H | Mnemonic | Description |
|---|---|---|---|
| 0 | 1 | LDRH/STRH | Unsigned halfword |
| 1 | 0 | LDRSB | Signed byte |
| 1 | 1 | LDRSH | Signed halfword |

### 4.3 Double-Word Load/Store (LDRD/STRD)

```
31  28 27    24 23 22 21 20 19    16 15    12 11  8  7  6 5  4  3     0
[cond] 0 0 0 P  U  1  W  0 [  Rn  ] [  Rd  ] [imm4H] 1 1 0  1 [imm4L]  (LDRD)
[cond] 0 0 0 P  U  1  W  0 [  Rn  ] [  Rd  ] [imm4H] 1 1 1  1 [imm4L]  (STRD)
```

- Rd must be even (R0, R2, R4, ..., R12)
- Loads/stores Rd and Rd+1
- 8-bit immediate offset (0-255)
- Address must be word-aligned (4-byte)

### 4.4 Load/Store Multiple (LDM/STM)

```
31  28 27 26 25 24 23 22 21 20 19    16 15                          0
[cond]  1  0  0  P  U  S  W  L [  Rn  ] [       register_list      ]
```

- **register_list** (bits [15:0]): Bitmask of registers to load/store.
  Bit N = 1 means register RN is included.
- **L**: 1 = LDM, 0 = STM
- **P**: 0 = post (include base address), 1 = pre (exclude base address)
- **U**: 1 = increment, 0 = decrement
- **W**: 1 = write back updated address to Rn

| P | U | Mode | Mnemonic | Stack equivalent |
|---|---|---|---|---|
| 0 | 1 | IA | LDMIA/STMIA | Pop / — |
| 1 | 1 | IB | LDMIB/STMIB | — / — |
| 0 | 0 | DA | LDMDA/STMDA | — / — |
| 1 | 0 | DB | LDMDB/STMDB | — / Push |

**PUSH = STMDB SP!, {regs}**
**POP = LDMIA SP!, {regs}**

---

## 5. Branch Instructions

### 5.1 Branch (B) and Branch with Link (BL)

```
31  28 27 26 25 24 23                                               0
[cond]  1  0  1  L  [              imm24                            ]
```

- **L** (bit 24): 0 = B (branch), 1 = BL (branch and link; LR = PC+4)
- **imm24**: Signed 24-bit offset, left-shifted by 2 → ±32 MB range
- **Effective target**: PC + 8 + (sign_extend(imm24) << 2)
  (PC is 8 bytes ahead of current instruction in ARM mode)

### 5.2 Branch and Exchange (BX) and Branch with Link and Exchange (BLX)

**BX Rm**:
```
31  28 27                     8  7  6  5  4  3        0
[cond] 0 0 0 1 0 0 1 0 1 1 1 1  1 1 1 1  0 0 0 1 [Rm]
```
Bit 0 of Rm determines mode: 0 = ARM, 1 = Thumb.

**BLX Rm** (register):
```
31  28 27                     8  7  6  5  4  3        0
[cond] 0 0 0 1 0 0 1 0 1 1 1 1  1 1 1 1  0 0 1 1 [Rm]
```

**BLX #offset** (immediate, unconditional):
```
31  28 27 26 25 24 23                                               0
 1111   1  0  1  H  [              imm24                            ]
```
- H (bit 24): added to offset as bit 1 (enables Thumb alignment)
- Target = PC + 8 + (sign_extend(imm24) << 2) + (H << 1)
- Always switches to Thumb mode

### 5.3 Compare and Branch (Thumb-2 only: CBZ/CBNZ)

These are Thumb-2 16-bit instructions, not available in ARM mode. See
Section 10 for Thumb encoding.

---

## 6. Multiply Instructions

### 6.1 32-bit Multiply

```
31  28 27    24 23 22 21 20 19    16 15    12 11   8  7  4  3      0
[cond] 0 0 0 0  0 0 0  S  [  Rd  ] [ SBZ  ] [  Rm ] 1001 [  Rn  ]
```

**MUL Rd, Rn, Rm**: Rd = Rn × Rm (low 32 bits). S=1 updates N and Z.

### 6.2 Multiply-Accumulate

```
[cond] 0 0 0 0  0 0 1  S  [  Rd  ] [  Ra  ] [  Rm ] 1001 [  Rn  ]
```

**MLA Rd, Rn, Rm, Ra**: Rd = Rn × Rm + Ra.

### 6.3 Multiply-Subtract (ARMv7)

```
[cond] 0 0 0 0  0 1 1  0  [  Rd  ] [  Ra  ] [  Rm ] 1001 [  Rn  ]
```

**MLS Rd, Rn, Rm, Ra**: Rd = Ra - Rn × Rm.

### 6.4 64-bit Multiply

```
[cond] 0 0 0 0  1 U A  S  [  RdHi] [ RdLo] [  Rm ] 1001 [  Rn  ]
```

| U | A | Mnemonic | Operation |
|---|---|---|---|
| 1 | 0 | UMULL | RdHi:RdLo = Rn × Rm (unsigned) |
| 1 | 1 | UMLAL | RdHi:RdLo += Rn × Rm (unsigned) |
| 0 | 0 | SMULL | RdHi:RdLo = Rn × Rm (signed) |
| 0 | 1 | SMLAL | RdHi:RdLo += Rn × Rm (signed) |

### 6.5 Signed Halfword Multiply (DSP extensions)

| Mnemonic | Operation | Encoding hint |
|---|---|---|
| SMULBB | Rd = Rn[15:0] × Rm[15:0] | op1=0001, op2=00 |
| SMULBT | Rd = Rn[15:0] × Rm[31:16] | op1=0001, op2=01 |
| SMULTB | Rd = Rn[31:16] × Rm[15:0] | op1=0001, op2=10 |
| SMULTT | Rd = Rn[31:16] × Rm[31:16] | op1=0001, op2=11 |
| SMLABB | Rd = Rn[15:0] × Rm[15:0] + Ra | Accumulate variants |
| SMULWB | Rd = (Rn × Rm[15:0]) >> 16 | Wide multiply |
| SMMUL | Rd = (Rn × Rm) >> 32 | High 32 bits |
| SMMLA | Rd = (Rn × Rm) >> 32 + Ra | High multiply-accumulate |

---

## 7. Miscellaneous Instructions

### 7.1 Move to/from Special Registers

**MRS** (read CPSR/SPSR):
```
[cond] 0 0 0 1 0  R  0 0  1 1 1 1  [  Rd  ] 0 0 0 0 0 0 0 0 0 0 0 0
```
R: 0 = CPSR, 1 = SPSR

**MSR** (write CPSR/SPSR):
```
[cond] 0 0  I  1 0  R  1 0  [mask]  1 1 1 1  [   operand    ]
```
mask: bits [19:16] = {f, s, x, c} fields to write

### 7.2 MOVW/MOVT (ARMv7)

**MOVW** (move wide, 16-bit immediate):
```
31  28 27    24 23 20 19    16 15    12 11                          0
[cond]  0 0 1 1  0 0 0 0  [imm4] [ Rd ] [          imm12          ]
```
Rd = imm4:imm12 (16-bit value, zero-extended)

**MOVT** (move top, 16-bit to upper halfword):
```
[cond]  0 0 1 1  0 1 0 0  [imm4] [ Rd ] [          imm12          ]
```
Rd[31:16] = imm4:imm12; Rd[15:0] unchanged

### 7.3 Bit Field Operations (ARMv7)

**BFC** (bit field clear):
```
[cond] 0 1 1 1 1 1 0 [msb] [  Rd  ] [  lsb  ] 0 0 1 1 1 1 1
```

**BFI** (bit field insert):
```
[cond] 0 1 1 1 1 1 0 [msb] [  Rd  ] [  lsb  ] 0 0 1 [  Rn  ]
```

**UBFX** (unsigned bit field extract):
```
[cond] 0 1 1 1 1 1 1 [widthm1] [Rd] [  lsb  ] 1 0 1 [  Rn  ]
```

**SBFX** (signed bit field extract):
```
[cond] 0 1 1 1 1 0 1 [widthm1] [Rd] [  lsb  ] 1 0 1 [  Rn  ]
```

### 7.4 Divide (ARMv7-A, optional: A7/A15/Krait)

**SDIV**:
```
[cond] 0 1 1 1 0 0 0 1 [  Rd  ] 1 1 1 1 [  Rm  ] 0 0 0 1 [  Rn  ]
```
Rd = Rn / Rm (signed). Rm = divisor, Rn = dividend.

**UDIV**:
```
[cond] 0 1 1 1 0 0 1 1 [  Rd  ] 1 1 1 1 [  Rm  ] 0 0 0 1 [  Rn  ]
```
Rd = Rn / Rm (unsigned).

**Note**: SDIV/UDIV are optional in ARMv7-A. Check ISAR0 register,
Divide_instrs field. Not present on Cortex-A8 or A9.

### 7.5 Saturating Arithmetic

| Mnemonic | Operation |
|---|---|
| QADD | Rd = sat(Rn + Rm) signed 32-bit |
| QSUB | Rd = sat(Rn - Rm) signed 32-bit |
| QDADD | Rd = sat(Rn + sat(Rm × 2)) |
| QDSUB | Rd = sat(Rn - sat(Rm × 2)) |
| USAT | Rd = usat(Rn, #width) unsigned saturation |
| SSAT | Rd = ssat(Rn, #width) signed saturation |

### 7.6 Reversal Instructions

| Mnemonic | Operation | Encoding |
|---|---|---|
| REV | Byte-reverse word | [cond] 0110 1011 1111 Rd 1111 0011 Rm |
| REV16 | Byte-reverse each halfword | [cond] 0110 1011 1111 Rd 1111 1011 Rm |
| REVSH | Byte-reverse signed halfword | [cond] 0110 1111 1111 Rd 1111 1011 Rm |
| RBIT | Reverse bits in word | [cond] 0110 1111 1111 Rd 1111 0011 Rm |
| CLZ | Count leading zeros | [cond] 0001 0110 1111 Rd 1111 0001 Rm |

### 7.7 Barrier Instructions

| Mnemonic | Encoding | Description |
|---|---|---|
| DMB | 1111 0101 0111 1111 1111 0000 0101 [opt] | Data Memory Barrier |
| DSB | 1111 0101 0111 1111 1111 0000 0100 [opt] | Data Synchronization Barrier |
| ISB | 1111 0101 0111 1111 1111 0000 0110 [opt] | Instruction Synch Barrier |
| PLD [addr] | 1111 0101 U101 Rn 1111 [offset12] | Preload Data |
| PLI [addr] | 1111 0100 U101 Rn 1111 [offset12] | Preload Instruction |

**DMB/DSB option** (bits [3:0]):

| Option | Value | Scope |
|---|---|---|
| SY | 1111 | Full system |
| ST | 1110 | Store only |
| ISH | 1011 | Inner shareable |
| ISHST | 1010 | Inner shareable, store only |
| NSH | 0111 | Non-shareable |
| NSHST | 0110 | Non-shareable, store only |
| OSH | 0011 | Outer shareable |
| OSHST | 0010 | Outer shareable, store only |

### 7.8 Exclusive Access

| Mnemonic | Encoding hint | Description |
|---|---|---|
| LDREX Rd, [Rn] | [cond] 0001 1001 Rn Rd 1111 1001 1111 | Load exclusive word |
| STREX Rd, Rm, [Rn] | [cond] 0001 1000 Rn Rd 1111 1001 Rm | Store exclusive word |
| LDREXB Rd, [Rn] | Similar, different op2 | Load exclusive byte |
| STREXB Rd, Rm, [Rn] | Similar, different op2 | Store exclusive byte |
| LDREXH Rd, [Rn] | Similar | Load exclusive halfword |
| STREXH Rd, Rm, [Rn] | Similar | Store exclusive halfword |
| LDREXD Rd, Rd2, [Rn] | Similar | Load exclusive doubleword |
| STREXD Rd, Rm, Rm2, [Rn] | Similar | Store exclusive doubleword |
| CLREX | 1111 0101 0111 1111 1111 0000 0001 1111 | Clear exclusive monitor |

---

## 8. VFP/NEON Instruction Encoding

### 8.1 VFP Register Encoding

VFP uses a 5-bit register specifier split across two fields:

**Single-precision (S0-S31)**: 5-bit index = Vd:D or Vn:N or Vm:M
- D/N/M is bit [22]/[7]/[5] respectively
- Vd/Vn/Vm is bits [15:12]/[19:16]/[3:0]
- Register = (Vx << 1) | bit

**Double-precision (D0-D31)**: 5-bit index = D:Vd or N:Vn or M:Vm
- D/N/M is bit [22]/[7]/[5]
- Vd/Vn/Vm is bits [15:12]/[19:16]/[3:0]
- Register = (bit << 4) | Vx

**Quad (Q0-Q15)**: Use D-register encoding with Vd[0]=0 (even D-reg).
Q0 = D0:D1, Q1 = D2:D3, etc.

### 8.2 VLDR/VSTR (FP Load/Store)

```
31  28 27  24 23 22 21 20 19    16 15    12 11   8  7        0
[cond] 1 1 0 1  U D 0  L  [  Rn  ] [  Vd  ] 1010 [  imm8   ]  (F32)
[cond] 1 1 0 1  U D 0  L  [  Rn  ] [  Vd  ] 1011 [  imm8   ]  (F64)
```

- L: 1 = load, 0 = store
- U: 1 = add offset, 0 = subtract offset
- Offset = imm8 × 4 (word-aligned, range ±1020)
- cp=1010 for single-precision, cp=1011 for double-precision

### 8.3 NEON Data Processing

NEON instructions use the unconditional encoding space (cond=1111):

```
31  28 27 26 25 24 23  22 21 20 19    16 15    12 11   8  7  6  5  4  3     0
 1111   0 0 1  U  0  D  [size] [  Vn  ] [  Vd  ] [opc ] N  Q  M  [op] [ Vm ]
```

| Field | Bits | Description |
|---|---|---|
| U | [24] | Unsigned (1) or signed (0) |
| D | [22] | High bit of Vd |
| size | [21:20] | 00=8bit, 01=16bit, 10=32bit, 11=64bit |
| Vn | [19:16] | First source register |
| Vd | [15:12] | Destination register |
| opc | [11:8] | Operation code |
| N | [7] | High bit of Vn |
| Q | [6] | 0=D-register (64-bit), 1=Q-register (128-bit) |
| M | [5] | High bit of Vm |
| Vm | [3:0] | Second source register |

### 8.4 Common NEON Operations

| Mnemonic | opc [11:8] | op [4] | Description |
|---|---|---|---|
| VHADD | 0000 | 0 | Halving add |
| VRHADD | 0001 | 0 | Rounding halving add |
| VQADD | 0000 | 1 | Saturating add |
| VADD (int) | 1000 | 0 | Integer add |
| VSUB (int) | 1000 | 1 | Integer subtract |
| VMUL (int) | 1001 | 1 | Integer multiply (U=0, polynomial if U=1) |
| VAND | 0001 | 1 | Bitwise AND (size field ignored) |
| VORR | 0001 | 1 | Bitwise OR (U=0, size encoding differs) |
| VEOR | 0001 | 1 | Bitwise XOR (U=1) |
| VBSL | 0001 | 1 | Bitwise select (U=1, size=01) |
| VADD.F32 | 1101 | 0 | FP add (size=00 for F32) |
| VSUB.F32 | 1101 | 0 | FP subtract (U=1) |
| VMUL.F32 | 1101 | 1 | FP multiply |
| VMLA.F32 | — | — | FP multiply-accumulate (separate encoding) |
| VCGT | 0011 | 0 | Compare greater than |
| VCEQ | 1000 | 1 | Compare equal |
| VMAX | 0110 | 0 | Maximum |
| VMIN | 0110 | 1 | Minimum |
| VPADD | 1011 | 1 | Pairwise add |

### 8.5 NEON Load/Store (VLDn/VSTn)

```
1111 0100 0 D 10 [Rn] [Vd] [type] [size] [align] [Rm]
```

| type | Operation | Registers |
|---|---|---|
| 0111 | VLD1/VST1 (1 register) | 1 D-reg |
| 1010 | VLD1/VST1 (2 registers) | 2 D-regs |
| 0110 | VLD1/VST1 (3 registers) | 3 D-regs |
| 0010 | VLD1/VST1 (4 registers) | 4 D-regs |
| 1000 | VLD2/VST2 (2 regs) | interleave-by-2 |
| 0100 | VLD3/VST3 (3 regs) | interleave-by-3 |
| 0000 | VLD4/VST4 (4 regs) | interleave-by-4 |

- **align** (bits [5:4]): 00=no alignment, 01=64-bit, 10=128-bit, 11=256-bit
- **Rm**: Post-increment register (0x1111=none, 0x1101=Rn+=size)

### 8.6 VMOV Immediate (NEON)

NEON supports immediate moves for specific patterns:

| Pattern | Encoding (cmode) | Example |
|---|---|---|
| 32-bit: 0x000000XX | 0000 | VMOV.I32 Dd, #0xFF |
| 32-bit: 0x0000XX00 | 0010 | VMOV.I32 Dd, #0xFF00 |
| 32-bit: 0x00XX0000 | 0100 | VMOV.I32 Dd, #0xFF0000 |
| 32-bit: 0xXX000000 | 0110 | VMOV.I32 Dd, #0xFF000000 |
| 16-bit: 0x00XX | 1000 | VMOV.I16 Dd, #0xFF |
| 16-bit: 0xXX00 | 1010 | VMOV.I16 Dd, #0xFF00 |
| 32-bit: 0x0000XXFF | 1100 | VMOV.I32 Dd, #0x00FFFF |
| 32-bit: 0x00XXFFFF | 1101 | VMOV.I32 Dd, #0xFFFFFF |
| 8-bit: 0xXX per byte | 1110 | VMOV.I8 Dd, #0xFF |
| 64-bit: byte mask | 1110 (op=1) | VMOV.I64 Dd, #... |
| F32 immediate | 1111 (op=0) | VMOV.F32 Dd, #imm |

### 8.7 VFP Immediate (VMOV.F32/F64)

**VMOV.F32 Sd, #imm** (VFPv3+):
```
[cond] 1110 1D11 [imm4H] [Vd] 1010 0000 [imm4L]
```

Encodes: `(-1)^sign × (16 + imm4:imm4L) × 2^(imm4H[2:0] - 3)`

Representable values: `±(1 + n/16) × 2^r` where n ∈ [0,15], r ∈ [-3,4].

Common values and their encoding:

| Value | imm8 (hex) | Notes |
|---|---|---|
| 0.5 | 0x60 | |
| 1.0 | 0x70 | |
| 2.0 | 0x00 | Actually encoded as 2^1 × 1.0 |
| -1.0 | 0xF0 | Sign bit set |
| 10.0 | 0x24 | |
| 0.0 | — | Use VMOV.I32 Dd, #0 (NEON) instead |

---

## 9. Register Conventions (AAPCS)

### 9.1 ARM Procedure Call Standard

| Register | Alias | Role | Callee-saved? |
|---|---|---|---|
| r0 | a1 | Argument 1 / Return value | No |
| r1 | a2 | Argument 2 / Return value (64-bit) | No |
| r2 | a3 | Argument 3 | No |
| r3 | a4 | Argument 4 | No |
| r4 | v1 | Variable 1 | **Yes** |
| r5 | v2 | Variable 2 | **Yes** |
| r6 | v3 | Variable 3 | **Yes** |
| r7 | v4 | Variable 4 (frame pointer in Thumb) | **Yes** |
| r8 | v5 | Variable 5 | **Yes** |
| r9 | v6/SB | Variable 6 / Static base (platform-dependent) | **Yes** (usually) |
| r10 | v7/SL | Variable 7 / Stack limit | **Yes** |
| r11 | v8/FP | Variable 8 / Frame pointer (ARM) | **Yes** |
| r12 | IP | Intra-procedure call scratch | No |
| r13 | SP | Stack pointer | **Yes** (special) |
| r14 | LR | Link register | No (call-clobbered) |
| r15 | PC | Program counter | — (not a GPR) |

### 9.2 VFP/NEON Register Conventions

| Register | Callee-saved? | Notes |
|---|---|---|
| s0-s15 (d0-d7) | No | Arguments, return values, scratch |
| s16-s31 (d8-d15) | **Yes** | Must be preserved across calls |
| d16-d31 | No | Scratch (NEON-only, not VFP addressable as S-regs) |

**Key**: Only d8-d15 (s16-s31) are callee-saved. The full d0-d7 and
d16-d31 are caller-saved and freely available as scratch.

### 9.3 Stack Alignment

The stack pointer (SP) must be **8-byte aligned** at public interfaces
(function calls). Within a function, it may be 4-byte aligned temporarily.

**At function entry**: SP is guaranteed 8-byte aligned.
**Before PUSH**: If pushing an odd number of registers, add one extra
register (e.g., push r14 even if not needed) to maintain 8-byte alignment.

---

## 10. Thumb-2 Encoding Notes

### 10.1 Overview

Thumb-2 uses a mix of 16-bit and 32-bit instructions. The ISA is
essentially feature-complete with ARM mode for ARMv7-A.

**16-bit instruction detection**: If bits [15:11] of a halfword are
`11101`, `11110`, or `11111`, it is the first halfword of a 32-bit
Thumb instruction. Otherwise, it is a standalone 16-bit instruction.

### 10.2 Key 16-bit Instructions

| Instruction | Bits [15:0] | Range/Notes |
|---|---|---|
| MOV Rd, Rm | 0100 0110 D Rm Rd | Any register |
| ADD Rd, Rn, #imm3 | 0001 110 imm3 Rn Rd | imm: 0-7 |
| ADD Rd, #imm8 | 0011 0 Rd imm8 | Rd = r0-r7 only |
| SUB Rd, Rn, #imm3 | 0001 111 imm3 Rn Rd | imm: 0-7 |
| CMP Rn, #imm8 | 0010 1 Rn imm8 | Rn = r0-r7 only |
| LDR Rd, [Rn, #imm5×4] | 0110 1 imm5 Rn Rd | imm: 0-124, r0-r7 |
| STR Rd, [Rn, #imm5×4] | 0110 0 imm5 Rn Rd | imm: 0-124, r0-r7 |
| LDR Rd, [SP, #imm8×4] | 1001 1 Rd imm8 | imm: 0-1020 |
| B #imm11 | 1110 0 imm11 | ±2KB unconditional |
| B.cond #imm8 | 1101 cond imm8 | ±256B conditional |
| PUSH {regs} | 1011 0 10 M rlist | r0-r7 + optional LR |
| POP {regs} | 1011 1 10 P rlist | r0-r7 + optional PC |
| CBZ Rn, #off | 1011 0 0 i 1 imm5 Rn | r0-r7, forward only |
| CBNZ Rn, #off | 1011 1 0 i 1 imm5 Rn | r0-r7, forward only |
| IT{x{y{z}}} cond | 1011 1111 cond mask | If-Then block |

**Key 16-bit constraints**: Most 16-bit Thumb instructions can only
access r0-r7 (low registers). High registers (r8-r15) require 32-bit
encoding or special forms.

### 10.3 IT (If-Then) Block

The IT instruction creates a block of up to 4 conditionally-executed
instructions:

```
IT{x{y{z}}} <cond>
```

- First instruction after IT uses `<cond>`
- Subsequent instructions use `<cond>` or its inverse
- x, y, z specify T (then = same condition) or E (else = inverse)

```asm
ite   eq              @ If-Then-Else with EQ condition
moveq r0, #1          @ executed if Z=1
movne r0, #0          @ executed if Z=0
```

**Encoding**: `1011 1111 [firstcond:4] [mask:4]`
- mask encodes the T/E pattern and block length
- mask = 1000 → IT (1 instruction)
- mask = x100 → ITx (2 instructions)
- mask = xy10 → ITxy (3 instructions)
- mask = xyz1 → IThyz (4 instructions)
- where each bit is 1 if matching firstcond, 0 if inverse

### 10.4 32-bit Thumb Encoding Differences

32-bit Thumb instructions are similar to ARM instructions but with
different bit layouts. Key differences:

- No per-instruction condition code (use IT blocks instead)
- Modified immediate constant encoding: different from ARM's rotated imm8
- Different branch offset encoding (larger range for B.W vs B)
- MOVW/MOVT have different bit positions than ARM encoding

**Thumb modified immediate**: Uses a 12-bit field (i:imm3:imm8) that
encodes more patterns than ARM's rotated immediate:
- Values 0x00-0xFF (no modification)
- 0x00XY00XY, 0xXY00XY00, 0xXYXYXYXY patterns
- Rotated 8-bit constant (similar to ARM but with more rotations)

---

## 11. Encoding Helpers and Patterns

### 11.1 NOP

**ARM**: `MOV R0, R0` = 0xE1A00000 (or dedicated NOP = 0xE320F000)
**Thumb-2**: `NOP` = 0xBF00 (16-bit)

### 11.2 PC-Relative Load (Literal Pool)

```asm
ldr   r0, [pc, #offset]    @ ARM: offset from PC+8
ldr   r0, [pc, #offset]    @ Thumb: offset from Align(PC+4, 4)
```

ARM: PC reads as current instruction + 8.
Thumb: PC reads as Align(current instruction + 4, 4).

### 11.3 Function Call Sequences

**Direct call** (within ±32MB):
```asm
bl    target              @ ARM: sets LR, branches
```

**Indirect call**:
```asm
blx   r12                 @ Call through register (ARM/Thumb interwork)
```

**Long-range call** (>32MB):
```asm
movw  r12, #:lower16:target
movt  r12, #:upper16:target
blx   r12
```

### 11.4 Switch/Jump Table

```asm
@ ARM mode switch via table
ldr   pc, [pc, r0, lsl #2]  @ jump table indexed by r0
.word case0
.word case1
.word case2

@ Thumb mode (TBB/TBH instructions)
tbb   [pc, r0]              @ byte offset table
.byte (case0 - base) / 2
.byte (case1 - base) / 2
```

**TBB/TBH** (Thumb-2 only): Table Branch Byte/Halfword. Efficient for
dense switch statements with entries within 510/131070 bytes.

### 11.5 64-Bit Arithmetic

ARMv7-A has no 64-bit registers. 64-bit operations use register pairs:

```asm
@ 64-bit add: {r1:r0} = {r3:r2} + {r5:r4}
adds  r0, r2, r4         @ low word with carry out
adc   r1, r3, r5         @ high word with carry in

@ 64-bit subtract: {r1:r0} = {r3:r2} - {r5:r4}
subs  r0, r2, r4         @ low word with borrow
sbc   r1, r3, r5         @ high word with borrow

@ 64-bit shift left by 1: {r1:r0} <<= 1
adds  r0, r0, r0         @ low word: LSL #1 via ADD, sets carry
adc   r1, r1, r1         @ high word: shift in carry
```

---

## 12. Emitter Implementation Checklist

Essential items for a correct and performant ARMv7-A code emitter:

1. **Immediate encoding validation**: Check if constant fits modified
   immediate format. If not, try MVN/negation. Fall back to MOVW/MOVT
   or literal pool.

2. **Addressing mode selection**: Use the most compact addressing mode.
   Prefer immediate offset (fits in 12 bits for word, 8 bits for
   halfword/double). Use shifted register for computed indices.

3. **Condition code threading**: Emit condition codes on instructions
   where possible (ARM mode) or group into IT blocks (Thumb-2).

4. **LDRD/STRD constraints**: Even register for Rd, word-aligned address,
   8-bit immediate offset only.

5. **Register allocation awareness**: Only 13 usable GPRs. Plan for
   frequent spilling. Minimize callee-saved register usage in leaf
   functions.

6. **Thumb-2 encoding selection**: Prefer 16-bit Thumb where possible
   (smaller code, better I-cache). Fall back to 32-bit Thumb for
   high registers, large immediates, or complex operations.

7. **Alignment directives**: Emit alignment for NEON data, literal pools,
   and function entry points. Minimum function alignment: 4 bytes (ARM)
   or 2 bytes (Thumb).

8. **Literal pool management**: Keep literal pools within ±4095 bytes of
   LDR instruction (ARM) or ±1020 bytes (Thumb). Emit pool entries at
   branch points or function boundaries.

9. **NEON register pairing**: Q-registers are pairs of consecutive
   D-registers (Q0=D0:D1). When emitting Q-register operations, ensure
   the D-register number is even.

10. **PC offset calculation**: ARM mode: PC = instruction address + 8.
    Thumb mode: PC = Align(instruction address + 4, 4). Get this wrong
    and every PC-relative reference is broken.
