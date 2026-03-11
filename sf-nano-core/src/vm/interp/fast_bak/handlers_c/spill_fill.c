// =============================================================================
// TOS Spill/Fill Handlers
// =============================================================================
//
// Spill: Write TOS registers to frame slots (operand stack in memory)
// Fill: Read frame slots into TOS registers
//
// 4 spill handlers (spill_1..4) × 4 variants (D1..D4) = 16 wrappers
// 4 fill handlers (fill_1..4) × 4 variants (D1..D4) = 16 wrappers
//
// The variant determines which physical registers map to positions 1-4.
// The wrapper passes the correct register pointers based on variant.
//
// Encoding: slot (16 bits) = frame_size + height - 1 (top of operand stack)
//
// Memory layout:
//   fp[slot]     = position 1 (top of stack)
//   fp[slot - 1] = position 2
//   fp[slot - 2] = position 3
//   fp[slot - 3] = position 4
//
// =============================================================================

#include <stdint.h>

#define fp (*pfp)

// =============================================================================
// Spill Handlers - Write TOS to Memory
// =============================================================================

// spill_1: Write 1 value (position 1) to fp[slot]
// tos_pattern = { pop = 1, push = 0 }
FORCE_INLINE struct Instruction* impl_spill_1(IMPL_PARAMS_POP1) {
    (void)ctx;    uint16_t s = spill_1_decode_slot(pc);
    fp[s] = *p_src;
    return pc_next(pc);
}

// spill_2: Write 2 values (positions 1, 2) to fp[slot], fp[slot-1]
// tos_pattern = { pop = 2, push = 0 }
FORCE_INLINE struct Instruction* impl_spill_2(IMPL_PARAMS_POP2) {
    (void)ctx;    uint16_t s = spill_2_decode_slot(pc);
    fp[s]     = *p_rhs;  // position 1 (rhs = top)
    fp[s - 1] = *p_lhs;  // position 2 (lhs = second)
    return pc_next(pc);
}

// spill_3: Write 3 values (positions 1, 2, 3) to memory
// tos_pattern = { pop = 3, push = 0 }
FORCE_INLINE struct Instruction* impl_spill_3(IMPL_PARAMS_POP3) {
    (void)ctx;    uint16_t s = spill_3_decode_slot(pc);
    fp[s]     = *p_c;    // position 1
    fp[s - 1] = *p_b;    // position 2
    fp[s - 2] = *p_a;    // position 3
    return pc_next(pc);
}

// spill_4: Write 4 values (positions 1, 2, 3, 4) to memory
// tos_pattern = { pop = 4, push = 0 }
FORCE_INLINE struct Instruction* impl_spill_4(IMPL_PARAMS_POP4) {
    (void)ctx;    uint16_t s = spill_4_decode_slot(pc);
    fp[s]     = *p_d;    // position 1
    fp[s - 1] = *p_c;    // position 2
    fp[s - 2] = *p_b;    // position 3
    fp[s - 3] = *p_a;    // position 4
    return pc_next(pc);
}

// =============================================================================
// Fill Handlers - Read Memory into TOS
// =============================================================================

// fill_1: Read 1 value from fp[slot] into position 1
// tos_pattern = { pop = 0, push = 1 }
FORCE_INLINE struct Instruction* impl_fill_1(IMPL_PARAMS_POP0_PUSH1) {
    (void)ctx;    uint16_t s = fill_1_decode_slot(pc);
    *p_dst = fp[s];
    return pc_next(pc);
}

// fill_2: Read 2 values from memory into positions 1, 2
// tos_pattern = { pop = 0, push = 2 }
FORCE_INLINE struct Instruction* impl_fill_2(IMPL_PARAMS_POP0_PUSH2) {
    (void)ctx;    uint16_t s = fill_2_decode_slot(pc);
    *p_dst1 = fp[s];      // position 1
    *p_dst0 = fp[s - 1];  // position 2
    return pc_next(pc);
}

// fill_3: Read 3 values from memory into positions 1, 2, 3
// tos_pattern = { pop = 0, push = 3 }
FORCE_INLINE struct Instruction* impl_fill_3(IMPL_PARAMS_POP0_PUSH3) {
    (void)ctx;    uint16_t s = fill_3_decode_slot(pc);
    *p_dst2 = fp[s];      // position 1
    *p_dst1 = fp[s - 1];  // position 2
    *p_dst0 = fp[s - 2];  // position 3
    return pc_next(pc);
}

// fill_4: Read 4 values from memory into positions 1, 2, 3, 4
// tos_pattern = { pop = 0, push = 4 }
FORCE_INLINE struct Instruction* impl_fill_4(IMPL_PARAMS_POP0_PUSH4) {
    (void)ctx;    uint16_t s = fill_4_decode_slot(pc);
    *p_dst3 = fp[s];      // position 1
    *p_dst2 = fp[s - 1];  // position 2
    *p_dst1 = fp[s - 2];  // position 3
    *p_dst0 = fp[s - 3];  // position 4
    return pc_next(pc);
}

#undef fp
