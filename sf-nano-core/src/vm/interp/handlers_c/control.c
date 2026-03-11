// Interpreter C handler implementations - Control flow operations
// These are FORCE_INLINE impl_* functions that get inlined into the generated wrappers.
//
// NOTE: Only `return` remains in Rust (needs call stack access).
// Branch operations (br, br_if, br_table) are implemented here.
//
// This file is #included in vm_trampoline.c before fast_c_wrappers.inc.
//
// Phase 3: TOS-Only Computation
// All handlers compute using TOS registers only. SP is not used.
// Frame access uses explicit fp-relative slots.

#include <stdint.h>
#include <string.h>

// Dereference the double pointers for direct access
#define fp (*pfp)

// =============================================================================
// Branch Fixup Helper (uses explicit fp-relative slots)
// =============================================================================

FORCE_INLINE void branch_fixup_frame(uint64_t* frame_ptr, uint16_t src_slot, uint16_t dst_slot, uint16_t count) {
    if (count == 0) {
        return;
    }
    for (uint16_t i = 0; i < count; i++) {
        frame_ptr[dst_slot + i] = frame_ptr[src_slot + i];
    }
}

FORCE_INLINE uint32_t read_u32_unaligned(const uint8_t* ptr, size_t offset) {
    uint32_t val;
    memcpy(&val, ptr + offset, sizeof(uint32_t));
    return val;
}

FORCE_INLINE int32_t read_i32_unaligned(const uint8_t* ptr, size_t offset) {
    int32_t val;
    memcpy(&val, ptr + offset, sizeof(int32_t));
    return val;
}

// =============================================================================
// Trivial Operations - tos_pattern = "none" (control flow, no TOS interaction)
// =============================================================================

FORCE_INLINE struct Instruction* impl_nop(IMPL_PARAMS_NONE) {
    (void)ctx; (void)pfp;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_end(IMPL_PARAMS_NONE) {
    (void)ctx; (void)pfp;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_block(IMPL_PARAMS_NONE) {
    (void)ctx; (void)pfp;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_loop(IMPL_PARAMS_NONE) {
    (void)ctx; (void)pfp;
    return pc_next(pc);
}

// =============================================================================
// Unreachable (trap) - tos_pattern = "none"
// =============================================================================

FORCE_INLINE struct Instruction* impl_unreachable(IMPL_PARAMS_NONE) {
    (void)pfp;
    return c_trap(ctx, "unreachable");
}

// =============================================================================
// Drop - tos_pattern = { pop = 1, push = 0 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_drop(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx; (void)pfp;
    (void)p_src;  // Value is dropped, TOS register will be reassigned
    return pc_next(pc);
}

// =============================================================================
// Select Operations - tos_pattern = { pop = 3, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_select(IMPL_PARAMS_POP3_PUSH1) {
    (void)ctx; (void)pfp;    *p_dst = ((uint32_t)*p_cond != 0) ? *p_val1 : *p_val2;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_select_t(IMPL_PARAMS_POP3_PUSH1) {
    (void)ctx; (void)pfp;    *p_dst = ((uint32_t)*p_cond != 0) ? *p_val1 : *p_val2;
    return pc_next(pc);
}

// =============================================================================
// Conditional Branch Entry Points - tos_pattern = { pop = 1, push = 0 }
// Condition is passed via TOS register (p_src)
// =============================================================================

FORCE_INLINE struct Instruction* impl_if_(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx; (void)pfp;    uint32_t cond = (uint32_t)*p_src;
    return (cond == 0) ? pc_alt(pc) : pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_else_(IMPL_PARAMS_NONE) {
    (void)ctx; (void)pfp;
    return pc_alt(pc);
}

// =============================================================================
// Branch Operations - tos_pattern = "none" (control flow)
// Use explicit fp-relative copy slots
// =============================================================================

FORCE_INLINE struct Instruction* impl_br(IMPL_PARAMS_NONE) {
    (void)ctx;
    branch_fixup_frame(fp, br_decode_src_slot(pc), br_decode_dst_slot(pc), br_decode_count(pc));

    return pc_branch_target(pc);
}

FORCE_INLINE struct Instruction* impl_br_if(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx;
    uint32_t cond = (uint32_t)*p_src;

    if (cond != 0) {
        branch_fixup_frame(
            fp,
            br_if_decode_src_slot(pc),
            br_if_decode_dst_slot(pc),
            br_if_decode_count(pc)
        );
        return pc_branch_target(pc);
    }

    return pc_next(pc);
}

// =============================================================================
// br_if_simple: Specialized for arity=0, stack_drop=0 (common loop back-edges)
// No branch fixup needed — just check condition and jump.
// =============================================================================

FORCE_INLINE struct Instruction* impl_br_if_simple(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx; (void)pfp;
    if ((uint32_t)*p_src != 0) {
        return pc_branch_target(pc);
    }
    return pc_next(pc);
}

// =============================================================================
// Branch Table - tos_pattern = { pop = 1, push = 0 }
// Index is passed via TOS register (p_src)
// =============================================================================

// Read br_table entry from inline data slots following the br_table instruction.
// Each 32-byte data pseudo-instruction holds 1 entry:
// - imm0 = rel (as i32)
// - imm1 = src_slot | (dst_slot << 16) | (count << 32)
FORCE_INLINE void read_br_table_entry(struct Instruction* pc, size_t entry_idx,
                                       int32_t* rel, uint16_t* src_slot, uint16_t* dst_slot, uint16_t* count) {
    struct Instruction* data_slot = pc + 1 + entry_idx;

    *rel = (int32_t)data_slot->imm0;
    uint64_t packed = data_slot->imm1;
    *src_slot = packed & 0xFFFF;
    *dst_slot = (packed >> 16) & 0xFFFF;
    *count = (packed >> 32) & 0xFFFF;
}

FORCE_INLINE struct Instruction* impl_br_table(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx;
    // Index comes from TOS register
    uint32_t idx = (uint32_t)*p_src;

    uint64_t entry_count = br_table_decode_entry_count(pc);

    // Clamp index to valid range (index >= entry_count takes default at entry[entry_count-1])
    size_t max_idx = entry_count > 0 ? entry_count - 1 : 0;
    size_t selected = (idx < max_idx) ? idx : max_idx;

    // Read selected entry
    int32_t rel;
    uint16_t src_slot, dst_slot, count;
    read_br_table_entry(pc, selected, &rel, &src_slot, &dst_slot, &count);
    branch_fixup_frame(fp, src_slot, dst_slot, count);

    // Branch to target
    return pc + rel;
}

#undef fp
