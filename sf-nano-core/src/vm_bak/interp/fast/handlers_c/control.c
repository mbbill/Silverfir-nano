// Fast interpreter C handler implementations - Control flow operations
// These are FORCE_INLINE impl_* functions that get inlined into the generated wrappers.
//
// NOTE: Only `return` remains in Rust (needs call stack access).
// Branch operations (br, br_if, br_table) are implemented here.
//
// This file is #included in vm_trampoline.c before fast_c_wrappers.inc.
//
// Phase 3: TOS-Only Computation
// All handlers compute using TOS registers only. SP is not used.
// Stack access uses fp-relative addressing with operand_base_offset.

#include <stdint.h>
#include <string.h>

// Dereference the double pointers for direct access
#define fp (*pfp)

// Operand stack access macros (fp-relative)
// operand_base = fp + operand_base_offset/8
// operand_base[index] = value at that stack slot
#define OPERAND_BASE(operand_base_offset) ((uint64_t*)((uint8_t*)fp + (operand_base_offset)))

// =============================================================================
// Branch Fixup Helper (uses fp-based operand stack access)
// =============================================================================

FORCE_INLINE void branch_fixup_frame(uint64_t* operand_base, size_t height, size_t stack_drop, size_t arity) {
    if (stack_drop == 0 || arity == 0) {
        return;
    }
    // Move arity values from [height-arity..height) to [height-stack_drop-arity..height-stack_drop)
    for (size_t i = 0; i < arity; i++) {
        operand_base[height - stack_drop - arity + i] = operand_base[height - arity + i];
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
// Use fp-based operand stack access with encoded operand_base_offset and height
// =============================================================================

FORCE_INLINE struct Instruction* impl_br(IMPL_PARAMS_NONE) {
    (void)ctx;
    size_t stack_drop = br_decode_stack_drop(pc);
    size_t arity = br_decode_arity(pc);
    uint16_t height = br_decode_height(pc);
    uint32_t operand_base_offset = br_decode_operand_base_offset(pc);

    // Branch fixup using fp-relative operand stack access
    uint64_t* operand_base = OPERAND_BASE(operand_base_offset);
    branch_fixup_frame(operand_base, height, stack_drop, arity);

    return pc_branch_target(pc);
}

FORCE_INLINE struct Instruction* impl_br_if(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx;
    uint32_t cond = (uint32_t)*p_src;

    if (cond != 0) {
        uint16_t arity = br_if_decode_arity(pc);
        uint16_t height = br_if_decode_height(pc);
        uint32_t operand_base_offset = br_if_decode_operand_base_offset(pc);
        size_t stack_drop = br_if_decode_stack_drop(pc);

        uint64_t* operand_base = OPERAND_BASE(operand_base_offset);
        // Note: height - 1 because we've conceptually popped the condition
        branch_fixup_frame(operand_base, height - 1, stack_drop, arity);
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
// Each 32-byte data pseudo-instruction holds 2 entries:
// - Entry 0: imm0 = rel (as i32), imm1 = (stack_drop << 16) | arity
// - Entry 1: imm2 = (rel << 32) | (stack_drop << 16) | arity
FORCE_INLINE void read_br_table_entry(struct Instruction* pc, size_t entry_idx,
                                       int32_t* rel, size_t* stack_drop, size_t* arity) {
    size_t slot_idx = entry_idx / 2;
    size_t entry_in_slot = entry_idx % 2;
    struct Instruction* data_slot = pc + 1 + slot_idx;

    if (entry_in_slot == 0) {
        // Entry 0: stored in imm0 and imm1
        *rel = (int32_t)data_slot->imm0;
        uint64_t packed = data_slot->imm1;
        *stack_drop = (packed >> 16) & 0xFFFF;
        *arity = packed & 0xFFFF;
    } else {
        // Entry 1: packed in imm2
        uint64_t packed = data_slot->imm2;
        *rel = (int32_t)(packed >> 32);
        *stack_drop = (packed >> 16) & 0xFFFF;
        *arity = packed & 0xFFFF;
    }
}

FORCE_INLINE struct Instruction* impl_br_table(IMPL_PARAMS_POP1_PUSH0) {
    (void)ctx;
    // Index comes from TOS register
    uint32_t idx = (uint32_t)*p_src;

    uint64_t entry_count = br_table_decode_entry_count(pc);
    uint32_t operand_base_offset = br_table_decode_operand_base_offset(pc);
    uint16_t height = br_table_decode_height(pc);

    // Clamp index to valid range (index >= entry_count takes default at entry[entry_count-1])
    size_t max_idx = entry_count > 0 ? entry_count - 1 : 0;
    size_t selected = (idx < max_idx) ? idx : max_idx;

    // Read selected entry
    int32_t rel;
    size_t stack_drop, arity;
    read_br_table_entry(pc, selected, &rel, &stack_drop, &arity);

    // Branch fixup: move arity values down by stack_drop
    // height - 1 because we conceptually popped the index
    uint64_t* operand_base = OPERAND_BASE(operand_base_offset);
    branch_fixup_frame(operand_base, height - 1, stack_drop, arity);

    // Branch to target
    return pc + rel;
}

#undef fp
#undef OPERAND_BASE
