// Fast interpreter C handler implementations - Bitwise operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>

// =============================================================================
// i32 Bitwise Operations
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_and(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_AND(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_or(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_OR(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_xor(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_XOR(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_shl(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_SHL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_shr_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_SHR_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_shr_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_SHR_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_rotl(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_ROTL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_rotr(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_ROTR(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// i64 Bitwise Operations
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i64_and(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_AND(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_or(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_OR(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_xor(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_XOR(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_shl(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_SHL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_shr_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_SHR_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_shr_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_SHR_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_rotl(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_ROTL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_rotr(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_ROTR(*p_lhs, *p_rhs);
    return pc_next(pc);
}
