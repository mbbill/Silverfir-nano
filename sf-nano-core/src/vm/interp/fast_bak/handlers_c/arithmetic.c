// Fast interpreter C handler implementations - Arithmetic operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>
#include <limits.h>

// =============================================================================
// i32 ARITHMETIC - SIMPLE (no traps)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_add(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_ADD(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_sub(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_SUB(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_mul(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_MUL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// i32 DIVISION/REMAINDER (with traps)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_div_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I32_DIV_S(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_div_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I32_DIV_U(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_rem_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I32_REM_S(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_rem_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I32_REM_U(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

// =============================================================================
// i64 ARITHMETIC - SIMPLE (no traps)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i64_add(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_ADD(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_sub(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_SUB(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_mul(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_MUL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// i64 DIVISION/REMAINDER (with traps)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_i64_div_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I64_DIV_S(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_div_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I64_DIV_U(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_rem_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I64_REM_S(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_rem_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)pfp;
    SEM_I64_REM_U(ctx, *p_lhs, *p_rhs, *p_dst);
    return pc_next(pc);
}

// =============================================================================
// f32 ARITHMETIC (no traps - IEEE 754 semantics)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_add(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_ADD(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_sub(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_SUB(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_mul(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_MUL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_div(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_DIV(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// f64 ARITHMETIC (no traps - IEEE 754 semantics)
// tos_pattern = { pop = 2, push = 1 }
// =============================================================================

FORCE_INLINE struct Instruction* impl_f64_add(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_ADD(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_sub(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_SUB(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_mul(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_MUL(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_div(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_DIV(*p_lhs, *p_rhs);
    return pc_next(pc);
}
