// Fast interpreter C handler implementations - Comparison operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>

// =============================================================================
// i32 Comparison Operations
// =============================================================================

// i32_eqz: pop1_push1
FORCE_INLINE struct Instruction* impl_i32_eqz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_EQZ(*p_src);
    return pc_next(pc);
}

// i32 binary comparisons: pop2_push1
FORCE_INLINE struct Instruction* impl_i32_eq(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_EQ(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_ne(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_NE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_lt_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_LT_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_lt_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_LT_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_gt_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_GT_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_gt_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_GT_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_le_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_LE_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_le_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_LE_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_ge_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_GE_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_ge_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_GE_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// i64 Comparison Operations
// =============================================================================

// i64_eqz: pop1_push1
FORCE_INLINE struct Instruction* impl_i64_eqz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EQZ(*p_src);
    return pc_next(pc);
}

// i64 binary comparisons: pop2_push1
FORCE_INLINE struct Instruction* impl_i64_eq(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EQ(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_ne(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_NE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_lt_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_LT_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_lt_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_LT_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_gt_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_GT_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_gt_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_GT_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_le_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_LE_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_le_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_LE_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_ge_s(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_GE_S(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_ge_u(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_GE_U(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// f32 Comparison Operations (pop2_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_eq(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_EQ(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_ne(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_NE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_lt(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_LT(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_gt(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_GT(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_le(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_LE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_ge(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_GE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// f64 Comparison Operations (pop2_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f64_eq(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_EQ(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_ne(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_NE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_lt(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_LT(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_gt(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_GT(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_le(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_LE(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_ge(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_GE(*p_lhs, *p_rhs);
    return pc_next(pc);
}
