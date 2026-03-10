// Fast interpreter C handler implementations - Unary operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>

// =============================================================================
// i32 Unary Operations (pop1_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_clz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_CLZ(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_ctz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_CTZ(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_popcnt(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_POPCNT(*p_src);
    return pc_next(pc);
}

// =============================================================================
// i64 Unary Operations (pop1_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_i64_clz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_CLZ(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_ctz(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_CTZ(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_popcnt(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_POPCNT(*p_src);
    return pc_next(pc);
}
