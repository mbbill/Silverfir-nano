// Fast interpreter C handler implementations - Float operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <math.h>

// Helper constants and functions for min/max/nearest with special NaN/zero semantics.
#define F32_NEG_ZERO 0x80000000u
#define F64_NEG_ZERO 0x8000000000000000ull

static inline uint32_t compute_f32_min(uint32_t lb, uint32_t rb) {
    float l = u32_to_f32(lb);
    float r = u32_to_f32(rb);
    if (isnan(l) || isnan(r)) return f32_to_u32(NAN);
    if (l == r) return (lb == F32_NEG_ZERO || rb == F32_NEG_ZERO) ? F32_NEG_ZERO : lb;
    return (l < r) ? lb : rb;
}

static inline uint32_t compute_f32_max(uint32_t lb, uint32_t rb) {
    float l = u32_to_f32(lb);
    float r = u32_to_f32(rb);
    if (isnan(l) || isnan(r)) return f32_to_u32(NAN);
    if (l == r) return (lb == 0 || rb == 0) ? 0 : lb;
    return (l > r) ? lb : rb;
}

static inline uint64_t compute_f64_min(uint64_t lb, uint64_t rb) {
    double l = u64_to_f64(lb);
    double r = u64_to_f64(rb);
    if (isnan(l) || isnan(r)) return f64_to_u64(NAN);
    if (l == r) return (lb == F64_NEG_ZERO || rb == F64_NEG_ZERO) ? F64_NEG_ZERO : lb;
    return (l < r) ? lb : rb;
}

static inline uint64_t compute_f64_max(uint64_t lb, uint64_t rb) {
    double l = u64_to_f64(lb);
    double r = u64_to_f64(rb);
    if (isnan(l) || isnan(r)) return f64_to_u64(NAN);
    if (l == r) return (lb == 0 || rb == 0) ? 0 : lb;
    return (l > r) ? lb : rb;
}

static inline uint32_t compute_f32_nearest(uint32_t bits) {
    float v = u32_to_f32(bits);
    if (!isfinite(v)) return bits;
    float fl = floorf(v);
    float diff = v - fl;
    float rounded;
    if (diff < 0.5f) rounded = fl;
    else if (diff > 0.5f) rounded = fl + 1.0f;
    else rounded = ((int64_t)fl % 2 == 0) ? fl : fl + 1.0f;
    return f32_to_u32(rounded);
}

static inline uint64_t compute_f64_nearest(uint64_t bits) {
    double v = u64_to_f64(bits);
    if (!isfinite(v)) return bits;
    double fl = floor(v);
    double diff = v - fl;
    double rounded;
    if (diff < 0.5) rounded = fl;
    else if (diff > 0.5) rounded = fl + 1.0;
    else rounded = (((int64_t)fl % 2) == 0) ? fl : fl + 1.0;
    return f64_to_u64(rounded);
}

// =============================================================================
// f32 Binary Operations (pop2_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_min(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_MIN(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_max(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_MAX(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_copysign(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_COPYSIGN(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// f32 Unary Operations (pop1_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_abs(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_ABS(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_neg(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_NEG(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_ceil(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_CEIL(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_floor(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_FLOOR(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_trunc(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_TRUNC(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_nearest(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_NEAREST(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_sqrt(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_SQRT(*p_src);
    return pc_next(pc);
}

// =============================================================================
// f64 Binary Operations (pop2_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f64_min(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_MIN(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_max(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_MAX(*p_lhs, *p_rhs);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_copysign(IMPL_PARAMS_POP2_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_COPYSIGN(*p_lhs, *p_rhs);
    return pc_next(pc);
}

// =============================================================================
// f64 Unary Operations (pop1_push1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f64_abs(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_ABS(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_neg(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_NEG(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_ceil(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_CEIL(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_floor(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_FLOOR(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_trunc(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_TRUNC(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_nearest(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_NEAREST(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_sqrt(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_SQRT(*p_src);
    return pc_next(pc);
}
