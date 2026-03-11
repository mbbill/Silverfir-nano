// Fast interpreter C handler implementations - Conversion operations
// Implementations use SEM_* macros from semantics.h where applicable.
// Trapping truncation and saturating truncation keep inline logic
// (too complex for single-expression macros).
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>
#include <math.h>

// =============================================================================
// Wrap / Extend (non-trapping)
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_wrap_i64(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_WRAP_I64(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_extend_i32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EXTEND_I32_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_extend_i32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EXTEND_I32_U(*p_src);
    return pc_next(pc);
}

// =============================================================================
// Sign-extend Operations (non-trapping)
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_extend8_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_EXTEND8_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_extend16_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_EXTEND16_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_extend8_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EXTEND8_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_extend16_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EXTEND16_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_extend32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_EXTEND32_S(*p_src);
    return pc_next(pc);
}

// =============================================================================
// Trapping Truncation (f32/f64 -> i32/i64)
// These have multi-step trap logic, kept inline.
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_trunc_f32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = (double)u32_to_f32((uint32_t)*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < (double)INT32_MIN || t > (double)INT32_MAX) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(uint32_t)(int32_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_f32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = (double)u32_to_f32((uint32_t)*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < 0.0 || t >= 4294967296.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(uint32_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_f64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = u64_to_f64(*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < (double)INT32_MIN || t > (double)INT32_MAX) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(uint32_t)(int32_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_f64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = u64_to_f64(*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < 0.0 || t >= 4294967296.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(uint32_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_f32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = (double)u32_to_f32((uint32_t)*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < -9223372036854775808.0 || t >= 9223372036854775808.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(int64_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_f32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = (double)u32_to_f32((uint32_t)*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < 0.0 || t >= 18446744073709551616.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_f64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = u64_to_f64(*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < -9223372036854775808.0 || t >= 9223372036854775808.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)(int64_t)t;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_f64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    double f = u64_to_f64(*p_src);
    if (!isfinite(f)) return c_trap(ctx, "integer overflow");
    double t = trunc(f);
    if (t < 0.0 || t >= 18446744073709551616.0) return c_trap(ctx, "integer overflow");
    *p_dst = (uint64_t)t;
    return pc_next(pc);
}

// =============================================================================
// Saturating Truncation (non-trapping) — kept inline (multi-branch logic)
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_trunc_sat_f32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    float f = u32_to_f32((uint32_t)*p_src);
    int32_t result;
    if (isnan(f)) result = 0;
    else if (f < (float)INT32_MIN) result = INT32_MIN;
    else if (f > (float)INT32_MAX) result = INT32_MAX;
    else result = (int32_t)f;
    *p_dst = (uint64_t)(uint32_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_sat_f32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    float f = u32_to_f32((uint32_t)*p_src);
    uint32_t result;
    if (isnan(f) || f < 0.0f) result = 0;
    else if (f > (float)UINT32_MAX) result = UINT32_MAX;
    else result = (uint32_t)f;
    *p_dst = (uint64_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_sat_f64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    double f = u64_to_f64(*p_src);
    int32_t result;
    if (isnan(f)) result = 0;
    else if (f < (double)INT32_MIN) result = INT32_MIN;
    else if (f > (double)INT32_MAX) result = INT32_MAX;
    else result = (int32_t)f;
    *p_dst = (uint64_t)(uint32_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_trunc_sat_f64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    double f = u64_to_f64(*p_src);
    uint32_t result;
    if (isnan(f) || f < 0.0) result = 0;
    else if (f > (double)UINT32_MAX) result = UINT32_MAX;
    else result = (uint32_t)f;
    *p_dst = (uint64_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_sat_f32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    float f = u32_to_f32((uint32_t)*p_src);
    int64_t result;
    if (isnan(f)) result = 0;
    else if (f < (float)INT64_MIN) result = INT64_MIN;
    else if (f > (float)INT64_MAX) result = INT64_MAX;
    else result = (int64_t)f;
    *p_dst = (uint64_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_sat_f32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    float f = u32_to_f32((uint32_t)*p_src);
    uint64_t result;
    if (isnan(f) || f < 0.0f) result = 0;
    else if (f > (float)UINT64_MAX) result = UINT64_MAX;
    else result = (uint64_t)f;
    *p_dst = result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_sat_f64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    double f = u64_to_f64(*p_src);
    int64_t result;
    if (isnan(f)) result = 0;
    else if (f < (double)INT64_MIN) result = INT64_MIN;
    else if (f > (double)INT64_MAX) result = INT64_MAX;
    else result = (int64_t)f;
    *p_dst = (uint64_t)result;
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_trunc_sat_f64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    double f = u64_to_f64(*p_src);
    uint64_t result;
    if (isnan(f) || f < 0.0) result = 0;
    else if (f > (double)UINT64_MAX) result = UINT64_MAX;
    else result = (uint64_t)f;
    *p_dst = result;
    return pc_next(pc);
}

// =============================================================================
// Float conversions from integers (non-trapping)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_convert_i32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_CONVERT_I32_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_convert_i32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_CONVERT_I32_U(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_convert_i64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_CONVERT_I64_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_convert_i64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_CONVERT_I64_U(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_convert_i32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_CONVERT_I32_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_convert_i32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_CONVERT_I32_U(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_convert_i64_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_CONVERT_I64_S(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_convert_i64_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_CONVERT_I64_U(*p_src);
    return pc_next(pc);
}

// =============================================================================
// Float demote/promote (non-trapping)
// =============================================================================

FORCE_INLINE struct Instruction* impl_f32_demote_f64(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_DEMOTE_F64(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_promote_f32(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_PROMOTE_F32(*p_src);
    return pc_next(pc);
}

// =============================================================================
// Reinterpret (non-trapping) - just bitwise copy
// =============================================================================

FORCE_INLINE struct Instruction* impl_i32_reinterpret_f32(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I32_REINTERPRET_F32(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_reinterpret_f64(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_I64_REINTERPRET_F64(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f32_reinterpret_i32(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F32_REINTERPRET_I32(*p_src);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_reinterpret_i64(IMPL_PARAMS_POP1_PUSH1) {
    (void)ctx; (void)pfp;
    *p_dst = SEM_F64_REINTERPRET_I64(*p_src);
    return pc_next(pc);
}
