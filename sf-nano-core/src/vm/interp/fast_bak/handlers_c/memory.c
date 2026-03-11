// Fast interpreter C handler implementations - Memory load/store operations
// Implementations use SEM_* macros from semantics.h (single source of truth).
//
// NOTE: These handlers ONLY support memory index 0 (the common fast path).
// The compiler selects between C fast path (mem_idx == 0) and Rust slow path
// (mem_idx != 0) at compile time. No memidx decoding needed here.
//
// This file is #included in vm_trampoline.c after semantics.h.

#include <stdint.h>
#include <string.h>

// =============================================================================
// LOAD OPERATIONS (pop1_push1)
// =============================================================================

// --- i32 loads ---

FORCE_INLINE struct Instruction* impl_i32_load(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I32_LOAD(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_load8_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I32_LOAD8_S(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_load8_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I32_LOAD8_U(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_load16_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I32_LOAD16_S(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_load16_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I32_LOAD16_U(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

// --- i64 loads ---

FORCE_INLINE struct Instruction* impl_i64_load(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load8_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD8_S(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load8_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD8_U(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load16_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD16_S(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load16_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD16_U(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load32_s(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD32_S(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_load32_u(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_I64_LOAD32_U(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

// --- f32/f64 loads ---

FORCE_INLINE struct Instruction* impl_f32_load(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_F32_LOAD(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_load(IMPL_PARAMS_POP1_PUSH1) {
    (void)pfp;
    uint32_t offset = load_decode_offset(pc);
    SEM_F64_LOAD(ctx, *p_src, offset, *p_dst);
    return pc_next(pc);
}

// =============================================================================
// STORE OPERATIONS (pop2_push0)
// =============================================================================

// --- i32 stores ---

FORCE_INLINE struct Instruction* impl_i32_store(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I32_STORE(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_store8(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I32_STORE8(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i32_store16(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I32_STORE16(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

// --- i64 stores ---

FORCE_INLINE struct Instruction* impl_i64_store(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I64_STORE(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_store8(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I64_STORE8(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_store16(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I64_STORE16(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_i64_store32(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_I64_STORE32(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

// --- f32/f64 stores ---

FORCE_INLINE struct Instruction* impl_f32_store(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_F32_STORE(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}

FORCE_INLINE struct Instruction* impl_f64_store(IMPL_PARAMS_POP2_PUSH0) {
    (void)pfp;
    uint32_t offset = store_decode_offset(pc);
    SEM_F64_STORE(ctx, *p_addr, offset, *p_val);
    return pc_next(pc);
}
