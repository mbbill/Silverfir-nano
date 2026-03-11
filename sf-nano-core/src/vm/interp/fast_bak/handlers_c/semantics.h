// Instruction semantics building blocks for handler implementation and fusion.
//
// Each SEM_* macro captures the exact computation of a base instruction.
// This is the SINGLE SOURCE OF TRUTH for instruction semantics:
//   - Standalone handlers use SEM_* directly
//   - The fusion code generator composes SEM_* into fused super-instructions
//
// Macro shapes:
//   Expression:  result = SEM_*(args)           — pure, no side effects
//   Stmt + OUT:  SEM_*(ctx, ..., OUT)           — may trap via `return`
//   Stmt:        SEM_*(ctx, ...)                — may trap, no output
//   Side effect: SEM_LOCAL_SET(fp, idx, val)    — writes to frame

#pragma once
#include <stdint.h>
#include <limits.h>

// =============================================================================
// i32 Arithmetic (pure expressions)
// =============================================================================

#define SEM_I32_ADD(a, b)   ((uint64_t)((uint32_t)(a) + (uint32_t)(b)))
#define SEM_I32_SUB(a, b)   ((uint64_t)((uint32_t)(a) - (uint32_t)(b)))
#define SEM_I32_MUL(a, b)   ((uint64_t)((uint32_t)(a) * (uint32_t)(b)))

// i32 Division/Remainder (trapping — statement form with OUT)
// NOTE: Flat expansion (no do-while) is critical for LLVM guard-check elimination.
#define SEM_I32_DIV_S(ctx, a, b, OUT) \
    int32_t a_ = (int32_t)(a); int32_t b_ = (int32_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    if (unlikely(a_ == INT32_MIN && b_ == -1)) return c_trap(ctx, "integer overflow"); \
    (OUT) = (uint64_t)(uint32_t)(a_ / b_);

#define SEM_I32_DIV_U(ctx, a, b, OUT) \
    uint32_t a_ = (uint32_t)(a); uint32_t b_ = (uint32_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    (OUT) = (uint64_t)(a_ / b_);

#define SEM_I32_REM_S(ctx, a, b, OUT) \
    int32_t a_ = (int32_t)(a); int32_t b_ = (int32_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    if (unlikely(a_ == INT32_MIN && b_ == -1)) { (OUT) = 0; } \
    else { (OUT) = (uint64_t)(uint32_t)(a_ % b_); }

#define SEM_I32_REM_U(ctx, a, b, OUT) \
    uint32_t a_ = (uint32_t)(a); uint32_t b_ = (uint32_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    (OUT) = (uint64_t)(a_ % b_);

// =============================================================================
// i64 Arithmetic (pure expressions)
// =============================================================================

#define SEM_I64_ADD(a, b)   ((uint64_t)((a) + (b)))
#define SEM_I64_SUB(a, b)   ((uint64_t)((a) - (b)))
#define SEM_I64_MUL(a, b)   ((uint64_t)((a) * (b)))

// i64 Division/Remainder (trapping)
#define SEM_I64_DIV_S(ctx, a, b, OUT) \
    int64_t a_ = (int64_t)(a); int64_t b_ = (int64_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    if (unlikely(a_ == INT64_MIN && b_ == -1)) return c_trap(ctx, "integer overflow"); \
    (OUT) = (uint64_t)(a_ / b_);

#define SEM_I64_DIV_U(ctx, a, b, OUT) \
    uint64_t a_ = (uint64_t)(a); uint64_t b_ = (uint64_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    (OUT) = a_ / b_;

#define SEM_I64_REM_S(ctx, a, b, OUT) \
    int64_t a_ = (int64_t)(a); int64_t b_ = (int64_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    if (unlikely(a_ == INT64_MIN && b_ == -1)) { (OUT) = 0; } \
    else { (OUT) = (uint64_t)(a_ % b_); }

#define SEM_I64_REM_U(ctx, a, b, OUT) \
    uint64_t a_ = (uint64_t)(a); uint64_t b_ = (uint64_t)(b); \
    if (unlikely(b_ == 0)) return c_trap(ctx, "integer divide by zero"); \
    (OUT) = a_ % b_;

// =============================================================================
// i32 Bitwise (pure expressions)
// =============================================================================

#define SEM_I32_AND(a, b)    ((uint64_t)((uint32_t)(a) & (uint32_t)(b)))
#define SEM_I32_OR(a, b)     ((uint64_t)((uint32_t)(a) | (uint32_t)(b)))
#define SEM_I32_XOR(a, b)    ((uint64_t)((uint32_t)(a) ^ (uint32_t)(b)))
#define SEM_I32_SHL(a, b)    ((uint64_t)((uint32_t)(a) << ((uint32_t)(b) & 31)))
#define SEM_I32_SHR_U(a, b)  ((uint64_t)((uint32_t)(a) >> ((uint32_t)(b) & 31)))
#define SEM_I32_SHR_S(a, b)  ((uint64_t)(uint32_t)((int32_t)(a) >> ((uint32_t)(b) & 31)))
#define SEM_I32_ROTL(a, b)   ((uint64_t)sem_rotl32((uint32_t)(a), (uint32_t)(b)))
#define SEM_I32_ROTR(a, b)   ((uint64_t)sem_rotr32((uint32_t)(a), (uint32_t)(b)))

static inline uint32_t sem_rotl32(uint32_t v, uint32_t s) { s &= 31; return (v << s) | (v >> (32 - s)); }
static inline uint32_t sem_rotr32(uint32_t v, uint32_t s) { s &= 31; return (v >> s) | (v << (32 - s)); }

// =============================================================================
// i64 Bitwise (pure expressions)
// =============================================================================

#define SEM_I64_AND(a, b)    ((uint64_t)((a) & (b)))
#define SEM_I64_OR(a, b)     ((uint64_t)((a) | (b)))
#define SEM_I64_XOR(a, b)    ((uint64_t)((a) ^ (b)))
#define SEM_I64_SHL(a, b)    ((uint64_t)((a) << ((b) & 63)))
#define SEM_I64_SHR_U(a, b)  ((uint64_t)((a) >> ((b) & 63)))
#define SEM_I64_SHR_S(a, b)  ((uint64_t)((int64_t)(a) >> ((b) & 63)))
#define SEM_I64_ROTL(a, b)   (sem_rotl64((a), (b)))
#define SEM_I64_ROTR(a, b)   (sem_rotr64((a), (b)))

static inline uint64_t sem_rotl64(uint64_t v, uint64_t s) { s &= 63; return (v << s) | (v >> (64 - s)); }
static inline uint64_t sem_rotr64(uint64_t v, uint64_t s) { s &= 63; return (v >> s) | (v << (64 - s)); }

// =============================================================================
// i32 Comparisons (pure expressions, result is 0 or 1)
// =============================================================================

#define SEM_I32_EQZ(a)       ((uint64_t)((uint32_t)(a) == 0 ? 1 : 0))
#define SEM_I32_EQ(a, b)     ((uint64_t)((uint32_t)(a) == (uint32_t)(b) ? 1 : 0))
#define SEM_I32_NE(a, b)     ((uint64_t)((uint32_t)(a) != (uint32_t)(b) ? 1 : 0))
#define SEM_I32_LT_S(a, b)   ((uint64_t)((int32_t)(a) <  (int32_t)(b) ? 1 : 0))
#define SEM_I32_LT_U(a, b)   ((uint64_t)((uint32_t)(a) <  (uint32_t)(b) ? 1 : 0))
#define SEM_I32_GT_S(a, b)   ((uint64_t)((int32_t)(a) >  (int32_t)(b) ? 1 : 0))
#define SEM_I32_GT_U(a, b)   ((uint64_t)((uint32_t)(a) >  (uint32_t)(b) ? 1 : 0))
#define SEM_I32_LE_S(a, b)   ((uint64_t)((int32_t)(a) <= (int32_t)(b) ? 1 : 0))
#define SEM_I32_LE_U(a, b)   ((uint64_t)((uint32_t)(a) <= (uint32_t)(b) ? 1 : 0))
#define SEM_I32_GE_S(a, b)   ((uint64_t)((int32_t)(a) >= (int32_t)(b) ? 1 : 0))
#define SEM_I32_GE_U(a, b)   ((uint64_t)((uint32_t)(a) >= (uint32_t)(b) ? 1 : 0))

// =============================================================================
// i64 Comparisons (pure expressions, result is 0 or 1)
// =============================================================================

#define SEM_I64_EQZ(a)       ((uint64_t)((a) == 0 ? 1 : 0))
#define SEM_I64_EQ(a, b)     ((uint64_t)((a) == (b) ? 1 : 0))
#define SEM_I64_NE(a, b)     ((uint64_t)((a) != (b) ? 1 : 0))
#define SEM_I64_LT_S(a, b)   ((uint64_t)((int64_t)(a) <  (int64_t)(b) ? 1 : 0))
#define SEM_I64_LT_U(a, b)   ((uint64_t)((a) <  (b) ? 1 : 0))
#define SEM_I64_GT_S(a, b)   ((uint64_t)((int64_t)(a) >  (int64_t)(b) ? 1 : 0))
#define SEM_I64_GT_U(a, b)   ((uint64_t)((a) >  (b) ? 1 : 0))
#define SEM_I64_LE_S(a, b)   ((uint64_t)((int64_t)(a) <= (int64_t)(b) ? 1 : 0))
#define SEM_I64_LE_U(a, b)   ((uint64_t)((a) <= (b) ? 1 : 0))
#define SEM_I64_GE_S(a, b)   ((uint64_t)((int64_t)(a) >= (int64_t)(b) ? 1 : 0))
#define SEM_I64_GE_U(a, b)   ((uint64_t)((a) >= (b) ? 1 : 0))

// =============================================================================
// i32 Unary (pure expressions)
// =============================================================================

#define SEM_I32_CLZ(a)       ((uint64_t)((uint32_t)(a) == 0 ? 32 : (uint32_t)__builtin_clz((uint32_t)(a))))
#define SEM_I32_CTZ(a)       ((uint64_t)((uint32_t)(a) == 0 ? 32 : (uint32_t)__builtin_ctz((uint32_t)(a))))
#define SEM_I32_POPCNT(a)    ((uint64_t)(uint32_t)__builtin_popcount((uint32_t)(a)))

// =============================================================================
// i64 Unary (pure expressions)
// =============================================================================

#define SEM_I64_CLZ(a)       ((uint64_t)((a) == 0 ? 64 : (uint64_t)__builtin_clzll((a))))
#define SEM_I64_CTZ(a)       ((uint64_t)((a) == 0 ? 64 : (uint64_t)__builtin_ctzll((a))))
#define SEM_I64_POPCNT(a)    ((uint64_t)__builtin_popcountll((a)))

// =============================================================================
// Locals (frame access)
// =============================================================================

#define SEM_LOCAL_GET(fp, idx)          ((fp)[(idx)])
#define SEM_LOCAL_SET(fp, idx, val)     do { (fp)[(idx)] = (uint64_t)(val); } while(0)

// =============================================================================
// Memory — generic load/store base macros
// =============================================================================
//
// SEM_LOAD(ctx, addr, offset, byte_count, load_body):
//   Computes effective address, bounds-checks, then executes load_body.
//   load_body has `ea_` (effective address) and `base_` (mem0 pointer) in scope.
//
// SEM_STORE(ctx, addr, offset, byte_count, store_body):
//   Same, but for stores.

#define SEM_LOAD(ctx, addr, offset, byte_count, load_body) \
    uint64_t ea_ = (uint64_t)(uint32_t)(addr) + (uint32_t)(offset); \
    if (unlikely(ea_ + (byte_count) > ctx_mem0_size(ctx))) \
        return c_trap(ctx, "out of bounds memory access"); \
    uint8_t* base_ = ctx_mem0_base(ctx); \
    load_body

#define SEM_STORE(ctx, addr, offset, byte_count, store_body) \
    uint64_t ea_ = (uint64_t)(uint32_t)(addr) + (uint32_t)(offset); \
    if (unlikely(ea_ + (byte_count) > ctx_mem0_size(ctx))) \
        return c_trap(ctx, "out of bounds memory access"); \
    uint8_t* base_ = ctx_mem0_base(ctx); \
    store_body

// --- Load specializations ---

// i32 loads
#define SEM_I32_LOAD(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 4, { int32_t v_; __builtin_memcpy(&v_, base_ + ea_, 4); (OUT) = (uint64_t)(uint32_t)v_; })

#define SEM_I32_LOAD8_S(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 1, { (OUT) = (uint64_t)(uint32_t)(int32_t)(int8_t)(*(base_ + ea_)); })

#define SEM_I32_LOAD8_U(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 1, { (OUT) = (uint64_t)(uint32_t)(*(base_ + ea_)); })

#define SEM_I32_LOAD16_S(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 2, { int16_t v_; __builtin_memcpy(&v_, base_ + ea_, 2); (OUT) = (uint64_t)(uint32_t)(int32_t)v_; })

#define SEM_I32_LOAD16_U(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 2, { uint16_t v_; __builtin_memcpy(&v_, base_ + ea_, 2); (OUT) = (uint64_t)(uint32_t)v_; })

// i64 loads
#define SEM_I64_LOAD(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 8, { int64_t v_; __builtin_memcpy(&v_, base_ + ea_, 8); (OUT) = (uint64_t)v_; })

#define SEM_I64_LOAD8_S(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 1, { (OUT) = (uint64_t)(int64_t)(int8_t)(*(base_ + ea_)); })

#define SEM_I64_LOAD8_U(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 1, { (OUT) = (uint64_t)(*(base_ + ea_)); })

#define SEM_I64_LOAD16_S(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 2, { int16_t v_; __builtin_memcpy(&v_, base_ + ea_, 2); (OUT) = (uint64_t)(int64_t)v_; })

#define SEM_I64_LOAD16_U(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 2, { uint16_t v_; __builtin_memcpy(&v_, base_ + ea_, 2); (OUT) = (uint64_t)v_; })

#define SEM_I64_LOAD32_S(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 4, { int32_t v_; __builtin_memcpy(&v_, base_ + ea_, 4); (OUT) = (uint64_t)(int64_t)v_; })

#define SEM_I64_LOAD32_U(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 4, { uint32_t v_; __builtin_memcpy(&v_, base_ + ea_, 4); (OUT) = (uint64_t)v_; })

// f32/f64 loads
#define SEM_F32_LOAD(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 4, { uint32_t v_; __builtin_memcpy(&v_, base_ + ea_, 4); (OUT) = (uint64_t)v_; })

#define SEM_F64_LOAD(ctx, addr, offset, OUT) \
    SEM_LOAD(ctx, addr, offset, 8, { uint64_t v_; __builtin_memcpy(&v_, base_ + ea_, 8); (OUT) = v_; })

// --- Store specializations ---

// i32 stores
#define SEM_I32_STORE(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 4, { uint32_t v_ = (uint32_t)(val); __builtin_memcpy(base_ + ea_, &v_, 4); })

#define SEM_I32_STORE8(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 1, { uint8_t v_ = (uint8_t)(val); *(base_ + ea_) = v_; })

#define SEM_I32_STORE16(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 2, { uint16_t v_ = (uint16_t)(val); __builtin_memcpy(base_ + ea_, &v_, 2); })

// i64 stores
#define SEM_I64_STORE(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 8, { uint64_t v_ = (uint64_t)(val); __builtin_memcpy(base_ + ea_, &v_, 8); })

#define SEM_I64_STORE8(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 1, { uint8_t v_ = (uint8_t)(val); *(base_ + ea_) = v_; })

#define SEM_I64_STORE16(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 2, { uint16_t v_ = (uint16_t)(val); __builtin_memcpy(base_ + ea_, &v_, 2); })

#define SEM_I64_STORE32(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 4, { uint32_t v_ = (uint32_t)(val); __builtin_memcpy(base_ + ea_, &v_, 4); })

// f32/f64 stores
#define SEM_F32_STORE(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 4, { uint32_t v_ = (uint32_t)(val); __builtin_memcpy(base_ + ea_, &v_, 4); })

#define SEM_F64_STORE(ctx, addr, offset, val) \
    SEM_STORE(ctx, addr, offset, 8, { uint64_t v_ = (uint64_t)(val); __builtin_memcpy(base_ + ea_, &v_, 8); })

// =============================================================================
// Conversions — wrap/extend (pure expressions)
// =============================================================================

#define SEM_I32_WRAP_I64(a)         ((uint64_t)(uint32_t)(a))
#define SEM_I64_EXTEND_I32_S(a)     ((uint64_t)(int64_t)(int32_t)(a))
#define SEM_I64_EXTEND_I32_U(a)     ((uint64_t)(uint32_t)(a))

// Sign-extension
#define SEM_I32_EXTEND8_S(a)        ((uint64_t)(uint32_t)(int32_t)(int8_t)(a))
#define SEM_I32_EXTEND16_S(a)       ((uint64_t)(uint32_t)(int32_t)(int16_t)(a))
#define SEM_I64_EXTEND8_S(a)        ((uint64_t)(int64_t)(int8_t)(a))
#define SEM_I64_EXTEND16_S(a)       ((uint64_t)(int64_t)(int16_t)(a))
#define SEM_I64_EXTEND32_S(a)       ((uint64_t)(int64_t)(int32_t)(a))

// Reinterpret (bitwise copy, already stored as bits in uint64_t slots)
#define SEM_I32_REINTERPRET_F32(a)  ((uint64_t)(uint32_t)(a))
#define SEM_I64_REINTERPRET_F64(a)  ((uint64_t)(a))
#define SEM_F32_REINTERPRET_I32(a)  ((uint64_t)(uint32_t)(a))
#define SEM_F64_REINTERPRET_I64(a)  ((uint64_t)(a))

// =============================================================================
// Float arithmetic (pure expressions via bit-punning helpers)
// Requires u32_to_f32/f32_to_u32/u64_to_f64/f64_to_u64 from vm_trampoline.h
// =============================================================================

// FP_MUL_BARRIER: prevents FP contraction (mul+add → FMA) in fused handlers.
// WASM requires strict sequential float evaluation, but -ffast-math enables
// contraction that #pragma cannot reliably override. The empty asm makes the
// compiler treat the uint64_t result as opaque, preventing it from eliding
// the f64_to_u64/u64_to_f64 roundtrip needed for FMA contraction.
// Only emitted by the code generator in fused patterns with mul+add/sub.
#define FP_MUL_BARRIER(v) __asm__ __volatile__("" : "+r"(v))

#define SEM_F32_ADD(a, b)   ((uint64_t)f32_to_u32(u32_to_f32((uint32_t)(a)) + u32_to_f32((uint32_t)(b))))
#define SEM_F32_SUB(a, b)   ((uint64_t)f32_to_u32(u32_to_f32((uint32_t)(a)) - u32_to_f32((uint32_t)(b))))
#define SEM_F32_MUL(a, b)   ((uint64_t)f32_to_u32(u32_to_f32((uint32_t)(a)) * u32_to_f32((uint32_t)(b))))
#define SEM_F32_DIV(a, b)   ((uint64_t)f32_to_u32(u32_to_f32((uint32_t)(a)) / u32_to_f32((uint32_t)(b))))

#define SEM_F64_ADD(a, b)   (f64_to_u64(u64_to_f64((a)) + u64_to_f64((b))))
#define SEM_F64_SUB(a, b)   (f64_to_u64(u64_to_f64((a)) - u64_to_f64((b))))
#define SEM_F64_MUL(a, b)   (f64_to_u64(u64_to_f64((a)) * u64_to_f64((b))))
#define SEM_F64_DIV(a, b)   (f64_to_u64(u64_to_f64((a)) / u64_to_f64((b))))

// Float min/max/copysign
// NOTE: compute_f32_min/max, compute_f64_min/max, compute_f32/f64_nearest
// are defined in float_ops.c (before handler functions that expand these macros).
#define SEM_F32_MIN(a, b)      ((uint64_t)compute_f32_min((uint32_t)(a), (uint32_t)(b)))
#define SEM_F32_MAX(a, b)      ((uint64_t)compute_f32_max((uint32_t)(a), (uint32_t)(b)))
#define SEM_F32_COPYSIGN(a, b) ((uint64_t)f32_to_u32(copysignf(u32_to_f32((uint32_t)(a)), u32_to_f32((uint32_t)(b)))))
#define SEM_F64_MIN(a, b)      (compute_f64_min((a), (b)))
#define SEM_F64_MAX(a, b)      (compute_f64_max((a), (b)))
#define SEM_F64_COPYSIGN(a, b) (f64_to_u64(copysign(u64_to_f64((a)), u64_to_f64((b)))))

// Float comparisons
#define SEM_F32_EQ(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) == u32_to_f32((uint32_t)(b)) ? 1 : 0))
#define SEM_F32_NE(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) != u32_to_f32((uint32_t)(b)) ? 1 : 0))
#define SEM_F32_LT(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) <  u32_to_f32((uint32_t)(b)) ? 1 : 0))
#define SEM_F32_GT(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) >  u32_to_f32((uint32_t)(b)) ? 1 : 0))
#define SEM_F32_LE(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) <= u32_to_f32((uint32_t)(b)) ? 1 : 0))
#define SEM_F32_GE(a, b)    ((uint64_t)(u32_to_f32((uint32_t)(a)) >= u32_to_f32((uint32_t)(b)) ? 1 : 0))

#define SEM_F64_EQ(a, b)    ((uint64_t)(u64_to_f64((a)) == u64_to_f64((b)) ? 1 : 0))
#define SEM_F64_NE(a, b)    ((uint64_t)(u64_to_f64((a)) != u64_to_f64((b)) ? 1 : 0))
#define SEM_F64_LT(a, b)    ((uint64_t)(u64_to_f64((a)) <  u64_to_f64((b)) ? 1 : 0))
#define SEM_F64_GT(a, b)    ((uint64_t)(u64_to_f64((a)) >  u64_to_f64((b)) ? 1 : 0))
#define SEM_F64_LE(a, b)    ((uint64_t)(u64_to_f64((a)) <= u64_to_f64((b)) ? 1 : 0))
#define SEM_F64_GE(a, b)    ((uint64_t)(u64_to_f64((a)) >= u64_to_f64((b)) ? 1 : 0))

// Float unary
#define SEM_F32_ABS(a)      ((uint64_t)f32_to_u32(fabsf(u32_to_f32((uint32_t)(a)))))
#define SEM_F32_NEG(a)      ((uint64_t)f32_to_u32(-u32_to_f32((uint32_t)(a))))
#define SEM_F32_CEIL(a)     ((uint64_t)f32_to_u32(ceilf(u32_to_f32((uint32_t)(a)))))
#define SEM_F32_FLOOR(a)    ((uint64_t)f32_to_u32(floorf(u32_to_f32((uint32_t)(a)))))
#define SEM_F32_TRUNC(a)    ((uint64_t)f32_to_u32(truncf(u32_to_f32((uint32_t)(a)))))
#define SEM_F32_NEAREST(a)  ((uint64_t)compute_f32_nearest((uint32_t)(a)))
#define SEM_F32_SQRT(a)     ((uint64_t)f32_to_u32(sqrtf(u32_to_f32((uint32_t)(a)))))

#define SEM_F64_ABS(a)      (f64_to_u64(fabs(u64_to_f64((a)))))
#define SEM_F64_NEG(a)      (f64_to_u64(-u64_to_f64((a))))
#define SEM_F64_CEIL(a)     (f64_to_u64(ceil(u64_to_f64((a)))))
#define SEM_F64_FLOOR(a)    (f64_to_u64(floor(u64_to_f64((a)))))
#define SEM_F64_TRUNC(a)    (f64_to_u64(trunc(u64_to_f64((a)))))
#define SEM_F64_NEAREST(a)  (compute_f64_nearest((a)))
#define SEM_F64_SQRT(a)     (f64_to_u64(sqrt(u64_to_f64((a)))))

// Float conversions from integers
#define SEM_F32_CONVERT_I32_S(a)  ((uint64_t)f32_to_u32((float)(int32_t)(a)))
#define SEM_F32_CONVERT_I32_U(a)  ((uint64_t)f32_to_u32((float)(uint32_t)(a)))
#define SEM_F32_CONVERT_I64_S(a)  ((uint64_t)f32_to_u32((float)(int64_t)(a)))
#define SEM_F32_CONVERT_I64_U(a)  ((uint64_t)f32_to_u32((float)(uint64_t)(a)))
#define SEM_F64_CONVERT_I32_S(a)  (f64_to_u64((double)(int32_t)(a)))
#define SEM_F64_CONVERT_I32_U(a)  (f64_to_u64((double)(uint32_t)(a)))
#define SEM_F64_CONVERT_I64_S(a)  (f64_to_u64((double)(int64_t)(a)))
#define SEM_F64_CONVERT_I64_U(a)  (f64_to_u64((double)(uint64_t)(a)))

// Float demote/promote
#define SEM_F32_DEMOTE_F64(a)     ((uint64_t)f32_to_u32((float)u64_to_f64((a))))
#define SEM_F64_PROMOTE_F32(a)    (f64_to_u64((double)u32_to_f32((uint32_t)(a))))
