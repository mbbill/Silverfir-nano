#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Branch prediction hints
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Force inline - guarantees inlining in LLVM
#define FORCE_INLINE __attribute__((always_inline)) static inline
#define PRESERVE_NONE __attribute__((preserve_none))

// Forward declarations
struct Ctx;  // opaque
struct Instruction;

// =============================================================================
// Context Access (C-visible portion)
// =============================================================================

// C-accessible portion of Context (first fields only).
// SAFETY: Must match Rust Context layout exactly (see context.rs).
//
// Layout: stack_end(8B) | call_depth(8B) | mem0_base(8B) | mem0_size(8B) | ...opaque...
struct CtxHot {
    uint64_t* stack_end;    // offset 0: stack overflow check pointer
    uint64_t call_depth;    // offset 8: call depth counter
    uint8_t* mem0_base;     // offset 16: direct pointer to memory 0
    uint64_t mem0_size;     // offset 24: size of memory 0 in bytes
    const char* trap_message;       // offset 32: deferred trap message (NULL = no trap)
    struct Instruction* term_inst;  // offset 40: cached TERM_INST pointer
};

_Static_assert(offsetof(struct CtxHot, stack_end) == 0, "CtxHot.stack_end offset mismatch");
_Static_assert(offsetof(struct CtxHot, call_depth) == 8, "CtxHot.call_depth offset mismatch");
_Static_assert(offsetof(struct CtxHot, mem0_base) == 16, "CtxHot.mem0_base offset mismatch");
_Static_assert(offsetof(struct CtxHot, mem0_size) == 24, "CtxHot.mem0_size offset mismatch");
_Static_assert(offsetof(struct CtxHot, trap_message) == 32, "CtxHot.trap_message offset mismatch");
_Static_assert(offsetof(struct CtxHot, term_inst) == 40, "CtxHot.term_inst offset mismatch");
_Static_assert(sizeof(struct CtxHot) == 48, "CtxHot size mismatch");

#define MAX_CALL_DEPTH 300

// Access macros for Context fields
#define ctx_stack_end(ctx)   (((struct CtxHot*)(ctx))->stack_end)
#define ctx_call_depth(ctx)  (((struct CtxHot*)(ctx))->call_depth)
#define ctx_mem0_base(ctx)   (((struct CtxHot*)(ctx))->mem0_base)
#define ctx_mem0_size(ctx)   (((struct CtxHot*)(ctx))->mem0_size)
#define ctx_trap_message(ctx) (((struct CtxHot*)(ctx))->trap_message)
#define ctx_term_inst(ctx)    (((struct CtxHot*)(ctx))->term_inst)

// Inline trap: stores message and returns TERM_INST with zero function calls.
// This keeps the calling handler a leaf function (no prologue/epilogue).
FORCE_INLINE struct Instruction* c_trap(struct Ctx* ctx, const char* msg) {
    ctx_trap_message(ctx) = msg;
    return ctx_term_inst(ctx);
}

// Opaque next-handler type for preloaded dispatch.
// Semantically an OpHandler, declared separately to avoid recursive typedef.
// All function pointers are pointer-sized, so this is ABI-compatible.
typedef void (*NextHandler)(void);

// Handler function pointer type (preserve_none calling convention)
// MUST be defined after struct Instruction forward declaration
// Wrappers receive TOS registers BY VALUE - they're passed in CPU registers
// Phase 3: sp removed from signature - handlers use TOS registers only
// (handlers that need sp access it via ctx_sp)
// nh: preloaded handler for the instruction after the current one's successor.
// See next-handler preloading design for dispatch categories.
PRESERVE_NONE
typedef void (*OpHandler)(
    struct Ctx* ctx,
    struct Instruction* pc,
    uint64_t* fp,
    uint64_t l0,               // Local register cache (fp[0] cached)
    uint64_t l1,               // Local register cache (fp[1] cached)
    uint64_t l2,               // Local register cache (fp[2] cached)
    uint64_t t0,               // TOS register 0 (by value)
    uint64_t t1,               // TOS register 1 (by value)
    uint64_t t2,               // TOS register 2 (by value)
    uint64_t t3,               // TOS register 3 (by value)
    NextHandler nh             // preloaded next handler
);

// Mirrors the Rust Instruction struct layout
// SAFETY: Keep in sync with Rust definition in instruction.rs
//
// Layout (32 bytes):
//   handler(8B) | imm0(8B) | imm1(8B) | imm2(8B)
//
// All semantic encoding (slot indices, branch targets, etc.) is defined
// by pattern-specific decode macros. Branch targets are stored in imm0
// as pointers when needed.
struct Instruction {
    OpHandler handler;   // 8B - handler function pointer
    uint64_t  imm0;      // 8B - primary immediate / branch target pointer
    uint64_t  imm1;      // 8B - secondary immediate
    uint64_t  imm2;      // 8B - extra payload
};

// Handler parameters (preserve_none calling convention)
// ctx: opaque pointer to Rust Context (contains mem0_base/mem0_size)
// pc: current instruction pointer
// fp: pointer to current frame's locals (params, then locals)
// t0-t3: TOS (Top-of-Stack) registers passed by value (4 TOS registers)
// nh: preloaded next handler (NextHandler)
#define PARAMS struct Ctx* ctx, struct Instruction* pc, uint64_t* fp, uint64_t l0, uint64_t l1, uint64_t l2, uint64_t t0, uint64_t t1, uint64_t t2, uint64_t t3, NextHandler nh
#define ARGS ctx, pc, fp, l0, l1, l2, t0, t1, t2, t3, nh

// Entry: starts execution at pc using its handler. Pointers are updated in place.
// ctx is opaque to C and passed through unchanged.
void run_trampoline(PARAMS);

// Trap delegation function for C handlers
// Pass a null-terminated error message string
// Returns TERM_INST pointer (never NULL)
extern struct Instruction* fast_c_trap(struct Ctx* ctx, const char* message);

// Helper to get next instruction (fallthrough)
#define pc_next(pc) ((pc) + 1)

// Helper macros for reading raw instruction immediates
#define pc_imm0(pc) ((pc)->imm0)
#define pc_imm1(pc) ((pc)->imm1)
#define pc_imm2(pc) ((pc)->imm2)

// =============================================================================
// Pattern-specific decoding macros
// Generated from encoding.toml - see fast_encoding.h for all patterns
// =============================================================================

// Branch target (stored as pointer in imm0)
#define pc_branch_target(pc) ((struct Instruction*)((pc)->imm0))

// Backward compatibility alias (will be removed after full migration)
#define pc_alt(pc) pc_branch_target(pc)

// =============================================================================
// Compatibility macros for binop/unop patterns (slots in imm0)
// Most C handlers use these patterns. See fast_encoding.h for pattern-specific macros.
// =============================================================================
#define pc_slot_dest(pc)  ((uint16_t)(pc)->imm0)
#define pc_slot_a(pc)     ((uint16_t)((pc)->imm0 >> 16))
#define pc_slot_b(pc)     ((uint16_t)((pc)->imm0 >> 32))
#define pc_slot_extra(pc) ((uint16_t)((pc)->imm0 >> 48))

// Frame access helpers
#define frame_read(fp, slot) ((fp)[slot])
#define frame_write(fp, slot, val) ((fp)[slot] = (val))

// =============================================================================
// Shared type conversion helpers
// =============================================================================

static inline float u32_to_f32(uint32_t bits) {
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

static inline uint32_t f32_to_u32(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(uint32_t));
    return bits;
}

static inline double u64_to_f64(uint64_t bits) {
    double d;
    __builtin_memcpy(&d, &bits, sizeof(double));
    return d;
}

static inline uint64_t f64_to_u64(double d) {
    uint64_t bits;
    __builtin_memcpy(&bits, &d, sizeof(uint64_t));
    return bits;
}

#ifdef __cplusplus
}
#endif
