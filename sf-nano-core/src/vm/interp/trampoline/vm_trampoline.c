// =============================================================================
// Fast Interpreter Trampoline
// =============================================================================
//
// This file implements the tail-call dispatch loop for the fast interpreter.
// It uses the preserve_none calling convention for efficient register usage.
//
// Architecture:
//   - Wrapper layer (op_*): Handler entry points with params passed by value
//   - Implementation layer (impl_*): Core logic with mutable refs via pointers
//
// Dispatch flow:
//   op_name(PARAMS) → impl_name(IMPL_ARGS_*) → np = next pc → np->handler(ARGS_NEXT)
//
// =============================================================================

#include "vm_trampoline.h"
#include "fast_encoding.h"
#include "../frame_layout.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Debug Instrumentation
// =============================================================================

// Trace hook - enabled via FAST_TRACE_ENABLED
// Note: fast_trace_instruction receives PARAMS (including nh) but ignores nh.
#if defined(FAST_TRACE_ENABLED)
void fast_trace_instruction(const char* name, PARAMS) {
    (void)ctx; (void)pc; (void)fp; (void)l0; (void)l1; (void)l2;
    (void)t0; (void)t1; (void)t2; (void)t3;
    (void)nh;
    fprintf(stderr, "%s\n", name);
}
#define FAST_TRACE_HOOK(op_name) \
    do { fast_trace_instruction(#op_name, ARGS); } while (0)
#else
#define FAST_TRACE_HOOK(op_name) do { } while (0)
#endif

// Profile hook - enabled via FAST_PROFILE_ENABLED
#if defined(FAST_PROFILE_ENABLED)
extern void fast_profile_record(const char* name);
#define FAST_PROFILE_HOOK(op_name) \
    do { fast_profile_record(#op_name); } while (0)

#else
#define FAST_PROFILE_HOOK(op_name) do { } while (0)
#endif

// =============================================================================
// Implementation Parameter Macros
// =============================================================================
//
// These macros define parameter lists for impl_* functions. The impl layer
// uses a double-pointer for fp (so it can be modified) and single
// pointers for TOS registers (so values can be updated).
// Memory 0 is accessed via ctx (ctx_mem0_base/ctx_mem0_size macros).
//
// Hierarchy:
//   IMPL_PARAMS_BASE         - 3 base params (ctx, pc, pfp)
//   IMPL_PARAMS_NONE         - same as BASE (for tos_pattern = "none")
//   IMPL_PARAMS_POP2_PUSH1   - BASE + 3 operand pointers (binary ops)
//   IMPL_PARAMS_POP1_PUSH1   - BASE + 2 operand pointers (unary ops)
//   IMPL_PARAMS_POP0_PUSH1   - BASE + 1 operand pointer  (push ops)
//   IMPL_PARAMS_POP2_PUSH0   - BASE + 2 operand pointers (store ops)
//   IMPL_PARAMS_POP1_PUSH0   - BASE + 1 operand pointer  (pop ops)
//   IMPL_PARAMS_POP3_PUSH1   - BASE + 4 operand pointers (select)
//
// =============================================================================

// Base parameters for all impl_* functions
// - ctx: opaque context pointer (contains mem0_base/mem0_size; passed to Rust for traps, calls, etc.)
// - pc: current instruction pointer
// - pfp: double-pointer to frame pointer (impl can modify fp)
#define IMPL_PARAMS_BASE \
    struct Ctx* ctx, struct Instruction* pc, \
    uint64_t** pfp, __attribute__((unused)) uint64_t* p_l0, \
    __attribute__((unused)) uint64_t* p_l1, \
    __attribute__((unused)) uint64_t* p_l2

#define IMPL_ARGS_BASE ctx, pc, &fp, &l0, &l1, &l2

// -----------------------------------------------------------------------------
// TOS Pattern Parameter Extensions
// Each pattern adds operand pointers for the TOS registers involved.
// Position P at stack depth D uses register t[(D - P) % 4].
// -----------------------------------------------------------------------------

// Pattern: none - no TOS interaction (control flow, structural ops)
#define IMPL_PARAMS_NONE IMPL_PARAMS_BASE
#define IMPL_ARGS_NONE   IMPL_ARGS_BASE

// Pattern: pop2_push1 - binary ops (add, sub, mul, div, etc.)
// Operands: lhs (pos2), rhs (pos1), dst (pos2, overwrites lhs)
#define IMPL_PARAMS_POP2_PUSH1 IMPL_PARAMS_BASE, uint64_t* p_lhs, uint64_t* p_rhs, uint64_t* p_dst
#define VOID_POP2_PUSH1 (void)p_lhs; (void)p_rhs; (void)p_dst

// Pattern: pop1_push1 - unary ops (clz, ctz, loads, conversions)
// Operands: src (pos1), dst (pos1, same register for in-place)
#define IMPL_PARAMS_POP1_PUSH1 IMPL_PARAMS_BASE, uint64_t* p_src, uint64_t* p_dst
#define VOID_POP1_PUSH1 (void)p_src; (void)p_dst

// Pattern: pop0_push1 - push ops (const, local_get, global_get)
// Operands: dst (new top after push)
#define IMPL_PARAMS_POP0_PUSH1 IMPL_PARAMS_BASE, uint64_t* p_dst
#define VOID_POP0_PUSH1 (void)p_dst

// Pattern: pop2_push0 - store ops (i32.store, i64.store, etc.)
// Operands: addr (pos2), val (pos1)
#define IMPL_PARAMS_POP2_PUSH0 IMPL_PARAMS_BASE, uint64_t* p_addr, uint64_t* p_val
#define VOID_POP2_PUSH0 (void)p_addr; (void)p_val

// Pattern: pop1_push0 - pop ops (local_set, drop)
// Operands: src (pos1)
#define IMPL_PARAMS_POP1_PUSH0 IMPL_PARAMS_BASE, uint64_t* p_src
#define VOID_POP1_PUSH0 (void)p_src

// Pattern: pop3_push1 - select op
// Operands: val1 (pos3), val2 (pos2), cond (pos1), dst (pos3)
#define IMPL_PARAMS_POP3_PUSH1 IMPL_PARAMS_BASE, uint64_t* p_val1, uint64_t* p_val2, uint64_t* p_cond, uint64_t* p_dst
#define VOID_POP3_PUSH1 (void)p_val1; (void)p_val2; (void)p_cond; (void)p_dst

// Pattern: pop1 - alias for pop1_push0 (spill_1)
#define IMPL_PARAMS_POP1 IMPL_PARAMS_POP1_PUSH0

// Pattern: pop2 - alias for pop2_push0 with different naming (spill_2)
// Operands: lhs (pos2), rhs (pos1)
#define IMPL_PARAMS_POP2 IMPL_PARAMS_BASE, uint64_t* p_lhs, uint64_t* p_rhs

// Pattern: pop3 - for spill_3
// Operands: a (pos3), b (pos2), c (pos1)
#define IMPL_PARAMS_POP3 IMPL_PARAMS_BASE, uint64_t* p_a, uint64_t* p_b, uint64_t* p_c

// Pattern: pop4 - for spill_4
// Operands: a (pos4), b (pos3), c (pos2), d (pos1)
#define IMPL_PARAMS_POP4 IMPL_PARAMS_BASE, uint64_t* p_a, uint64_t* p_b, uint64_t* p_c, uint64_t* p_d

// Pattern: pop0_push2 - for fill_2
// Operands: dst0 (pos2), dst1 (pos1)
#define IMPL_PARAMS_POP0_PUSH2 IMPL_PARAMS_BASE, uint64_t* p_dst0, uint64_t* p_dst1

// Pattern: pop0_push3 - for fill_3
// Operands: dst0 (pos3), dst1 (pos2), dst2 (pos1)
#define IMPL_PARAMS_POP0_PUSH3 IMPL_PARAMS_BASE, uint64_t* p_dst0, uint64_t* p_dst1, uint64_t* p_dst2

// Pattern: pop0_push4 - for fill_4
// Operands: dst0 (pos4), dst1 (pos3), dst2 (pos2), dst3 (pos1)
#define IMPL_PARAMS_POP0_PUSH4 IMPL_PARAMS_BASE, uint64_t* p_dst0, uint64_t* p_dst1, uint64_t* p_dst2, uint64_t* p_dst3

// Pattern: pop1_push2 - for fused ops like tee_xor_get
// Operands: src (pos2), dst0 (pos2), dst1 (pos1)
// Note: src and dst0 share pos2 since we read first then write
#define IMPL_PARAMS_POP1_PUSH2 IMPL_PARAMS_BASE, uint64_t* p_src, uint64_t* p_dst0, uint64_t* p_dst1

// Pattern: pop2_push2 - for fused ops like add_tee_get, add_set_get_const
// Operands: lhs (pos2), rhs (pos1), dst0 (pos2), dst1 (pos1)
// Note: positions overlap since we read first then write
#define IMPL_PARAMS_POP2_PUSH2 IMPL_PARAMS_BASE, uint64_t* p_lhs, uint64_t* p_rhs, uint64_t* p_dst0, uint64_t* p_dst1

// =============================================================================
// Tail-Call Dispatch Macros
// =============================================================================

// ARGS_NEXT: Arguments for tail-call to next handler (guard-check linear path)
// Like ARGS but with pc replaced by np and nh replaced by new_nh.
// The dispatch target is the preloaded nh (from previous handler).
// Requires: np, new_nh in scope.
#define ARGS_NEXT ctx, np, fp, l0, l1, l2, t0, t1, t2, t3, new_nh

// ARGS_NEXT_RELOAD: Arguments for tail-call after reload (nonlinear/trap path)
// Like ARGS_NEXT but used when dispatching to np->handler (freshly loaded).
// Requires: np, new_nh in scope.
#define ARGS_NEXT_RELOAD ARGS_NEXT

// =============================================================================
// Terminal Handler
// =============================================================================

// op_term: Breaks the tail-call chain and returns to trampoline caller.
// All frame cleanup is handled by impl_return; op_term just terminates.
// Error state (if any) is stored in ctx->error.
PRESERVE_NONE
void op_term(PARAMS)
{
    (void)ctx; (void)pc; (void)fp; (void)l0; (void)l1; (void)l2;
    (void)t0; (void)t1; (void)t2; (void)t3;
    (void)nh;
    return;
}

// =============================================================================
// Trampoline Entry Point
// =============================================================================

// Entry point: starts execution at pc using its handler.
// Called from Rust with initial register values.
// The nh parameter is the preloaded handler for pc+1, computed by the caller.
void run_trampoline(PARAMS)
{
    OpHandler entry = pc->handler;
    entry(ARGS);
}

// =============================================================================
// C Handler Implementations
// =============================================================================
//
// These files contain FORCE_INLINE impl_* functions.
// Must be included BEFORE fast_c_wrappers.inc so they can be inlined.
//

#include "../handlers_c/semantics.h"
#include "../handlers_c/spill_fill.c"
#include "../handlers_c/const_local.c"
#include "../handlers_c/arithmetic.c"
#include "../handlers_c/bitwise.c"
#include "../handlers_c/comparison.c"
#include "../handlers_c/unary.c"
#include "../handlers_c/float_ops.c"
#include "../handlers_c/conversion.c"
#include "../handlers_c/memory.c"
#include "../handlers_c/control.c"
#include "../handlers_c/call.c"
#ifdef FUSION_ENABLED
#include "fast_interp/fast_fused_handlers.inc"
#endif

// =============================================================================
// Generated Wrappers
// =============================================================================
//
// Generated by build/fast_interp/gen_c_wrappers.rs from handlers.toml.
// Contains op_* wrapper functions for all handlers.
//

#include "fast_interp/fast_c_wrappers.inc"
