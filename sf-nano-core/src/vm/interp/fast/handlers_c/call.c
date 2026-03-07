// Fast interpreter C handler implementations - Call operations
// These are FORCE_INLINE impl_* functions that get inlined into the generated wrappers.
//
// UNIFIED STACK DESIGN:
//   - No native stack frames per call (no run_trampoline per call)
//   - Call metadata (return_pc, saved_fp, saved_module) stored on value stack
//   - Calls are tail-call jumps, returns restore from value stack
//
// Stack layout after call:
//   [callee operands...]  <- conceptual sp (tracked via height)
//   [saved_module]        <- fp[frame_size + 2]
//   [saved_fp]            <- fp[frame_size + 1]
//   [return_pc]           <- fp[frame_size]
//   [locals...]           <- fp[params_count..frame_size-1]
//   [params...]           <- fp[0..params_count-1] (args become params in-place)
//   [caller operands...]
//
// Phase 3: TOS-Only Computation
// All handlers compute using TOS registers only. SP is not used.
// Stack access uses fp-relative addressing with operand_base_offset.
//
// This file is #included in vm_trampoline.c before fast_c_wrappers.inc.

#include <stdint.h>
#include <string.h>

// External Rust helpers for slow paths
extern void fast_return_cross_module(struct Ctx* ctx, uint64_t saved_module_ptr);

// Dereference the double pointer for direct access
#define fp (*pfp)

// Operand stack access macros (fp-relative)
#define OPERAND_BASE(operand_base_offset) ((uint64_t*)((uint8_t*)fp + (operand_base_offset)))

// =============================================================================
// impl_call_local - Unified stack call (no run_trampoline!)
// =============================================================================

// Encoding: entry(64), params_count(16), locals_count(16), operand_base_offset(32), height(16)
// Args are at operand_base[height-params_count..height), become params in-place
FORCE_INLINE struct Instruction* impl_call_local(IMPL_PARAMS_NONE) {
    struct Instruction* entry = (struct Instruction*)call_local_decode_entry(pc);
    uint16_t params_count = call_local_decode_params_count(pc);
    uint16_t locals_count = call_local_decode_locals_count(pc);
    uint32_t operand_base_offset = call_local_decode_operand_base_offset(pc);
    uint16_t height = call_local_decode_height(pc);
    uint16_t frame_size = params_count + locals_count;

    // Spill l0/l1/l2 before frame setup
    fp[0] = *p_l0;
    fp[1] = *p_l1;
    fp[2] = *p_l2;

    // Compute operand base for current frame
    uint64_t* operand_base = OPERAND_BASE(operand_base_offset);

    // Args are at operand_base[height-params_count..height)
    // Callee fp = operand_base + (height - params_count)
    uint64_t* callee_fp = operand_base + (height - params_count);

    // Stack overflow check: need space for locals + metadata slots
    uint64_t* new_stack_top = callee_fp + frame_size + FRAME_METADATA_SLOTS;
    if (unlikely(new_stack_top > ctx_stack_end(ctx))) {
        return c_trap(ctx, "stack overflow");
    }

    // Call depth check
    if (unlikely(ctx_call_depth(ctx) >= MAX_CALL_DEPTH)) {
        return c_trap(ctx, "call stack exhausted");
    }
    ctx_call_depth(ctx)++;

    // Zero callee's locals.
    //
    // Spell this as an explicit pair-store friendly loop instead of a generic
    // memset pattern: on AArch64 clang reliably lowers the 2-at-a-time body to
    // `stp xzr, xzr`, which halves the store count in local-heavy callers like
    // c-ray without turning the handler into a non-leaf `_bzero` call.
    uint64_t* local_dst = callee_fp + params_count;
    uint16_t remaining = locals_count;
    while (remaining >= 2) {
        local_dst[0] = 0;
        local_dst[1] = 0;
        local_dst += 2;
        remaining -= 2;
    }
    if (remaining != 0) {
        local_dst[0] = 0;
    }

    // Push metadata AFTER frame (at callee_fp[frame_size], [frame_size+1], [frame_size+2])
    callee_fp[frame_size]     = (uint64_t)pc_next(pc);  // return_pc
    callee_fp[frame_size + 1] = (uint64_t)fp;           // saved_fp
    callee_fp[frame_size + 2] = 0;                      // saved_module (same-module)

    // Update fp
    fp = callee_fp;

    // Tail-call to callee entry (no run_trampoline!)
    return entry;
}

// =============================================================================
// Return: shared epilogue (restore fp, cross-module check, dispatch)
// =============================================================================

FORCE_INLINE struct Instruction* return_epilogue(
    struct Ctx* ctx, uint64_t** pfp, uint64_t* p_l0, uint64_t* p_l1, uint64_t* p_l2,
    struct Instruction* return_pc, uint64_t* saved_fp, uint64_t saved_module
) {
    // Decrement call depth
    ctx_call_depth(ctx)--;

    // Check for entry frame sentinel (saved_fp == NULL)
    if (unlikely(saved_fp == NULL)) {
        return ctx_term_inst(ctx);
    }

    // Restore fp
    *pfp = saved_fp;

    // Fill l0/l1/l2 from restored caller frame
    *p_l0 = (*pfp)[0];
    *p_l1 = (*pfp)[1];
    *p_l2 = (*pfp)[2];

    // Cross-module return: restore caller's module context
    if (unlikely(saved_module != 0)) {
        fast_return_cross_module(ctx, saved_module);
    }

    return return_pc;
}

// =============================================================================
// impl_return - General case (arity >= 2)
// =============================================================================

FORCE_INLINE struct Instruction* impl_return(IMPL_PARAMS_NONE) {
    uint16_t arity = return_decode_arity(pc);
    uint16_t frame_size = return_decode_frame_size(pc);
    uint32_t operand_base_offset = return_decode_operand_base_offset(pc);
    uint16_t height = return_decode_height(pc);

    uint64_t* operand_base = OPERAND_BASE(operand_base_offset);

    struct Instruction* return_pc = (struct Instruction*)fp[frame_size];
    uint64_t* saved_fp = (uint64_t*)fp[frame_size + 1];
    uint64_t saved_module = fp[frame_size + 2];

    for (uint16_t i = 0; i < arity; i++) {
        fp[i] = operand_base[height - arity + i];
    }

    return return_epilogue(ctx, pfp, p_l0, p_l1, p_l2, return_pc, saved_fp, saved_module);
}

// =============================================================================
// impl_return_void - Specialized: no return values (arity == 0)
// =============================================================================

FORCE_INLINE struct Instruction* impl_return_void(IMPL_PARAMS_NONE) {
    uint16_t frame_size = return_void_decode_frame_size(pc);

    struct Instruction* return_pc = (struct Instruction*)fp[frame_size];
    uint64_t* saved_fp = (uint64_t*)fp[frame_size + 1];
    uint64_t saved_module = fp[frame_size + 2];

    return return_epilogue(ctx, pfp, p_l0, p_l1, p_l2, return_pc, saved_fp, saved_module);
}

// =============================================================================
// impl_return_one - Specialized: single return value (arity == 1)
// =============================================================================

FORCE_INLINE struct Instruction* impl_return_one(IMPL_PARAMS_NONE) {
    uint16_t frame_size = return_one_decode_frame_size(pc);
    uint32_t operand_base_offset = return_one_decode_operand_base_offset(pc);
    uint16_t height = return_one_decode_height(pc);

    uint64_t* operand_base = OPERAND_BASE(operand_base_offset);

    struct Instruction* return_pc = (struct Instruction*)fp[frame_size];
    uint64_t* saved_fp = (uint64_t*)fp[frame_size + 1];
    uint64_t saved_module = fp[frame_size + 2];

    // Single result: fp[0] = operand_base[height - 1]
    fp[0] = operand_base[height - 1];

    return return_epilogue(ctx, pfp, p_l0, p_l1, p_l2, return_pc, saved_fp, saved_module);
}

#undef fp
#undef OPERAND_BASE
