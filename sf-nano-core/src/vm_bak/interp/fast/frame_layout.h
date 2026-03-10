// =============================================================================
// Frame Layout - Shared between Rust and C
// =============================================================================
//
// This header defines the frame layout for the fast interpreter.
// Keep in sync with frame_layout.rs!
//
// Frame structure (addresses grow upward):
//
//   fp[frame_size + 2]            →  [saved_module]    ← metadata (fixed offset)
//   fp[frame_size + 1]            →  [saved_fp]        ← metadata (fixed offset)
//   fp[frame_size]                →  [return_pc]       ← metadata (fixed offset)
//   fp[frame_size - 1]            →  [local N-1]
//   ...                           →  [locals...]
//   fp[params_count]              →  [local 0]
//   ...                           →  [params...]
//   fp[0]                         →  [param 0]         ← frame pointer
//   fp[frame_size + 3]            →  [operand[0]]      ← operand stack base
//   fp[frame_size + 3 + h-1]      →  [operand[h-1]]    ← top at height h
//
// Key insight: metadata is at FIXED offsets (frame_size, frame_size+1, frame_size+2),
// operand stack starts AFTER metadata at frame_size + 3.
//
// TOS Register Assignment (cyclic):
//   - 4 registers: t0, t1, t2, t3
//   - Position P (1=top) at height H uses register: t[(H - P) % 4]
//   - Example at height 5: top=t0, pos2=t3, pos3=t2, pos4=t1
//
// =============================================================================

#pragma once
#include <stdint.h>

// Metadata slots between frame and operand stack (return_pc + saved_fp + saved_module)
#define FRAME_METADATA_SLOTS 3

// Calculate operand stack base as slot index from fp
// operand[0] is at fp[frame_size + FRAME_METADATA_SLOTS]
#define OPERAND_STACK_BASE(frame_size) ((frame_size) + FRAME_METADATA_SLOTS)

// Calculate operand base offset from fp (in bytes)
#define OPERAND_BASE_OFFSET(frame_size) (OPERAND_STACK_BASE(frame_size) * 8)

// Calculate slot address in operand stack
#define OPERAND_SLOT(fp, operand_base_offset, index) \
    ((uint64_t*)((uint8_t*)(fp) + (operand_base_offset) + (index) * 8))

// Slot index for return_pc (relative to fp)
#define RETURN_PC_SLOT(frame_size) (frame_size)

// Slot index for saved_fp (relative to fp)
#define SAVED_FP_SLOT(frame_size) ((frame_size) + 1)

// Slot index for saved_module (relative to fp)
#define SAVED_MODULE_SLOT(frame_size) ((frame_size) + 2)

// TOS_REGISTER_MASK = TOS_REGISTER_COUNT - 1
// When TOS_REGISTER_COUNT = 4, mask = 3
// This should match build/fast_interp/tos_config.rs:TOS_REGISTER_COUNT
#ifndef TOS_REGISTER_MASK
#define TOS_REGISTER_MASK 3
#endif

// Calculate which TOS register holds position P at stack height H
// Position 1 = top of stack, Position 2 = second from top, etc.
#define TOS_REGISTER(height, position) (((height) - (position)) & TOS_REGISTER_MASK)

// Access TOS register by index (0-3)
#define TOS_REG(regs, idx) ((regs)[(idx)])
