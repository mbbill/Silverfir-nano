//! Table operations (Phase 3 TOS-only).
//!
//! Handlers for: table.get, table.set, table.size, table.grow, table.fill, table.init, table.copy, elem.drop
//!
//! Phase 3 TOS-only:
//! - table_get (pop1_push1): Read index from p_src, write ref to p_dst
//! - table_set (pop2_push0): Read index from p_addr, ref from p_val
//! - table_size (pop0_push1): Write size to p_dst
//! - table_grow (pop2_push1): Read init_ref from p_lhs, delta from p_rhs, write old size to p_dst
//! - table_fill (pop3_push0): Read start from p_a, val from p_b, size from p_c
//! - table_init (pop3_push0): Read dst from p_a, src from p_b, size from p_c
//! - table_copy (pop3_push0): Read dst from p_a, src from p_b, size from p_c
//! - elem_drop: tos_pattern = "none" - no TOS interaction

use super::common::*;
use super::trap_with;
use crate::vm::interp::fast::encoding::{drop_op, table_get, table_grow, table_set, table_size, ternary};
use crate::error::WasmError;

// =============================================================================
// Table Get / Set (Phase 3 TOS-only)
// =============================================================================

/// table.get: read index from TOS, write ref to TOS (1→1)
/// Encoding: see encoding.toml "table_op" pattern
/// tos_pattern = { pop = 1, push = 1 } - uses p_src for input, p_dst for output
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_get(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers
    p_src: *mut u64,
    p_dst: *mut u64,
) -> *mut Instruction {
    let table_idx = table_get::decode_table_idx(pc) as usize;

    let store_ref = ctx_store(ctx);
    let table = store_ref.table(table_idx);
    // Read index from TOS
    let elem_index = unsafe { *p_src } as usize;
    if elem_index >= table.elements.len() {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }
    let r = table.elements[elem_index];
    let raw: usize = r.into();

    // Write result to TOS
    unsafe { *p_dst = raw as u64 };

    pc_fallthrough(pc)
}

/// table.set: read index and ref from TOS (2→0)
/// Encoding: see encoding.toml "table_set" pattern
/// tos_pattern = { pop = 2, push = 0 } - uses p_addr (index), p_val (ref)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_set(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos2=index, pos1=ref)
    p_addr: *mut u64,
    p_val: *mut u64,
) -> *mut Instruction {
    let table_idx = table_set::decode_table_idx(pc) as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let table = store_mut_ref.table_mut(table_idx);
    // Read from TOS: p_val=ref (pos1), p_addr=index (pos2)
    let func_ref_raw = unsafe { *p_val } as usize;
    let elem_index = unsafe { *p_addr } as usize;
    if elem_index >= table.elements.len() {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }
    table.elements[elem_index] = VmRefHandle::new(func_ref_raw);
    pc_fallthrough(pc)
}

// =============================================================================
// Table Size / Grow (Phase 3 TOS-only)
// =============================================================================

/// table.size: write size to TOS (0→1)
/// Encoding: see encoding.toml "table_size" pattern
/// tos_pattern = { pop = 0, push = 1 } - uses p_dst for output
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_size(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for output
    p_dst: *mut u64,
) -> *mut Instruction {
    let table_idx = table_size::decode_table_idx(pc) as usize;

    let store_ref = ctx_store(ctx);
    let table = store_ref.table(table_idx);
    let size = table.size() as u64;

    // Write result to TOS
    unsafe { *p_dst = size };

    pc_fallthrough(pc)
}

/// table.grow: read init_ref and delta from TOS, write old size (2→1)
/// Encoding: see encoding.toml "table_grow" pattern
/// tos_pattern = { pop = 2, push = 1 } - uses p_lhs (init_ref), p_rhs (delta), p_dst (output)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_grow(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos2=init_ref, pos1=delta)
    p_lhs: *mut u64,
    p_rhs: *mut u64,
    p_dst: *mut u64,
) -> *mut Instruction {
    let table_idx = table_grow::decode_table_idx(pc) as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let table = store_mut_ref.table_mut(table_idx);
    // Read from TOS: p_rhs=delta (pos1), p_lhs=init_ref (pos2)
    let size = unsafe { *p_rhs } as usize;
    let val = VmRefHandle::new(unsafe { *p_lhs } as usize);
    let new_size = table.elements.len().checked_add(size).unwrap_or(usize::MAX);
    if new_size > table.limits.get_max() {
        // Return -1 (as u32::MAX)
        unsafe { *p_dst = u32::MAX as u64 };
        return pc_fallthrough(pc);
    }
    let len = table.elements.len();
    table.elements.resize_with(len + size, || val);

    // Write old size to TOS
    unsafe { *p_dst = len as u64 };

    pc_fallthrough(pc)
}

// =============================================================================
// Table Fill / Init / Copy / Elem Drop (Phase 3 TOS-only)
// =============================================================================

/// table.fill: read start, val, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - uses p_a (start), p_b (val), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_fill(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos3=start, pos2=val, pos1=size)
    p_a: *mut u64,
    p_b: *mut u64,
    p_c: *mut u64,
) -> *mut Instruction {
    let table_idx = ternary::decode_imm0(pc) as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let table = store_mut_ref.table_mut(table_idx);
    // Read from TOS: p_c=size (pos1), p_b=val (pos2), p_a=start (pos3)
    let size = unsafe { *p_c } as usize;
    let val = VmRefHandle::new(unsafe { *p_b } as usize);
    let start = unsafe { *p_a } as usize;
    if start.saturating_add(size) > table.elements.len() {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }
    table.elements[start..start + size].fill(val);
    pc_fallthrough(pc)
}

/// table.init: read dst, src, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - uses p_a (dst), p_b (src), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_init(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos3=dst, pos2=src, pos1=size)
    p_a: *mut u64,
    p_b: *mut u64,
    p_c: *mut u64,
) -> *mut Instruction {
    let table_idx = ternary::decode_imm0(pc) as usize;
    let elem_idx = ternary::decode_imm1(pc) as usize;

    // Read from TOS: p_c=size (pos1), p_b=src (pos2), p_a=dst (pos3)
    let size = unsafe { *p_c } as usize;
    let src = unsafe { *p_b } as usize;
    let dst = unsafe { *p_a } as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let module = store_mut_ref.module_mut();

    if elem_idx >= module.elements.len() {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }

    let elem_inst = &module.elements[elem_idx];

    // Zero-length init: perform bounds checks but do not trap on dropped segments
    if size == 0 {
        if src > elem_inst.refs.len() || dst > module.tables[table_idx].elements.len() {
            return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
        }
        return pc_fallthrough(pc);
    }
    if src.saturating_add(size) > elem_inst.refs.len()
        || dst.saturating_add(size) > module.tables[table_idx].elements.len()
    {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }
    if elem_inst.is_dropped() {
        return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
    }

    // Copy element refs into table
    for i in 0..size {
        module.tables[table_idx].elements[dst + i] = module.elements[elem_idx].refs[src + i];
    }

    pc_fallthrough(pc)
}

/// table.copy: read dst, src, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - uses p_a (dst), p_b (src), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_table_copy(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos3=dst, pos2=src, pos1=size)
    p_a: *mut u64,
    p_b: *mut u64,
    p_c: *mut u64,
) -> *mut Instruction {
    let dstidx = ternary::decode_imm0(pc) as usize;
    let srcidx = ternary::decode_imm1(pc) as usize;

    // Read from TOS: p_c=size (pos1), p_b=src (pos2), p_a=dst (pos3)
    let size = unsafe { *p_c } as usize;
    let src = unsafe { *p_b } as usize;
    let dst = unsafe { *p_a } as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable

    if dstidx == srcidx {
        let table = store_mut_ref.table_mut(dstidx);
        if src.saturating_add(size) > table.elements.len()
            || dst.saturating_add(size) > table.elements.len()
        {
            return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
        }
        table.elements.copy_within(src..src + size, dst);
    } else {
        let module = store_mut_ref.module_mut();
        let (src_table, dst_table) = if srcidx < dstidx {
            let (left, right) = module.tables.split_at_mut(dstidx);
            (&left[srcidx], &mut right[0])
        } else {
            let (left, right) = module.tables.split_at_mut(srcidx);
            (&right[0] as &TableInst, &mut left[dstidx])
        };
        if src.saturating_add(size) > src_table.elements.len()
            || dst.saturating_add(size) > dst_table.elements.len()
        {
            return trap_with(ctx, WasmError::trap("out of bounds table access".into()));
        }
        dst_table.elements[dst..dst + size].copy_from_slice(&src_table.elements[src..src + size]);
    }
    pc_fallthrough(pc)
}

/// elem.drop: no stack effect (0→0)
/// Encoding: see encoding.toml "drop_op" pattern
/// tos_pattern = "none" - no TOS interaction
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_elem_drop(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
) -> *mut Instruction {
    let elem_idx = drop_op::decode_idx(pc) as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let module = store_mut_ref.module_mut();
    if elem_idx < module.elements.len() {
        module.elements[elem_idx].drop_segment();
    }
    pc_fallthrough(pc)
}
