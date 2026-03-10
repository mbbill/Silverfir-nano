//! Memory operations.
//!
//! Handlers for: memory.size, memory.grow, memory.init, memory.copy, memory.fill, data.drop
//!
//! NOTE: Load/store operations are implemented in handlers_c/memory.c for performance.
//!
//! Phase 3 TOS-only:
//! - memory_size (pop0_push1): Write result to p_dst
//! - memory_grow (pop1_push1): Read from p_src, write result to p_dst
//! - memory_init/copy/fill (pop3_push0): Read from p_a, p_b, p_c

use super::common::*;
use super::trap_with;
use crate::vm::interp::fast::encoding::{drop_op, load, memory_grow, memory_size, store, ternary};
use crate::error::WasmError;

// =============================================================================
// Memory Size / Grow (Phase 3 TOS-only)
// =============================================================================

/// memory.size: write page count to TOS (0 → 1)
/// Encoding: see encoding.toml "memory_size" pattern
/// tos_pattern = { pop = 0, push = 1 } - writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_memory_size(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointer for output
    p_dst: *mut u64,
) -> *mut Instruction {
    let mem_idx = memory_size::decode_mem_idx(pc) as usize;

    // Output: pages
    let pages = if mem_idx == 0 {
        let (_, heap_len) = heap_info(ctx);
        (heap_len / crate::constants::WASM_PAGE_SIZE) as u64
    } else {
        let store_ref = ctx_store(ctx);
        let mem = store_ref.memory(mem_idx);
        mem.current_pages() as u64
    };

    // Write result to TOS
    unsafe { *p_dst = pages };

    pc_fallthrough(pc)
}

/// memory.grow: read delta from TOS, grow memory, write result (1 → 1)
/// Encoding: see encoding.toml "memory_grow" pattern
/// tos_pattern = { pop = 1, push = 1 } - reads from p_src, writes to p_dst
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_memory_grow(
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
    let mem_idx = memory_grow::decode_mem_idx(pc) as usize;

    // Read delta from TOS
    let delta_pages_raw = unsafe { *p_src };

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let mem = store_mut_ref.memory_mut(mem_idx);
    let is_64 = mem.limits.is64;
    let error_value = if is_64 { u64::MAX } else { u32::MAX as u64 };

    let delta_pages = if is_64 {
        delta_pages_raw as i64 as usize
    } else {
        let d = delta_pages_raw as i32;
        if d < 0 {
            // Write error to TOS
            unsafe { *p_dst = error_value };
            return pc_fallthrough(pc);
        }
        d as usize
    };

    let old_pages = mem.current_pages();
    let new_pages = old_pages + delta_pages;
    let max_pages = mem.limits.get_max();
    let result = if new_pages > max_pages {
        error_value
    } else {
        let new_len_bytes = new_pages * crate::constants::WASM_PAGE_SIZE;
        mem.data.resize(new_len_bytes, 0);
        if mem_idx == 0 {
            let p = mem.data.as_mut_ptr();
            let len = mem.data.len() as u64;
            write_mem0(ctx, p, len);
        }
        old_pages as u64
    };

    // Write result to TOS
    unsafe { *p_dst = result };

    pc_fallthrough(pc)
}

// =============================================================================
// Bulk Memory Operations (Phase 3 TOS-only)
// =============================================================================

/// memory.init: read dst, src, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - reads from p_a (dst), p_b (src), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_memory_init(
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
    let mem_idx = ternary::decode_imm0(pc) as usize;
    let data_idx = ternary::decode_imm1(pc) as usize;

    // Read from TOS: p_c=size (pos1), p_b=src (pos2), p_a=dst (pos3)
    let size = unsafe { *p_c } as usize;
    let src = unsafe { *p_b } as usize;
    let dst = unsafe { *p_a } as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    let module = store_mut_ref.module_mut();

    if data_idx >= module.data.len() {
        return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
    }

    let data_bytes = &module.data[data_idx].bytes;
    let data_dropped = module.data[data_idx].is_dropped();

    if size == 0 {
        if src > data_bytes.len() || dst > module.memories[mem_idx].data.len() {
            return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
        }
        return pc_fallthrough(pc);
    }
    if data_dropped {
        return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
    }
    if src.saturating_add(size) > data_bytes.len() {
        return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
    }

    // Use raw pointers to avoid borrow conflict
    let src_ptr = data_bytes[src..].as_ptr();
    let mem_data = &mut module.memories[mem_idx].data;
    if dst.saturating_add(size) > mem_data.len() {
        return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
    }
    unsafe {
        core::ptr::copy_nonoverlapping(src_ptr, mem_data[dst..].as_mut_ptr(), size);
    }

    pc_fallthrough(pc)
}

/// Encoding: see encoding.toml "drop_op" pattern
/// tos_pattern = "none" - no TOS interaction
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_data_drop(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
) -> *mut Instruction {
    let data_idx = drop_op::decode_idx(pc) as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let module = store_mut_ref.module_mut();
    if data_idx < module.data.len() {
        module.data[data_idx].drop_segment();
    }
    pc_fallthrough(pc)
}

/// memory.copy: read dst, src, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - reads from p_a (dst), p_b (src), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_memory_copy(
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
    let dst_idx = ternary::decode_imm0(pc) as usize;
    let src_idx = ternary::decode_imm1(pc) as usize;

    // Read from TOS: p_c=size (pos1), p_b=src (pos2), p_a=dst (pos3)
    let size = unsafe { *p_c } as usize;
    let src = unsafe { *p_b } as usize;
    let dst = unsafe { *p_a } as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable

    if dst_idx == src_idx {
        let mem_data = &mut store_mut_ref.memory_mut(dst_idx).data;
        if src.saturating_add(size) > mem_data.len()
            || dst.saturating_add(size) > mem_data.len()
        {
            return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
        }
        mem_data.copy_within(src..src + size, dst);
    } else {
        // Cross-memory copy: need to handle aliasing
        let module = store_mut_ref.module_mut();
        let (src_mem, dst_mem) = if src_idx < dst_idx {
            let (left, right) = module.memories.split_at_mut(dst_idx);
            (&left[src_idx], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(src_idx);
            (&right[0] as &MemInst, &mut left[dst_idx])
        };
        if src.saturating_add(size) > src_mem.data.len()
            || dst.saturating_add(size) > dst_mem.data.len()
        {
            return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
        }
        dst_mem.data[dst..dst + size].copy_from_slice(&src_mem.data[src..src + size]);
    }

    pc_fallthrough(pc)
}

/// memory.fill: read dst, value, size from TOS (3→0)
/// Encoding: see encoding.toml "ternary" pattern
/// tos_pattern = { pop = 3, push = 0 } - reads from p_a (dst), p_b (val), p_c (size)
#[no_mangle]
#[inline(always)]
pub extern "C" fn impl_memory_fill(
    ctx: *mut Context,
    pc: *mut Instruction,
    _fp_pp: *mut *mut u64,
    _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
    // Phase 3: Operand pointers (pos3=dst, pos2=value, pos1=size)
    p_a: *mut u64,
    p_b: *mut u64,
    p_c: *mut u64,
) -> *mut Instruction {
    let mem_idx = ternary::decode_imm0(pc) as usize;

    // Read from TOS: p_c=size (pos1), p_b=value (pos2), p_a=dst (pos3)
    let size = unsafe { *p_c } as usize;
    let value = unsafe { *p_b } as u8;
    let dst = unsafe { *p_a } as usize;

    let store_mut_ref = ctx_store_mut(ctx);
    // store_mut_ref already mutable
    let mem_data = &mut store_mut_ref.memory_mut(mem_idx).data;
    if dst.saturating_add(size) > mem_data.len() {
        return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
    }
    mem_data[dst..dst + size].fill(value);

    pc_fallthrough(pc)
}

// =============================================================================
// Multi-Memory Slow Path Handlers (Rust implementations for mem_idx >= 1)
// =============================================================================
// These handlers support arbitrary memory indices by accessing memory through
// the Store. They are used when mem_idx != 0, while the C fast path handles mem_idx == 0.
//
// Phase 3 TOS-only:
// - load_mm (pop1_push1): Reads address from p_src, writes result to p_dst
// - store_mm (pop2_push0): Reads address from p_addr, value from p_val

/// Generic multi-memory load handler implementation (Phase 3 TOS-only)
/// Multi-memory load handler - Phase 3: TOS-only computation
/// Encoding: see encoding.toml "load" pattern
/// tos_pattern = { pop = 1, push = 1 } - uses p_src for input, p_dst for output
macro_rules! impl_mm_load {
    ($name:ident, $load_type:ty, $result_expr:expr) => {
        #[no_mangle]
        #[inline(always)]
        pub extern "C" fn $name(
            ctx: *mut Context,
            pc: *mut Instruction,
            _fp_pp: *mut *mut u64,
            _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
            // Phase 3: Operand pointers for TOS computation
            p_src: *mut u64,
            p_dst: *mut u64,
        ) -> *mut Instruction {
            let mem_idx = load::decode_memidx(pc) as usize;
            let offset = load::decode_offset(pc) as usize;

            // Read address from TOS (p_src)
            let index = unsafe { *p_src } as u32 as usize;
            let addr = index.saturating_add(offset);
            let size = core::mem::size_of::<$load_type>();

            let store_ref = ctx_store(ctx);
            let mem = store_ref.memory(mem_idx);
            let mem_data = &mem.data;
            if addr.saturating_add(size) > mem_data.len() {
                return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
            }

            // SAFETY: addr is bounds-checked above
            let value: $load_type = unsafe {
                let base = mem_data.as_ptr().add(addr) as *const $load_type;
                core::ptr::read_unaligned(base)
            };

            // Write result to TOS (p_dst)
            let result = $result_expr(value);
            unsafe { *p_dst = result };

            pc_fallthrough(pc)
        }
    };
}

/// Multi-memory store handler - Phase 3: TOS-only computation
/// Encoding: see encoding.toml "store" pattern
/// tos_pattern = { pop = 2, push = 0 } - uses p_addr and p_val for inputs
macro_rules! impl_mm_store {
    ($name:ident, $store_type:ty, $value_expr:expr) => {
        #[no_mangle]
        #[inline(always)]
        pub extern "C" fn $name(
            ctx: *mut Context,
            pc: *mut Instruction,
            _fp_pp: *mut *mut u64,
            _p_l0: *mut u64,
    _p_l1: *mut u64,
    _p_l2: *mut u64,
            // Phase 3: Operand pointers for TOS computation
            p_addr: *mut u64,
            p_val: *mut u64,
        ) -> *mut Instruction {
            let mem_idx = store::decode_memidx(pc) as usize;
            let offset = store::decode_offset(pc) as usize;

            // Read address and value from TOS operand pointers
            let index = unsafe { *p_addr } as u32 as usize;
            let raw_value = unsafe { *p_val };
            let addr = index.saturating_add(offset);
            let size = core::mem::size_of::<$store_type>();

            let store_mut_ref = ctx_store_mut(ctx);
            // store_mut_ref already mutable
            let mem_data = &mut store_mut_ref.memory_mut(mem_idx).data;
            if addr.saturating_add(size) > mem_data.len() {
                return trap_with(ctx, WasmError::trap("out of bounds memory access".into()));
            }

            let value: $store_type = $value_expr(raw_value);
            // SAFETY: addr is bounds-checked above
            unsafe {
                let base = mem_data.as_mut_ptr().add(addr) as *mut $store_type;
                core::ptr::write_unaligned(base, value);
            }

            pc_fallthrough(pc)
        }
    };
}

// Multi-memory load handlers (slow path for mem_idx >= 1)
impl_mm_load!(impl_i32_load_mm, i32, |v: i32| v as u32 as u64);
impl_mm_load!(impl_i64_load_mm, i64, |v: i64| v as u64);
impl_mm_load!(impl_f32_load_mm, u32, |v: u32| v as u64);
impl_mm_load!(impl_f64_load_mm, u64, |v: u64| v);

impl_mm_load!(impl_i32_load8_s_mm, i8, |v: i8| v as i32 as u32 as u64);
impl_mm_load!(impl_i32_load8_u_mm, u8, |v: u8| v as u32 as u64);
impl_mm_load!(impl_i32_load16_s_mm, i16, |v: i16| v as i32 as u32 as u64);
impl_mm_load!(impl_i32_load16_u_mm, u16, |v: u16| v as u32 as u64);

impl_mm_load!(impl_i64_load8_s_mm, i8, |v: i8| v as i64 as u64);
impl_mm_load!(impl_i64_load8_u_mm, u8, |v: u8| v as u64);
impl_mm_load!(impl_i64_load16_s_mm, i16, |v: i16| v as i64 as u64);
impl_mm_load!(impl_i64_load16_u_mm, u16, |v: u16| v as u64);
impl_mm_load!(impl_i64_load32_s_mm, i32, |v: i32| v as i64 as u64);
impl_mm_load!(impl_i64_load32_u_mm, u32, |v: u32| v as u64);

// Multi-memory store handlers (slow path for mem_idx >= 1)
impl_mm_store!(impl_i32_store_mm, i32, |v: u64| v as i32);
impl_mm_store!(impl_i64_store_mm, i64, |v: u64| v as i64);
impl_mm_store!(impl_f32_store_mm, u32, |v: u64| v as u32);
impl_mm_store!(impl_f64_store_mm, u64, |v: u64| v);

impl_mm_store!(impl_i32_store8_mm, u8, |v: u64| v as u8);
impl_mm_store!(impl_i32_store16_mm, u16, |v: u64| v as u16);
impl_mm_store!(impl_i64_store8_mm, u8, |v: u64| v as u8);
impl_mm_store!(impl_i64_store16_mm, u16, |v: u64| v as u16);
impl_mm_store!(impl_i64_store32_mm, u32, |v: u64| v as u32);
