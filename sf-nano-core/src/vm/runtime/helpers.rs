//! Native runtime helper wrappers.
//!
//! These are the concrete Rust entrypoints behind helper-backed runtime
//! boundaries. Every helper uses the same ABI:
//! `helper(ctx, frame, metadata) -> status`.

use alloc::{rc::Rc, vec, vec::Vec};
use core::mem::align_of;

use crate::{
    constants::WASM_PAGE_SIZE,
    error::WasmError,
    vm::{
        entities::{Caller, FunctionInst, MemInst, TableInst},
        machine::machine_ir::MachineHelperSymbol,
        raw_value::{raw_to_value, value_to_raw},
        runtime::{
            context::NativeContext,
            helper_meta::{
                CallExternalMeta, CallIndirectExternalMeta, HelperFrameRegion,
            },
        },
        value::{RefHandle, Value},
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum NativeHelperStatus {
    Ok = 0,
    Error = 1,
}

pub(crate) type NativeHelperEntry =
    unsafe extern "C" fn(ctx: *mut NativeContext, frame: *mut u64, metadata: *const u8) -> u32;

#[inline]
pub(crate) fn resolve_helper_entry(symbol: MachineHelperSymbol) -> NativeHelperEntry {
    match symbol {
        MachineHelperSymbol::CallExternal => call_external_entry,
        MachineHelperSymbol::CallIndirectExternal => call_indirect_external_entry,
    }
}

#[inline]
fn set_ctx_error(ctx: &mut NativeContext, error: WasmError) {
    #[cfg(feature = "function-trace")]
    crate::vm::debug::function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
}

#[inline]
fn internal_error(message: &str) -> WasmError {
    WasmError::internal(message.into())
}

#[inline]
fn trap_error(message: &str) -> WasmError {
    WasmError::trap(message.into())
}

#[inline]
fn require_region_slots(
    region: HelperFrameRegion,
    expected: u16,
    label: &str,
) -> Result<(), WasmError> {
    if region.slots != expected {
        return Err(internal_error(label));
    }
    Ok(())
}

#[inline]
unsafe fn decode_meta<'a, T>(metadata: *const u8) -> Result<&'a T, WasmError> {
    if metadata.is_null() {
        return Err(internal_error(
            "native helper received null metadata pointer",
        ));
    }
    if (metadata as usize) % align_of::<T>() != 0 {
        return Err(internal_error(
            "native helper received misaligned metadata pointer",
        ));
    }
    Ok(unsafe { &*metadata.cast::<T>() })
}

#[inline]
unsafe fn frame_read(frame: *mut u64, slot: u16) -> u64 {
    unsafe { *frame.add(slot as usize) }
}

#[inline]
unsafe fn frame_write(frame: *mut u64, slot: u16, value: u64) {
    unsafe {
        *frame.add(slot as usize) = value;
    }
}

#[inline]
fn region_slot(region: HelperFrameRegion, index: u16, label: &str) -> Result<u16, WasmError> {
    if index >= region.slots {
        return Err(internal_error(label));
    }
    region
        .base_slot
        .checked_add(index)
        .ok_or_else(|| internal_error("helper frame slot overflow"))
}

#[inline]
unsafe fn region_read(
    frame: *mut u64,
    region: HelperFrameRegion,
    index: u16,
    label: &str,
) -> Result<u64, WasmError> {
    Ok(unsafe { frame_read(frame, region_slot(region, index, label)?) })
}

#[inline]
unsafe fn region_write(
    frame: *mut u64,
    region: HelperFrameRegion,
    index: u16,
    value: u64,
    label: &str,
) -> Result<(), WasmError> {
    unsafe {
        frame_write(frame, region_slot(region, index, label)?, value);
    }
    Ok(())
}

#[inline]
fn memory_mut(ctx: &mut NativeContext, mem_idx: u32) -> Result<&mut MemInst, WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    store
        .module_mut()
        .memories
        .get_mut(mem_idx as usize)
        .ok_or_else(|| internal_error("native helper referenced invalid memory index"))
}

#[inline]
fn table_mut(ctx: &mut NativeContext, table_idx: u32) -> Result<&mut TableInst, WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    store
        .module_mut()
        .tables
        .get_mut(table_idx as usize)
        .ok_or_else(|| internal_error("native helper referenced invalid table index"))
}

#[inline]
fn run_helper<T>(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
    body: impl FnOnce(&mut NativeContext, *mut u64, &T) -> Result<(), WasmError>,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeHelperStatus::Error as u32;
    };

    ctx.error = None;

    let result = if frame.is_null() {
        Err(internal_error("native helper received null frame pointer"))
    } else {
        unsafe { decode_meta::<T>(metadata) }.and_then(|meta| body(ctx, frame, meta))
    };

    match result {
        Ok(()) => NativeHelperStatus::Ok as u32,
        Err(error) => {
            set_ctx_error(ctx, error);
            NativeHelperStatus::Error as u32
        }
    }
}

unsafe extern "C" fn call_external_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, call_external_helper)
}

unsafe extern "C" fn call_indirect_external_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, call_indirect_external_helper)
}

fn call_external_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &CallExternalMeta,
) -> Result<(), WasmError> {
    call_external_by_index(ctx, frame, meta.func_idx, meta.args, meta.results)
}

fn call_indirect_external_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &CallIndirectExternalMeta,
) -> Result<(), WasmError> {
    let func_idx_slot = u16::try_from(meta.func_idx_slot)
        .map_err(|_| internal_error("call_indirect external helper slot exceeds u16"))?;
    let func_idx = unsafe { frame_read(frame, func_idx_slot) } as u32;
    call_external_by_index(ctx, frame, func_idx, meta.args, meta.results)
}

fn call_external_by_index(
    ctx: &mut NativeContext,
    frame: *mut u64,
    func_idx: u32,
    args_region: HelperFrameRegion,
    results_region: HelperFrameRegion,
) -> Result<(), WasmError> {
    let (func_type, callback) = {
        let store = ctx
            .store()
            .ok_or_else(|| internal_error("native helper context is missing store"))?;
        let callee = store
            .module()
            .functions
            .get(func_idx as usize)
            .ok_or_else(|| internal_error("native helper referenced invalid function index"))?;
        match callee {
            FunctionInst::External {
                func_type,
                callback,
            } => (Rc::clone(func_type), *callback),
            FunctionInst::Local { .. } => {
                return Err(internal_error(
                    "native external-call helper received a local function target",
                ))
            }
        }
    };

    require_region_slots(
        args_region,
        func_type.params().len() as u16,
        "call_external args span does not match function arity",
    )?;
    require_region_slots(
        results_region,
        func_type.results().len() as u16,
        "call_external result span does not match function arity",
    )?;

    let args: Vec<Value> = func_type
        .params()
        .iter()
        .enumerate()
        .map(|(index, ty)| unsafe {
            region_read(
                frame,
                args_region,
                index as u16,
                "call_external arg slot is out of bounds",
            )
            .map(|raw| raw_to_value(raw, *ty))
        })
        .collect::<Result<_, _>>()?;
    let mut ret_vals = vec![Value::default(); func_type.results().len()];

    let mem_slice = {
        let store = ctx
            .store_mut()
            .ok_or_else(|| internal_error("native helper context is missing store"))?;
        if store.module().memories.is_empty() {
            None
        } else {
            let mem = store.memory_mut(0);
            let ptr = mem.memory_ptr();
            let len = mem.memory_len();
            if len == 0 {
                None
            } else {
                Some(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
            }
        }
    };
    let mut caller = Caller::new(mem_slice);
    callback(&mut caller, &args, &mut ret_vals)?;

    if ret_vals.len() != func_type.results().len() {
        return Err(internal_error(
            "external callback returned an unexpected result count",
        ));
    }

    for (index, value) in ret_vals.into_iter().enumerate() {
        unsafe {
            region_write(
                frame,
                results_region,
                index as u16,
                value_to_raw(value),
                "call_external result slot is out of bounds",
            )?;
        }
    }

    Ok(())
}

/// Core memory-grow logic shared by both the legacy boundary helper and the
/// preserved-helper path.  Returns the Wasm result value (old page count on
/// success, or the appropriate error sentinel).
fn do_memory_grow(ctx: &mut NativeContext, mem_idx: u32, delta_raw: u64) -> Result<u64, WasmError> {
    let mem = memory_mut(ctx, mem_idx)?;
    let is_64 = mem.limits.is64;
    let error_value = memory_grow_error_value(is_64);
    let delta_pages = decode_memory_grow_delta(delta_raw, is_64);

    let old_pages = mem.current_pages();
    let result = match old_pages.checked_add(delta_pages) {
        None => error_value,
        Some(new_pages) if new_pages > mem.limits.get_max() => error_value,
        Some(new_pages) => {
            #[cfg(has_guard_pages)]
            {
                if let Some(guard) = mem.guard_mut() {
                    match guard.grow(delta_pages) {
                        Ok(_) => old_pages as u64,
                        Err(_) => error_value,
                    }
                } else {
                    grow_heap(mem, new_pages, old_pages, error_value)
                }
            }
            #[cfg(not(has_guard_pages))]
            {
                grow_heap(mem, new_pages, old_pages, error_value)
            }
        }
    };

    if memory_grow_succeeded(result, error_value) {
        ctx.refresh_memory_views();
    }
    Ok(result)
}

fn grow_heap(mem: &mut MemInst, new_pages: usize, old_pages: usize, error_value: u64) -> u64 {
    if let Some(new_len) = new_pages.checked_mul(WASM_PAGE_SIZE) {
        let additional = new_len.saturating_sub(mem.data.len());
        if mem.data.try_reserve(additional).is_err() {
            error_value
        } else {
            mem.data.resize(new_len, 0);
            old_pages as u64
        }
    } else {
        error_value
    }
}

#[inline]
fn memory_grow_error_value(is_64: bool) -> u64 {
    if is_64 {
        u64::MAX
    } else {
        u32::MAX as u64
    }
}

#[inline]
fn decode_memory_grow_delta(delta_pages_raw: u64, is_64: bool) -> usize {
    if is_64 {
        delta_pages_raw as usize
    } else {
        (delta_pages_raw as u32) as usize
    }
}

#[inline]
fn memory_grow_succeeded(result: u64, error_value: u64) -> bool {
    result != error_value
}

// ── Preserved-helper ABI ────────────────────────────────────────────────────
//
// Uniform calling convention for arch-local helper calls.  Every helper uses
// the same C signature:
//
//     fn(ctx: *mut NativeContext, op_code: u32, io: *mut u64) -> u32
//
// `io` points to a small I/O area on the native stack.  The layout is
// standardised across all operations:

/// Preserved-helper I/O slot indices.
pub(crate) mod preserved_io {
    /// First immediate (mem_idx, table_idx, data_idx, elem_idx).
    pub const IMM0: usize = 0;
    /// Second immediate (src_mem_idx, src_table_idx) — zero when unused.
    pub const IMM1: usize = 1;
    /// First Wasm stack operand.
    pub const ARG0: usize = 2;
    /// Second Wasm stack operand.
    pub const ARG1: usize = 3;
    /// Third Wasm stack operand.
    pub const ARG2: usize = 4;
    /// Result written by the helper.
    pub const RET0: usize = 5;
    /// Total number of 8-byte slots in the I/O area.
    pub const SLOT_COUNT: usize = 8;
}

/// Op-code constants for [`preserved_helper`].
pub(crate) mod preserved_op {
    pub const MEMORY_GROW: u32 = 0;
    pub const MEMORY_FILL: u32 = 1;
    pub const MEMORY_COPY: u32 = 2;
    pub const MEMORY_INIT: u32 = 3;
    pub const DATA_DROP: u32 = 4;
    pub const TABLE_GROW: u32 = 5;
    pub const TABLE_FILL: u32 = 6;
    pub const TABLE_COPY: u32 = 7;
    pub const TABLE_INIT: u32 = 8;
    pub const ELEM_DROP: u32 = 9;
}

/// Trap entry point called from generated code.  Sets `ctx.error` to the
/// appropriate trap error and returns nonzero.
///
/// # Safety
///
/// `ctx` must point to a valid `NativeContext`.
pub(crate) unsafe extern "C" fn raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = match kind {
        0 => WasmError::trap("unreachable executed".into()),
        1 => WasmError::trap("out of bounds memory access".into()),
        2 => WasmError::trap("out of bounds table access".into()),
        3 => WasmError::trap("invalid function reference".into()),
        4 => WasmError::trap("indirect call type mismatch".into()),
        5 => WasmError::trap("integer divide by zero".into()),
        6 => WasmError::trap("integer overflow".into()),
        7 => WasmError::trap("invalid conversion to integer".into()),
        8 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native helper failed".into()),
    };
    set_ctx_error(ctx, error);
    1
}

/// Unified preserved-helper entry point called from generated code.
///
/// # Safety
///
/// `ctx` must point to a valid `NativeContext` and `io` must point to at
/// least `preserved_io::SLOT_COUNT` writable `u64` slots.
pub(crate) unsafe extern "C" fn preserved_helper(
    ctx: *mut NativeContext,
    op_code: u32,
    io: *mut u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeHelperStatus::Error as u32;
    };
    let result = unsafe { preserved_dispatch(ctx, op_code, io) };
    match result {
        Ok(()) => NativeHelperStatus::Ok as u32,
        Err(err) => {
            set_ctx_error(ctx, err);
            NativeHelperStatus::Error as u32
        }
    }
}

unsafe fn preserved_dispatch(
    ctx: &mut NativeContext,
    op_code: u32,
    io: *mut u64,
) -> Result<(), WasmError> {
    use preserved_io::*;
    match op_code {
        preserved_op::MEMORY_GROW => {
            let mem_idx = unsafe { *io.add(IMM0) } as u32;
            let delta = unsafe { *io.add(ARG0) };
            let result = do_memory_grow(ctx, mem_idx, delta)?;
            unsafe { *io.add(RET0) = result; }
            Ok(())
        }
        preserved_op::MEMORY_FILL => {
            let mem_idx = unsafe { *io.add(IMM0) } as u32;
            let dest = unsafe { *io.add(ARG0) } as usize;
            let val = unsafe { *io.add(ARG1) } as u8;
            let len = unsafe { *io.add(ARG2) } as usize;
            let mem = memory_mut(ctx, mem_idx)?;
            let mem_len = mem.memory_len();
            if dest.saturating_add(len) > mem_len {
                return Err(trap_error("out of bounds memory access"));
            }
            unsafe {
                let slice = core::slice::from_raw_parts_mut(mem.memory_ptr(), mem_len);
                slice[dest..dest + len].fill(val);
            }
            Ok(())
        }
        preserved_op::MEMORY_COPY => {
            let dst_mem_idx = unsafe { *io.add(IMM0) } as u32;
            let src_mem_idx = unsafe { *io.add(IMM1) } as u32;
            let dest = unsafe { *io.add(ARG0) } as usize;
            let src = unsafe { *io.add(ARG1) } as usize;
            let len = unsafe { *io.add(ARG2) } as usize;
            do_memory_copy(ctx, dst_mem_idx, src_mem_idx, dest, src, len)
        }
        preserved_op::MEMORY_INIT => {
            let mem_idx = unsafe { *io.add(IMM0) } as u32;
            let data_idx = unsafe { *io.add(IMM1) } as u32;
            let dest = unsafe { *io.add(ARG0) } as usize;
            let src = unsafe { *io.add(ARG1) } as usize;
            let len = unsafe { *io.add(ARG2) } as usize;
            do_memory_init(ctx, mem_idx, data_idx, dest, src, len)
        }
        preserved_op::DATA_DROP => {
            let data_idx = unsafe { *io.add(IMM0) } as u32;
            let store = ctx.store_mut()
                .ok_or_else(|| internal_error("native helper context is missing store"))?;
            if let Some(data) = store.module_mut().data.get_mut(data_idx as usize) {
                data.drop_segment();
            }
            Ok(())
        }
        preserved_op::TABLE_GROW => {
            let table_idx = unsafe { *io.add(IMM0) } as u32;
            let init_val = unsafe { *io.add(ARG0) };
            let delta = unsafe { *io.add(ARG1) } as usize;
            let result = do_table_grow(ctx, table_idx, init_val, delta)?;
            unsafe { *io.add(RET0) = result; }
            Ok(())
        }
        preserved_op::TABLE_FILL => {
            let table_idx = unsafe { *io.add(IMM0) } as u32;
            let start = unsafe { *io.add(ARG0) } as usize;
            let val = unsafe { *io.add(ARG1) };
            let len = unsafe { *io.add(ARG2) } as usize;
            let table = table_mut(ctx, table_idx)?;
            if start.saturating_add(len) > table.elements.len() {
                return Err(trap_error("out of bounds table access"));
            }
            table.elements[start..start + len].fill(RefHandle::new(val as usize));
            Ok(())
        }
        preserved_op::TABLE_COPY => {
            let dst_tbl = unsafe { *io.add(IMM0) } as u32;
            let src_tbl = unsafe { *io.add(IMM1) } as u32;
            let dest = unsafe { *io.add(ARG0) } as usize;
            let src = unsafe { *io.add(ARG1) } as usize;
            let len = unsafe { *io.add(ARG2) } as usize;
            do_table_copy(ctx, dst_tbl, src_tbl, dest, src, len)
        }
        preserved_op::TABLE_INIT => {
            let table_idx = unsafe { *io.add(IMM0) } as u32;
            let elem_idx = unsafe { *io.add(IMM1) } as u32;
            let dest = unsafe { *io.add(ARG0) } as usize;
            let src = unsafe { *io.add(ARG1) } as usize;
            let len = unsafe { *io.add(ARG2) } as usize;
            do_table_init(ctx, table_idx, elem_idx, dest, src, len)
        }
        preserved_op::ELEM_DROP => {
            let elem_idx = unsafe { *io.add(IMM0) } as u32;
            let store = ctx.store_mut()
                .ok_or_else(|| internal_error("native helper context is missing store"))?;
            if let Some(elem) = store.module_mut().elements.get_mut(elem_idx as usize) {
                elem.drop_segment();
            }
            Ok(())
        }
        _ => Err(internal_error("unknown preserved-helper op code")),
    }
}

/// Core memory.copy logic shared by boundary and preserved-helper paths.
fn do_memory_copy(
    ctx: &mut NativeContext,
    dst_mem_idx: u32,
    src_mem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx.store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let di = dst_mem_idx as usize;
    let si = src_mem_idx as usize;
    if di >= store.module().memories.len() || si >= store.module().memories.len() {
        return Err(internal_error("native helper referenced invalid memory index"));
    }
    if di == si {
        let mem = store.memory_mut(di);
        let mem_len = mem.memory_len();
        if src.saturating_add(len) > mem_len || dest.saturating_add(len) > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        let data = unsafe { core::slice::from_raw_parts_mut(mem.memory_ptr(), mem_len) };
        data.copy_within(src..src + len, dest);
    } else {
        let module = store.module_mut();
        let (src_mem, dst_mem) = if si < di {
            let (left, right) = module.memories.split_at_mut(di);
            (&left[si], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(si);
            (&right[0] as &MemInst, &mut left[di])
        };
        let sl = src_mem.memory_len();
        let dl = dst_mem.memory_len();
        if src.saturating_add(len) > sl || dest.saturating_add(len) > dl {
            return Err(trap_error("out of bounds memory access"));
        }
        let ss = unsafe { core::slice::from_raw_parts(src_mem.memory_ptr(), sl) };
        let ds = unsafe { core::slice::from_raw_parts_mut(dst_mem.memory_ptr(), dl) };
        ds[dest..dest + len].copy_from_slice(&ss[src..src + len]);
    }
    Ok(())
}

/// Core memory.init logic.
fn do_memory_init(
    ctx: &mut NativeContext,
    mem_idx: u32,
    data_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx.store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let module = store.module_mut();
    let mi = mem_idx as usize;
    let di = data_idx as usize;
    if mi >= module.memories.len() {
        return Err(internal_error("native helper referenced invalid memory index"));
    }
    if di >= module.data.len() {
        return Err(trap_error("out of bounds memory access"));
    }
    let data_bytes = &module.data[di].bytes;
    let data_dropped = module.data[di].is_dropped();
    let mem_len = module.memories[mi].memory_len();
    if len == 0 {
        if src > data_bytes.len() || dest > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        return Ok(());
    }
    if data_dropped {
        return Err(trap_error("out of bounds memory access"));
    }
    if src.saturating_add(len) > data_bytes.len() {
        return Err(trap_error("out of bounds memory access"));
    }
    if dest.saturating_add(len) > mem_len {
        return Err(trap_error("out of bounds memory access"));
    }
    unsafe {
        core::ptr::copy_nonoverlapping(
            data_bytes[src..].as_ptr(),
            module.memories[mi].memory_ptr().add(dest),
            len,
        );
    }
    Ok(())
}

/// Core table.grow logic.
fn do_table_grow(
    ctx: &mut NativeContext,
    table_idx: u32,
    init_val_raw: u64,
    delta: usize,
) -> Result<u64, WasmError> {
    let fill = RefHandle::new(init_val_raw as usize);
    let table = table_mut(ctx, table_idx)?;
    let old_len = table.elements.len();
    let result = match old_len.checked_add(delta) {
        None => u32::MAX as u64,
        Some(new_len) if new_len > table.limits.get_max() => u32::MAX as u64,
        Some(_) if table.elements.try_reserve(delta).is_err() => u32::MAX as u64,
        Some(new_len) => {
            table.elements.resize_with(new_len, || fill);
            old_len as u64
        }
    };
    if result != u32::MAX as u64 {
        ctx.refresh_table_views();
    }
    Ok(result)
}

/// Core table.copy logic.
fn do_table_copy(
    ctx: &mut NativeContext,
    dst_tbl: u32,
    src_tbl: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx.store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let di = dst_tbl as usize;
    let si = src_tbl as usize;
    if di >= store.module().tables.len() || si >= store.module().tables.len() {
        return Err(internal_error("native helper referenced invalid table index"));
    }
    if di == si {
        let table = store.table_mut(di);
        if src.saturating_add(len) > table.elements.len()
            || dest.saturating_add(len) > table.elements.len()
        {
            return Err(trap_error("out of bounds table access"));
        }
        table.elements.copy_within(src..src + len, dest);
    } else {
        let module = store.module_mut();
        let (src_table, dst_table) = if si < di {
            let (left, right) = module.tables.split_at_mut(di);
            (&left[si], &mut right[0])
        } else {
            let (left, right) = module.tables.split_at_mut(si);
            (&right[0] as &TableInst, &mut left[di])
        };
        if src.saturating_add(len) > src_table.elements.len()
            || dest.saturating_add(len) > dst_table.elements.len()
        {
            return Err(trap_error("out of bounds table access"));
        }
        dst_table.elements[dest..dest + len]
            .copy_from_slice(&src_table.elements[src..src + len]);
    }
    Ok(())
}

/// Core table.init logic.
fn do_table_init(
    ctx: &mut NativeContext,
    table_idx: u32,
    elem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx.store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let module = store.module_mut();
    let ti = table_idx as usize;
    let ei = elem_idx as usize;
    if ti >= module.tables.len() {
        return Err(internal_error("native helper referenced invalid table index"));
    }
    if ei >= module.elements.len() {
        return Err(trap_error("out of bounds table access"));
    }
    let elem = &module.elements[ei];
    if len == 0 {
        if src > elem.refs.len() || dest > module.tables[ti].elements.len() {
            return Err(trap_error("out of bounds table access"));
        }
        return Ok(());
    }
    if src.saturating_add(len) > elem.refs.len()
        || dest.saturating_add(len) > module.tables[ti].elements.len()
    {
        return Err(trap_error("out of bounds table access"));
    }
    if elem.is_dropped() {
        return Err(trap_error("out of bounds table access"));
    }
    for offset in 0..len {
        module.tables[ti].elements[dest + offset] = module.elements[ei].refs[src + offset];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::*;
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
        utils::limits::Limits,
        value_type::ValueType,
        vm::{
            entities::{ExternalFn, MemInst, ModuleInst},
            store::Store,
        },
    };

    fn test_context(module: ModuleInst) -> (Box<Store>, NativeContext) {
        let mut store = Box::new(Store::new(module));
        let ctx = NativeContext::new((&mut *store) as *mut Store, core::ptr::null_mut());
        (store, ctx)
    }

    fn call_helper<T: Copy>(
        symbol: MachineHelperSymbol,
        ctx: &mut NativeContext,
        frame: &mut [u64],
        meta: &T,
    ) -> u32 {
        let entry = resolve_helper_entry(symbol);
        unsafe { entry(ctx, frame.as_mut_ptr(), (meta as *const T).cast::<u8>()) }
    }

    #[test]
    fn memory_grow_uses_unsigned_delta_for_32_bit_memories() {
        assert_eq!(
            decode_memory_grow_delta(u32::MAX as u64, false),
            u32::MAX as usize
        );
        assert_eq!(
            decode_memory_grow_delta(0x8000_0000u64, false),
            0x8000_0000usize
        );
    }

    #[test]
    fn memory_grow_success_check_uses_selected_error_sentinel() {
        assert!(memory_grow_succeeded(u32::MAX as u64, u64::MAX));
        assert!(!memory_grow_succeeded(u64::MAX, u64::MAX));
        assert!(!memory_grow_succeeded(u32::MAX as u64, u32::MAX as u64));
    }

    #[test]
    fn call_external_helper_marshals_frame_slots() {
        fn host_add(
            _caller: &mut Caller,
            args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            let lhs = match args[0] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            let rhs = match args[1] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            results[0] = Value::I32(lhs + rhs);
            Ok(())
        }

        let func_type = Rc::new(FunctionType::new(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
        ));
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.functions.push(FunctionInst::External {
            func_type,
            callback: host_add as ExternalFn,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = CallExternalMeta {
            func_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5];

        let status = call_helper(
            MachineHelperSymbol::CallExternal,
            &mut ctx,
            &mut frame,
            &meta,
        );

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn call_indirect_external_helper_reads_dynamic_target_slot() {
        fn host_add(
            _caller: &mut Caller,
            args: &[Value],
            results: &mut [Value],
        ) -> Result<(), WasmError> {
            let lhs = match args[0] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            let rhs = match args[1] {
                Value::I32(value) => value,
                _ => panic!("unexpected arg"),
            };
            results[0] = Value::I32(lhs + rhs);
            Ok(())
        }

        let func_type = Rc::new(FunctionType::new(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
        ));
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.functions.push(FunctionInst::External {
            func_type,
            callback: host_add as ExternalFn,
        });
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = CallIndirectExternalMeta {
            func_idx_slot: 2,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [7, 5, 0];

        let status = call_helper(
            MachineHelperSymbol::CallIndirectExternal,
            &mut ctx,
            &mut frame,
            &meta,
        );

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], 12);
        assert!(ctx.error.is_none());
    }

}
