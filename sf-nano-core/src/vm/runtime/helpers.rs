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
                CallExternalMeta, CallIndirectExternalMeta, DataDropMeta, ElemDropMeta,
                HelperFrameRegion, MemoryCopyMeta, MemoryFillMeta, MemoryGrowMeta, MemoryInitMeta,
                TableCopyMeta, TableFillMeta, TableGrowMeta, TableInitMeta,
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
        MachineHelperSymbol::MemoryGrow => memory_grow_entry,
        MachineHelperSymbol::MemoryFill => memory_fill_entry,
        MachineHelperSymbol::MemoryCopy => memory_copy_entry,
        MachineHelperSymbol::TableGrow => table_grow_entry,
        MachineHelperSymbol::TableFill => table_fill_entry,
        MachineHelperSymbol::TableCopy => table_copy_entry,
        MachineHelperSymbol::MemoryInit => memory_init_entry,
        MachineHelperSymbol::DataDrop => data_drop_entry,
        MachineHelperSymbol::TableInit => table_init_entry,
        MachineHelperSymbol::ElemDrop => elem_drop_entry,
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

unsafe extern "C" fn memory_grow_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, memory_grow_helper)
}

unsafe extern "C" fn call_indirect_external_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, call_indirect_external_helper)
}

unsafe extern "C" fn memory_fill_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, memory_fill_helper)
}

unsafe extern "C" fn memory_copy_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, memory_copy_helper)
}

unsafe extern "C" fn table_grow_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, table_grow_helper)
}

unsafe extern "C" fn table_fill_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, table_fill_helper)
}

unsafe extern "C" fn table_copy_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, table_copy_helper)
}

unsafe extern "C" fn memory_init_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, memory_init_helper)
}

unsafe extern "C" fn data_drop_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, data_drop_helper)
}

unsafe extern "C" fn table_init_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, table_init_helper)
}

unsafe extern "C" fn elem_drop_entry(
    ctx: *mut NativeContext,
    frame: *mut u64,
    metadata: *const u8,
) -> u32 {
    run_helper(ctx, frame, metadata, elem_drop_helper)
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

fn memory_grow_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &MemoryGrowMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.io, 1, "memory.grow expects one io slot")?;

    let delta_pages_raw =
        unsafe { region_read(frame, meta.io, 0, "memory.grow io slot is out of bounds")? };

    let (result, error_value) = {
        let mem = memory_mut(ctx, meta.mem_idx)?;
        let is_64 = mem.limits.is64;
        let error_value = memory_grow_error_value(is_64);
        let delta_pages = decode_memory_grow_delta(delta_pages_raw, is_64);

        let old_pages = mem.current_pages();
        match old_pages.checked_add(delta_pages) {
            None => (error_value, error_value),
            Some(new_pages) if new_pages > mem.limits.get_max() => (error_value, error_value),
            Some(new_pages) => {
                #[cfg(has_guard_pages)]
                {
                    if let Some(guard) = mem.guard_mut() {
                        match guard.grow(delta_pages) {
                            Ok(_) => (old_pages as u64, error_value),
                            Err(_) => (error_value, error_value),
                        }
                    } else {
                        if let Some(new_len) = new_pages.checked_mul(WASM_PAGE_SIZE) {
                            let additional = new_len.saturating_sub(mem.data.len());
                            if mem.data.try_reserve(additional).is_err() {
                                (error_value, error_value)
                            } else {
                                mem.data.resize(new_len, 0);
                                (old_pages as u64, error_value)
                            }
                        } else {
                            (error_value, error_value)
                        }
                    }
                }
                #[cfg(not(has_guard_pages))]
                {
                    if let Some(new_len) = new_pages.checked_mul(WASM_PAGE_SIZE) {
                        let additional = new_len.saturating_sub(mem.data.len());
                        if mem.data.try_reserve(additional).is_err() {
                            (error_value, error_value)
                        } else {
                            mem.data.resize(new_len, 0);
                            (old_pages as u64, error_value)
                        }
                    } else {
                        (error_value, error_value)
                    }
                }
            }
        }
    };

    unsafe {
        region_write(
            frame,
            meta.io,
            0,
            result,
            "memory.grow io slot is out of bounds",
        )?;
    }
    if memory_grow_succeeded(result, error_value) {
        ctx.refresh_memory_views();
    }
    Ok(())
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

fn memory_fill_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &MemoryFillMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "memory.fill expects three args")?;
    let dst = unsafe {
        region_read(frame, meta.args, 0, "memory.fill dst slot is out of bounds")? as usize
    };
    let value = unsafe {
        region_read(
            frame,
            meta.args,
            1,
            "memory.fill value slot is out of bounds",
        )? as u8
    };
    let size = unsafe {
        region_read(
            frame,
            meta.args,
            2,
            "memory.fill size slot is out of bounds",
        )? as usize
    };

    let mem = memory_mut(ctx, meta.mem_idx)?;
    let mem_len = mem.memory_len();
    let mem_ptr = mem.memory_ptr();
    if dst.saturating_add(size) > mem_len {
        return Err(trap_error("out of bounds memory access"));
    }
    unsafe {
        let slice = core::slice::from_raw_parts_mut(mem_ptr, mem_len);
        slice[dst..dst + size].fill(value);
    }
    Ok(())
}

fn memory_copy_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &MemoryCopyMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "memory.copy expects three args")?;
    let dst = unsafe {
        region_read(frame, meta.args, 0, "memory.copy dst slot is out of bounds")? as usize
    };
    let src = unsafe {
        region_read(frame, meta.args, 1, "memory.copy src slot is out of bounds")? as usize
    };
    let size = unsafe {
        region_read(
            frame,
            meta.args,
            2,
            "memory.copy size slot is out of bounds",
        )? as usize
    };

    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;

    let dst_idx = meta.dst_mem_idx as usize;
    let src_idx = meta.src_mem_idx as usize;
    if dst_idx >= store.module().memories.len() || src_idx >= store.module().memories.len() {
        return Err(internal_error(
            "native helper referenced invalid memory index",
        ));
    }

    if dst_idx == src_idx {
        let mem = store.memory_mut(dst_idx);
        let mem_len = mem.memory_len();
        let mem_ptr = mem.memory_ptr();
        if src.saturating_add(size) > mem_len || dst.saturating_add(size) > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        let mem_data = unsafe { core::slice::from_raw_parts_mut(mem_ptr, mem_len) };
        mem_data.copy_within(src..src + size, dst);
    } else {
        let module = store.module_mut();
        let (src_mem, dst_mem) = if src_idx < dst_idx {
            let (left, right) = module.memories.split_at_mut(dst_idx);
            (&left[src_idx], &mut right[0])
        } else {
            let (left, right) = module.memories.split_at_mut(src_idx);
            (&right[0] as &MemInst, &mut left[dst_idx])
        };
        let src_len = src_mem.memory_len();
        let dst_len = dst_mem.memory_len();
        if src.saturating_add(size) > src_len || dst.saturating_add(size) > dst_len {
            return Err(trap_error("out of bounds memory access"));
        }
        let src_slice = unsafe { core::slice::from_raw_parts(src_mem.memory_ptr(), src_len) };
        let dst_slice = unsafe { core::slice::from_raw_parts_mut(dst_mem.memory_ptr(), dst_len) };
        dst_slice[dst..dst + size].copy_from_slice(&src_slice[src..src + size]);
    }

    Ok(())
}

fn table_grow_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &TableGrowMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 2, "table.grow expects two args")?;
    require_region_slots(meta.results, 1, "table.grow expects one result slot")?;

    let fill = RefHandle::new(unsafe {
        region_read(
            frame,
            meta.args,
            0,
            "table.grow value slot is out of bounds",
        )? as usize
    });
    let delta = unsafe {
        region_read(
            frame,
            meta.args,
            1,
            "table.grow delta slot is out of bounds",
        )? as usize
    };

    let result = {
        let table = table_mut(ctx, meta.table_idx)?;
        let old_len = table.elements.len();
        match old_len.checked_add(delta) {
            None => u32::MAX as u64,
            Some(new_len) if new_len > table.limits.get_max() => u32::MAX as u64,
            Some(new_len) if table.elements.try_reserve(delta).is_err() => u32::MAX as u64,
            Some(new_len) => {
                table.elements.resize_with(new_len, || fill);
                old_len as u64
            }
        }
    };

    unsafe {
        region_write(
            frame,
            meta.results,
            0,
            result,
            "table.grow result slot is out of bounds",
        )?;
    }
    if result != u32::MAX as u64 {
        ctx.refresh_table_views();
    }
    Ok(())
}

fn table_fill_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &TableFillMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "table.fill expects three args")?;
    let start = unsafe {
        region_read(
            frame,
            meta.args,
            0,
            "table.fill start slot is out of bounds",
        )? as usize
    };
    let value = RefHandle::new(unsafe {
        region_read(
            frame,
            meta.args,
            1,
            "table.fill value slot is out of bounds",
        )? as usize
    });
    let size = unsafe {
        region_read(frame, meta.args, 2, "table.fill size slot is out of bounds")? as usize
    };

    let table = table_mut(ctx, meta.table_idx)?;
    if start.saturating_add(size) > table.elements.len() {
        return Err(trap_error("out of bounds table access"));
    }
    table.elements[start..start + size].fill(value);
    Ok(())
}

fn table_copy_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &TableCopyMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "table.copy expects three args")?;
    let dst = unsafe {
        region_read(frame, meta.args, 0, "table.copy dst slot is out of bounds")? as usize
    };
    let src = unsafe {
        region_read(frame, meta.args, 1, "table.copy src slot is out of bounds")? as usize
    };
    let size = unsafe {
        region_read(frame, meta.args, 2, "table.copy size slot is out of bounds")? as usize
    };

    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;

    let dst_idx = meta.dst_table_idx as usize;
    let src_idx = meta.src_table_idx as usize;
    if dst_idx >= store.module().tables.len() || src_idx >= store.module().tables.len() {
        return Err(internal_error(
            "native helper referenced invalid table index",
        ));
    }

    if dst_idx == src_idx {
        let table = store.table_mut(dst_idx);
        if src.saturating_add(size) > table.elements.len()
            || dst.saturating_add(size) > table.elements.len()
        {
            return Err(trap_error("out of bounds table access"));
        }
        table.elements.copy_within(src..src + size, dst);
    } else {
        let module = store.module_mut();
        let (src_table, dst_table) = if src_idx < dst_idx {
            let (left, right) = module.tables.split_at_mut(dst_idx);
            (&left[src_idx], &mut right[0])
        } else {
            let (left, right) = module.tables.split_at_mut(src_idx);
            (&right[0] as &TableInst, &mut left[dst_idx])
        };
        if src.saturating_add(size) > src_table.elements.len()
            || dst.saturating_add(size) > dst_table.elements.len()
        {
            return Err(trap_error("out of bounds table access"));
        }
        dst_table.elements[dst..dst + size].copy_from_slice(&src_table.elements[src..src + size]);
    }

    Ok(())
}

fn memory_init_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &MemoryInitMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "memory.init expects three args")?;
    let dst = unsafe {
        region_read(frame, meta.args, 0, "memory.init dst slot is out of bounds")? as usize
    };
    let src = unsafe {
        region_read(frame, meta.args, 1, "memory.init src slot is out of bounds")? as usize
    };
    let size = unsafe {
        region_read(
            frame,
            meta.args,
            2,
            "memory.init size slot is out of bounds",
        )? as usize
    };

    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let module = store.module_mut();
    let mem_idx = meta.mem_idx as usize;
    let data_idx = meta.data_idx as usize;

    if mem_idx >= module.memories.len() {
        return Err(internal_error(
            "native helper referenced invalid memory index",
        ));
    }
    if data_idx >= module.data.len() {
        return Err(trap_error("out of bounds memory access"));
    }

    let data_bytes = &module.data[data_idx].bytes;
    let data_dropped = module.data[data_idx].is_dropped();

    let mem_len = module.memories[mem_idx].memory_len();

    if size == 0 {
        if src > data_bytes.len() || dst > mem_len {
            return Err(trap_error("out of bounds memory access"));
        }
        return Ok(());
    }
    if data_dropped {
        return Err(trap_error("out of bounds memory access"));
    }
    if src.saturating_add(size) > data_bytes.len() {
        return Err(trap_error("out of bounds memory access"));
    }

    let src_ptr = data_bytes[src..].as_ptr();
    let mem_ptr = module.memories[mem_idx].memory_ptr();
    if dst.saturating_add(size) > mem_len {
        return Err(trap_error("out of bounds memory access"));
    }
    unsafe {
        core::ptr::copy_nonoverlapping(src_ptr, mem_ptr.add(dst), size);
    }

    Ok(())
}

fn data_drop_helper(
    ctx: &mut NativeContext,
    _frame: *mut u64,
    meta: &DataDropMeta,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    if let Some(data) = store.module_mut().data.get_mut(meta.data_idx as usize) {
        data.drop_segment();
    }
    Ok(())
}

fn table_init_helper(
    ctx: &mut NativeContext,
    frame: *mut u64,
    meta: &TableInitMeta,
) -> Result<(), WasmError> {
    require_region_slots(meta.args, 3, "table.init expects three args")?;
    let dst = unsafe {
        region_read(frame, meta.args, 0, "table.init dst slot is out of bounds")? as usize
    };
    let src = unsafe {
        region_read(frame, meta.args, 1, "table.init src slot is out of bounds")? as usize
    };
    let size = unsafe {
        region_read(frame, meta.args, 2, "table.init size slot is out of bounds")? as usize
    };

    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    let module = store.module_mut();
    let table_idx = meta.table_idx as usize;
    let elem_idx = meta.elem_idx as usize;

    if table_idx >= module.tables.len() {
        return Err(internal_error(
            "native helper referenced invalid table index",
        ));
    }
    if elem_idx >= module.elements.len() {
        return Err(trap_error("out of bounds table access"));
    }

    let elem_inst = &module.elements[elem_idx];

    if size == 0 {
        if src > elem_inst.refs.len() || dst > module.tables[table_idx].elements.len() {
            return Err(trap_error("out of bounds table access"));
        }
        return Ok(());
    }
    if src.saturating_add(size) > elem_inst.refs.len()
        || dst.saturating_add(size) > module.tables[table_idx].elements.len()
    {
        return Err(trap_error("out of bounds table access"));
    }
    if elem_inst.is_dropped() {
        return Err(trap_error("out of bounds table access"));
    }

    for offset in 0..size {
        module.tables[table_idx].elements[dst + offset] =
            module.elements[elem_idx].refs[src + offset];
    }

    Ok(())
}

fn elem_drop_helper(
    ctx: &mut NativeContext,
    _frame: *mut u64,
    meta: &ElemDropMeta,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("native helper context is missing store"))?;
    if let Some(elem) = store.module_mut().elements.get_mut(meta.elem_idx as usize) {
        elem.drop_segment();
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
        value_type::{RefType, ValueType},
        vm::{
            entities::{DataInst, ElementInst, ExternalFn, MemInst, ModuleInst, TableInst},
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
    fn memory_grow_helper_refreshes_cached_views() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(3)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = MemoryGrowMeta {
            mem_idx: 0,
            io: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [1u64];

        let status = call_helper(MachineHelperSymbol::MemoryGrow, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], 1);
        assert_eq!(ctx.mem0_size, (2 * WASM_PAGE_SIZE) as u64);
        assert_eq!(ctx.memory_views_len, 1);
        assert!(ctx.error.is_none());
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
    fn table_grow_helper_refreshes_table_views() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.tables.push(TableInst::new(
            Limits::new(1, Some(4)).unwrap(),
            ValueType::Ref(RefType::funcref()),
        ));
        let (_store, mut ctx) = test_context(module);
        let meta = TableGrowMeta {
            table_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [RefHandle::externref(7).0 as u64, 2];

        let status = call_helper(MachineHelperSymbol::TableGrow, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], 1);
        assert_eq!(ctx.table_views_len, 1);
        let view = unsafe { &*ctx.table_views_base };
        assert_eq!(view.elements_len, 3);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn table_grow_helper_returns_minus_one_on_unallocatable_growth() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.tables.push(TableInst::new(
            Limits::new(0x10, Some(u32::MAX as usize)).unwrap(),
            ValueType::Ref(RefType::funcref()),
        ));
        let (_store, mut ctx) = test_context(module);
        let meta = TableGrowMeta {
            table_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 2,
            },
            results: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [RefHandle::null().0 as u64, 0xffff_fff0];

        let status = call_helper(MachineHelperSymbol::TableGrow, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], u32::MAX as u64);
        assert_eq!(ctx.table_views_len, 1);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn memory_grow_helper_returns_minus_one_on_unallocatable_growth() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module
            .memories
            .push(MemInst::new(Limits::new(0, None).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = MemoryGrowMeta {
            mem_idx: 0,
            io: HelperFrameRegion {
                base_slot: 0,
                slots: 1,
            },
        };
        let mut frame = [u32::MAX as u64];

        let status = call_helper(MachineHelperSymbol::MemoryGrow, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(frame[0], u32::MAX as u64);
        assert_eq!(ctx.mem0_size, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    fn memory_copy_helper_copies_between_memories() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        let mut src = MemInst::new(Limits::new(1, Some(2)).unwrap());
        src.data[4..8].copy_from_slice(&[1, 2, 3, 4]);
        let dst = MemInst::new(Limits::new(1, Some(2)).unwrap());
        module.memories.push(dst);
        module.memories.push(src);
        let (mut store, mut ctx) = test_context(module);
        let meta = MemoryCopyMeta {
            dst_mem_idx: 0,
            src_mem_idx: 1,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 3,
            },
        };
        let mut frame = [10, 4, 4];

        let status = call_helper(MachineHelperSymbol::MemoryCopy, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Ok as u32);
        assert_eq!(&store.memory(0).data[10..14], &[1, 2, 3, 4]);
        assert!(ctx.error.is_none());
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

    #[test]
    fn memory_init_and_data_drop_helpers_follow_segment_semantics() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        module.data.push(DataInst::new(vec![9, 8, 7, 6]));
        let (mut store, mut ctx) = test_context(module);

        let init_meta = MemoryInitMeta {
            data_idx: 0,
            mem_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 3,
            },
        };
        let mut init_frame = [2, 1, 2];
        let init_status = call_helper(
            MachineHelperSymbol::MemoryInit,
            &mut ctx,
            &mut init_frame,
            &init_meta,
        );
        assert_eq!(init_status, NativeHelperStatus::Ok as u32);
        assert_eq!(&store.memory(0).data[2..4], &[8, 7]);

        let drop_meta = DataDropMeta { data_idx: 0 };
        let mut drop_frame = [0];
        let drop_status = call_helper(
            MachineHelperSymbol::DataDrop,
            &mut ctx,
            &mut drop_frame,
            &drop_meta,
        );
        assert_eq!(drop_status, NativeHelperStatus::Ok as u32);
        assert!(store.module().data[0].is_dropped());
    }

    #[test]
    fn table_init_and_elem_drop_helpers_follow_segment_semantics() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module.tables.push(TableInst::new(
            Limits::new(3, Some(3)).unwrap(),
            ValueType::Ref(RefType::funcref()),
        ));
        module.elements.push(ElementInst::new(
            vec![RefHandle::new(11), RefHandle::new(22)],
            ValueType::Ref(RefType::funcref()),
        ));
        let (mut store, mut ctx) = test_context(module);

        let init_meta = TableInitMeta {
            elem_idx: 0,
            table_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 3,
            },
        };
        let mut init_frame = [1, 0, 2];
        let init_status = call_helper(
            MachineHelperSymbol::TableInit,
            &mut ctx,
            &mut init_frame,
            &init_meta,
        );
        assert_eq!(init_status, NativeHelperStatus::Ok as u32);
        assert_eq!(store.table(0).elements[1], RefHandle::new(11));
        assert_eq!(store.table(0).elements[2], RefHandle::new(22));

        let drop_meta = ElemDropMeta { elem_idx: 0 };
        let mut drop_frame = [0];
        let drop_status = call_helper(
            MachineHelperSymbol::ElemDrop,
            &mut ctx,
            &mut drop_frame,
            &drop_meta,
        );
        assert_eq!(drop_status, NativeHelperStatus::Ok as u32);
        assert!(store.module().elements[0].is_dropped());
    }

    #[test]
    fn helper_failure_sets_ctx_error_and_returns_error_status() {
        let mut module = ModuleInst::new(String::from("m"), TypeContext::empty());
        module
            .memories
            .push(MemInst::new(Limits::new(1, Some(1)).unwrap()));
        let (_store, mut ctx) = test_context(module);
        let meta = MemoryCopyMeta {
            dst_mem_idx: 0,
            src_mem_idx: 0,
            args: HelperFrameRegion {
                base_slot: 0,
                slots: 3,
            },
        };
        let mut frame = [0, 0, (WASM_PAGE_SIZE as u64) + 1];

        let status = call_helper(MachineHelperSymbol::MemoryCopy, &mut ctx, &mut frame, &meta);

        assert_eq!(status, NativeHelperStatus::Error as u32);
        assert!(ctx.error.as_ref().is_some_and(WasmError::is_trap));
    }
}
