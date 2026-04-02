use crate::{
    constants::WASM_PAGE_SIZE,
    error::WasmError,
    vm::{
        entities::{MemInst, TableInst},
        runtime::{
            common::{internal_error, trap_error},
            context::NativeContext,
        },
        value::RefHandle,
    },
};

#[inline]
pub(super) fn memory_mut(ctx: &mut NativeContext, mem_idx: u32) -> Result<&mut MemInst, WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    store
        .module_mut()
        .memories
        .get_mut(mem_idx as usize)
        .ok_or_else(|| internal_error("preserved helper referenced invalid memory index"))
}

#[inline]
pub(super) fn table_mut(
    ctx: &mut NativeContext,
    table_idx: u32,
) -> Result<&mut TableInst, WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    store
        .module_mut()
        .tables
        .get_mut(table_idx as usize)
        .ok_or_else(|| internal_error("preserved helper referenced invalid table index"))
}

/// Core memory-grow logic owned by the preserved-helper system. Returns the
/// Wasm result value (old page count on success, or the appropriate sentinel).
pub(super) fn do_memory_grow(
    ctx: &mut NativeContext,
    mem_idx: u32,
    delta_raw: u64,
) -> Result<u64, WasmError> {
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

pub(super) fn do_memory_copy(
    ctx: &mut NativeContext,
    dst_mem_idx: u32,
    src_mem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let di = dst_mem_idx as usize;
    let si = src_mem_idx as usize;
    if di >= store.module().memories.len() || si >= store.module().memories.len() {
        return Err(internal_error(
            "preserved helper referenced invalid memory index",
        ));
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

pub(super) fn do_memory_init(
    ctx: &mut NativeContext,
    mem_idx: u32,
    data_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let module = store.module_mut();
    let mi = mem_idx as usize;
    let di = data_idx as usize;
    if mi >= module.memories.len() {
        return Err(internal_error(
            "preserved helper referenced invalid memory index",
        ));
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

pub(super) fn do_table_grow(
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

pub(super) fn do_table_copy(
    ctx: &mut NativeContext,
    dst_tbl: u32,
    src_tbl: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let di = dst_tbl as usize;
    let si = src_tbl as usize;
    if di >= store.module().tables.len() || si >= store.module().tables.len() {
        return Err(internal_error(
            "preserved helper referenced invalid table index",
        ));
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
        dst_table.elements[dest..dest + len].copy_from_slice(&src_table.elements[src..src + len]);
    }
    Ok(())
}

pub(super) fn do_table_init(
    ctx: &mut NativeContext,
    table_idx: u32,
    elem_idx: u32,
    dest: usize,
    src: usize,
    len: usize,
) -> Result<(), WasmError> {
    let store = ctx
        .store_mut()
        .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
    let module = store.module_mut();
    let ti = table_idx as usize;
    let ei = elem_idx as usize;
    if ti >= module.tables.len() {
        return Err(internal_error(
            "preserved helper referenced invalid table index",
        ));
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
    use super::{decode_memory_grow_delta, memory_grow_succeeded};

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
}
