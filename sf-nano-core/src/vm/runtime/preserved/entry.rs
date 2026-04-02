use crate::{
    error::WasmError,
    vm::runtime::{
        common::{internal_error, set_ctx_error, NativeCallStatus},
        context::NativeContext,
    },
};

use super::{
    abi::{io, op},
    ops::{
        do_memory_copy, do_memory_grow, do_memory_init, do_table_copy, do_table_grow, do_table_init,
    },
};

/// Unified preserved-helper entry point called from generated code.
///
/// # Safety
///
/// `ctx` must point to a valid `NativeContext` and `io` must point to at
/// least `io::SLOT_COUNT` writable `u64` slots.
pub(crate) unsafe extern "C" fn preserved_entry(
    ctx: *mut NativeContext,
    op_code: u32,
    io: *mut u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return NativeCallStatus::Error as u32;
    };
    let result = unsafe { dispatch_preserved(ctx, op_code, io) };
    match result {
        Ok(()) => NativeCallStatus::Ok as u32,
        Err(err) => {
            set_ctx_error(ctx, err);
            NativeCallStatus::Error as u32
        }
    }
}

unsafe fn dispatch_preserved(
    ctx: &mut NativeContext,
    op_code: u32,
    io_ptr: *mut u64,
) -> Result<(), WasmError> {
    match op_code {
        op::MEMORY_GROW => {
            let mem_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let delta = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_memory_grow(ctx, mem_idx, delta)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::MEMORY_FILL => {
            let mem_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let dest = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let val = unsafe { *io_ptr.add(io::ARG1) } as u8;
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            let mem = super::ops::memory_mut(ctx, mem_idx)?;
            let mem_len = mem.memory_len();
            if dest.saturating_add(len) > mem_len {
                return Err(crate::vm::runtime::common::trap_error(
                    "out of bounds memory access",
                ));
            }
            unsafe {
                let slice = core::slice::from_raw_parts_mut(mem.memory_ptr(), mem_len);
                slice[dest..dest + len].fill(val);
            }
            Ok(())
        }
        op::MEMORY_COPY => {
            let dst_mem_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let src_mem_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let dest = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let src = unsafe { *io_ptr.add(io::ARG1) } as usize;
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            do_memory_copy(ctx, dst_mem_idx, src_mem_idx, dest, src, len)
        }
        op::MEMORY_INIT => {
            let mem_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let data_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let dest = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let src = unsafe { *io_ptr.add(io::ARG1) } as usize;
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            do_memory_init(ctx, mem_idx, data_idx, dest, src, len)
        }
        op::DATA_DROP => {
            let data_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let store = ctx
                .store_mut()
                .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
            if let Some(data) = store.module_mut().data.get_mut(data_idx as usize) {
                data.drop_segment();
            }
            Ok(())
        }
        op::TABLE_GROW => {
            let table_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let init_val = unsafe { *io_ptr.add(io::ARG0) };
            let delta = unsafe { *io_ptr.add(io::ARG1) } as usize;
            let result = do_table_grow(ctx, table_idx, init_val, delta)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::TABLE_FILL => {
            let table_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let start = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let val = unsafe { *io_ptr.add(io::ARG1) };
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            let table = super::ops::table_mut(ctx, table_idx)?;
            if start.saturating_add(len) > table.elements.len() {
                return Err(crate::vm::runtime::common::trap_error(
                    "out of bounds table access",
                ));
            }
            table.elements[start..start + len].fill(crate::vm::value::RefHandle::new(val as usize));
            Ok(())
        }
        op::TABLE_COPY => {
            let dst_tbl = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let src_tbl = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let dest = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let src = unsafe { *io_ptr.add(io::ARG1) } as usize;
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            do_table_copy(ctx, dst_tbl, src_tbl, dest, src, len)
        }
        op::TABLE_INIT => {
            let table_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let elem_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let dest = unsafe { *io_ptr.add(io::ARG0) } as usize;
            let src = unsafe { *io_ptr.add(io::ARG1) } as usize;
            let len = unsafe { *io_ptr.add(io::ARG2) } as usize;
            do_table_init(ctx, table_idx, elem_idx, dest, src, len)
        }
        op::ELEM_DROP => {
            let elem_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let store = ctx
                .store_mut()
                .ok_or_else(|| internal_error("preserved helper context is missing store"))?;
            if let Some(elem) = store.module_mut().elements.get_mut(elem_idx as usize) {
                elem.drop_segment();
            }
            Ok(())
        }
        _ => Err(internal_error("unknown preserved-helper op code")),
    }
}
