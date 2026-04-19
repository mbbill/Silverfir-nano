use crate::{
    error::WasmError,
    vm::{
        runtime::{
            common::{internal_error, set_ctx_error, trap_error, NativeCallStatus},
            context::NativeContext,
        },
        value_encoding::machine_raw_to_ref,
    },
};

use super::{
    abi::{io, op},
    ops::{
        do_any_convert_extern, do_array_copy, do_array_fill, do_array_get, do_array_init_data,
        do_array_init_elem, do_array_len, do_array_new, do_array_new_data, do_array_new_default,
        do_array_new_elem, do_array_new_fixed, do_array_set, do_extern_convert_any, do_i31_get_s,
        do_i31_get_u, do_memory_copy, do_memory_grow, do_memory_init, do_ref_as_non_null,
        do_ref_cast, do_ref_eq, do_ref_func, do_ref_i31, do_ref_test, do_struct_get, do_struct_new,
        do_struct_new_default, do_struct_set, do_table_copy, do_table_grow, do_table_init,
        do_v128_from_raw, do_v128_to_raw,
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
                return Err(trap_error("out of bounds memory access"));
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
            let delta = unsafe { *io_ptr.add(io::ARG1) };
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
            let mut elements = table.elements_mut();
            if start.saturating_add(len) > elements.len() {
                return Err(trap_error("out of bounds table access"));
            }
            elements[start..start + len]
                .fill(machine_raw_to_ref(val, super::ops::active_gp_unit_bytes()));
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
        op::REF_FUNC => {
            let func_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let result = do_ref_func(ctx, func_idx)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::REF_AS_NON_NULL => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_ref_as_non_null(raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::REF_EQ => {
            let lhs = unsafe { *io_ptr.add(io::ARG0) };
            let rhs = unsafe { *io_ptr.add(io::ARG1) };
            unsafe {
                *io_ptr.add(io::RET0) = do_ref_eq(lhs, rhs);
            }
            Ok(())
        }
        op::REF_I31 => {
            let value = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_ref_i31(ctx, value)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::I31_GET_S => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_i31_get_s(ctx, raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::I31_GET_U => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_i31_get_u(ctx, raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ANY_CONVERT_EXTERN => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_any_convert_extern(raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::EXTERN_CONVERT_ANY => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_extern_convert_any(raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_NEW => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let count = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let payload_ptr = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_struct_new(ctx, type_idx, payload_ptr, count)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_NEW_DEFAULT => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let result = do_struct_new_default(ctx, type_idx)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_NEW => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let init = unsafe { *io_ptr.add(io::ARG0) };
            let len = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_new(ctx, type_idx, init, len)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_NEW_DEFAULT => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let len = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_array_new_default(ctx, type_idx, len)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::REF_TEST => {
            let imm0 = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let imm1 = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_ref_test(ctx, imm0, imm1, raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::REF_CAST => {
            let imm0 = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let imm1 = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_ref_cast(ctx, imm0, imm1, raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_GET => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let field_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_struct_get(ctx, type_idx, field_idx, raw_ref, None)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_GET_S => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let field_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_struct_get(ctx, type_idx, field_idx, raw_ref, Some(true))?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_GET_U => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let field_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_struct_get(ctx, type_idx, field_idx, raw_ref, Some(false))?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::STRUCT_SET => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let field_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let raw_value = unsafe { *io_ptr.add(io::ARG1) };
            do_struct_set(ctx, type_idx, field_idx, raw_ref, raw_value)
        }
        op::ARRAY_GET => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let index = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_get(ctx, type_idx, raw_ref, index, None)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_GET_S => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let index = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_get(ctx, type_idx, raw_ref, index, Some(true))?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_GET_U => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let index = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_get(ctx, type_idx, raw_ref, index, Some(false))?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_LEN => {
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_array_len(ctx, raw_ref)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_SET => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let index = unsafe { *io_ptr.add(io::ARG1) };
            let raw_value = unsafe { *io_ptr.add(io::ARG2) };
            do_array_set(ctx, type_idx, raw_ref, index, raw_value)
        }
        op::ARRAY_NEW_FIXED => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let count = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let payload_ptr = unsafe { *io_ptr.add(io::ARG0) };
            let result = do_array_new_fixed(ctx, type_idx, payload_ptr, count)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_FILL => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let index = unsafe { *io_ptr.add(io::ARG1) };
            let raw_value = unsafe { *io_ptr.add(io::ARG2) };
            let len = unsafe { *io_ptr.add(io::ARG3) };
            do_array_fill(ctx, type_idx, raw_ref, index, raw_value, len)
        }
        op::ARRAY_COPY => {
            let dst_type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let src_type_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let dst_ref = unsafe { *io_ptr.add(io::ARG0) };
            let dst_index = unsafe { *io_ptr.add(io::ARG1) };
            let src_ref = unsafe { *io_ptr.add(io::ARG2) };
            let src_index = unsafe { *io_ptr.add(io::ARG3) };
            let len = unsafe { *io_ptr.add(io::ARG4) };
            do_array_copy(
                ctx,
                dst_type_idx,
                src_type_idx,
                dst_ref,
                dst_index,
                src_ref,
                src_index,
                len,
            )
        }
        op::ARRAY_INIT_DATA => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let data_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let dst_index = unsafe { *io_ptr.add(io::ARG1) };
            let src_index = unsafe { *io_ptr.add(io::ARG2) };
            let len = unsafe { *io_ptr.add(io::ARG3) };
            do_array_init_data(ctx, type_idx, data_idx, raw_ref, dst_index, src_index, len)
        }
        op::ARRAY_INIT_ELEM => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let elem_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let raw_ref = unsafe { *io_ptr.add(io::ARG0) };
            let dst_index = unsafe { *io_ptr.add(io::ARG1) };
            let src_index = unsafe { *io_ptr.add(io::ARG2) };
            let len = unsafe { *io_ptr.add(io::ARG3) };
            do_array_init_elem(ctx, type_idx, elem_idx, raw_ref, dst_index, src_index, len)
        }
        op::ARRAY_NEW_DATA => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let data_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let src_index = unsafe { *io_ptr.add(io::ARG0) };
            let len = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_new_data(ctx, type_idx, data_idx, src_index, len)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        op::ARRAY_NEW_ELEM => {
            let type_idx = unsafe { *io_ptr.add(io::IMM0) } as u32;
            let elem_idx = unsafe { *io_ptr.add(io::IMM1) } as u32;
            let src_index = unsafe { *io_ptr.add(io::ARG0) };
            let len = unsafe { *io_ptr.add(io::ARG1) };
            let result = do_array_new_elem(ctx, type_idx, elem_idx, src_index, len)?;
            unsafe {
                *io_ptr.add(io::RET0) = result;
            }
            Ok(())
        }
        // These two ops are storage-ABI bridges, not SIMD arithmetic helpers.
        //
        // Native SIMD code keeps a live v128 in an FP/Q register, but most of
        // nano's runtime storage ABI still assumes one value fits in one
        // 64-bit raw slot. Frame slots and globals therefore store a `u64`
        // handle into the shared SIMD registry instead of the 16-byte payload
        // itself.
        //
        // The preserved-helper I/O area only has 8-byte ARG/RET slots, so the
        // arm64 backend uses the 16-byte prefix area as an out-of-band payload
        // channel: `v128.to_raw` reads the bytes from that prefix, interns
        // them, and writes the resulting raw handle to RET0; `v128.from_raw`
        // does the inverse by resolving the handle and writing the 16-byte
        // payload back to the prefix area for the caller to reload into a Q
        // register.
        #[cfg(sf_has_simd)]
        op::V128_TO_RAW => {
            let prefix_ptr = unsafe { (io_ptr as *const u8).sub(16) };
            let mut bytes = [0u8; 16];
            unsafe {
                core::ptr::copy_nonoverlapping(prefix_ptr, bytes.as_mut_ptr(), 16);
                *io_ptr.add(io::RET0) = do_v128_to_raw(ctx, bytes)?;
            }
            Ok(())
        }
        #[cfg(sf_has_simd)]
        op::V128_FROM_RAW => {
            let prefix_ptr = unsafe { (io_ptr as *mut u8).sub(16) };
            let raw = unsafe { *io_ptr.add(io::ARG0) };
            let bytes = do_v128_from_raw(ctx, raw)?;
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), prefix_ptr, 16);
            }
            Ok(())
        }
        _ => Err(internal_error("unknown preserved-helper op code")),
    }
}
