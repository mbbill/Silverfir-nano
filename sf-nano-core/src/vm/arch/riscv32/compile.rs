//! RV32 module compilation and linking.
//!
//! Per-function lowering uses the common pipeline. Linking is RV32-specific
//! because direct-call and continuation literals are pointer-sized 32-bit
//! words, and local-call metadata uses `NativeLocalCallInfo32`.

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        arch::common::{
            backend::ArchBackend, helpers::page_align_function, pipeline, types::FunctionArtifact,
        },
        entities::ModuleInst,
        runtime::{
            code::{CompiledNativeModule, NativeRootEntry},
            dispatch_view::NativeLocalCallInfo32,
        },
    },
};

use super::backend::{CompiledRiscv32Entry, Riscv32Backend};

type Riscv32FunctionInfo = NativeLocalCallInfo32;
const RISCV32_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<Riscv32FunctionInfo>();

pub(crate) fn compile_module(
    module: &ModuleInst,
    compiled: &CompiledNativeModule,
) -> Result<collections::Vec<Option<CompiledRiscv32Entry>>, WasmError> {
    let mut artifacts: collections::Vec<FunctionArtifact> =
        collections::Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        artifacts.push(pipeline::compile_function::<Riscv32Backend>(
            compiled, function,
        )?);
    }

    let mut base_offsets = collections::Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        running_offset = page_align_function(running_offset, artifact.text.len());
        base_offsets.push(running_offset);
        running_offset = running_offset.saturating_add(artifact.text.len());
    }
    let function_info_table_offset = running_offset;

    let base_ptr = {
        let executable = module
            .native_code_buffer()
            .map_err(|err| WasmError::internal(err))?;
        executable.as_ptr()
    };

    let mut internal_entry_addrs = collections::Vec::with_capacity(artifacts.len());
    for (i, base_offset) in base_offsets.iter().enumerate() {
        internal_entry_addrs.push(unsafe {
            base_ptr.add(*base_offset + artifacts[i].internal_entry_offset)
        } as usize);
    }

    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u32;
            artifact.text.patch_u32(patch.literal_offset, target_addr);
        }
        for patch in &artifact.direct_call_patches {
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| WasmError::internal("rv32 direct callee address is out of range"))?
                as u32;
            artifact.text.patch_u32(patch.literal_offset, callee_addr);
        }
    }

    let mut function_info_bytes = collections::Vec::with_capacity(
        compiled.abi().functions.len() * RISCV32_FUNCTION_INFO_SIZE,
    );
    for (func_idx, runtime) in compiled.abi().functions.iter().enumerate() {
        let info = Riscv32FunctionInfo {
            entry: *internal_entry_addrs
                .get(func_idx)
                .ok_or_else(|| WasmError::internal("rv32 function entry is out of range"))?
                as u32,
            total_frame_bytes: u32::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u32::from(runtime.frame_prefix_slots),
        };
        function_info_bytes.extend_from_slice(&info.entry.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
    }

    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut entries = collections::Vec::with_capacity(artifacts.len());
    for (func_idx, artifact) in artifacts.into_iter().enumerate() {
        let current = executable.len() - written_start;
        let expected = base_offsets[func_idx];
        debug_assert!(expected >= current);
        let padding = expected - current;
        if padding > 0 {
            Riscv32Backend::emit_nop_padding(&mut executable, padding);
        }
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        #[cfg(sf_has_debug_regions)]
        let debug_regions = artifact.debug_regions;
        let offset = executable.emit_bytes(&text_bytes);
        let entry = unsafe { executable.fn_ptr::<NativeRootEntry>(offset) };
        entries.push(Some(CompiledRiscv32Entry {
            entry,
            text_len,
            #[cfg(sf_has_debug_regions)]
            debug_regions,
        }));
    }
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len().saturating_sub(written_start);
    executable.finish_write(written_start, written_len);
    compiled
        .publish_local_call_infos(unsafe { executable.as_ptr().add(function_info_table_offset) });

    #[cfg(sf_jitdump)]
    {
        let module_name = &module.name;
        for (func_idx, entry) in entries.iter().enumerate() {
            if let Some(entry) = entry {
                let func_base = entry.entry as *const u8;
                for region in &entry.debug_regions {
                    if region.len > 0 {
                        let region_start = unsafe { func_base.add(region.offset) };
                        let code_bytes =
                            unsafe { core::slice::from_raw_parts(region_start, region.len) };
                        let symbol = tracked_alloc::format!(
                            "jit::{}::func{}::{}",
                            module_name,
                            func_idx,
                            region.label
                        );
                        crate::vm::debug::jitdump::record_function(
                            region_start,
                            code_bytes,
                            &symbol,
                        );
                    }
                }
            }
        }
    }

    Ok(entries)
}
