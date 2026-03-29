//! ARMv7-A module compilation and linking.
//!
//! Per-function compilation is delegated to the common pipeline via
//! `compile_function::<Arm32Backend>`. This module handles ARM32-specific
//! linking: MOVW/MOVT address patching and the 32-bit function info table.

use alloc::vec::Vec;

use crate::{
    error::WasmError,
    vm::{
        arch::common::{
            helpers::page_align_function,
            pipeline,
            text_emitter::TextEmitter,
            types::FunctionArtifact,
        },
        entities::ModuleInst,
        runtime::code::{NativeRootEntry, CompiledNativeModule},
    },
};

use crate::vm::arch::common::backend::ArchBackend;
use super::backend::{Arm32Backend, CompiledArm32Entry};
use super::enc;
use super::reg::Arm32Reg;

// ── ARM32-specific patch helper ──────────────────────────────────────────────

/// Patch a MOVW/MOVT pair at `movw_offset` with a 32-bit address.
fn patch_movw_movt(text: &mut TextEmitter, movw_offset: usize, addr: u32) {
    let existing = u32::from_le_bytes([
        text.byte(movw_offset),
        text.byte(movw_offset + 1),
        text.byte(movw_offset + 2),
        text.byte(movw_offset + 3),
    ]);
    let rd_bits = (existing >> 12) & 0xF;
    let rd = Arm32Reg::from_idx(rd_bits);
    text.patch_u32(movw_offset, enc::movw(rd, addr as u16));
    text.patch_u32(movw_offset + 4, enc::movt(rd, (addr >> 16) as u16));
}

// ── ARM32 function info table ────────────────────────────────────────────────

/// Per-function metadata: 4x u32 = 16 bytes (ARM32 uses 32-bit pointers).
#[repr(C)]
struct Arm32FunctionInfo {
    entry: u32,
    total_frame_bytes: u32,
    frame_prefix_slots: u32,
    call_scratch_base_slot: u32,
}

const ARM32_FUNCTION_INFO_SIZE: usize = core::mem::size_of::<Arm32FunctionInfo>();

// ── Module compilation ───────────────────────────────────────────────────────

pub(crate) fn compile_module(
    module: &ModuleInst,
    compiled: &CompiledNativeModule,
) -> Result<Vec<Option<CompiledArm32Entry>>, WasmError> {
    // Pass 1: compile each function via common pipeline
    let mut artifacts: Vec<FunctionArtifact> = Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        artifacts.push(pipeline::compile_function::<Arm32Backend>(compiled, function)?);
    }

    // Pass 2: compute page-aligned base offsets
    let mut base_offsets = Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        running_offset = page_align_function(running_offset, artifact.text.len());
        base_offsets.push(running_offset);
        running_offset = running_offset.saturating_add(artifact.text.len());
    }
    let function_info_table_offset = running_offset;

    // Get executable buffer base pointer
    let base_ptr = {
        let executable = module
            .native_code_buffer()
            .map_err(|err| WasmError::internal(err.into()))?;
        executable.as_ptr()
    };

    let mut internal_entry_addrs = Vec::with_capacity(artifacts.len());
    for (i, base_offset) in base_offsets.iter().enumerate() {
        internal_entry_addrs.push(unsafe {
            base_ptr.add(*base_offset + artifacts[i].internal_entry_offset)
        } as usize);
    }

    // Build ARM32-specific function info table (4x u32 = 16 bytes per entry)
    let mut function_info_bytes =
        Vec::with_capacity(compiled.runtime().functions.len() * ARM32_FUNCTION_INFO_SIZE);
    for (func_idx, runtime) in compiled.runtime().functions.iter().enumerate() {
        let info = Arm32FunctionInfo {
            entry: *internal_entry_addrs.get(func_idx).ok_or_else(|| {
                WasmError::internal("armv7a function entry is out of range".into())
            })? as u32,
            total_frame_bytes: u32::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u32::from(runtime.frame_prefix_slots),
            call_scratch_base_slot: u32::from(
                runtime
                    .call_scratch
                    .map(|region| region.base_slot)
                    .unwrap_or(0),
            ),
        };
        function_info_bytes.extend_from_slice(&info.entry.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.call_scratch_base_slot.to_le_bytes());
    }

    // Pass 2.5: patch MOVW/MOVT addresses in artifacts
    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        // Patch local pointers (continuation addresses)
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u32;
            patch_movw_movt(&mut artifact.text, patch.literal_offset, target_addr);
        }
        // Patch direct call targets
        for patch in &artifact.direct_call_patches {
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| {
                    WasmError::internal("armv7a direct callee address is out of range".into())
                })? as u32;
            patch_movw_movt(&mut artifact.text, patch.literal_offset, callee_addr);
        }
        // Patch function table references
        for &literal_offset in &artifact.function_table_patches {
            let table_addr = unsafe { base_ptr.add(function_info_table_offset) } as u32;
            patch_movw_movt(&mut artifact.text, literal_offset, table_addr);
        }
    }

    // Pass 3: write everything to the shared CodeBuffer
    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut entries = Vec::with_capacity(artifacts.len());
    for (func_idx, artifact) in artifacts.into_iter().enumerate() {
        let current = executable.len() - written_start;
        let expected = base_offsets[func_idx];
        debug_assert!(expected >= current);
        let padding = expected - current;
        if padding > 0 {
            Arm32Backend::emit_nop_padding(&mut executable, padding);
        }
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        let debug_regions = artifact.debug_regions;
        let offset = executable.emit_bytes(&text_bytes);
        let entry = unsafe { executable.fn_ptr::<NativeRootEntry>(offset) };
        let root_return = unsafe { executable.ptr(offset + artifact.root_return_offset) };
        #[cfg(has_guard_pages)]
        let return_error = unsafe { executable.ptr(offset + artifact.return_error_offset) };
        entries.push(Some(CompiledArm32Entry {
            entry,
            root_return,
            #[cfg(has_guard_pages)]
            return_error,
            text_len,
            debug_regions,
        }));
    }
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len().saturating_sub(written_start);
    executable.finish_write(written_start, written_len);

    // Record profiler symbols
    let module_name = &module.name;
    for (func_idx, entry) in entries.iter().enumerate() {
        if let Some(entry) = entry {
            let func_base = entry.entry as *const u8;
            for region in &entry.debug_regions {
                if region.len > 0 {
                    let region_start = unsafe { func_base.add(region.offset) };
                    let code_bytes =
                        unsafe { core::slice::from_raw_parts(region_start, region.len) };
                    let symbol =
                        alloc::format!("jit::{}::func{}::{}", module_name, func_idx, region.label);
                    crate::vm::runtime::profiler::record_function(region_start, code_bytes, &symbol);
                }
            }
        }
    }

    // Register guard-pages JIT ranges
    #[cfg(has_guard_pages)]
    {
        let ranges: Vec<_> = entries
            .iter()
            .flatten()
            .map(|e| {
                (
                    e.entry as usize,
                    e.entry as usize + e.text_len,
                    e.return_error as usize,
                )
            })
            .collect();
        crate::vm::runtime::trap_signal::register_jit_ranges(&ranges);
    }

    Ok(entries)
}
