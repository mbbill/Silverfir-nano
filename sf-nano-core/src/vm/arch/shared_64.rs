use alloc::format;

use crate::collections;

use crate::{
    error::WasmError,
    vm::{
        arch::common::{
            backend::ArchBackend, helpers::page_align_function, pipeline::compile_function,
        },
        entities::ModuleInst,
        machine::machine_ir::{MachineBlock, MachineBlockId, MachineReg, MachineValue},
        runtime::{
            code::CompiledNativeModule, code_buf::CodeBuffer, dispatch_view::NativeLocalCallInfo64,
        },
    },
};

#[cfg(sf_has_debug_regions)]
use crate::vm::arch::common::types::DebugRegion;

/// Metadata needed by the shared 64-bit module linker for each emitted function.
pub(crate) struct EmittedFunction64 {
    pub text_offset: usize,
    pub text_len: usize,
    /// Offset of `body_local_error_label` within the function's text. Used
    /// by the guard-page signal handler to redirect a faulting PC to the
    /// trap propagation tail.
    #[cfg(sf_has_guard_pages)]
    pub body_local_error_offset: usize,
    #[cfg(sf_has_debug_regions)]
    pub debug_regions: collections::Vec<DebugRegion>,
}

/// 64-bit backends that use the shared module linker.
pub(crate) trait ModuleLinkBackend64<'a>: ArchBackend<'a> {
    type CompiledEntry: Clone + core::fmt::Debug;

    fn make_entry(buf: &CodeBuffer, emitted: &EmittedFunction64) -> Self::CompiledEntry;
}

type NativeFunctionInfo64 = NativeLocalCallInfo64;

const NATIVE_FUNCTION_INFO64_SIZE: usize = core::mem::size_of::<NativeFunctionInfo64>();

pub(crate) fn compile_module_64<'a, A: ModuleLinkBackend64<'a>>(
    module: &ModuleInst,
    compiled: &'a CompiledNativeModule,
) -> Result<collections::Vec<Option<A::CompiledEntry>>, WasmError> {
    let mut artifacts = collections::Vec::with_capacity(compiled.module().functions.len());
    for function in &compiled.module().functions {
        artifacts.push(compile_function::<A>(compiled, function)?);
    }

    let mut base_offsets = collections::Vec::with_capacity(artifacts.len());
    let mut running_offset = 0usize;
    for artifact in &artifacts {
        running_offset = page_align_function(running_offset, artifact.text.len());
        base_offsets.push(running_offset);
        running_offset = running_offset.saturating_add(artifact.text.len());
    }
    let function_info_table_offset = running_offset;

    let mut internal_entry_addrs = collections::Vec::with_capacity(artifacts.len());
    let base_ptr = {
        let executable = module
            .native_code_buffer()
            .map_err(|err| WasmError::internal(err.into()))?;
        executable.as_ptr()
    };
    for (i, base_offset) in base_offsets.iter().enumerate() {
        internal_entry_addrs.push(unsafe {
            base_ptr.add(*base_offset + artifacts[i].internal_entry_offset)
        } as usize);
    }

    for (index, artifact) in artifacts.iter_mut().enumerate() {
        let function_base = base_offsets[index];
        for patch in &artifact.local_ptr_patches {
            let target_addr = unsafe { base_ptr.add(function_base + patch.target_offset) } as u64;
            artifact.text.patch_u64(patch.literal_offset, target_addr);
        }
        for patch in &artifact.direct_call_patches {
            let callee_addr = *internal_entry_addrs
                .get(patch.callee.0 as usize)
                .ok_or_else(|| {
                    WasmError::internal("direct callee address is out of range".into())
                })? as u64;
            artifact.text.patch_u64(patch.literal_offset, callee_addr);
        }
    }

    let mut function_info_bytes =
        collections::Vec::with_capacity(artifacts.len() * NATIVE_FUNCTION_INFO64_SIZE);
    for (func_idx, runtime) in compiled.abi().functions.iter().enumerate() {
        let info = NativeFunctionInfo64 {
            entry: *internal_entry_addrs
                .get(func_idx)
                .ok_or_else(|| WasmError::internal("function entry is out of range".into()))?
                as u64,
            total_frame_bytes: u64::from(runtime.total_frame_slots) * 8,
            frame_prefix_slots: u64::from(runtime.frame_prefix_slots),
        };
        function_info_bytes.extend_from_slice(&info.entry.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.total_frame_bytes.to_le_bytes());
        function_info_bytes.extend_from_slice(&info.frame_prefix_slots.to_le_bytes());
    }

    let mut executable = module
        .native_code_buffer()
        .map_err(|err| WasmError::internal(err.into()))?;
    executable.begin_write();
    executable.reset();

    let written_start = executable.len();
    let mut emitted = collections::Vec::with_capacity(artifacts.len());
    for (func_idx, artifact) in artifacts.into_iter().enumerate() {
        let current = executable.len() - written_start;
        let expected = base_offsets[func_idx];
        debug_assert!(expected >= current);
        let padding = expected - current;
        if padding > 0 {
            A::emit_nop_padding(&mut executable, padding);
        }
        let text_bytes = artifact.text.finish();
        let text_len = text_bytes.len();
        let text_offset = executable.emit_bytes(&text_bytes);
        emitted.push(EmittedFunction64 {
            text_offset,
            text_len,
            #[cfg(sf_has_guard_pages)]
            body_local_error_offset: artifact.body_local_error_offset,
            #[cfg(sf_has_debug_regions)]
            debug_regions: artifact.debug_regions,
        });
    }
    executable.emit_bytes(&function_info_bytes);
    let written_len = executable.len().saturating_sub(written_start);
    executable.finish_write(written_start, written_len);
    compiled
        .publish_local_call_infos(unsafe { executable.as_ptr().add(function_info_table_offset) });

    let mut entries = collections::Vec::with_capacity(emitted.len());
    for ef in &emitted {
        entries.push(Some(A::make_entry(&executable, ef)));
    }

    #[cfg(sf_jitdump)]
    {
        let module_name = &module.name;
        for (func_idx, ef) in emitted.iter().enumerate() {
            let func_base = unsafe { base_ptr.add(base_offsets[func_idx]) };
            for region in &ef.debug_regions {
                if region.len > 0 {
                    let region_start = unsafe { func_base.add(region.offset) };
                    let code_bytes =
                        unsafe { core::slice::from_raw_parts(region_start, region.len) };
                    let symbol =
                        format!("jit::{}::func{}::{}", module_name, func_idx, region.label);
                    crate::vm::debug::jitdump::record_function(region_start, code_bytes, &symbol);
                }
            }
        }
    }

    #[cfg(sf_has_guard_pages)]
    {
        let ranges: collections::Vec<_> = emitted
            .iter()
            .enumerate()
            .map(|(func_idx, ef)| {
                let func_start = unsafe { base_ptr.add(base_offsets[func_idx]) } as usize;
                let func_end = func_start + ef.text_len;
                let body_local_error = func_start + ef.body_local_error_offset;
                (func_start, func_end, body_local_error)
            })
            .collect();
        crate::vm::runtime::trap_signal::register_jit_ranges(&ranges);
    }

    Ok(entries)
}

/// Validate that `reg` is a GP register (not FP). Returns the reg unchanged.
pub(crate) fn validate_gp_reg<'a, A: ArchBackend<'a>>(
    backend: &A,
    reg: MachineReg,
) -> Result<MachineReg, WasmError> {
    if backend.core().is_fp_reg(reg) {
        return Err(WasmError::invalid(format!(
            "expected GP register, got FP machine reg {}",
            reg.0
        )));
    }
    Ok(reg)
}

/// Returns true if jumping to `target` with `args` can be elided because the
/// target is the physical fallthrough and the args are an identity mapping.
pub(crate) fn is_fallthrough_edge(
    target: MachineBlockId,
    args: &[MachineValue],
    fallthrough: Option<MachineBlockId>,
    blocks: &[MachineBlock],
) -> bool {
    if fallthrough != Some(target) {
        return false;
    }
    let Some(block) = blocks.get(target.as_usize()) else {
        return false;
    };
    if block.params.len() != args.len() {
        return false;
    }
    block
        .params
        .iter()
        .zip(args.iter())
        .all(|(param, arg)| match arg {
            MachineValue::Reg(reg) | MachineValue::ReservedReg(reg) => *reg == param.reg,
            MachineValue::Imm64(_) => false,
        })
}

#[cfg(test)]
mod tests {
    use super::is_fallthrough_edge;
    use crate::collections;
    use crate::vm::machine::machine_ir::{
        MachineBlock, MachineBlockId, MachineBlockParam, MachineReg, MachineRegOwner,
        MachineTerminator, MachineValue,
    };

    fn block(id: u32, params: &[MachineBlockParam]) -> MachineBlock {
        MachineBlock {
            id: MachineBlockId(id),
            params: params.to_vec().into(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::Return,
        }
    }

    #[test]
    fn reserved_regs_count_as_identity_for_fallthrough_edges() {
        let params = [
            MachineBlockParam::gp_word(MachineReg(4)).with_owner(MachineRegOwner::CachedLocal),
            MachineBlockParam::gp_word(MachineReg(5)).with_owner(MachineRegOwner::CachedLocal),
        ];
        let blocks = [block(0, &[]), block(1, &params)];

        assert!(is_fallthrough_edge(
            MachineBlockId(1),
            &[
                MachineValue::ReservedReg(MachineReg(4)),
                MachineValue::ReservedReg(MachineReg(5)),
            ],
            Some(MachineBlockId(1)),
            &blocks,
        ));
    }

    #[test]
    fn mismatched_reserved_reg_is_not_identity_for_fallthrough_edges() {
        let params =
            [MachineBlockParam::gp_word(MachineReg(4)).with_owner(MachineRegOwner::CachedLocal)];
        let blocks = [block(0, &[]), block(1, &params)];

        assert!(!is_fallthrough_edge(
            MachineBlockId(1),
            &[MachineValue::ReservedReg(MachineReg(7))],
            Some(MachineBlockId(1)),
            &blocks,
        ));
    }
}
