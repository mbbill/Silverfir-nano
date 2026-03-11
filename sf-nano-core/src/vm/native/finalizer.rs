//! Native finalization and target patching.
//!
//! This packages finalized native execution metadata for the runtime-facing
//! native entry ABI. The current bring-up path points every compiled function
//! at one shared native entry and keeps execution in terms of finalized native
//! IR, not interpreter descriptors. Direct ARM64 block emission can replace
//! that shared entry later without changing the backend boundary.

use alloc::vec::Vec;

use crate::error::WasmError;

use super::{
    arch::reference,
    code::{DirectCallPatch, NativeCode},
    entry::NativeEntry,
    helper_meta::{ColdHelperKind, HelperMetadataArena},
    ir::{NativeBlockId, NativeInstKind, NativeProgram, NativeReg, NativeTerminator},
    resolve::ResolvedNativeEntry,
};

#[derive(Clone, Debug)]
struct FinalEntryInfo {
    resolved_index: usize,
    block: Option<NativeBlockId>,
    entry: Option<NativeEntry>,
    cold_helper: Option<ColdHelperKind>,
}

/// Finalization output for one native function.
#[derive(Debug, Default)]
pub struct NativeFinalized {
    pub code: NativeCode,
}

pub fn finalize_native(
    program: &NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeFinalized, WasmError> {
    let _entries = build_final_entry_table(resolved);
    Ok(NativeFinalized {
        code: NativeCode::from_parts(
            Some(shared_native_entry()),
            Vec::new(),
            HelperMetadataArena::new(),
            Vec::<DirectCallPatch>::new(),
            Some(program.clone()),
            count_vregs(program),
        ),
    })
}

#[inline]
fn shared_native_entry() -> NativeEntry {
    reference::shared_native_entry
}

fn build_final_entry_table(resolved: &[ResolvedNativeEntry]) -> Vec<FinalEntryInfo> {
    resolved
        .iter()
        .enumerate()
        .map(|(resolved_index, entry)| FinalEntryInfo {
            resolved_index,
            block: entry.block,
            entry: entry.entry,
            cold_helper: entry.cold_helper,
        })
        .collect()
}

fn count_vregs(program: &NativeProgram) -> u32 {
    let mut next = 0u32;
    for block in &program.blocks {
        for param in &block.params {
            next = next.max(param.0.saturating_add(1));
        }
        for op in &block.ops {
            match &op.kind {
                NativeInstKind::Leaf { args, results, .. } => {
                    next = max_reg_slice(next, args);
                    next = max_reg_slice(next, results);
                }
                NativeInstKind::ReadOperandSlot { dst, .. }
                | NativeInstKind::ReadHotLocal { dst, .. }
                | NativeInstKind::ReadFrameLocal { dst, .. } => {
                    next = next.max(dst.0.saturating_add(1));
                }
                NativeInstKind::WriteOperandSlot { src, .. }
                | NativeInstKind::WriteHotLocal { src, .. }
                | NativeInstKind::WriteFrameLocal { src, .. } => {
                    next = next.max(src.0.saturating_add(1));
                }
                NativeInstKind::CallExternal { args, results, .. }
                | NativeInstKind::CallInternal { args, results, .. } => {
                    next = max_reg_slice(next, args);
                    next = max_reg_slice(next, results);
                }
                NativeInstKind::CallIndirect {
                    index,
                    args,
                    results,
                    ..
                } => {
                    next = next.max(index.0.saturating_add(1));
                    next = max_reg_slice(next, args);
                    next = max_reg_slice(next, results);
                }
            }
        }
        match &block.terminator {
            NativeTerminator::Goto(edge) => {
                for copy in &edge.copies {
                    next = next.max(copy.src.0.saturating_add(1));
                    next = next.max(copy.dst.0.saturating_add(1));
                }
            }
            NativeTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                next = next.max(cond.0.saturating_add(1));
                for edge in [then_edge, else_edge] {
                    for copy in &edge.copies {
                        next = next.max(copy.src.0.saturating_add(1));
                        next = next.max(copy.dst.0.saturating_add(1));
                    }
                }
            }
            NativeTerminator::BrTable { index, entries } => {
                next = next.max(index.0.saturating_add(1));
                for edge in entries {
                    for copy in &edge.copies {
                        next = next.max(copy.src.0.saturating_add(1));
                        next = next.max(copy.dst.0.saturating_add(1));
                    }
                }
            }
            NativeTerminator::Return { values } => {
                next = max_reg_slice(next, values);
            }
            NativeTerminator::TrapUnreachable => {}
        }
    }
    next
}

fn max_reg_slice(mut next: u32, regs: &[NativeReg]) -> u32 {
    for reg in regs {
        next = next.max(reg.0.saturating_add(1));
    }
    next
}
