//! Reference-backend code packaging.
//!
//! This is the live bring-up compilation target for `NativeProgram`. It keeps
//! code packaging and entry selection inside the architecture layer so the
//! generic native pipeline does not depend on `reference` details.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::native::{
    code::{DirectCallPatch, NativeCode},
    helper_meta::{ColdHelperKind, HelperMetadataArena},
    ir::NativeBlockId,
    resolve::ResolvedNativeEntry,
};

use super::entry::shared_native_entry;

#[derive(Clone, Debug)]
struct FinalEntryInfo {
    resolved_index: usize,
    block: Option<NativeBlockId>,
    entry: Option<crate::vm::native::entry::NativeEntry>,
    cold_helper: Option<ColdHelperKind>,
}

pub fn compile_program(
    program: &crate::vm::native::ir::NativeProgram,
    resolved: &[ResolvedNativeEntry],
) -> Result<NativeCode, WasmError> {
    let _entries = build_final_entry_table(resolved);
    Ok(NativeCode::from_parts(
        Some(shared_native_entry),
        None,
        Vec::new(),
        None,
        HelperMetadataArena::new(),
        Vec::<DirectCallPatch>::new(),
        Some(program.clone()),
    ))
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
