//! Export binding metadata comes from the parsed module and live interpreter state.
use super::exec::InterpInstance;
use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::entities::TagDef;
use crate::vm::imports::{
    memory_export, table_export, ImportValue, ImportedFunction, ImportedGlobal,
    ImportedGlobalState, ImportedTableState, ImportedTagState,
};
use alloc::string::String;

impl InterpInstance {
    pub(crate) fn export_names(&self) -> Vec<String> {
        let module = self.module();
        module
            .functions()
            .iter()
            .flat_map(|v| v.export_names())
            .chain(module.tables().iter().flat_map(|v| v.export_names()))
            .chain(module.memories().iter().flat_map(|v| v.export_names()))
            .chain(module.globals().iter().flat_map(|v| v.export_names()))
            .chain(module.tags().iter().flat_map(|v| v.export_names()))
            .cloned()
            .collect()
    }

    pub(crate) fn export_value(&self, name: &str) -> Result<Option<ImportValue>, WasmError> {
        let module = self.module();
        let missing = || WasmError::invalid("exported entity is unavailable");
        if let Some(idx) = self.find_export(name) {
            let function = &module.functions()[idx];
            return Ok(Some(ImportValue::Func(ImportedFunction::Linked {
                handle: self.function_handle_at(idx).ok_or_else(missing)?,
                func_type: function.func_type().clone(),
                type_index: function.type_index(),
                type_ctx: Some(module.types().clone()),
            })));
        }
        if let Some(idx) = module
            .tables()
            .iter()
            .position(|v| v.export_names().iter().any(|n| n == name))
        {
            return table_export(ImportedTableState {
                table: self.table_state_at(idx).ok_or_else(missing)?,
                type_ctx: Some(module.types().clone()),
            })
            .map(Some);
        }
        if let Some(idx) = module
            .memories()
            .iter()
            .position(|v| v.export_names().iter().any(|n| n == name))
        {
            return memory_export(self.shared_memory_at(idx).ok_or_else(missing)?).map(Some);
        }
        if let Some(idx) = self.find_export_global(name) {
            let global = self.global_state_at(idx).ok_or_else(missing)?;
            let mutable = global.mutable;
            return Ok(Some(ImportValue::Global(
                ImportedGlobal::State(ImportedGlobalState {
                    global,
                    type_ctx: Some(module.types().clone()),
                }),
                mutable,
            )));
        }
        if let Some((idx, tag)) = module
            .tags()
            .iter()
            .enumerate()
            .find(|(_, v)| v.export_names().iter().any(|n| n == name))
        {
            return Ok(Some(ImportValue::Tag(ImportedTagState {
                handle: self.tag_identity_at(idx).ok_or_else(missing)?,
                func_type: tag.func_type().clone(),
                type_index: match tag.def() {
                    TagDef::Local(spec) => spec.type_index(),
                    TagDef::Import { type_index, .. } => *type_index,
                },
                type_ctx: Some(module.types().clone()),
            })));
        }
        Ok(None)
    }
}
