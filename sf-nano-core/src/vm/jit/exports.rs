//! Export binding metadata comes from the instantiated JIT module.
use super::instantiate::{ExportKind, JitInstanceLease};
use crate::collections::Vec;
use crate::error::WasmError;
use crate::vm::imports::{
    memory_export, table_export, ImportValue, ImportedFunction, ImportedGlobal, ImportedTagState,
};
use alloc::string::String;

impl JitInstanceLease {
    pub(crate) fn export_names(&self) -> Vec<String> {
        self.store()
            .exports()
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect()
    }

    pub(crate) fn export_value(&self, name: &str) -> Result<Option<ImportValue>, WasmError> {
        let store = self.store();
        let Some((_, kind, idx)) = store.exports().iter().find(|(n, _, _)| n == name) else {
            return Ok(None);
        };
        let module = store.module();
        let missing = || WasmError::invalid("exported entity is unavailable");
        let value = match kind {
            ExportKind::Func => {
                let function = module.functions.get(*idx).ok_or_else(missing)?;
                ImportValue::Func(ImportedFunction::Linked {
                    handle: self.function_handle_at(*idx).ok_or_else(missing)?,
                    func_type: function.func_type().clone(),
                    type_index: function.type_index(),
                    type_ctx: Some(module.types.clone()),
                })
            }
            ExportKind::Table => {
                table_export(self.shared_table_state_at(*idx).ok_or_else(missing)?)?
            }
            ExportKind::Memory => memory_export(self.shared_memory_at(*idx).ok_or_else(missing)?)?,
            ExportKind::Global => {
                let state = self.shared_global_state_at(*idx).ok_or_else(missing)?;
                let mutable = state.global.mutable;
                ImportValue::Global(ImportedGlobal::State(state), mutable)
            }
            ExportKind::Tag => {
                let tag = module.tags.get(*idx).ok_or_else(missing)?;
                ImportValue::Tag(ImportedTagState {
                    handle: tag.handle,
                    func_type: module
                        .types
                        .get_function_type(tag.type_index)
                        .ok_or_else(missing)?
                        .as_ref()
                        .clone(),
                    type_index: tag.type_index,
                    type_ctx: Some(module.types.clone()),
                })
            }
        };
        Ok(Some(value))
    }
}
