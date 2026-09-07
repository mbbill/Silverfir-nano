//! Linking fixtures for interpreter dispatch-hook tests.
use crate::module::{type_context::TypeContext, type_defs::FunctionType};
use crate::vm::imports::{Import, ImportValue, ImportedFunction};
use crate::vm::value::RefValue;
use alloc::string::ToString;

// The instance-table Miri fixtures inspect memory inside a scoped token
// materialization. Production embedding access uses MemoryView guards instead.
impl super::InterpInstance {
    pub(crate) fn memory(&self) -> Option<&[u8]> {
        let memory = self.shared_memory_at(0)?;
        Some(unsafe { core::slice::from_raw_parts(memory.memory_ptr(), memory.memory_len()) })
    }

    pub(crate) fn memory_mut(&mut self) -> Option<&mut [u8]> {
        let memory = self.shared_memory_at(0)?;
        Some(unsafe { core::slice::from_raw_parts_mut(memory.memory_ptr(), memory.memory_len()) })
    }
}

impl Import {
    pub(crate) fn linked_func_typed_with_context_and_index(
        module: &str,
        name: &str,
        handle: RefValue,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: TypeContext,
    ) -> Self {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Linked {
                handle,
                func_type,
                type_index,
                type_ctx: Some(type_ctx),
            }),
        }
    }
}
