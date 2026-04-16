//! WebAssembly Module Validator (WASM 2.0, no_std)
//!
//! Pure validation — checks semantic correctness of a parsed module.
//! No side effects (no jump tables, no max_stack_height computation).
//!
//! Validation phases:
//! 1. Type reference bounds checking
//! 2. Function body type checking (instruction operand/result types)
//! 3. Control flow validity (block/loop/if/else/end nesting, branch targets)
//! 4. Export name uniqueness, data/element segment validity

use self::{expressions::ValidationContext, functions::FunctionValidator};
use crate::{
    error::WasmError,
    module::{entities::ElementInit, Module},
    op_decoder,
    utils::limits::Limitable,
    value_type::{HeapType, ValueType},
};
use tracked_alloc::collections::BTreeSet;
use tracked_alloc::string::String;

use super::entities::{Data, Element};

mod expressions;
mod functions;

pub struct Validator<'a> {
    module: &'a Module,
}

impl<'a> Validator<'a> {
    pub fn new(module: &'a Module) -> Self {
        Validator { module }
    }

    pub fn validate(&mut self) -> Result<(), WasmError> {
        if self.module.version() != 1 {
            return Err(WasmError::malformed("unknown binary version"));
        }

        // Phase 1: Validate type references in entities
        self.validate_entity_type_references()?;

        // Phase 1b: Validate start function
        if let Some(start_idx) = self.module.start_function_index() {
            if start_idx >= self.module.functions().len() {
                return Err(WasmError::invalid("unknown function"));
            }
            let func = &self.module.functions()[start_idx];
            let ft = func.func_type();
            if !ft.params().is_empty() || !ft.results().is_empty() {
                return Err(WasmError::invalid("start function"));
            }
        }

        // Phase 2: Validate function bodies
        self.module
            .functions()
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.is_import())
            .try_for_each(|(_func_idx, f)| {
                let spec = f.spec().ok_or_else(|| {
                    WasmError::invalid("Function validation failed: not a local function")
                })?;
                let code = spec.code();
                let mut validator = FunctionValidator::new(self.module, spec)?;
                let mut decoder = op_decoder::Decoder::new(code);
                decoder.add_handler(&mut validator);
                decoder.decode_function()?;
                Ok::<_, WasmError>(())
            })?;

        // Phase 3: Export name uniqueness
        let mut name_pool = BTreeSet::new();
        Self::check_unique_export_names(
            &mut name_pool,
            self.module.functions().iter().map(|f| f.export_names()),
        )?;
        Self::check_unique_export_names(
            &mut name_pool,
            self.module.tables().iter().map(|t| t.export_names()),
        )?;
        Self::check_unique_export_names(
            &mut name_pool,
            self.module.memories().iter().map(|m| m.export_names()),
        )?;
        Self::check_unique_export_names(
            &mut name_pool,
            self.module.globals().iter().map(|g| g.export_names()),
        )?;

        // Phase 4: Validate global initializers
        self.module
            .globals()
            .iter()
            .enumerate()
            .filter(|&(_, g)| !g.is_import())
            .try_for_each(|(global_idx, g)| {
                let global_spec = g
                    .spec()
                    .ok_or_else(|| WasmError::invalid("Global is not a local global"))?;
                let ctx = ValidationContext::global(global_idx);
                let expr_type = global_spec
                    .init_expr()
                    .validate_in_context(self.module, &ctx)?;
                let global_type = global_spec.value_type();
                if !expr_type.can_initialize(&global_type, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                Ok(())
            })?;

        // Phase 5: Validate table initializers
        self.module
            .tables()
            .iter()
            .filter(|t| !t.is_import())
            .try_for_each(|t| -> Result<(), WasmError> {
                let table_type = t.value_type();
                let spec = t.spec();

                if !table_type.is_defaultable() && spec.init_expr().is_none() {
                    return Err(WasmError::invalid("type mismatch"));
                }

                if let Some(init_expr) = spec.init_expr() {
                    let ctx = ValidationContext::table_init();
                    let expr_type = init_expr.validate_in_context(self.module, &ctx)?;
                    if !expr_type.can_initialize(&table_type, self.module.types()) {
                        return Err(WasmError::invalid("type mismatch"));
                    }
                }
                Ok(())
            })?;

        // Phase 6: Validate element segments
        self.module.elements().iter().try_for_each(|e| {
            if let ElementInit::InitExprs { value_type, exprs } = e.get_init() {
                let ctx = if matches!(e, Element::Passive { .. }) {
                    ValidationContext::passive()
                } else {
                    ValidationContext::active()
                };
                for expr in exprs {
                    let expr_type = expr.validate_in_context(self.module, &ctx)?;
                    if !expr_type.can_initialize(value_type, self.module.types()) {
                        return Err(WasmError::invalid(
                            "element init expression must return , got",
                        ));
                    }
                }
            }
            if let Element::Active {
                offset_expr,
                table_index,
                init,
            } = e
            {
                let ctx = ValidationContext::active();
                let offset_type = offset_expr.validate_in_context(self.module, &ctx)?;

                if *table_index < self.module.tables().len() {
                    let table = &self.module.tables()[*table_index];
                    let is_table64 = table.spec().limits().is64;
                    let expected_type = if is_table64 {
                        ValueType::I64
                    } else {
                        ValueType::I32
                    };
                    if offset_type != expected_type {
                        return Err(WasmError::invalid(
                            "element offset expression must return for",
                        ));
                    }
                } else if offset_type != ValueType::I32 {
                    return Err(WasmError::invalid(
                        "element offset expression must return i32".into(),
                    ));
                }

                // Check element type compatibility with table type
                if *table_index < self.module.tables().len() {
                    let table = &self.module.tables()[*table_index];
                    let table_elem_type = table.value_type();
                    let init_type = match init {
                        ElementInit::FunctionIndexes(_) => ValueType::funcref().to_non_nullable(),
                        ElementInit::InitExprs { value_type, .. } => *value_type,
                    };
                    if !init_type.can_initialize(&table_elem_type, self.module.types()) {
                        return Err(WasmError::invalid("type mismatch"));
                    }
                }
            }
            Ok(())
        })?;

        // Phase 7: Validate data segments
        self.module.data().iter().try_for_each(|d| {
            let Data::Active {
                offset_expr,
                memory_index,
                ..
            } = d
            else {
                return Ok(());
            };
            let ctx = ValidationContext::active();
            let offset_type = offset_expr.validate_in_context(self.module, &ctx)?;

            if *memory_index >= self.module.memories().len() {
                return Err(WasmError::invalid("unknown memory"));
            }

            let mem = &self.module.memories()[*memory_index];
            let is_mem64 = mem.spec().limits().is64;
            let expected_type = if is_mem64 {
                ValueType::I64
            } else {
                ValueType::I32
            };

            if offset_type == expected_type {
                Ok(())
            } else {
                Err(WasmError::invalid("data offset expression must return for"))
            }
        })?;

        Ok(())
    }

    fn check_unique_export_names<'b, I>(
        name_pool: &mut BTreeSet<String>,
        items: I,
    ) -> Result<(), WasmError>
    where
        I: Iterator<Item = &'b [String]>,
    {
        for export_names in items {
            for name in export_names {
                if !name_pool.insert(name.clone()) {
                    return Err(WasmError::invalid("export path not unique"));
                }
            }
        }
        Ok(())
    }

    /// Validate type references in globals, tables, elements, and function locals
    fn validate_entity_type_references(&self) -> Result<(), WasmError> {
        let type_count = self.module.types().len();

        for (idx, def_type) in self.module.types().as_slice().iter().enumerate() {
            def_type.validate_type_references(idx, type_count, self.module.types().as_slice())?;
        }

        // Validate function type indices
        for (_idx, func) in self.module.functions().iter().enumerate() {
            let ti = func.type_index() as usize;
            if ti >= type_count {
                return Err(WasmError::invalid("Function : type index out of bounds ()"));
            }
        }

        // Validate that table value types reference valid type indices
        for (_idx, table) in self.module.tables().iter().enumerate() {
            validate_valtype_ref(&table.value_type(), type_count)
                .map_err(|_e| WasmError::invalid("Table"))?;
        }

        // Validate global value types
        for (_idx, global) in self.module.globals().iter().enumerate() {
            validate_valtype_ref(&global.value_type(), type_count)
                .map_err(|_e| WasmError::invalid("Global"))?;
        }

        // Validate element segment value types
        for (_idx, element) in self.module.elements().iter().enumerate() {
            if let ElementInit::InitExprs { value_type, .. } = element.get_init() {
                validate_valtype_ref(value_type, type_count)
                    .map_err(|_e| WasmError::invalid("Element"))?;
            }
        }

        // Validate function locals
        for (_idx, func) in self.module.functions().iter().enumerate() {
            if let Some(spec) = func.spec() {
                for local in spec.locals() {
                    validate_valtype_ref(local, type_count)
                        .map_err(|_e| WasmError::invalid("Local"))?;
                }
            }
        }

        Ok(())
    }
}

/// Validate that any concrete heap type index in a value type is within bounds.
fn validate_valtype_ref(vt: &ValueType, type_count: usize) -> Result<(), WasmError> {
    if let ValueType::Ref(rt) = vt {
        if let HeapType::Concrete(idx) = rt.heap_type {
            if (idx as usize) >= type_count {
                return Err(WasmError::invalid("type index out of bounds ()"));
            }
        }
    }
    Ok(())
}
