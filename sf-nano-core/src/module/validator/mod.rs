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
    value_type::{AbstractHeapType, HeapType, ValueType},
};
use alloc::collections::BTreeSet;
use alloc::string::String;

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
            return Err(WasmError::malformed(alloc::format!(
                "unknown binary version: {}",
                self.module.version(),
            )));
        }

        // Phase 1: Validate type references in entities
        self.validate_entity_type_references()?;

        // Phase 1b: WASM 2.0 limits — at most one memory
        if self.module.memories().len() > 1 {
            return Err(WasmError::invalid("multiple memories".into()));
        }

        // Phase 1c: Validate start function
        if let Some(start_idx) = self.module.start_function_index() {
            if start_idx >= self.module.functions().len() {
                return Err(WasmError::invalid("unknown function".into()));
            }
            let func = &self.module.functions()[start_idx];
            let ft = &self.module.types()[func.type_index() as usize];
            if !ft.params().is_empty() || !ft.results().is_empty() {
                return Err(WasmError::invalid("start function".into()));
            }
        }

        // Phase 2: Validate function bodies
        self.module
            .functions()
            .iter()
            .enumerate()
            .filter(|(_, f)| !f.is_import())
            .try_for_each(|(func_idx, f)| {
                let spec = f.spec().ok_or_else(|| {
                    WasmError::invalid(alloc::format!(
                        "Function {} validation failed: not a local function",
                        func_idx
                    ))
                })?;
                let code = spec.code();
                let mut validator = FunctionValidator::new(self.module, spec).map_err(|e| {
                    WasmError::invalid(alloc::format!(
                        "Function {} validator setup failed: {}",
                        func_idx,
                        e
                    ))
                })?;
                let mut decoder = op_decoder::Decoder::new(code);
                decoder.add_handler(&mut validator);
                decoder.decode_function().map_err(|e| {
                    WasmError::invalid(alloc::format!("Function {} decode failed: {}", func_idx, e))
                })?;
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
                let global_spec = g.spec().ok_or_else(|| {
                    WasmError::invalid(alloc::format!(
                        "Global {} is not a local global",
                        global_idx
                    ))
                })?;
                let ctx = ValidationContext::global(global_idx);
                let expr_type = global_spec
                    .init_expr()
                    .validate_in_context(self.module, &ctx)?;
                let global_type = global_spec.value_type();
                if !is_type_compatible(expr_type, global_type) {
                    return Err(WasmError::invalid("type mismatch".into()));
                }
                Ok(())
            })?;

        // Phase 5: Validate element segments
        self.module.elements().iter().try_for_each(|e| {
            if let ElementInit::InitExprs { value_type, exprs } = e.get_init() {
                let ctx = if matches!(e, Element::Passive { .. }) {
                    ValidationContext::passive()
                } else {
                    ValidationContext::active()
                };
                for expr in exprs {
                    let expr_type = expr.validate_in_context(self.module, &ctx)?;
                    if !is_type_compatible(expr_type, *value_type) {
                        return Err(WasmError::invalid(alloc::format!(
                            "element init expression must return {:?}, got {:?}",
                            value_type,
                            expr_type
                        )));
                    }
                }
            }
            if let Element::Active {
                offset_expr,
                table_index,
                init,
            } = e
            {
                let ctx = ValidationContext::table_init(); // offset uses only imported globals
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
                        return Err(WasmError::invalid(alloc::format!(
                            "element offset expression must return {} for {}",
                            expected_type,
                            if is_table64 { "table64" } else { "table" }
                        )));
                    }
                } else if offset_type != ValueType::I32 {
                    return Err(WasmError::invalid(
                        "element offset expression must return i32".into(),
                    ));
                }

                // Check element type compatibility with table type
                if *table_index < self.module.tables().len() {
                    if let ElementInit::InitExprs { value_type, .. } = init {
                        let table = &self.module.tables()[*table_index];
                        let table_elem_type = table.value_type();
                        if !is_type_compatible(*value_type, table_elem_type) {
                            return Err(WasmError::invalid("type mismatch".into()));
                        }
                    }
                }
            }
            Ok(())
        })?;

        // Phase 6: Validate data segments
        self.module.data().iter().try_for_each(|d| {
            let Data::Active {
                offset_expr,
                memory_index,
                ..
            } = d
            else {
                return Ok(());
            };
            let ctx = ValidationContext::table_init(); // offset uses only imported globals
            let offset_type = offset_expr.validate_in_context(self.module, &ctx)?;

            if *memory_index >= self.module.memories().len() {
                return Err(WasmError::invalid("unknown memory".into()));
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
                Err(WasmError::invalid(alloc::format!(
                    "data offset expression must return {} for {}",
                    expected_type,
                    if is_mem64 { "memory64" } else { "memory" }
                )))
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
                    return Err(WasmError::invalid("export path not unique".into()));
                }
            }
        }
        Ok(())
    }

    /// Validate type references in globals, tables, elements, and function locals
    fn validate_entity_type_references(&self) -> Result<(), WasmError> {
        let type_count = self.module.types().len();

        // Validate function type indices
        for (idx, func) in self.module.functions().iter().enumerate() {
            let ti = func.type_index() as usize;
            if ti >= type_count {
                return Err(WasmError::invalid(alloc::format!(
                    "Function {}: type index {} out of bounds ({})",
                    idx,
                    ti,
                    type_count
                )));
            }
        }

        // Validate that table value types reference valid type indices
        for (idx, table) in self.module.tables().iter().enumerate() {
            validate_valtype_ref(&table.value_type(), type_count)
                .map_err(|e| WasmError::invalid(alloc::format!("Table {}: {}", idx, e)))?;
        }

        // Validate element segment value types
        for (idx, element) in self.module.elements().iter().enumerate() {
            if let ElementInit::InitExprs { value_type, .. } = element.get_init() {
                validate_valtype_ref(value_type, type_count)
                    .map_err(|e| WasmError::invalid(alloc::format!("Element {}: {}", idx, e)))?;
            }
        }

        Ok(())
    }
}

/// WASM 2.0 type compatibility with reference subtyping.
/// Unknown (bot) is compatible with everything.
/// Ref subtyping rules:
///   - non-nullable ≤ nullable (for compatible heap types)
///   - Concrete(idx) ≤ Abstract(Func) (all concrete types are function types)
fn is_type_compatible(actual: ValueType, expected: ValueType) -> bool {
    if actual == expected || actual == ValueType::Unknown || expected == ValueType::Unknown {
        return true;
    }
    // Reference subtyping
    if let (ValueType::Ref(actual_ref), ValueType::Ref(expected_ref)) = (actual, expected) {
        // Non-nullable is subtype of nullable (not the reverse)
        if !expected_ref.nullable && actual_ref.nullable {
            return false;
        }
        // Same heap type is compatible
        if actual_ref.heap_type == expected_ref.heap_type {
            return true;
        }
        // Concrete(idx) is a subtype of Abstract(Func) — all defined types are func types in WASM 2.0
        if let (HeapType::Concrete(_), HeapType::Abstract(AbstractHeapType::Func)) =
            (actual_ref.heap_type, expected_ref.heap_type)
        {
            return true;
        }
    }
    false
}

/// Validate that any concrete heap type index in a value type is within bounds.
fn validate_valtype_ref(vt: &ValueType, type_count: usize) -> Result<(), WasmError> {
    if let ValueType::Ref(rt) = vt {
        if let crate::value_type::HeapType::Concrete(idx) = rt.heap_type {
            if (idx as usize) >= type_count {
                return Err(WasmError::invalid(alloc::format!(
                    "type index {} out of bounds ({})",
                    idx,
                    type_count
                )));
            }
        }
    }
    Ok(())
}
