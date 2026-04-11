//! Type context for type resolution and equivalence checking
//!
//! This module provides a type context that ensures type indices are always
//! resolved within the correct module scope, as required by the WebAssembly spec.
//!
//! TypeContext uses Rc internally, so cloning is cheap (just incrementing reference counts).

use crate::collections;

use alloc::rc::Rc;

use crate::module::type_defs::FunctionType;
use crate::value_type::{HeapType, ValueType};

/// Type context provides access to type definitions within a module
///
/// This ensures type indices are always resolved within the correct module scope.
/// Per WebAssembly spec, each module instance maintains its own type array,
/// and type indices are module-local.
///
/// ## Performance
/// TypeContext uses `Rc` internally, so cloning is cheap (O(1) - just incrementing
/// reference counts). This allows passing TypeContext by value without performance concerns.
#[derive(Clone)]
pub struct TypeContext {
    types: Rc<[Rc<FunctionType>]>,
}

impl TypeContext {
    /// Create a new type context from type definitions
    pub fn new(types: collections::Vec<Rc<FunctionType>>) -> Self {
        let types: alloc::vec::Vec<_> = types.into();
        Self {
            types: types.into(),
        }
    }

    /// Create an empty type context (for modules with no type section)
    pub fn empty() -> Self {
        Self { types: Rc::new([]) }
    }

    /// Get a type definition by index
    pub fn get(&self, idx: u32) -> Option<&Rc<FunctionType>> {
        self.types.get(idx as usize)
    }

    /// Get all type definitions as a slice
    pub fn as_slice(&self) -> &[Rc<FunctionType>] {
        &self.types
    }

    /// Number of types in this context
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if this context is empty
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Check if two function types (by index) are equivalent
    ///
    /// In WASM 2.0, all defined types are function types. Two function types
    /// are equivalent if they have the same parameter and result types.
    pub fn types_equivalent(&self, idx1: u32, idx2: u32) -> bool {
        if idx1 == idx2 {
            return true;
        }

        let Some(type1) = self.get(idx1) else {
            return false;
        };
        let Some(type2) = self.get(idx2) else {
            return false;
        };

        function_types_structurally_equal(type1, type2)
    }
}

impl core::ops::Deref for TypeContext {
    type Target = [Rc<FunctionType>];

    fn deref(&self) -> &Self::Target {
        &self.types
    }
}

impl AsRef<[Rc<FunctionType>]> for TypeContext {
    fn as_ref(&self) -> &[Rc<FunctionType>] {
        &self.types
    }
}

impl core::fmt::Debug for TypeContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TypeContext")
            .field("num_types", &self.types.len())
            .finish()
    }
}

// ============================================================================
// Cross-Module Type Equivalence Functions
// ============================================================================

/// Check if two function types are equivalent for cross-module import matching
///
/// This handles cross-module imports where type indices might differ between modules.
/// Two function types are equivalent if:
/// - Same number of parameters and results
/// - Each parameter/result type is equivalent (recursively for reference types)
///
/// # Arguments
/// * `export_type` - The function type from the exporting module
/// * `import_type` - The function type from the importing module
/// * `export_type_ctx` - The type context of the exporting module
pub fn check_function_types_equivalent(
    export_type: &FunctionType,
    import_type: &FunctionType,
    export_type_ctx: &TypeContext,
) -> bool {
    if export_type.params().len() != import_type.params().len()
        || export_type.results().len() != import_type.results().len()
    {
        return false;
    }

    for (exp_param, imp_param) in export_type.params().iter().zip(import_type.params().iter()) {
        if !value_types_equivalent_cross_module(exp_param, imp_param, export_type_ctx) {
            return false;
        }
    }

    for (exp_result, imp_result) in export_type
        .results()
        .iter()
        .zip(import_type.results().iter())
    {
        if !value_types_equivalent_cross_module(exp_result, imp_result, export_type_ctx) {
            return false;
        }
    }

    true
}

/// Check if two value types are equivalent across modules
///
/// For cross-module type matching, type indices are module-local so we need
/// special handling for concrete reference types.
pub fn value_types_equivalent_cross_module(
    exp_type: &ValueType,
    imp_type: &ValueType,
    export_type_ctx: &TypeContext,
) -> bool {
    match (exp_type, imp_type) {
        (ValueType::I32, ValueType::I32) => true,
        (ValueType::I64, ValueType::I64) => true,
        (ValueType::F32, ValueType::F32) => true,
        (ValueType::F64, ValueType::F64) => true,
        (ValueType::V128, ValueType::V128) => true,

        (ValueType::Ref(exp_ref), ValueType::Ref(imp_ref)) => {
            if exp_ref.nullable != imp_ref.nullable {
                return false;
            }

            match (&exp_ref.heap_type, &imp_ref.heap_type) {
                (HeapType::Abstract(a1), HeapType::Abstract(a2)) => a1 == a2,
                (HeapType::Concrete(exp_idx), HeapType::Concrete(imp_idx)) => {
                    if exp_idx == imp_idx {
                        true
                    } else {
                        export_type_ctx.types_equivalent(*exp_idx, *imp_idx)
                    }
                }
                _ => false,
            }
        }

        _ => false,
    }
}

/// Check if two function types are structurally equal (same params and results)
fn function_types_structurally_equal(f1: &FunctionType, f2: &FunctionType) -> bool {
    f1.params() == f2.params() && f1.results() == f2.results()
}
