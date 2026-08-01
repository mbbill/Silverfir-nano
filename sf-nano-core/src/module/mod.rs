use tracked_alloc::string::String;

use crate::collections;
use crate::op_decoder::{Decoder, Immediate, OpStream, OpcodeHandler};
use crate::opcodes::{Opcode, WasmOpcode};

#[cfg(not(sf_has_simd))]
use self::type_defs::{CompositeType, StorageType};
#[cfg(not(sf_has_simd))]
use crate::value_type::ValueType;
#[cfg(not(sf_has_simd))]
use entities::FunctionType;
use entities::{Data, Element, ElementInit, Function, Global, Memory, Table, Tag};

pub mod builder;
pub mod entities;
pub(crate) mod parser;
pub mod type_context;
pub mod type_defs;
#[cfg(sf_module_validator)]
pub mod validator;

use crate::error::WasmError;

#[derive(Debug)]
pub struct Module {
    name: String,
    binary_version: u32,
    types: type_context::TypeContext,
    functions: collections::Vec<Function>,
    tables: collections::Vec<Table>,
    memories: collections::Vec<Memory>,
    globals: collections::Vec<Global>,
    tags: collections::Vec<Tag>,
    elements: collections::Vec<Element>,
    data: collections::Vec<Data>,
    start_func_index: Option<usize>,
    data_count: Option<usize>,
}

impl Module {
    pub fn new(name: &str, bin: &[u8]) -> Result<Self, WasmError> {
        let module = parser::parse_module(name, bin)?;
        module.ensure_simd_supported()?;
        Ok(module)
    }

    pub(crate) fn ensure_simd_supported(&self) -> Result<(), WasmError> {
        #[cfg(not(sf_has_simd))]
        if self.requires_simd() {
            return Err(WasmError::invalid("SIMD is not supported on this CPU"));
        }
        Ok(())
    }

    #[cfg(not(sf_has_simd))]
    pub(crate) fn requires_simd(&self) -> bool {
        self.types
            .as_slice()
            .iter()
            .any(|def_type| match &def_type.composite {
                CompositeType::Func(func_type) => function_type_requires_simd(func_type),
                CompositeType::Struct(struct_type) => struct_type
                    .fields
                    .iter()
                    .any(|field| storage_type_requires_simd(&field.storage)),
                CompositeType::Array(array_type) => {
                    storage_type_requires_simd(&array_type.element.storage)
                }
            })
            || self
                .functions
                .iter()
                .any(|func| function_requires_simd(func))
            || self
                .globals
                .iter()
                .any(|global| matches!(global.value_type(), ValueType::V128))
            || self
                .tags
                .iter()
                .any(|tag| function_type_requires_simd(tag.func_type()))
            || self.elements.iter().any(element_requires_simd)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> u32 {
        self.binary_version
    }

    pub fn types(&self) -> &type_context::TypeContext {
        &self.types
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn memories(&self) -> &[Memory] {
        &self.memories
    }

    pub fn tables(&self) -> &[Table] {
        &self.tables
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn data(&self) -> &[Data] {
        &self.data
    }

    pub fn start_function_index(&self) -> Option<usize> {
        self.start_func_index
    }

    pub fn data_count(&self) -> Option<usize> {
        self.data_count
    }

    /// Functions whose identity may leave direct-call-only code.
    ///
    /// Exactly the module's declared-reference set: function indices found
    /// in element segments, exports, and table/global constant initializers.
    /// A valid module's code-section `ref.func` operands are required to be
    /// a subset of this set, so function bodies are never scanned here; the
    /// validator enforces that requirement with this same set.
    pub(crate) fn escapable_functions(&self) -> Result<collections::Vec<bool>, WasmError> {
        let mut escapable = collections::vec![false; self.functions.len()];

        for (index, function) in self.functions.iter().enumerate() {
            if !function.export_names().is_empty() {
                escapable[index] = true;
            }
        }

        for element in &self.elements {
            match element.get_init() {
                ElementInit::FunctionIndexes(indices) => {
                    for &index in indices {
                        mark_escapable(&mut escapable, index)?;
                    }
                }
                ElementInit::InitExprs { exprs, .. } => {
                    for expr in exprs {
                        scan_ref_funcs(expr, &mut escapable)?;
                    }
                }
            }
        }

        for table in &self.tables {
            if let Some(expr) = table.spec().init_expr() {
                scan_ref_funcs(expr, &mut escapable)?;
            }
        }
        for global in &self.globals {
            if let Some(spec) = global.spec() {
                scan_ref_funcs(spec.init_expr(), &mut escapable)?;
            }
        }

        Ok(escapable)
    }

    /// Consume the module, returning all internal fields.
    pub fn into_parts(
        self,
    ) -> (
        type_context::TypeContext,
        collections::Vec<Function>,
        collections::Vec<Table>,
        collections::Vec<Memory>,
        collections::Vec<Global>,
        collections::Vec<Tag>,
        collections::Vec<Element>,
        collections::Vec<Data>,
        Option<usize>,
    ) {
        (
            self.types,
            self.functions,
            self.tables,
            self.memories,
            self.globals,
            self.tags,
            self.elements,
            self.data,
            self.start_func_index,
        )
    }
}

fn mark_escapable(escapable: &mut [bool], index: usize) -> Result<(), WasmError> {
    let slot = escapable
        .get_mut(index)
        .ok_or_else(|| WasmError::invalid("ref.func: function index out of range"))?;
    *slot = true;
    Ok(())
}

struct RefFuncScan<'a> {
    escapable: &'a mut [bool],
}

impl OpcodeHandler for RefFuncScan<'_> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            if let (WasmOpcode::OP(Opcode::REF_FUNC), Immediate::FunctionIndex(function_index)) =
                (decoded.wasm_op, &decoded.imm)
            {
                mark_escapable(self.escapable, *function_index as usize)?;
            }
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        Ok(())
    }
}

fn scan_ref_funcs(code: &[u8], escapable: &mut [bool]) -> Result<(), WasmError> {
    let mut scan = RefFuncScan { escapable };
    let mut decoder = Decoder::new(code);
    decoder.add_handler(&mut scan);
    decoder.decode_function()
}

#[cfg(not(sf_has_simd))]
#[inline]
fn function_type_requires_simd(func_type: &FunctionType) -> bool {
    func_type
        .params()
        .iter()
        .chain(func_type.results().iter())
        .any(|value_type| matches!(value_type, ValueType::V128))
}

#[cfg(not(sf_has_simd))]
#[inline]
fn storage_type_requires_simd(storage: &StorageType) -> bool {
    matches!(storage, StorageType::Val(ValueType::V128))
}

#[cfg(not(sf_has_simd))]
#[inline]
fn function_requires_simd(func: &Function) -> bool {
    function_type_requires_simd(func.func_type())
        || func.spec().is_some_and(|spec| {
            spec.locals()
                .iter()
                .any(|value_type| matches!(value_type, ValueType::V128))
        })
}

#[cfg(not(sf_has_simd))]
#[inline]
fn element_requires_simd(element: &Element) -> bool {
    matches!(
        element.get_init(),
        ElementInit::InitExprs {
            value_type: ValueType::V128,
            ..
        }
    )
}

#[cfg(test)]
mod tests {
    use super::Module;

    #[test]
    fn parser_rejects_simd_value_types() {
        let wasm = wat::parse_str("(module (type (func (param v128))))")
            .expect("wat should encode a module with a v128 parameter");

        #[cfg(not(sf_has_simd))]
        {
            let err = Module::new("simd-types", &wasm)
                .expect_err("non-SIMD builds should reject v128 value types");
            assert_eq!(
                err,
                crate::WasmError::invalid("SIMD is not supported on this CPU")
            );
        }

        #[cfg(sf_has_simd)]
        {
            Module::new("simd-types", &wasm)
                .expect("SIMD-enabled builds should accept v128 type syntax");
        }
    }
}
