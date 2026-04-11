//! WebAssembly 2.0 Module Parser (no_std)
//!
//! Handles binary format parsing and structural validation only.
//! Validates binary format correctness, structural requirements, and basic range checks.
//! Returns **malformed** errors for structural/format issues.

use crate::collections;

use alloc::rc::Rc;
use alloc::string::{String, ToString};

use crate::{
    constants,
    error::WasmError,
    module::{
        entities::{
            Bytecode, ConstExpr, Data, Element, ElementInit, Function, FunctionType, Global,
            Memory, Table,
        },
        type_context::TypeContext,
        Module,
    },
    opcodes::Opcode,
    utils::{limits::Limits, payload::Payload},
    value_type::ValueType,
};

// ============================================================================
// Section and External Kind enums
// ============================================================================

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum WasmSection {
    Custom = 0,
    Type = 1,
    Import = 2,
    Function = 3,
    Table = 4,
    Memory = 5,
    Global = 6,
    Export = 7,
    Start = 8,
    Element = 9,
    Code = 10,
    Data = 11,
    DataCount = 12,
}

impl TryFrom<u8> for WasmSection {
    type Error = WasmError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(WasmSection::Custom),
            1 => Ok(WasmSection::Type),
            2 => Ok(WasmSection::Import),
            3 => Ok(WasmSection::Function),
            4 => Ok(WasmSection::Table),
            5 => Ok(WasmSection::Memory),
            6 => Ok(WasmSection::Global),
            7 => Ok(WasmSection::Export),
            8 => Ok(WasmSection::Start),
            9 => Ok(WasmSection::Element),
            10 => Ok(WasmSection::Code),
            11 => Ok(WasmSection::Data),
            12 => Ok(WasmSection::DataCount),
            _ => Err(WasmError::malformed("Invalid section id")),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExternalKind {
    Function = 0,
    Table = 1,
    Memory = 2,
    Global = 3,
}

impl TryFrom<u8> for ExternalKind {
    type Error = WasmError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ExternalKind::Function),
            1 => Ok(ExternalKind::Table),
            2 => Ok(ExternalKind::Memory),
            3 => Ok(ExternalKind::Global),
            _ => Err(WasmError::malformed("Invalid external kind")),
        }
    }
}

// ============================================================================
// Constants
// ============================================================================

const WASM_MAGIC_NUMBER: [u8; 4] = [0x00, 0x61, 0x73, 0x6d]; // "\0asm"

// Element section kind constants
const ELEM_ACTIVE_FUNCIDX: u8 = 0x00;
const ELEM_PASSIVE_FUNCIDX: u8 = 0x01;
const ELEM_ACTIVE_TABLEIDX_FUNCIDX: u8 = 0x02;
const ELEM_DECLARATIVE_FUNCIDX: u8 = 0x03;
const ELEM_ACTIVE_EXPR: u8 = 0x04;
const ELEM_PASSIVE_EXPR: u8 = 0x05;
const ELEM_ACTIVE_TABLEIDX_EXPR: u8 = 0x06;
const ELEM_DECLARATIVE_EXPR: u8 = 0x07;

// Data section kind constants
const DATA_ACTIVE: u32 = 0x00;
const DATA_PASSIVE: u32 = 0x01;
const DATA_ACTIVE_MEMIDX: u32 = 0x02;

// ============================================================================
// Main parse entry point
// ============================================================================

pub(crate) fn parse_module(name: &str, bin: &[u8]) -> Result<Module, WasmError> {
    let mut payload: Payload = bin.into();

    let magic = payload.read_bytes(4)?;
    if magic != WASM_MAGIC_NUMBER {
        return Err(WasmError::malformed("Invalid magic number"));
    }

    let mut previous_section = WasmSection::Custom;
    let mut check_section_order = |current: WasmSection| -> Result<(), WasmError> {
        if current == WasmSection::Custom {
            return Ok(());
        }

        let valid = match (previous_section, current) {
            (WasmSection::Custom, _) => true,
            (WasmSection::DataCount, WasmSection::Code | WasmSection::Data) => true,
            _ => {
                fn section_order(s: WasmSection) -> u8 {
                    match s {
                        WasmSection::Custom => 0,
                        WasmSection::Type => 1,
                        WasmSection::Import => 2,
                        WasmSection::Function => 3,
                        WasmSection::Table => 4,
                        WasmSection::Memory => 5,
                        WasmSection::Global => 6,
                        WasmSection::Export => 7,
                        WasmSection::Start => 8,
                        WasmSection::Element => 9,
                        WasmSection::Code => 10,
                        WasmSection::Data => 11,
                        WasmSection::DataCount => 12,
                    }
                }
                section_order(previous_section) < section_order(current)
            }
        };

        if !valid {
            return Err(WasmError::malformed("Invalid section order"));
        }

        previous_section = current;
        Ok(())
    };

    let version = u32::from_le_bytes(
        payload
            .read_bytes(4)?
            .try_into()
            .map_err(|_| WasmError::malformed("Invalid version number"))?,
    );

    if version != 1 {
        return Err(WasmError::malformed(
            "Invalid WebAssembly version: . Expected version 1.",
        ));
    }

    let mut types: collections::Vec<Rc<FunctionType>> = collections::Vec::new();
    let mut functions: collections::Vec<Function> = collections::Vec::new();
    let mut tables: collections::Vec<Table> = collections::Vec::new();
    let mut memories: collections::Vec<Memory> = collections::Vec::new();
    let mut globals: collections::Vec<Global> = collections::Vec::new();
    let mut elements: collections::Vec<Element> = collections::Vec::new();
    let mut data_segments: collections::Vec<Data> = collections::Vec::new();
    let mut start_func_index: Option<usize> = None;
    let mut data_count: Option<usize> = None;
    let mut export_names: collections::Vec<String> = collections::Vec::new();

    loop {
        if payload.is_empty() {
            break;
        }
        let section_id = payload.read_u8()?;
        let section_len = payload.read_leb128_u32()? as usize;
        let payload_offset = payload.position();
        let mut section_payload: Payload = payload.advance_and_split_at(section_len)?.into();
        let section = WasmSection::try_from(section_id)?;

        check_section_order(section)?;

        match section {
            WasmSection::Custom => {
                parse_custom_section(&mut section_payload)?;
            }
            WasmSection::Type => {
                types = parse_type_section(&mut section_payload)?;
            }
            WasmSection::Import => {
                parse_import_section(
                    &types,
                    &mut functions,
                    &mut tables,
                    &mut memories,
                    &mut globals,
                    &mut section_payload,
                )?;
            }
            WasmSection::Function => {
                parse_function_section(&types, &mut functions, &mut section_payload)?;
            }
            WasmSection::Table => {
                parse_table_section(&mut tables, &mut section_payload)?;
            }
            WasmSection::Memory => {
                parse_memory_section(&mut memories, &mut section_payload)?;
            }
            WasmSection::Global => {
                parse_global_section(&mut globals, &mut section_payload)?;
            }
            WasmSection::Export => {
                parse_export_section(
                    &mut functions,
                    &mut tables,
                    &mut memories,
                    &mut globals,
                    &mut export_names,
                    &mut section_payload,
                )?;
            }
            WasmSection::Start => {
                let index = section_payload.read_leb128_u32()? as usize;
                if index >= functions.len() {
                    return Err(WasmError::malformed("Invalid start function index"));
                }
                start_func_index = Some(index);
            }
            WasmSection::Element => {
                elements = parse_vec(&mut section_payload, parse_element)?;
            }
            WasmSection::DataCount => {
                data_count = Some(section_payload.read_leb128_u32()? as usize);
            }
            WasmSection::Code => {
                parse_code_section(&mut functions, &mut section_payload, payload_offset)?;
            }
            WasmSection::Data => {
                data_segments = parse_data_section(data_count, &mut section_payload)?;
            }
        }
        if !section_payload.is_empty() {
            return Err(WasmError::malformed("Invalid section length"));
        }
    }

    // Validate data count consistency
    if let Some(dc) = data_count {
        if dc > 0 && data_segments.is_empty() {
            return Err(WasmError::malformed(
                "data count and data section have inconsistent lengths".into(),
            ));
        }
    }

    Ok(Module {
        name: name.into(),
        binary_version: version,
        types: TypeContext::new(types),
        functions,
        tables,
        memories,
        globals,
        elements,
        data: data_segments,
        start_func_index,
        data_count,
    })
}

// ============================================================================
// Helpers
// ============================================================================

fn parse_vec<'a, T: 'a>(
    payload: &mut Payload<'a>,
    parser: fn(&mut Payload<'a>) -> Result<T, WasmError>,
) -> Result<collections::Vec<T>, WasmError> {
    let count = payload.read_leb128_u32()?;
    let mut vec = collections::Vec::with_capacity(count as usize);
    for _ in 0..count {
        vec.push(parser(payload)?);
    }
    Ok(vec)
}

fn parse_indices(payload: &mut Payload) -> Result<usize, WasmError> {
    Ok(payload.read_leb128_u32()? as usize)
}

fn parse_valtype(payload: &mut Payload) -> Result<ValueType, WasmError> {
    ValueType::parse(payload)
}

fn parse_resulttype(payload: &mut Payload) -> Result<collections::Vec<ValueType>, WasmError> {
    parse_vec(payload, parse_valtype)
}

fn parse_limits(payload: &mut Payload) -> Result<Limits, WasmError> {
    let tag = payload.read_u8()?;
    match tag {
        0x00 => {
            let min = payload.read_leb128_u32()? as usize;
            Ok(Limits::new(min, None)?)
        }
        0x01 => {
            let min = payload.read_leb128_u32()? as usize;
            let max = payload.read_leb128_u32()? as usize;
            Ok(Limits::new(min, Some(max))?)
        }
        0x04 => {
            let min = payload.read_leb128_u64()? as usize;
            Ok(Limits::new_64(min, None)?)
        }
        0x05 => {
            let min = payload.read_leb128_u64()? as usize;
            let max = payload.read_leb128_u64()? as usize;
            Ok(Limits::new_64(min, Some(max))?)
        }
        _ => Err(WasmError::malformed("malformed table limits flag: 0x")),
    }
}

fn parse_constexpr(payload: &mut Payload) -> Result<ConstExpr, WasmError> {
    use Opcode::*;

    let mut code: Payload = payload.remaining_slice().into();
    let position = 'outer: loop {
        if code.is_empty() {
            break Err(WasmError::malformed("Empty constexpr"));
        }
        let op: Opcode = code.read_u8()?.try_into()?;
        match op {
            I32_CONST => {
                code.read_leb128_i32()?;
            }
            I64_CONST => {
                code.read_leb128_i64()?;
            }
            F32_CONST => {
                code.read_f32()?;
            }
            F64_CONST => {
                code.read_f64()?;
            }
            REF_NULL => {
                code.read_u8()?;
            }
            REF_FUNC | GLOBAL_GET => {
                code.read_leb128_u32()?;
            }
            // Extended constant expressions (WASM 2.0)
            I32_ADD | I32_SUB | I32_MUL | I64_ADD | I64_SUB | I64_MUL => {
                // Binary operations have no immediates
            }
            END => {
                break 'outer Ok(code.position());
            }
            _ => {
                break 'outer Err(WasmError::malformed("Invalid opcode in constexpr"));
            }
        }
    }?;
    Ok(payload.advance_and_split_at(position)?.into())
}

fn parse_globaltype(payload: &mut Payload) -> Result<(ValueType, bool), WasmError> {
    let value_type = parse_valtype(payload)?;
    let is_mutable = match payload.read_u8()? {
        0 => false,
        1 => true,
        _ => {
            return Err(WasmError::malformed(
                "malformed global mutable value".into(),
            ));
        }
    };
    Ok((value_type, is_mutable))
}

fn parse_tabletype(payload: &mut Payload) -> Result<(ValueType, Limits), WasmError> {
    let value_type = parse_valtype(payload)?;
    if !value_type.is_ref() {
        return Err(WasmError::invalid("Invalid table type"));
    }
    let limits = parse_limits(payload)?;
    Ok((value_type, limits))
}

// ============================================================================
// Section parsers
// ============================================================================

fn parse_type_section(
    payload: &mut Payload,
) -> Result<collections::Vec<Rc<FunctionType>>, WasmError> {
    let count = payload.read_leb128_u32()?;
    let mut types = collections::Vec::with_capacity(count as usize);

    for _ in 0..count {
        let tag = payload.read_u8()?;
        if tag != 0x60 {
            return Err(WasmError::malformed(
                "Invalid type tag: 0x (expected 0x60 for function type)",
            ));
        }
        let params = parse_resulttype(payload)?;
        let results = parse_resulttype(payload)?;
        types.push(Rc::new(FunctionType::new(params, results)));
    }

    Ok(types)
}

fn parse_import_section(
    types: &[Rc<FunctionType>],
    functions: &mut collections::Vec<Function>,
    tables: &mut collections::Vec<Table>,
    memories: &mut collections::Vec<Memory>,
    globals: &mut collections::Vec<Global>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let module_name = payload.read_length_prefixed_utf8()?.to_string();
        let field_name = payload.read_length_prefixed_utf8()?.to_string();

        let kind = ExternalKind::try_from(payload.read_u8()?)?;

        match kind {
            ExternalKind::Function => {
                let type_index = payload.read_leb128_u32()?;
                let func_type = types
                    .get(type_index as usize)
                    .ok_or_else(|| WasmError::invalid("Invalid function type index"))?;
                functions.push(Function::new_import(
                    module_name,
                    field_name,
                    func_type.clone(),
                    type_index,
                ));
            }
            ExternalKind::Table => {
                let (value_type, limits) = parse_tabletype(payload)?;
                tables.push(Table::new_import(
                    module_name,
                    field_name,
                    value_type,
                    limits,
                )?);
            }
            ExternalKind::Memory => {
                let limits = parse_limits(payload)?;
                memories.push(Memory::new_import(module_name, field_name, limits)?);
            }
            ExternalKind::Global => {
                let (value_type, mutable) = parse_globaltype(payload)?;
                globals.push(Global::new_import(
                    module_name,
                    field_name,
                    value_type,
                    mutable,
                ));
            }
        }
    }
    Ok(())
}

fn parse_function_section(
    types: &[Rc<FunctionType>],
    functions: &mut collections::Vec<Function>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let indices = parse_vec(payload, parse_indices)?;
    for index in indices {
        let func_type = types
            .get(index)
            .ok_or_else(|| WasmError::invalid("Invalid function type index"))?;
        functions.push(Function::new_local(func_type.clone(), index as u32));
    }
    Ok(())
}

fn parse_table_section(
    tables: &mut collections::Vec<Table>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let (value_type, limits) = parse_tabletype(payload)?;
        tables.push(Table::new_local(value_type, limits)?);
    }
    Ok(())
}

fn parse_memory_section(
    memories: &mut collections::Vec<Memory>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let mem_limits = parse_vec(payload, parse_limits)?;
    for limits in mem_limits {
        memories.push(Memory::new_local(limits)?);
    }
    Ok(())
}

fn parse_global<'a>(payload: &mut Payload<'a>) -> Result<Global, WasmError> {
    let (global_type, is_mutable) = parse_globaltype(payload)?;
    let init = parse_constexpr(payload)?;
    Ok(Global::new_local(global_type, is_mutable, init))
}

fn parse_global_section(
    globals: &mut collections::Vec<Global>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let parsed = parse_vec(payload, parse_global)?;
    for global in parsed {
        globals.push(global);
    }
    Ok(())
}

fn parse_export_section(
    functions: &mut [Function],
    tables: &mut [Table],
    memories: &mut [Memory],
    globals: &mut [Global],
    export_names: &mut collections::Vec<String>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let name = payload.read_length_prefixed_utf8()?.to_string();
        let kind = ExternalKind::try_from(payload.read_u8()?)
            .map_err(|_| WasmError::malformed("Invalid export kind"))?;
        let index = payload.read_leb128_u32()? as usize;

        // Validate unique export name
        if export_names.iter().any(|n| n == &name) {
            return Err(WasmError::invalid("duplicate export name"));
        }
        export_names.push(name.clone());

        match kind {
            ExternalKind::Function => {
                let f = functions
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("Invalid export function index"))?;
                f.add_export_name(name);
            }
            ExternalKind::Table => {
                let t = tables
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("Invalid export table index"))?;
                t.add_export_name(name);
            }
            ExternalKind::Memory => {
                let m = memories
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("Invalid export memory index"))?;
                m.add_export_name(name);
            }
            ExternalKind::Global => {
                let g = globals
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("Invalid export global index"))?;
                g.add_export_name(name);
            }
        }
    }
    Ok(())
}

fn parse_validate_reftype(payload: &mut Payload) -> Result<ValueType, WasmError> {
    let valtype = parse_valtype(payload)?;
    if !valtype.is_ref() {
        return Err(WasmError::malformed("Invalid element kind"));
    }
    Ok(valtype)
}

fn parse_validate_element_kind(payload: &mut Payload) -> Result<(), WasmError> {
    let element_kind = payload.read_u8()?;
    if element_kind != 0x0 {
        return Err(WasmError::invalid("Invalid element kind"));
    }
    Ok(())
}

fn parse_element<'a>(payload: &mut Payload<'a>) -> Result<Element, WasmError> {
    let kind = payload.read_leb128_u32()? as u8;

    match kind {
        ELEM_ACTIVE_FUNCIDX => {
            let expr = parse_constexpr(payload)?;
            let func_indices = parse_vec(payload, parse_indices)?;
            Ok(Element::new_active(
                0,
                expr,
                ElementInit::FunctionIndexes(func_indices),
            ))
        }
        ELEM_PASSIVE_FUNCIDX => {
            parse_validate_element_kind(payload)?;
            let func_indices = parse_vec(payload, parse_indices)?;
            Ok(Element::new_passive(ElementInit::FunctionIndexes(
                func_indices,
            )))
        }
        ELEM_ACTIVE_TABLEIDX_FUNCIDX => {
            let table_index = payload.read_leb128_u32()? as usize;
            let expr = parse_constexpr(payload)?;
            parse_validate_element_kind(payload)?;
            let func_indices = parse_vec(payload, parse_indices)?;
            Ok(Element::new_active(
                table_index,
                expr,
                ElementInit::FunctionIndexes(func_indices),
            ))
        }
        ELEM_DECLARATIVE_FUNCIDX => {
            parse_validate_element_kind(payload)?;
            let func_indices = parse_vec(payload, parse_indices)?;
            Ok(Element::new_declarative(ElementInit::FunctionIndexes(
                func_indices,
            )))
        }
        ELEM_ACTIVE_EXPR => {
            let expr = parse_constexpr(payload)?;
            let exprs = parse_vec(payload, parse_constexpr)?;
            Ok(Element::new_active(
                0,
                expr,
                ElementInit::InitExprs {
                    value_type: ValueType::funcref(),
                    exprs,
                },
            ))
        }
        ELEM_PASSIVE_EXPR => {
            let value_type = parse_validate_reftype(payload)?;
            let exprs = parse_vec(payload, parse_constexpr)?;
            Ok(Element::new_passive(ElementInit::InitExprs {
                value_type,
                exprs,
            }))
        }
        ELEM_ACTIVE_TABLEIDX_EXPR => {
            let table_index = payload.read_leb128_u32()? as usize;
            let expr = parse_constexpr(payload)?;
            let value_type = parse_validate_reftype(payload)?;
            let exprs = parse_vec(payload, parse_constexpr)?;
            Ok(Element::new_active(
                table_index,
                expr,
                ElementInit::InitExprs { value_type, exprs },
            ))
        }
        ELEM_DECLARATIVE_EXPR => {
            let value_type = parse_validate_reftype(payload)?;
            let exprs = parse_vec(payload, parse_constexpr)?;
            Ok(Element::new_declarative(ElementInit::InitExprs {
                value_type,
                exprs,
            }))
        }
        _ => Err(WasmError::malformed("Invalid element kind")),
    }
}

fn parse_code(
    payload: &mut Payload,
) -> Result<(collections::Vec<ValueType>, Bytecode, usize), WasmError> {
    let total_size = payload.read_leb128_u32()? as usize;
    let local_begin = payload.position();

    let num_local_groups = payload.read_leb128_u32()?;
    let mut total_locals = 0u32;
    let mut local_groups = collections::Vec::with_capacity(num_local_groups as usize);

    for _ in 0..num_local_groups {
        let count = payload.read_leb128_u32()?;
        total_locals = total_locals
            .checked_add(count)
            .ok_or_else(|| WasmError::malformed("Too many locals (overflow)"))?;
        if total_locals > constants::MAX_LOCALS {
            return Err(WasmError::malformed("Too many locals"));
        }
        let value_type = parse_valtype(payload)?;
        local_groups.push((count, value_type));
    }

    let mut locals = collections::Vec::with_capacity(total_locals as usize);
    for (count, value_type) in local_groups {
        for _ in 0..count {
            locals.push(value_type);
        }
    }

    let code_begin = payload.position();
    let code_size = total_size - (code_begin - local_begin);
    let code = payload.advance_and_split_at(code_size)?;
    // WASM spec: every function body must end with the END (0x0b) opcode
    if code.last() != Some(&0x0b) {
        return Err(WasmError::malformed("END opcode expected"));
    }
    Ok((locals, code.into(), code_begin))
}

fn parse_code_section(
    functions: &mut [Function],
    payload: &mut Payload,
    payload_offset: usize,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()? as usize;
    let imported_count = functions.iter().filter(|f| f.is_import()).count();

    for index in 0..count {
        let (locals, code, mut code_offset) = parse_code(payload)?;
        code_offset += payload_offset;
        let function_index = index
            .checked_add(imported_count)
            .ok_or_else(|| WasmError::malformed("Function index overflow"))?;
        let function = functions
            .get_mut(function_index)
            .ok_or_else(|| WasmError::malformed("Invalid function index in code section"))?;
        let spec = function
            .spec_mut()
            .ok_or_else(|| WasmError::invalid("Expected local function for code section"))?;
        spec.set_locals(locals);
        spec.set_code(code);
        spec.set_code_offset(code_offset);
    }
    Ok(())
}

fn parse_data<'a>(payload: &mut Payload<'a>) -> Result<Data, WasmError> {
    let kind = payload.read_leb128_u32()?;
    match kind {
        DATA_ACTIVE => {
            let offset_expr = parse_constexpr(payload)?;
            let bytes = payload.read_leb128_u32()? as usize;
            let init = payload.advance_and_split_at(bytes)?;
            Ok(Data::new_active(0, offset_expr, init))
        }
        DATA_PASSIVE => {
            let bytes = payload.read_leb128_u32()? as usize;
            let init = payload.advance_and_split_at(bytes)?;
            Ok(Data::new_passive(0, init))
        }
        DATA_ACTIVE_MEMIDX => {
            let memory_index = payload.read_leb128_u32()? as usize;
            let offset_expr = parse_constexpr(payload)?;
            let bytes = payload.read_leb128_u32()? as usize;
            let init = payload.advance_and_split_at(bytes)?;
            Ok(Data::new_active(memory_index, offset_expr, init))
        }
        _ => Err(WasmError::malformed("Invalid data kind")),
    }
}

fn parse_data_section(
    data_count: Option<usize>,
    payload: &mut Payload,
) -> Result<collections::Vec<Data>, WasmError> {
    let data = parse_vec(payload, parse_data)?;
    if let Some(dc) = data_count {
        if data.len() != dc {
            return Err(WasmError::malformed("Invalid data count"));
        }
    }
    Ok(data)
}

fn parse_custom_section<'a>(payload: &mut Payload<'a>) -> Result<(), WasmError> {
    let _name = payload.read_length_prefixed_utf8()?;
    let _ = payload.advance_and_split_at(payload.remaining_slice().len())?;
    Ok(())
}
