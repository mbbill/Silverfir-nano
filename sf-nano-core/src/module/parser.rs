//! WebAssembly 2.0 Module Parser (no_std)
//!
//! Handles binary format parsing and structural validation only.
//! Validates binary format correctness, structural requirements, and basic range checks.
//! Returns **malformed** errors for structural/format issues.

use crate::collections;

use tracked_alloc::rc::Rc;
use tracked_alloc::string::ToString;

use crate::{
    constants,
    error::WasmError,
    module::{
        entities::{
            Bytecode, ConstExpr, Data, Element, ElementInit, Function, FunctionType, Global,
            Memory, Table, Tag,
        },
        type_context::TypeContext,
        type_defs::{
            ArrayType, CompositeType, DefType, FieldType, PackedType, StorageType, StructType,
        },
        Module,
    },
    opcodes::{Opcode, OpcodeFB, OpcodeFD},
    utils::{
        limits::{Limits, LimitsError},
        payload::Payload,
    },
    value_type::{HeapType, ValueType},
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
    Tag = 13,
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
            13 => Ok(WasmSection::Tag),
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
    Tag = 4,
}

impl TryFrom<u8> for ExternalKind {
    type Error = WasmError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ExternalKind::Function),
            1 => Ok(ExternalKind::Table),
            2 => Ok(ExternalKind::Memory),
            3 => Ok(ExternalKind::Global),
            4 => Ok(ExternalKind::Tag),
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ModuleCapacities {
    functions: usize,
    tables: usize,
    memories: usize,
    globals: usize,
    tags: usize,
}

impl ModuleCapacities {
    fn add(slot: &mut usize, count: usize) -> Result<(), WasmError> {
        *slot = slot
            .checked_add(count)
            .ok_or_else(|| WasmError::malformed("Module entity count overflow"))?;
        Ok(())
    }
}

/// Read a section vector count without trusting it to allocate more storage
/// than the section can possibly contain. Every encoded entry consumes at
/// least one byte, so this also keeps malformed tiny modules from turning the
/// capacity pass into an oversized allocation request.
fn planned_section_count(payload: &mut Payload) -> Result<usize, WasmError> {
    let count = payload.read_leb128_u32()? as usize;
    if count > payload.remaining_slice().len() {
        return Err(WasmError::malformed("Section count exceeds section length"));
    }
    Ok(count)
}

fn skip_name(payload: &mut Payload) -> Result<(), WasmError> {
    let len = payload.read_leb128_u32()? as usize;
    payload.advance_and_split_at(len)?;
    Ok(())
}

/// Count each imported entity kind without allocating or retaining anything.
/// Import is the only heterogeneous section, so its single vector count is
/// insufficient to reserve the five destination vectors exactly.
fn plan_import_capacities(
    payload: &mut Payload,
    capacities: &mut ModuleCapacities,
) -> Result<(), WasmError> {
    let count = planned_section_count(payload)?;
    for _ in 0..count {
        skip_name(payload)?;
        skip_name(payload)?;
        match ExternalKind::try_from(payload.read_u8()?)? {
            ExternalKind::Function => {
                payload.read_leb128_u32()?;
                ModuleCapacities::add(&mut capacities.functions, 1)?;
            }
            ExternalKind::Table => {
                parse_tabletype(payload)?;
                ModuleCapacities::add(&mut capacities.tables, 1)?;
            }
            ExternalKind::Memory => {
                parse_memory_limits(payload)?;
                ModuleCapacities::add(&mut capacities.memories, 1)?;
            }
            ExternalKind::Global => {
                parse_globaltype(payload)?;
                ModuleCapacities::add(&mut capacities.globals, 1)?;
            }
            ExternalKind::Tag => {
                if payload.read_u8()? != 0x00 {
                    return Err(WasmError::malformed("Invalid tag attribute"));
                }
                payload.read_leb128_u32()?;
                ModuleCapacities::add(&mut capacities.tags, 1)?;
            }
        }
    }
    Ok(())
}

/// Walk only section envelopes and counts before the real parse. Code and data
/// payloads are skipped in O(number of sections); only import descriptors need
/// a small second decode because their entities are heterogeneous. The result
/// gives every long-lived top-level entity vector its final capacity before
/// the first push, eliminating geometric growth and copies during startup.
fn plan_module_capacities(mut payload: Payload<'_>) -> Result<ModuleCapacities, WasmError> {
    let mut capacities = ModuleCapacities::default();
    while !payload.is_empty() {
        let section = WasmSection::try_from(payload.read_u8()?)?;
        let section_len = payload.read_leb128_u32()? as usize;
        let mut section_payload: Payload = payload.advance_and_split_at(section_len)?.into();
        match section {
            WasmSection::Import => {
                plan_import_capacities(&mut section_payload, &mut capacities)?;
            }
            WasmSection::Function => {
                let count = planned_section_count(&mut section_payload)?;
                ModuleCapacities::add(&mut capacities.functions, count)?;
            }
            WasmSection::Table => {
                let count = planned_section_count(&mut section_payload)?;
                ModuleCapacities::add(&mut capacities.tables, count)?;
            }
            WasmSection::Memory => {
                let count = planned_section_count(&mut section_payload)?;
                ModuleCapacities::add(&mut capacities.memories, count)?;
            }
            WasmSection::Global => {
                let count = planned_section_count(&mut section_payload)?;
                ModuleCapacities::add(&mut capacities.globals, count)?;
            }
            WasmSection::Tag => {
                let count = planned_section_count(&mut section_payload)?;
                ModuleCapacities::add(&mut capacities.tags, count)?;
            }
            WasmSection::Custom
            | WasmSection::Type
            | WasmSection::Export
            | WasmSection::Start
            | WasmSection::Element
            | WasmSection::Code
            | WasmSection::Data
            | WasmSection::DataCount => {}
        }
    }
    Ok(capacities)
}

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
            (WasmSection::Memory, WasmSection::Tag) => true,
            (WasmSection::Tag, WasmSection::Global) => true,
            _ => {
                fn section_order(s: WasmSection) -> u8 {
                    match s {
                        WasmSection::Custom => 0,
                        WasmSection::Type => 1,
                        WasmSection::Import => 2,
                        WasmSection::Function => 3,
                        WasmSection::Table => 4,
                        WasmSection::Memory => 5,
                        WasmSection::Tag => 5,
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

    let capacities = plan_module_capacities(payload.clone())?;
    let mut types: collections::Vec<Rc<DefType>> = collections::Vec::new();
    let mut functions: collections::Vec<Function> =
        collections::Vec::with_capacity(capacities.functions);
    let mut tables: collections::Vec<Table> = collections::Vec::with_capacity(capacities.tables);
    let mut memories: collections::Vec<Memory> =
        collections::Vec::with_capacity(capacities.memories);
    let mut globals: collections::Vec<Global> = collections::Vec::with_capacity(capacities.globals);
    let mut tags: collections::Vec<Tag> = collections::Vec::with_capacity(capacities.tags);
    let mut elements: collections::Vec<Element> = collections::Vec::new();
    let mut data_segments: collections::Vec<Data> = collections::Vec::new();
    let mut start_func_index: Option<usize> = None;
    let mut data_count: Option<usize> = None;

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
                    &mut tags,
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
            WasmSection::Tag => {
                parse_tag_section(&types, &mut tags, &mut section_payload)?;
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
                    &mut tags,
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
        tags,
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
    let value_type = ValueType::parse(payload)?;
    value_type.ensure_enabled_in_build()?;
    Ok(value_type)
}

fn parse_resulttype(payload: &mut Payload) -> Result<collections::Vec<ValueType>, WasmError> {
    parse_vec(payload, parse_valtype)
}

fn narrow_memory_limit_u32(value: u32) -> Result<usize, WasmError> {
    let value = value as usize;
    if value > constants::MAX_MEM_PAGES_SPEC {
        return Err(WasmError::invalid("memory size"));
    }
    Ok(value.min(constants::MAX_MEM_PAGES))
}

fn narrow_memory_limit_u64(value: u64) -> Result<usize, WasmError> {
    if value > constants::MAX_MEM_PAGES_64_SPEC {
        return Err(WasmError::invalid("memory size"));
    }
    Ok(value.min(constants::MAX_MEM_PAGES_64 as u64) as usize)
}

fn narrow_table_limit_u64(value: u64) -> usize {
    value.min(usize::MAX as u64) as usize
}

#[inline]
fn validate_limit_order<T: Ord>(min: T, max: T) -> Result<(), WasmError> {
    if max < min {
        return Err(LimitsError::MinLargerThanMax.into());
    }
    Ok(())
}

fn parse_memory_limits(payload: &mut Payload) -> Result<Limits, WasmError> {
    let tag = payload.read_u8()?;
    match tag {
        0x00 => {
            let min = narrow_memory_limit_u32(payload.read_leb128_u32()?)?;
            Ok(Limits::new(min, None)?)
        }
        0x01 => {
            let min_raw = payload.read_leb128_u32()?;
            let max_raw = payload.read_leb128_u32()?;
            validate_limit_order(min_raw, max_raw)?;
            let min = narrow_memory_limit_u32(min_raw)?;
            let max = narrow_memory_limit_u32(max_raw)?;
            Ok(Limits::new(min, Some(max))?)
        }
        0x04 => {
            let min = narrow_memory_limit_u64(payload.read_leb128_u64()?)?;
            Ok(Limits::new_64(min, None)?)
        }
        0x05 => {
            let min_raw = payload.read_leb128_u64()?;
            let max_raw = payload.read_leb128_u64()?;
            validate_limit_order(min_raw, max_raw)?;
            let min = narrow_memory_limit_u64(min_raw)?;
            let max = narrow_memory_limit_u64(max_raw)?;
            Ok(Limits::new_64(min, Some(max))?)
        }
        // The message used to end at "0x" with nothing after it: the error
        // type carries a &'static str, so the byte it saw could not be
        // formatted in. Naming the proposal each unsupported flag belongs
        // to is more use than the byte would have been.
        0x02 | 0x03 => Err(WasmError::malformed(
            "memory limits flag selects shared memory, which is not supported",
        )),
        0x08 | 0x09 => Err(WasmError::malformed(
            "memory limits flag selects a custom page size, which is not supported",
        )),
        _ => Err(WasmError::malformed("malformed memory limits flag")),
    }
}

fn parse_table_limits(payload: &mut Payload) -> Result<Limits, WasmError> {
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
            let min = narrow_table_limit_u64(payload.read_leb128_u64()?);
            Ok(Limits::new_64(min, None)?)
        }
        0x05 => {
            let min_raw = payload.read_leb128_u64()?;
            let max_raw = payload.read_leb128_u64()?;
            validate_limit_order(min_raw, max_raw)?;
            let min = narrow_table_limit_u64(min_raw);
            let max = narrow_table_limit_u64(max_raw);
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
                HeapType::parse(&mut code)?;
            }
            REF_FUNC | GLOBAL_GET => {
                code.read_leb128_u32()?;
            }
            // Extended constant expressions (WASM 2.0)
            I32_ADD | I32_SUB | I32_MUL | I64_ADD | I64_SUB | I64_MUL => {
                // Binary operations have no immediates
            }
            PREFIX_FB => {
                let op_ext: OpcodeFB = code.read_leb128_u32()?.try_into()?;
                use OpcodeFB::*;
                match op_ext {
                    REF_I31 | ANY_CONVERT_EXTERN | EXTERN_CONVERT_ANY => {}
                    STRUCT_NEW | STRUCT_NEW_DEFAULT => {
                        code.read_leb128_u32()?;
                    }
                    ARRAY_NEW | ARRAY_NEW_DEFAULT => {
                        code.read_leb128_u32()?;
                    }
                    ARRAY_NEW_FIXED => {
                        code.read_leb128_u32()?;
                        code.read_leb128_u32()?;
                    }
                    _ => {
                        break 'outer Err(WasmError::malformed("Invalid opcode in constexpr"));
                    }
                }
            }
            PREFIX_FD => {
                let op_ext: OpcodeFD = code.read_leb128_u32()?.try_into()?;
                use OpcodeFD::*;
                match op_ext {
                    V128_CONST => {
                        for _ in 0..16 {
                            code.read_u8()?;
                        }
                    }
                    _ => {
                        break 'outer Err(WasmError::malformed("Invalid opcode in constexpr"));
                    }
                }
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
    let limits = parse_table_limits(payload)?;
    Ok((value_type, limits))
}

// ============================================================================
// Section parsers
// ============================================================================

fn parse_type(payload: &mut Payload) -> Result<Rc<DefType>, WasmError> {
    let peek_byte = payload.peek_u8()?;
    match peek_byte {
        0x4E => Err(WasmError::malformed(
            "Recursive type group should be handled by parse_type_section",
        )),
        0x4F | 0x50 => parse_subtype(payload, None),
        _ => parse_comptype(payload, collections::vec![], true, None),
    }
}

fn parse_subtype(payload: &mut Payload, rec_group: Option<u32>) -> Result<Rc<DefType>, WasmError> {
    match payload.peek_u8()? {
        0x4F | 0x50 => {
            let tag = payload.read_u8()?;
            let is_final = tag == 0x4F;
            let supertype_count = payload.read_leb128_u32()?;
            let mut supertypes = collections::Vec::with_capacity(supertype_count as usize);
            for _ in 0..supertype_count {
                supertypes.push(payload.read_leb128_u32()?);
            }
            parse_comptype(payload, supertypes, is_final, rec_group)
        }
        _ => parse_comptype(payload, collections::vec![], true, rec_group),
    }
}

fn parse_comptype(
    payload: &mut Payload,
    supertypes: collections::Vec<u32>,
    is_final: bool,
    rec_group: Option<u32>,
) -> Result<Rc<DefType>, WasmError> {
    let tag = payload.read_u8()?;

    let composite = match tag {
        0x60 => {
            let params = parse_resulttype(payload)?;
            let results = parse_resulttype(payload)?;
            CompositeType::Func(Rc::new(FunctionType::new(params, results)))
        }
        0x5F => {
            let field_count = payload.read_leb128_u32()?;
            let mut fields = collections::Vec::with_capacity(field_count as usize);
            for _ in 0..field_count {
                fields.push(parse_fieldtype(payload)?);
            }
            CompositeType::Struct(StructType::new(fields))
        }
        0x5E => CompositeType::Array(ArrayType::new(parse_fieldtype(payload)?)),
        _ => return Err(WasmError::malformed("Invalid composite type tag")),
    };

    Ok(Rc::new(DefType {
        composite,
        supertypes,
        is_final,
        rec_group,
    }))
}

fn parse_fieldtype(payload: &mut Payload) -> Result<FieldType, WasmError> {
    let storage = parse_storagetype(payload)?;
    let mutable = match payload.read_u8()? {
        0x00 => false,
        0x01 => true,
        _ => return Err(WasmError::malformed("Invalid mutability flag")),
    };
    Ok(FieldType::new(storage, mutable))
}

fn parse_storagetype(payload: &mut Payload) -> Result<StorageType, WasmError> {
    match payload.peek_u8()? {
        0x77 => {
            payload.read_u8()?;
            Ok(StorageType::Packed(PackedType::I16))
        }
        0x78 => {
            payload.read_u8()?;
            Ok(StorageType::Packed(PackedType::I8))
        }
        _ => Ok(StorageType::Val(parse_valtype(payload)?)),
    }
}

fn parse_type_section(payload: &mut Payload) -> Result<collections::Vec<Rc<DefType>>, WasmError> {
    let count = payload.read_leb128_u32()?;
    let mut types = collections::Vec::with_capacity(count as usize);
    let mut i = 0;
    let mut next_rec_group_id = 0u32;

    while i < count {
        if payload.peek_u8()? == 0x4E {
            payload.read_u8()?;
            let group_count = payload.read_leb128_u32()?;
            if group_count as usize > payload.remaining_slice().len() {
                return Err(WasmError::malformed(
                    "Recursive type count exceeds section length",
                ));
            }
            // The section count counts this recursive group as one entry,
            // while the destination stores every subtype. Reserve the exact
            // expansion before pushing so a large group cannot grow in steps.
            types.reserve(group_count as usize);
            let rec_group_id = next_rec_group_id;
            next_rec_group_id += 1;

            for _ in 0..group_count {
                types.push(parse_subtype(payload, Some(rec_group_id))?);
            }
            i += 1;
        } else {
            types.push(parse_type(payload)?);
            i += 1;
        }
    }

    Ok(types)
}

fn get_function_type(types: &[Rc<DefType>], index: u32) -> Result<Rc<FunctionType>, WasmError> {
    let def_type = types
        .get(index as usize)
        .ok_or_else(|| WasmError::invalid("Invalid function type index"))?;
    match &def_type.composite {
        CompositeType::Func(func_type) => Ok(func_type.clone()),
        _ => Err(WasmError::invalid(
            "Type index does not reference a function type",
        )),
    }
}

fn parse_import_section(
    types: &[Rc<DefType>],
    functions: &mut collections::Vec<Function>,
    tables: &mut collections::Vec<Table>,
    memories: &mut collections::Vec<Memory>,
    globals: &mut collections::Vec<Global>,
    tags: &mut collections::Vec<Tag>,
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
                let func_type = get_function_type(types, type_index)?;
                functions.push(Function::new_import(
                    module_name,
                    field_name,
                    func_type,
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
                let limits = parse_memory_limits(payload)?;
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
            ExternalKind::Tag => {
                let attribute = payload.read_u8()?;
                if attribute != 0x00 {
                    return Err(WasmError::malformed("Invalid tag attribute"));
                }
                let type_index = payload.read_leb128_u32()?;
                let func_type = get_function_type(types, type_index)?;
                tags.push(Tag::new_import(
                    module_name,
                    field_name,
                    func_type,
                    type_index,
                ));
            }
        }
    }
    Ok(())
}

fn parse_function_section(
    types: &[Rc<DefType>],
    functions: &mut collections::Vec<Function>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let index = parse_indices(payload)?;
        let func_type = get_function_type(types, index as u32)?;
        functions.push(Function::new_local(func_type, index as u32));
    }
    Ok(())
}

fn parse_table_section(
    tables: &mut collections::Vec<Table>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let first_byte = payload.read_u8()?;
        if first_byte == 0x40 {
            let reserved = payload.read_u8()?;
            if reserved != 0x00 {
                return Err(WasmError::malformed(
                    "Invalid reserved byte after table initializer marker",
                ));
            }
            let (value_type, limits) = parse_tabletype(payload)?;
            let init_expr = parse_constexpr(payload)?;
            tables.push(Table::new_local_with_init(value_type, limits, init_expr)?);
        } else {
            payload.rewind(1).map_err(WasmError::from)?;
            let (value_type, limits) = parse_tabletype(payload)?;
            tables.push(Table::new_local(value_type, limits)?);
        }
    }
    Ok(())
}

fn parse_memory_section(
    memories: &mut collections::Vec<Memory>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let limits = parse_memory_limits(payload)?;
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
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        globals.push(parse_global(payload)?);
    }
    Ok(())
}

fn parse_tag_section(
    types: &[Rc<DefType>],
    tags: &mut collections::Vec<Tag>,
    payload: &mut Payload,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    for _ in 0..count {
        let attribute = payload.read_u8()?;
        if attribute != 0x00 {
            return Err(WasmError::malformed("Invalid tag attribute"));
        }
        let type_index = payload.read_leb128_u32()?;
        let func_type = get_function_type(types, type_index)?;
        tags.push(Tag::new_local(func_type, type_index));
    }
    Ok(())
}

fn parse_export_section<'a>(
    functions: &mut [Function],
    tables: &mut [Table],
    memories: &mut [Memory],
    globals: &mut [Global],
    tags: &mut [Tag],
    payload: &mut Payload<'a>,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()?;
    // The borrowed names are only a parser-time uniqueness set. Keeping
    // slices into the module bytes avoids cloning every export String into a
    // temporary owner that is dropped at the end of this section.
    let mut export_names: collections::Vec<&'a str> =
        collections::Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = payload.read_length_prefixed_utf8()?;
        let kind = ExternalKind::try_from(payload.read_u8()?)
            .map_err(|_| WasmError::malformed("Invalid export kind"))?;
        let index = payload.read_leb128_u32()? as usize;

        // Validate unique export name
        if export_names.contains(&name) {
            return Err(WasmError::invalid("duplicate export name"));
        }
        export_names.push(name);
        let name = name.to_string();

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
            ExternalKind::Tag => {
                let tag = tags
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("Invalid export tag index"))?;
                tag.add_export_name(name);
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
) -> Result<(collections::Vec<ValueType>, usize, usize), WasmError> {
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
    let code_size = total_size
        .checked_sub(code_begin - local_begin)
        .ok_or_else(|| WasmError::malformed("Function locals exceed body size"))?;
    let code = payload.advance_and_split_at(code_size)?;
    // WASM spec: every function body must end with the END (0x0b) opcode
    if code.last() != Some(&0x0b) {
        return Err(WasmError::malformed("END opcode expected"));
    }
    Ok((locals, code_begin, code_size))
}

fn parse_code_section(
    functions: &mut [Function],
    payload: &mut Payload,
    payload_offset: usize,
) -> Result<(), WasmError> {
    let count = payload.read_leb128_u32()? as usize;
    let imported_count = functions.iter().filter(|f| f.is_import()).count();
    // Own the code section once. Each FunctionSpec below keeps a cheap Rc
    // clone plus a range instead of allocating and copying one Rc<[u8]> per
    // body. The parser still advances over the caller's bytes, so section
    // length/error behavior is unchanged.
    let arena_base = payload.position();
    let code_arena: Rc<[u8]> = Rc::from(payload.remaining_slice());

    for index in 0..count {
        let (locals, code_begin, code_size) = parse_code(payload)?;
        let arena_start = code_begin
            .checked_sub(arena_base)
            .ok_or_else(|| WasmError::malformed("Invalid function body range"))?;
        let arena_end = arena_start
            .checked_add(code_size)
            .ok_or_else(|| WasmError::malformed("Function body range overflow"))?;
        let code = Bytecode::from_shared(code_arena.clone(), arena_start, arena_end)?;
        let mut code_offset = code_begin;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn encode_uleb_u64(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
        out
    }

    fn encoded_limits(tag: u8, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + values.len() * 5);
        out.push(tag);
        for &value in values {
            out.extend(encode_uleb_u64(value));
        }
        out
    }

    #[test]
    fn recursive_type_group_expands_the_section_entry_capacity() {
        let bytes = wat::parse_str(
            r#"(module
                (rec
                    (type $a (func (param i32)))
                    (type $b (func (result i64))))
                (func (type $a) local.get 0 drop))"#,
        )
        .expect("encode recursive type group");
        let module = Module::new("recursive-type-capacity", &bytes).expect("parse module");
        assert_eq!(module.types().len(), 2);
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn coremark_parser_capacity_plan_eliminates_allocator_growth() {
        let bytes = include_bytes!("../../../benchmarks/wasi/coremark/coremark.wasm");
        let (module, census) = crate::test_alloc::measure(|| Module::new("coremark", bytes));
        module.expect("parse coremark");
        assert_eq!(
            census.reallocations, 0,
            "the unplanned parser made 5 real realloc calls on CoreMark: {census:?}"
        );
        assert_eq!(census.reallocated_bytes, 0);
        if cfg!(not(sf_jit)) {
            assert!(
                census.allocations <= 321,
                "the interpreter-only unplanned parser made 326 real alloc calls: {census:?}"
            );
            assert!(
                census.allocated_bytes <= 70_273,
                "the interpreter-only unplanned parser moved 88,109 allocator bytes: {census:?}"
            );
        }
    }

    #[test]
    fn function_bodies_are_ranges_in_one_code_section_arena() {
        let wasm = wat::parse_str(
            r#"(module
                (func (result i32) i32.const 7)
                (func (param i64) local.get 0 drop))"#,
        )
        .expect("wat");
        let module = parse_module("shared-code", &wasm).expect("module");
        let first = module.functions()[0].spec().expect("first local").code();
        let second = module.functions()[1].spec().expect("second local").code();

        assert_eq!(&**first, &[Opcode::I32_CONST as u8, 7, Opcode::END as u8]);
        assert_eq!(
            &**second,
            &[
                Opcode::LOCAL_GET as u8,
                0,
                Opcode::DROP as u8,
                Opcode::END as u8,
            ]
        );
        assert!(first.shares_storage_with(second));
    }

    #[test]
    fn parse_memory64_limits_rejects_raw_min_larger_than_max_before_clamp() {
        let bytes = encoded_limits(
            0x05,
            &[
                constants::MAX_MEM_PAGES_64_SPEC,
                constants::MAX_MEM_PAGES_64_SPEC - 1,
            ],
        );
        let mut payload: Payload = bytes.as_slice().into();

        let err = parse_memory_limits(&mut payload).unwrap_err();

        assert_eq!(err, WasmError::invalid("min larger than max"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn parse_memory32_limits_rejects_raw_min_larger_than_max_before_clamp() {
        let bytes = encoded_limits(0x01, &[constants::MAX_MEM_PAGES_SPEC as u64, 65535]);
        let mut payload: Payload = bytes.as_slice().into();

        let err = parse_memory_limits(&mut payload).unwrap_err();

        assert_eq!(err, WasmError::invalid("min larger than max"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn parse_table64_limits_rejects_raw_min_larger_than_max_before_clamp() {
        let bytes = encoded_limits(0x05, &[u64::from(u32::MAX) + 1, u64::from(u32::MAX)]);
        let mut payload: Payload = bytes.as_slice().into();

        let err = parse_table_limits(&mut payload).unwrap_err();

        assert_eq!(err, WasmError::invalid("min larger than max"));
    }
}
