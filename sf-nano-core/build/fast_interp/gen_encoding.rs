// Fast interpreter encoding codegen
// Generates encode/decode functions from handlers.toml

use super::types::{bits_to_rust_type, to_pascal_case, Field, FieldKind, HandlersFile};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Computed field layout information
#[derive(Debug)]
struct FieldLayout {
    name: String,
    bits: u32,
    kind: FieldKind,
    imm_index: usize, // 0, 1, or 2 (for imm0, imm1, imm2)
    bit_offset: u32,  // offset within the imm field
}

/// Compute the layout of fields across imm0, imm1, imm2
fn compute_layout(fields: &[Field], handler_name: &str) -> Vec<FieldLayout> {
    let mut layouts = Vec::new();
    let mut current_imm = 0usize;
    let mut current_bit = 0u32;

    for field in fields {
        // Check if this field fits in current imm
        if current_bit + field.bits > 64 {
            // Move to next imm
            current_imm += 1;
            current_bit = 0;

            if current_imm > 2 {
                panic!(
                    "Handler '{}': Field '{}' exceeds available imm space (3 x 64 bits)",
                    handler_name, field.name
                );
            }
        }

        layouts.push(FieldLayout {
            name: field.name.clone(),
            bits: field.bits,
            kind: field.kind.clone(),
            imm_index: current_imm,
            bit_offset: current_bit,
        });

        current_bit += field.bits;
    }

    layouts
}

/// Rust keywords that need escaping with r#
const RUST_KEYWORDS: &[&str] = &[
    "const", "return", "type", "match", "loop", "move", "ref", "self", "super", "static", "unsafe",
    "where", "async", "await", "dyn", "abstract", "become", "box", "do", "final", "macro",
    "override", "priv", "try", "typeof", "unsized", "virtual", "yield", "fn", "let", "mut", "pub",
    "struct", "enum", "trait", "impl",
];

/// Escape Rust keyword with r# prefix if needed
fn escape_keyword(name: &str) -> String {
    if RUST_KEYWORDS.contains(&name) {
        format!("r#{}", name)
    } else {
        name.to_string()
    }
}

/// Generate Rust code for a pattern module (encode/decode functions)
fn generate_rust_pattern(name: &str, fields: &[Field]) -> String {
    let layouts = compute_layout(fields, name);
    let mut code = String::new();

    // Module documentation
    code.push_str("/// Layout: ");

    // Document the layout
    let mut imm_contents: HashMap<usize, Vec<String>> = HashMap::new();
    for layout in &layouts {
        imm_contents
            .entry(layout.imm_index)
            .or_default()
            .push(format!(
                "{}[{}:{}]",
                layout.name,
                layout.bit_offset,
                layout.bit_offset + layout.bits
            ));
    }

    for imm_idx in 0..=2 {
        if let Some(contents) = imm_contents.get(&imm_idx) {
            code.push_str(&format!("imm{}: {}, ", imm_idx, contents.join(", ")));
        }
    }
    code.push('\n');

    code.push_str("#[allow(dead_code)]\n");
    code.push_str(&format!("pub mod {} {{\n", escape_keyword(name)));
    // Only import super::* when we have fields (decode functions need Instruction type)
    if !fields.is_empty() {
        code.push_str("    use super::*;\n\n");
    }

    // Generate encode function
    code.push_str("    #[inline(always)]\n");
    code.push_str("    pub fn encode(");

    // Function parameters
    let params: Vec<String> = fields
        .iter()
        .map(|f| {
            let rust_type = bits_to_rust_type(f.bits);
            format!("{}: {}", f.name, rust_type)
        })
        .collect();
    code.push_str(&params.join(", "));
    code.push_str(") -> (u64, u64, u64) {\n");

    // Build the imm values
    let mut imm_exprs: [Vec<String>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for layout in &layouts {
        let cast = if layout.bits < 64 {
            format!("{} as u64", layout.name)
        } else {
            layout.name.clone()
        };

        let shifted = if layout.bit_offset > 0 {
            format!("(({}) << {})", cast, layout.bit_offset)
        } else {
            cast
        };

        imm_exprs[layout.imm_index].push(shifted);
    }

    for (idx, exprs) in imm_exprs.iter().enumerate() {
        if exprs.is_empty() {
            code.push_str(&format!("        let imm{} = 0u64;\n", idx));
        } else if exprs.len() == 1 {
            // Single expression - no need for | operator
            code.push_str(&format!("        let imm{} = {};\n", idx, exprs[0]));
        } else {
            code.push_str(&format!(
                "        let imm{} = {};\n",
                idx,
                exprs.join(" | ")
            ));
        }
    }

    code.push_str("        (imm0, imm1, imm2)\n");
    code.push_str("    }\n\n");

    // Generate decode functions for each field
    for layout in &layouts {
        let is_slot = layout.kind == FieldKind::Slot;
        let rust_type = if is_slot {
            "usize"
        } else {
            bits_to_rust_type(layout.bits)
        };

        code.push_str("    #[inline(always)]\n");
        code.push_str(&format!(
            "    pub fn decode_{}(pc: *const Instruction) -> {} {{\n",
            layout.name, rust_type
        ));

        let imm_access = format!("(*pc).imm{}", layout.imm_index);

        if layout.bits == 64 && layout.bit_offset == 0 {
            // Full 64-bit field, no shifting needed
            if is_slot {
                code.push_str(&format!("        unsafe {{ {} as usize }}\n", imm_access));
            } else {
                code.push_str(&format!("        unsafe {{ {} }}\n", imm_access));
            }
        } else {
            let shifted = if layout.bit_offset > 0 {
                format!("({} >> {})", imm_access, layout.bit_offset)
            } else {
                imm_access
            };

            let intermediate_type = bits_to_rust_type(layout.bits);

            if is_slot {
                code.push_str(&format!(
                    "        unsafe {{ {} as {} as usize }}\n",
                    shifted, intermediate_type
                ));
            } else {
                code.push_str(&format!(
                    "        unsafe {{ {} as {} }}\n",
                    shifted, intermediate_type
                ));
            }
        }

        code.push_str("    }\n\n");
    }

    code.push_str("}\n\n");
    code
}

/// Generate C macros for a pattern
fn generate_c_pattern(name: &str, fields: &[Field]) -> String {
    let layouts = compute_layout(fields, name);
    let mut code = String::new();

    // Comment
    code.push_str(&format!("// {}\n", name));

    // Generate decode macros for each field
    for layout in &layouts {
        let c_type = match layout.bits {
            8 => "uint8_t",
            16 => "uint16_t",
            32 => "uint32_t",
            64 => "uint64_t",
            _ => panic!("Unsupported bit width: {}", layout.bits),
        };

        let imm_access = format!("(pc)->imm{}", layout.imm_index);

        let decode_expr = if layout.bits == 64 && layout.bit_offset == 0 {
            // Full 64-bit field
            imm_access
        } else if layout.bit_offset > 0 {
            format!("(({})({} >> {}))", c_type, imm_access, layout.bit_offset)
        } else {
            format!("(({}){})", c_type, imm_access)
        };

        code.push_str(&format!(
            "#define {}_decode_{}(pc) {}\n",
            name, layout.name, decode_expr
        ));
    }

    code.push('\n');
    code
}

/// Generate PatternKind enum
fn generate_pattern_kind_enum(patterns: &[(String, Vec<Field>)]) -> String {
    let mut code = String::new();

    code.push_str("/// Pattern kind for instruction encoding\n");
    code.push_str("#[derive(Copy, Clone, Debug, PartialEq, Eq)]\n");
    code.push_str("#[repr(u16)]\n");
    code.push_str("pub enum PatternKind {\n");

    for (i, (name, _)) in patterns.iter().enumerate() {
        let variant = to_pascal_case(name);
        code.push_str(&format!("    {} = {},\n", variant, i));
    }

    code.push_str("}\n\n");
    code
}

/// Generate PatternData enum - stores logical values (not encoded imm values)
fn generate_pattern_data_enum(patterns: &[(String, Vec<Field>)]) -> String {
    let mut code = String::new();

    code.push_str("/// Pattern data storing logical field values (not encoded).\n");
    code.push_str("/// Used by TempInst during compilation. Encoding happens at finalization.\n");
    code.push_str("#[derive(Clone, Debug)]\n");
    code.push_str("pub enum PatternData {\n");

    for (name, fields) in patterns {
        let variant = to_pascal_case(name);

        // Collect non-target, non-reserved fields
        let visible_fields: Vec<_> = fields
            .iter()
            .filter(|f| f.kind != FieldKind::Target && !f.name.starts_with('_'))
            .collect();

        if visible_fields.is_empty() {
            // No visible fields - unit variant
            code.push_str(&format!("    {},\n", variant));
        } else {
            code.push_str(&format!("    {} {{\n", variant));
            for field in &visible_fields {
                // Slots are always u16, constants use their native type
                let rust_type = if field.kind == FieldKind::Slot {
                    "u16"
                } else {
                    bits_to_rust_type(field.bits)
                };
                code.push_str(&format!("        {}: {},\n", field.name, rust_type));
            }
            code.push_str("    },\n");
        }
    }

    code.push_str("}\n\n");
    code
}

/// Generate the finalize_pattern_data function
fn generate_finalize_function(patterns: &[(String, Vec<Field>)]) -> String {
    let mut code = String::new();

    code.push_str("/// Finalize PatternData into encoded (imm0, imm1, imm2).\n");
    code.push_str("/// This is the ONLY place where logical values are encoded into imm format.\n");
    code.push_str("#[inline]\n");
    code.push_str("pub fn finalize_pattern_data<F>(data: &PatternData, target_ptr: u64, fix_slot: F) -> (u64, u64, u64)\n");
    code.push_str("where\n");
    code.push_str("    F: Fn(u16) -> u16,\n");
    code.push_str("{\n");
    code.push_str("    match data {\n");

    for (name, fields) in patterns {
        let variant = to_pascal_case(name);

        // Collect non-target, non-reserved fields for the match pattern
        let visible_fields: Vec<_> = fields
            .iter()
            .filter(|f| f.kind != FieldKind::Target && !f.name.starts_with('_'))
            .collect();

        if visible_fields.is_empty() {
            // Unit variant
            code.push_str(&format!("        PatternData::{} => {{\n", variant));
        } else {
            // Struct variant with fields
            let field_names: Vec<_> = visible_fields.iter().map(|f| f.name.as_str()).collect();
            code.push_str(&format!(
                "        PatternData::{} {{ {} }} => {{\n",
                variant,
                field_names.join(", ")
            ));
        }

        // Build the encode call
        let mut encode_args: Vec<String> = Vec::new();
        for field in fields {
            if field.kind == FieldKind::Target {
                encode_args.push("target_ptr".to_string());
            } else if field.name.starts_with('_') {
                // Reserved field - pass 0
                encode_args.push("0".to_string());
            } else if field.kind == FieldKind::Slot {
                // Slot needs fixup
                encode_args.push(format!("fix_slot(*{})", field.name));
            } else {
                // Constant - pass through
                encode_args.push(format!("*{}", field.name));
            }
        }

        if fields.is_empty() {
            // No fields - return zeros
            code.push_str("            (0, 0, 0)\n");
        } else {
            code.push_str(&format!(
                "            {}::encode({})\n",
                escape_keyword(name),
                encode_args.join(", ")
            ));
        }

        code.push_str("        }\n");
    }

    code.push_str("    }\n");
    code.push_str("}\n\n");
    code
}

/// Generate SlotField struct and helpers
fn generate_slot_helpers(patterns: &[(String, Vec<Field>)]) -> String {
    let mut code = String::new();

    // SlotField struct
    code.push_str("/// Describes a slot field's location within an instruction\n");
    code.push_str("#[derive(Copy, Clone, Debug)]\n");
    code.push_str("pub struct SlotField {\n");
    code.push_str("    pub imm_index: u8,\n");
    code.push_str("    pub bit_offset: u8,\n");
    code.push_str("}\n\n");

    // TargetField struct
    code.push_str("/// Describes a branch target field's location\n");
    code.push_str("#[derive(Copy, Clone, Debug)]\n");
    code.push_str("pub struct TargetField {\n");
    code.push_str("    pub imm_index: u8,\n");
    code.push_str("    pub bit_offset: u8,\n");
    code.push_str("}\n\n");

    // get_slot_fields function
    code.push_str("/// Get the slot fields for a pattern\n");
    code.push_str("pub const fn get_slot_fields(pattern: PatternKind) -> &'static [SlotField] {\n");
    code.push_str("    match pattern {\n");

    for (name, fields) in patterns {
        let layouts = compute_layout(fields, name);
        let slot_fields: Vec<_> = layouts
            .iter()
            .filter(|l| l.kind == FieldKind::Slot)
            .collect();

        let variant = to_pascal_case(name);

        if slot_fields.is_empty() {
            code.push_str(&format!("        PatternKind::{} => &[],\n", variant));
        } else {
            code.push_str(&format!("        PatternKind::{} => &[\n", variant));
            for sf in &slot_fields {
                code.push_str(&format!(
                    "            SlotField {{ imm_index: {}, bit_offset: {} }},\n",
                    sf.imm_index, sf.bit_offset
                ));
            }
            code.push_str("        ],\n");
        }
    }

    code.push_str("    }\n");
    code.push_str("}\n\n");

    // get_target_field function
    code.push_str("/// Get the target field for a pattern, if any\n");
    code.push_str("pub const fn get_target_field(pattern: PatternKind) -> Option<TargetField> {\n");
    code.push_str("    match pattern {\n");

    for (name, fields) in patterns {
        let layouts = compute_layout(fields, name);
        let target_field = layouts.iter().find(|l| l.kind == FieldKind::Target);

        let variant = to_pascal_case(name);

        if let Some(tf) = target_field {
            code.push_str(&format!(
                "        PatternKind::{} => Some(TargetField {{ imm_index: {}, bit_offset: {} }}),\n",
                variant, tf.imm_index, tf.bit_offset
            ));
        } else {
            code.push_str(&format!("        PatternKind::{} => None,\n", variant));
        }
    }

    code.push_str("    }\n");
    code.push_str("}\n\n");

    // read_slot_from_imms function
    code.push_str("/// Read a slot value from imm fields\n");
    code.push_str("#[inline(always)]\n");
    code.push_str(
        "pub fn read_slot_from_imms(imm0: u64, imm1: u64, imm2: u64, field: &SlotField) -> u16 {\n",
    );
    code.push_str("    let imm = match field.imm_index {\n");
    code.push_str("        0 => imm0,\n");
    code.push_str("        1 => imm1,\n");
    code.push_str("        2 => imm2,\n");
    code.push_str("        _ => unreachable!(),\n");
    code.push_str("    };\n");
    code.push_str("    ((imm >> field.bit_offset) & 0xFFFF) as u16\n");
    code.push_str("}\n\n");

    // write_slot_to_imms function
    code.push_str("/// Write a slot value to imm fields\n");
    code.push_str("#[inline(always)]\n");
    code.push_str("pub fn write_slot_to_imms(imm0: u64, imm1: u64, imm2: u64, field: &SlotField, value: u16) -> (u64, u64, u64) {\n");
    code.push_str("    let mask = 0xFFFFu64 << field.bit_offset;\n");
    code.push_str("    let shifted_value = (value as u64) << field.bit_offset;\n");
    code.push_str("    match field.imm_index {\n");
    code.push_str("        0 => ((imm0 & !mask) | shifted_value, imm1, imm2),\n");
    code.push_str("        1 => (imm0, (imm1 & !mask) | shifted_value, imm2),\n");
    code.push_str("        2 => (imm0, imm1, (imm2 & !mask) | shifted_value),\n");
    code.push_str("        _ => unreachable!(),\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    // write_target_to_imms function
    code.push_str("/// Write a target pointer to imm fields\n");
    code.push_str("#[inline(always)]\n");
    code.push_str("pub fn write_target_to_imms(imm0: u64, imm1: u64, imm2: u64, field: &TargetField, value: u64) -> (u64, u64, u64) {\n");
    code.push_str("    debug_assert_eq!(field.bit_offset, 0);\n");
    code.push_str("    match field.imm_index {\n");
    code.push_str("        0 => (value, imm1, imm2),\n");
    code.push_str("        1 => (imm0, value, imm2),\n");
    code.push_str("        2 => (imm0, imm1, value),\n");
    code.push_str("        _ => unreachable!(),\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    code
}

/// Generate encoding.rs
fn generate_rust(handlers: &HandlersFile, out_dir: &PathBuf) {
    let patterns = handlers.all_patterns();
    let mut code = String::new();

    // Header
    code.push_str("// Auto-generated by build.rs from handlers.toml\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("use super::instruction::Instruction;\n\n");

    // Generate PatternKind enum
    code.push_str(&generate_pattern_kind_enum(&patterns));

    // Generate PatternData enum
    code.push_str(&generate_pattern_data_enum(&patterns));

    // Generate slot helpers
    code.push_str(&generate_slot_helpers(&patterns));

    // Generate each pattern's encode/decode functions
    for (name, fields) in &patterns {
        code.push_str(&generate_rust_pattern(name, fields));
    }

    // Generate finalize_pattern_data function
    code.push_str(&generate_finalize_function(&patterns));

    let out_path = out_dir.join("fast_encoding.rs");
    fs::write(&out_path, code).unwrap_or_else(|_| panic!("Failed to write {:?}", out_path));
}

/// Generate encoding.h
fn generate_c(handlers: &HandlersFile, out_dir: &PathBuf) {
    let patterns = handlers.all_patterns();
    let mut code = String::new();

    // Header
    code.push_str("// Auto-generated by build.rs from handlers.toml\n");
    code.push_str("// DO NOT EDIT MANUALLY\n\n");
    code.push_str("#pragma once\n");
    code.push_str("#include <stdint.h>\n\n");

    // Generate each pattern
    for (name, fields) in &patterns {
        code.push_str(&generate_c_pattern(name, fields));
    }

    let out_path = out_dir.join("fast_encoding.h");
    fs::write(&out_path, code).unwrap_or_else(|_| panic!("Failed to write {:?}", out_path));
}

/// Main entry point - generates fast_encoding.rs and fast_encoding.h
pub fn generate(handlers: &HandlersFile, out_dir: &PathBuf) {
    generate_rust(handlers, out_dir);
    generate_c(handlers, out_dir);
}
