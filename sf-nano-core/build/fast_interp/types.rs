// Fast interpreter build types
// Data structures for handlers.toml (merged handler + encoding definitions)

use serde::Deserialize;
use std::fmt;

/// Field kind for encoding patterns
/// Determines how a field is treated during slot fixup and branch patching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    /// No fixup needed (default)
    Constant,
    /// Needs slot fixup (stack offset adjustment)
    Slot,
    /// Branch target pointer
    Target,
}

impl Default for FieldKind {
    fn default() -> Self {
        FieldKind::Constant
    }
}

/// Dispatch mode for next-handler preloading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchMode {
    /// Guard-check pattern: preloads next handler, checks if np == pc_next(pc)
    /// (default when dispatch is absent — represented as Option::None)
    GuardCheck,
    /// Always-reload pattern (for handlers that never return pc_next)
    Nonlinear,
}

/// WebAssembly value type for handler type expansion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WasmValType {
    I32,
    I64,
    F32,
    F64,
}

impl fmt::Display for WasmValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WasmValType::I32 => write!(f, "i32"),
            WasmValType::I64 => write!(f, "i64"),
            WasmValType::F32 => write!(f, "f32"),
            WasmValType::F64 => write!(f, "f64"),
        }
    }
}

impl WasmValType {
    /// Get uppercase form for opcode generation (e.g., "I32", "F64")
    pub fn to_uppercase(&self) -> &'static str {
        match self {
            WasmValType::I32 => "I32",
            WasmValType::I64 => "I64",
            WasmValType::F32 => "F32",
            WasmValType::F64 => "F64",
        }
    }
}

/// WebAssembly opcode prefix byte
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum OpcodePrefix {
    /// Regular opcodes (0x00-0xBF)
    #[serde(rename = "op")]
    Op,
    /// Saturating truncation and bulk memory (0xFC)
    #[serde(rename = "fc")]
    Fc,
    /// GC opcodes (0xFB)
    #[serde(rename = "fb")]
    Fb,
}

impl Default for OpcodePrefix {
    fn default() -> Self {
        OpcodePrefix::Op
    }
}

impl OpcodePrefix {
    /// Get the prefix string for generated code (e.g., "OP", "FC", "FB")
    pub fn as_str(&self) -> &'static str {
        match self {
            OpcodePrefix::Op => "OP",
            OpcodePrefix::Fc => "FC",
            OpcodePrefix::Fb => "FB",
        }
    }
}

/// Op category for handler classification.
/// Used by fusion code generation to determine how an op behaves
/// (side effects, stack effects, ctx requirements, immediate types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpCategory {
    /// Pure binop: pop 2, push 1, no side effects (e.g., i32_add, f64_mul)
    PureBinop,
    /// Trapping binop: pop 2, push 1, needs ctx (e.g., i32_div_s, i64_rem_u)
    TrappingBinop,
    /// Pure unary: pop 1, push 1, no side effects (e.g., i32_clz, f32_neg, conversions)
    PureUnary,
    /// Load: pop 1 (addr), push 1, needs ctx, has MemArg immediate
    Load,
    /// Store: pop 2 (addr, val), push 0, needs ctx, has MemArg immediate
    Store,
}

/// TOS stack-effect pattern for handler variants
/// Determines how many operands are popped/pushed for register allocation
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum TosPattern {
    /// Handler pops N operands and pushes M results
    /// { pop = N, push = M } in TOML
    PopPush { pop: u8, push: u8 },
    /// String patterns - "none" or "all"
    /// Note: PopPush must come first since untagged tries in order
    String(TosPatternString),
}

/// String-based TOS patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TosPatternString {
    None,
    All,
}

impl<'de> Deserialize<'de> for TosPatternString {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "none" => Ok(TosPatternString::None),
            "all" => Ok(TosPatternString::All),
            _ => Err(serde::de::Error::custom(format!(
                "expected 'none' or 'all', got '{}'", s
            ))),
        }
    }
}

#[allow(dead_code)]
impl TosPattern {
    /// Check if this is the "none" pattern
    pub fn is_none(&self) -> bool {
        matches!(self, TosPattern::String(TosPatternString::None))
    }

    /// Check if this is the "all" pattern (spill/fill)
    pub fn is_all(&self) -> bool {
        matches!(self, TosPattern::String(TosPatternString::All))
    }

    /// Get pop count (0 for String patterns)
    pub fn pop_count(&self) -> u8 {
        match self {
            TosPattern::String(_) => 0,
            TosPattern::PopPush { pop, .. } => *pop,
        }
    }

    /// Get push count (0 for String patterns)
    pub fn push_count(&self) -> u8 {
        match self {
            TosPattern::String(_) => 0,
            TosPattern::PopPush { push, .. } => *push,
        }
    }
}

/// Trait for handler types that generate C wrappers, extern decls, and lookup tables.
/// Implemented by both `HandlerDef` (regular handlers) and `FusedHandler` (super-instructions).
/// Eliminates duplicated handler-vs-fused iteration in the generators.
pub trait HandlerVariantSource {
    /// Get the expanded handler names (after type expansion).
    /// HandlerDef may return multiple names (e.g., ["i32_add", "i64_add"]),
    /// FusedHandler always returns a single name.
    fn expanded_names(&self) -> Vec<String>;

    /// Get the TOS stack-effect pattern, if any.
    fn tos_pattern(&self) -> Option<&TosPattern>;

    /// Whether the handler is implemented in C.
    fn c_impl(&self) -> bool;

    /// Get the dispatch mode.
    fn dispatch(&self) -> Option<DispatchMode>;

    /// Whether this handler needs D1-DN variant generation (PopPush pattern).
    fn needs_variants(&self) -> bool {
        matches!(self.tos_pattern(), Some(TosPattern::PopPush { .. }))
    }
}

impl HandlerVariantSource for HandlerDef {
    fn expanded_names(&self) -> Vec<String> {
        self.expand().into_iter().map(|e| e.name).collect()
    }

    fn tos_pattern(&self) -> Option<&TosPattern> {
        self.tos_pattern.as_ref()
    }

    fn c_impl(&self) -> bool {
        self.c_impl
    }

    fn dispatch(&self) -> Option<DispatchMode> {
        self.dispatch
    }
}

impl HandlerVariantSource for FusedHandler {
    fn expanded_names(&self) -> Vec<String> {
        vec![self.op.clone()]
    }

    fn tos_pattern(&self) -> Option<&TosPattern> {
        self.tos_pattern.as_ref()
    }

    fn c_impl(&self) -> bool {
        self.c_impl
    }

    fn dispatch(&self) -> Option<DispatchMode> {
        // Fused patterns ending with br_if always need nonlinear dispatch:
        // the taken branch target is never the next sequential instruction.
        // Patterns ending with if_ use guard-check dispatch (default):
        // the then-path falls through to pc_next(pc) which is linear.
        // Handles variant-qualified names (e.g., "br_if_d1").
        if self.dispatch.is_some() {
            self.dispatch
        } else if self.pattern.iter().any(|op| op == "br_if" || op.starts_with("br_if_d")) {
            Some(DispatchMode::Nonlinear)
        } else {
            // if_ patterns and everything else use guard-check (None = default)
            None
        }
    }
}

/// Field definition for encoding patterns
#[derive(Debug, Deserialize, Clone)]
pub struct Field {
    pub name: String,
    pub bits: u32,
    /// Field kind: Constant (no fixup, default), Slot (needs fixup), Target (branch target)
    #[serde(default)]
    pub kind: FieldKind,
    /// For fused handlers: index into `pattern[]` that this field's immediate comes from.
    /// E.g., `from = 0` means extract from the first instruction in the fused pattern.
    #[serde(default)]
    pub from: Option<usize>,
}

/// Shared pattern definition (for [[pattern]] entries)
#[derive(Debug, Deserialize, Clone)]
pub struct SharedPattern {
    pub name: String,
    pub fields: Vec<Field>,
}

/// Encoding specification for a handler
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum EncodingSpec {
    /// Inline field definition: encoding.fields = [...]
    Inline { fields: Vec<Field> },
}

/// Handler definition from handlers.toml
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct HandlerDef {
    pub op: String,
    /// Whether this is a custom/complex handler
    #[serde(default)]
    pub custom: bool,
    /// Type variants to expand (e.g., [i32, i64, f32, f64])
    #[serde(default)]
    pub types: Vec<WasmValType>,
    /// Optional wasm_op override for opcode mapping
    #[serde(default)]
    pub wasm_op: Option<String>,
    /// If true, the handler is implemented in C
    #[serde(default)]
    pub c_impl: bool,
    /// Encoding specification (inline fields or "stack_only")
    #[serde(default)]
    pub encoding: Option<EncodingSpec>,
    /// Reference to a shared pattern name
    #[serde(default)]
    pub pattern: Option<String>,
    /// TOS stack-effect pattern for variant generation
    /// Required for all handlers in TOS-enabled builds
    #[serde(default)]
    pub tos_pattern: Option<TosPattern>,
    /// Dispatch mode for next-handler preloading.
    /// - None (default): guard-check pattern (works for linear, trapping, and conditional handlers)
    /// - Some(Nonlinear): always-reload pattern (for handlers that never return pc_next)
    #[serde(default)]
    pub dispatch: Option<DispatchMode>,
    /// Opcode prefix byte for the handler map.
    /// Default is Op (regular opcodes). Use Fc for 0xFC-prefixed, Fb for 0xFB-prefixed.
    #[serde(default)]
    pub opcode_prefix: OpcodePrefix,
    /// Op category for fusion classification.
    /// Determines how this op behaves in fused instruction patterns.
    #[serde(default)]
    pub category: Option<OpCategory>,
}

/// Fused (super-instruction) handler definition from handlers.toml
/// Represents a sequence of base instructions fused into a single handler
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct FusedHandler {
    /// Handler name (e.g., "local_get_i32_const")
    pub op: String,
    /// Sequence of base operations to match (e.g., ["local_get", "i32_const"])
    pub pattern: Vec<String>,
    /// If true, the handler is implemented in C
    #[serde(default)]
    pub c_impl: bool,
    /// Encoding specification for the fused instruction
    #[serde(default)]
    pub encoding: Option<EncodingSpec>,
    /// TOS stack-effect pattern for variant generation
    /// Usually "none" for fused handlers since they manage stack internally
    #[serde(default)]
    pub tos_pattern: Option<TosPattern>,
    /// Dispatch mode for next-handler preloading.
    /// - None (default): guard-check pattern
    /// - Some(Nonlinear): always-reload pattern
    #[serde(default)]
    pub dispatch: Option<DispatchMode>,
}

/// Root structure for handlers.toml file
#[derive(Debug, Deserialize)]
pub struct HandlersFile {
    /// Shared patterns ([[pattern]] entries)
    #[serde(default)]
    pub pattern: Vec<SharedPattern>,
    /// Handler definitions ([[handler]] entries)
    pub handler: Vec<HandlerDef>,
    /// Fused instruction definitions ([[fused]] entries)
    /// Populated from handlers_fused.toml (auto-generated by discover-fusion)
    #[serde(default)]
    pub fused: Vec<FusedHandler>,
}

/// Separate file for auto-generated fused instructions (handlers_fused.toml)
#[derive(Debug, Deserialize)]
pub struct FusedFile {
    #[serde(default)]
    pub fused: Vec<FusedHandler>,
}

/// Expanded handler with concrete type (for code generation)
#[derive(Debug, Clone)]
pub struct ExpandedHandler {
    /// Full handler name (e.g., "i32_add", "f64_mul")
    pub name: String,
    /// The wasm opcode pattern (e.g., "I32_ADD")
    pub wasm_op: Option<String>,
    /// Opcode prefix (OP/FC/FB), propagated from HandlerDef
    pub opcode_prefix: OpcodePrefix,
}

impl HandlerDef {
    /// Expand this handler definition into concrete handlers
    /// If types is empty, returns single handler with op as-is
    /// Otherwise, returns one handler per type (e.g., i32_add, i64_add, etc.)
    pub fn expand(&self) -> Vec<ExpandedHandler> {
        if self.types.is_empty() {
            // No type expansion - use op directly
            vec![ExpandedHandler {
                name: self.op.clone(),
                wasm_op: self.wasm_op.clone(),
                opcode_prefix: self.opcode_prefix,
            }]
        } else {
            // Expand for each type
            self.types
                .iter()
                .map(|t| {
                    let name = format!("{}_{}", t, self.op);
                    let wasm_op = self.wasm_op.clone().unwrap_or_else(|| {
                        format!("{}_{}", t.to_uppercase(), self.op.to_uppercase())
                    });
                    ExpandedHandler {
                        name,
                        wasm_op: Some(wasm_op),
                        opcode_prefix: self.opcode_prefix,
                    }
                })
                .collect()
        }
    }

    /// Get the encoding pattern name for this handler
    /// Returns: pattern name (shared or inline) or None for handlers without immediates
    #[allow(dead_code)]
    pub fn encoding_pattern(&self) -> Option<String> {
        if let Some(ref pattern) = self.pattern {
            Some(pattern.clone())
        } else if let Some(EncodingSpec::Inline { .. }) = self.encoding {
            Some(self.op.clone()) // Use op name for inline patterns
        } else {
            None // No immediates
        }
    }

    /// Get fields for this handler (either from inline encoding or shared pattern)
    #[allow(dead_code)]
    pub fn get_fields<'a>(&'a self, patterns: &'a [SharedPattern]) -> &'a [Field] {
        if let Some(ref pattern_name) = self.pattern {
            // Reference to shared pattern
            patterns
                .iter()
                .find(|p| &p.name == pattern_name)
                .map(|p| p.fields.as_slice())
                .unwrap_or(&[])
        } else if let Some(EncodingSpec::Inline { ref fields }) = self.encoding {
            fields.as_slice()
        } else {
            &[]
        }
    }
}

impl FusedHandler {
    /// Get encoding fields for this fused handler
    #[allow(dead_code)]
    pub fn get_fields(&self) -> &[Field] {
        if let Some(EncodingSpec::Inline { ref fields }) = self.encoding {
            fields.as_slice()
        } else {
            &[]
        }
    }
}

impl HandlersFile {
    /// Iterate all handler variant sources (regular handlers then fused handlers).
    pub fn all_variant_sources(&self) -> impl Iterator<Item = &dyn HandlerVariantSource> {
        self.handler
            .iter()
            .map(|h| h as &dyn HandlerVariantSource)
            .chain(self.fused.iter().map(|f| f as &dyn HandlerVariantSource))
    }

    /// Validate the parsed handlers file.
    /// Panics with a clear message on any validation failure.
    /// Call after parsing and merging fused handlers.
    pub fn validate(&self) {
        use std::collections::HashSet;

        let mut errors: Vec<String> = Vec::new();

        // 1. Duplicate op names (across handlers and fused)
        let mut seen_ops: HashSet<&str> = HashSet::new();
        for h in &self.handler {
            if !seen_ops.insert(&h.op) {
                errors.push(format!("Duplicate handler op: '{}'", h.op));
            }
        }
        for f in &self.fused {
            if !seen_ops.insert(&f.op) {
                errors.push(format!("Duplicate fused handler op: '{}'", f.op));
            }
        }

        // 2. Duplicate shared pattern names
        let mut seen_patterns: HashSet<&str> = HashSet::new();
        for p in &self.pattern {
            if !seen_patterns.insert(&p.name) {
                errors.push(format!("Duplicate shared pattern name: '{}'", p.name));
            }
        }

        // 3. Invalid pattern references
        let pattern_names: HashSet<&str> = self.pattern.iter().map(|p| p.name.as_str()).collect();
        for h in &self.handler {
            if let Some(ref pat) = h.pattern {
                if !pattern_names.contains(pat.as_str()) {
                    errors.push(format!(
                        "Handler '{}': references unknown pattern '{}'",
                        h.op, pat
                    ));
                }
            }
        }

        // 4. Encoding bit overflow (max 192 bits = 3 × 64-bit imm slots)
        let validate_fields = |name: &str, fields: &[Field], errors: &mut Vec<String>| {
            let total_bits: u32 = fields.iter().map(|f| f.bits).sum();
            if total_bits > 192 {
                errors.push(format!(
                    "'{}': encoding uses {} bits, exceeds 192-bit limit (3 × 64)",
                    name, total_bits
                ));
            }
            for field in fields {
                if !matches!(field.bits, 8 | 16 | 32 | 64) {
                    errors.push(format!(
                        "'{}': field '{}' has unsupported bit width {}",
                        name, field.name, field.bits
                    ));
                }
            }
        };

        for p in &self.pattern {
            validate_fields(&p.name, &p.fields, &mut errors);
        }
        for h in &self.handler {
            if let Some(EncodingSpec::Inline { ref fields }) = h.encoding {
                validate_fields(&h.op, fields, &mut errors);
            }
        }
        for f in &self.fused {
            if let Some(EncodingSpec::Inline { ref fields }) = f.encoding {
                validate_fields(&f.op, fields, &mut errors);
            }
        }

        // 5. Fused field from-index bounds
        for f in &self.fused {
            let pattern_len = f.pattern.len();
            for field in f.get_fields() {
                if let Some(from_idx) = field.from {
                    if from_idx >= pattern_len {
                        errors.push(format!(
                            "Fused '{}': field '{}' has from={} but pattern has only {} ops",
                            f.op, field.name, from_idx, pattern_len
                        ));
                    }
                }
            }
        }

        // 6. Handler must not have both pattern and inline encoding
        for h in &self.handler {
            if h.pattern.is_some() && h.encoding.is_some() {
                errors.push(format!(
                    "Handler '{}': cannot have both 'pattern' and 'encoding'",
                    h.op
                ));
            }
        }

        if !errors.is_empty() {
            panic!(
                "handlers.toml validation failed ({} error{}):\n  - {}",
                errors.len(),
                if errors.len() == 1 { "" } else { "s" },
                errors.join("\n  - ")
            );
        }
    }

    /// Build a lookup map from expanded handler name to OpCategory.
    /// Only includes handlers that have a category set.
    pub fn category_map(&self) -> std::collections::HashMap<String, OpCategory> {
        let mut map = std::collections::HashMap::new();
        for h in &self.handler {
            if let Some(cat) = h.category {
                for expanded in h.expand() {
                    map.insert(expanded.name, cat);
                }
            }
        }
        map
    }

    /// Get all fused handlers
    #[allow(dead_code)]
    pub fn all_fused(&self) -> &[FusedHandler] {
        &self.fused
    }

    /// Get a shared pattern by name
    #[allow(dead_code)]
    pub fn get_pattern(&self, name: &str) -> Option<&SharedPattern> {
        self.pattern.iter().find(|p| p.name == name)
    }

    /// Collect all unique encoding patterns (shared + inline + fused)
    /// Returns (name, fields) pairs for generating PatternData enum
    /// Note: Handlers without pattern or encoding have no immediates (stack-only)
    pub fn all_patterns(&self) -> Vec<(String, Vec<Field>)> {
        use std::collections::HashMap;
        let mut patterns: HashMap<String, Vec<Field>> = HashMap::new();

        // Add shared patterns
        for p in &self.pattern {
            patterns.insert(p.name.clone(), p.fields.clone());
        }

        // Add inline patterns (use handler op name)
        for h in &self.handler {
            if h.pattern.is_none() {
                if let Some(EncodingSpec::Inline { ref fields }) = h.encoding {
                    patterns.insert(h.op.clone(), fields.clone());
                }
            }
        }

        // Add fused handler patterns (including zero-field patterns)
        for f in &self.fused {
            if let Some(EncodingSpec::Inline { ref fields }) = f.encoding {
                patterns.insert(f.op.clone(), fields.clone());
            } else {
                // Zero-field fused patterns (pure ops only, no immediates)
                patterns.insert(f.op.clone(), vec![]);
            }
        }

        // Sort for deterministic output
        let mut result: Vec<_> = patterns.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

// =============================================================================
// Shared utility functions
// =============================================================================

/// Convert snake_case to PascalCase for enum variants
pub fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Get the Rust type for a field based on its bit width
pub fn bits_to_rust_type(bits: u32) -> &'static str {
    match bits {
        8 => "u8",
        16 => "u16",
        32 => "u32",
        64 => "u64",
        _ => panic!("Unsupported bit width: {}", bits),
    }
}
