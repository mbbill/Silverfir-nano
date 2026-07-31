//! Wasm decoder to the decoded semantic function model.
//!
//! This stage stops at semantic structure:
//! - structured control targets
//! - abstract local / call / branch semantics
//! - semantic stack height for later frame sizing
//!
//! It must not:
//! - insert spill/fill
//! - assign frame slots
//! - assign transient-window handlers
//! - shape prepared execution blocks

use crate::collections;

use tracked_alloc::collections::BTreeMap;

use crate::{
    error::WasmError,
    op_decoder::{CatchClauseKind, Decoder, Immediate, OpStream, OpcodeHandler},
    opcodes::{Opcode, OpcodeFB, OpcodeFC, OpcodeFD, WasmOpcode},
    utils::payload::Payload,
    value_type::{HeapType, RefType, ValueType},
    vm::{entities::FunctionInst, tag::TagIdentity},
};

use super::{
    common::{BrTableEntry, SemanticIndex, SemanticTarget},
    context::CompileContext,
    primitive_op::{self, PrimitiveOpKind},
    semantic_ir::{SemanticCatchClause, SemanticOp, SemanticOpKind, SemanticProgram},
};

/// Semantic IR builder used by Wasm decode.
#[derive(Default)]
struct SemanticBuilder {
    ops: collections::Vec<SemanticOp>,
}

impl SemanticBuilder {
    #[inline]
    fn current_index(&self) -> SemanticIndex {
        SemanticIndex::new(self.ops.len())
    }

    fn push(&mut self, kind: impl Into<SemanticOpKind>) -> SemanticIndex {
        let idx = self.current_index();
        self.ops.push(SemanticOp { kind: kind.into() });
        idx
    }

    fn patch_target(&mut self, idx: SemanticIndex, target: SemanticTarget) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            match &mut op.kind {
                SemanticOpKind::If { else_target, .. } => *else_target = target,
                SemanticOpKind::Else { end_target } => *end_target = target,
                SemanticOpKind::Br { target: branch, .. }
                | SemanticOpKind::BrIf { target: branch, .. }
                | SemanticOpKind::BrOnNull { target: branch, .. }
                | SemanticOpKind::BrOnNonNull { target: branch, .. }
                | SemanticOpKind::BrOnCast { target: branch, .. }
                | SemanticOpKind::BrOnCastFail { target: branch, .. } => *branch = target,
                _ => {}
            }
        }
    }

    fn patch_br_table_target(
        &mut self,
        idx: SemanticIndex,
        entry_idx: usize,
        target: SemanticTarget,
    ) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            if let SemanticOpKind::BrTable { entries } = &mut op.kind {
                if let Some(entry) = entries.get_mut(entry_idx) {
                    entry.target = target;
                }
            }
        }
    }

    fn patch_try_table_catch_target(
        &mut self,
        idx: SemanticIndex,
        catch_idx: usize,
        target: SemanticTarget,
    ) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            if let SemanticOpKind::TryTable { catches, .. } = &mut op.kind {
                if let Some(catch) = catches.get_mut(catch_idx) {
                    catch.target = target;
                }
            }
        }
    }

    fn finish(
        self,
        params: u16,
        results: u16,
        local_count: u16,
        max_stack_height: u16,
        local_types: collections::Vec<ValueType>,
        result_types: collections::Vec<ValueType>,
        op_result_types: BTreeMap<usize, collections::Vec<ValueType>>,
    ) -> SemanticProgram {
        let mut ops = self.ops;
        ops.shrink_to_fit();
        SemanticProgram {
            params,
            results,
            local_count,
            max_stack_height,
            ops,
            local_types,
            result_types,
            op_result_types,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeBlockKind {
    Function,
    Block,
    Loop,
    If,
    TryTable,
}

#[derive(Clone, Copy, Debug)]
enum PendingFixupSite {
    Target,
    BrTableEntry(usize),
    TryTableCatch(usize),
}

#[derive(Clone, Debug)]
struct PendingBranchFixup {
    inst_idx: SemanticIndex,
    site: PendingFixupSite,
}

/// Catch clause info cached on a `TryTable` frame so that `throw` can match
/// statically against enclosing handlers at decode time, short-circuiting
/// the throw into a direct `Br` to the handler label.
#[derive(Clone, Debug)]
struct DecodeTryCatch {
    tag_idx: Option<u32>,
    kind: CatchClauseKind,
    /// Number of payload values the handler label expects for non-ref forms
    /// (tag's param arity for `catch`; 0 for `catch_all`). Ref forms add a
    /// trailing `(ref exn)` so their branch arity is payload_arity + 1.
    payload_arity: u16,
    /// Resolved semantic-target for the handler label. May be pending if
    /// the target block hasn't been decoded yet at try_table decode time.
    target: SemanticTarget,
    /// Start height of the handler's target frame at try_table decode time.
    /// Used to compute `stack_drop` when a throw short-circuits into a
    /// direct `Br` — drops everything above `target_start_height +
    /// payload_arity` at the throw site.
    target_start_height: u32,
    /// Original `label_idx` from the immediate — used to register a
    /// forward-branch fixup on the throw-as-Br when `target` is pending.
    /// The fixup needs to patch the emitted Br's target, which lives at a
    /// semantic-op index that isn't known until the throw is emitted.
    /// So at throw time we recompute the enclosing frame's depth and
    /// re-register rather than carry state here.
    label_idx: u32,
}

/// Result of statically matching a `throw $tag` against an enclosing
/// `try_table`. Populated by `match_throw_to_catch` when lowering a direct
/// throw to a `Br` is viable.
#[derive(Clone, Copy, Debug)]
struct StaticCatchMatch {
    stack_drop: u32,
    arity: u16,
    target: SemanticTarget,
    forwards_exn: bool,
    /// If `Some`, the target is still pending and a forward-branch fixup
    /// must be registered at the given frame depth (from the throw's
    /// control stack position).
    pending_fixup_depth: Option<u32>,
}

#[derive(Clone, Debug)]
enum InlineThrowPayloadOp {
    I32Const(u32),
    I64Const(u64),
    F32Const(u32),
    F64Const(u64),
    RefFunc(u32),
}

#[derive(Clone, Debug)]
struct InlineThrowWrapper {
    tag_idx: Option<u32>,
    tag_identity: TagIdentity,
    payload_ops: collections::Vec<InlineThrowPayloadOp>,
}

#[derive(Clone, Debug)]
struct InlineConditionalThrowWrapper {
    tag_idx: u32,
    compare_const: u32,
    result_const: u32,
}

#[derive(Clone, Debug)]
struct DecodeControlFrame {
    kind: DecodeBlockKind,
    start_height: usize,
    param_count: u16,
    result_count: u16,
    param_types: collections::Vec<ValueType>,
    result_types: collections::Vec<ValueType>,
    start_inst_idx: SemanticTarget,
    if_inst_idx: Option<SemanticIndex>,
    pending_fixups: collections::Vec<PendingBranchFixup>,
    /// Populated only for `DecodeBlockKind::TryTable` frames.
    try_catches: collections::Vec<DecodeTryCatch>,
}

#[inline]
fn decode_immediate_mismatch(message: &'static str) -> WasmError {
    WasmError::internal(message)
}

#[inline]
fn expect_function_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::FunctionIndex(func_idx) => Ok(*func_idx),
        _ => Err(decode_immediate_mismatch(
            "decoder expected function index immediate",
        )),
    }
}

#[inline]
fn expect_call_indirect_immediate(imm: &Immediate) -> Result<(u32, u32), WasmError> {
    match imm {
        Immediate::CallIndirectArgs { typeidx, tableidx } => Ok((*typeidx, *tableidx)),
        _ => Err(decode_immediate_mismatch(
            "decoder expected call_indirect immediate",
        )),
    }
}

#[inline]
fn expect_type_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::TypeIndex(typeidx) => Ok(*typeidx),
        _ => Err(decode_immediate_mismatch(
            "decoder expected type index immediate",
        )),
    }
}

#[inline]
fn expect_v128_immediate(imm: &Immediate) -> Result<[u8; 16], WasmError> {
    match imm {
        Immediate::V128(value) => Ok(*value),
        _ => Err(decode_immediate_mismatch("decoder expected v128 immediate")),
    }
}

#[inline]
fn expect_lane_index_immediate(imm: &Immediate) -> Result<u8, WasmError> {
    match imm {
        Immediate::LaneIndex(lane) => Ok(*lane),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD lane immediate",
        )),
    }
}

#[inline]
fn expect_shuffle_mask_immediate(imm: &Immediate) -> Result<[u8; 16], WasmError> {
    match imm {
        Immediate::ShuffleMask(lanes) => Ok(*lanes),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD shuffle immediate",
        )),
    }
}

#[inline]
fn expect_memarg_lane_immediate(imm: &Immediate) -> Result<(u64, u32, u8), WasmError> {
    match imm {
        Immediate::MemArgLane {
            offset,
            memidx,
            lane,
            ..
        } => Ok((*offset, *memidx, *lane)),
        _ => Err(decode_immediate_mismatch(
            "decoder expected SIMD memarg-lane immediate",
        )),
    }
}

#[inline]
fn is_relaxed_simd_opcode(op: OpcodeFD) -> bool {
    use OpcodeFD::*;

    matches!(
        op,
        I8X16_RELAXED_SWIZZLE
            | I32X4_RELAXED_TRUNC_F32X4_S
            | I32X4_RELAXED_TRUNC_F32X4_U
            | I32X4_RELAXED_TRUNC_F64X2_S_ZERO
            | I32X4_RELAXED_TRUNC_F64X2_U_ZERO
            | I16X8_RELAXED_Q15MULR_S
            | I8X16_RELAXED_LANESELECT
            | I16X8_RELAXED_LANESELECT
            | I32X4_RELAXED_LANESELECT
            | I64X2_RELAXED_LANESELECT
            | F32X4_RELAXED_MIN
            | F32X4_RELAXED_MAX
            | F64X2_RELAXED_MIN
            | F64X2_RELAXED_MAX
            | I16X8_RELAXED_DOT_I8X16_I7X16_S
            | F32X4_RELAXED_MADD
            | F32X4_RELAXED_NMADD
            | F64X2_RELAXED_MADD
            | F64X2_RELAXED_NMADD
            | I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S
    )
}

#[inline]
fn is_exn_ref_type(ty: ValueType) -> bool {
    let ValueType::Ref(ref_ty) = ty else {
        return false;
    };
    matches!(
        ref_ty.heap_type,
        HeapType::Abstract(crate::value_type::AbstractHeapType::Exn)
    )
}

fn inline_throw_tag(
    compile: CompileContext<'_>,
    owner_store: &crate::vm::jit::store::Store,
    tag_idx: u32,
) -> Option<(Option<u32>, TagIdentity, usize)> {
    let tag_inst = owner_store.module().tags.get(tag_idx as usize).copied()?;
    let tag_arity = owner_store
        .module()
        .types
        .get_function_type(tag_inst.type_index)?
        .params()
        .len();
    let caller_tag_idx = compile
        .store
        .module()
        .tags
        .iter()
        .position(|tag| tag.handle == tag_inst.handle)
        .map(|idx| idx as u32);
    Some((caller_tag_idx, tag_inst.handle, tag_arity))
}

fn with_inline_local_spec<R>(
    compile: CompileContext<'_>,
    func_idx: u32,
    analyze: impl for<'owner> FnOnce(
        &'owner crate::module::entities::FunctionSpec,
        &'owner crate::vm::jit::store::Store,
    ) -> Option<R>,
) -> Option<R> {
    match compile.store.function(func_idx as usize) {
        FunctionInst::Local { spec, .. } => analyze(spec, compile.store),
        FunctionInst::Linked { handle, .. } => {
            let entry = compile.store.function_entry_for_handle(*handle)?;
            if entry.owner == compile.store.instance_backref().self_id() {
                let FunctionInst::Local { spec, .. } =
                    compile.store.function(entry.local_index as usize)
                else {
                    return None;
                };
                return analyze(spec, compile.store);
            }

            let owner = compile.store.instance_backref().checkout(entry.owner)?;
            let owner_store = owner.jit()?;
            let FunctionInst::Local { spec, .. } = owner_store.function(entry.local_index as usize)
            else {
                return None;
            };
            analyze(spec, owner_store)
        }
        FunctionInst::Host { .. } => None,
    }
}

fn analyze_simple_inline_throw_wrapper(
    compile: CompileContext<'_>,
    spec: &crate::module::entities::FunctionSpec,
    owner_store: &crate::vm::jit::store::Store,
) -> Option<InlineThrowWrapper> {
    let params = spec.func_type().params().len() as u32;
    let bytes: &[u8] = spec.code();
    let mut code: Payload = bytes.into();
    let mut next_param = 0u32;
    let mut payload_ops = collections::Vec::new();
    let mut tag_idx = None;

    while !code.is_empty() {
        let op: Opcode = code.read_u8().ok()?.try_into().ok()?;
        match op {
            Opcode::LOCAL_GET => {
                let idx = code.read_leb128_u32().ok()?;
                if idx != next_param || idx >= params {
                    return None;
                }
                next_param = next_param.saturating_add(1);
            }
            Opcode::I32_CONST => {
                payload_ops.push(InlineThrowPayloadOp::I32Const(
                    code.read_leb128_i32().ok()? as u32
                ));
            }
            Opcode::I64_CONST => {
                payload_ops.push(InlineThrowPayloadOp::I64Const(
                    code.read_leb128_i64().ok()? as u64
                ));
            }
            Opcode::F32_CONST => {
                payload_ops.push(InlineThrowPayloadOp::F32Const(
                    code.read_f32().ok()?.to_bits(),
                ));
            }
            Opcode::F64_CONST => {
                payload_ops.push(InlineThrowPayloadOp::F64Const(
                    code.read_f64().ok()?.to_bits(),
                ));
            }
            Opcode::REF_FUNC => {
                payload_ops.push(InlineThrowPayloadOp::RefFunc(code.read_leb128_u32().ok()?));
            }
            Opcode::THROW => {
                tag_idx = Some(code.read_leb128_u32().ok()?);
                break;
            }
            _ => return None,
        }
    }

    let tag_idx = tag_idx?;
    while !code.is_empty() {
        let op: Opcode = code.read_u8().ok()?.try_into().ok()?;
        if op != Opcode::END {
            return None;
        }
    }

    let (caller_tag_idx, tag_identity, tag_arity) =
        inline_throw_tag(compile, owner_store, tag_idx)?;
    let payload_arity = next_param as usize + payload_ops.len();
    if payload_arity != tag_arity {
        return None;
    }

    Some(InlineThrowWrapper {
        tag_idx: caller_tag_idx,
        tag_identity,
        payload_ops,
    })
}

fn simple_inline_throw_wrapper(
    compile: CompileContext<'_>,
    func_idx: u32,
) -> Option<InlineThrowWrapper> {
    with_inline_local_spec(compile, func_idx, |spec, owner_store| {
        analyze_simple_inline_throw_wrapper(compile, spec, owner_store)
    })
}

fn analyze_simple_inline_conditional_throw_wrapper(
    compile: CompileContext<'_>,
    spec: &crate::module::entities::FunctionSpec,
    owner_store: &crate::vm::jit::store::Store,
) -> Option<InlineConditionalThrowWrapper> {
    if spec.func_type().params() != [ValueType::I32]
        || spec.func_type().results() != [ValueType::I32]
    {
        return None;
    }

    let bytes: &[u8] = spec.code();
    let mut code: Payload = bytes.into();

    if Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::LOCAL_GET
        || code.read_leb128_u32().ok()? != 0
        || Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::I32_CONST
    {
        return None;
    }
    let compare_const = code.read_leb128_i32().ok()? as u32;
    if Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::I32_NE
        || Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::IF
        || code.read_u8().ok()? != 0x40
        || Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::THROW
    {
        return None;
    }
    let raw_tag_idx = code.read_leb128_u32().ok()?;
    if Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::END
        || Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::I32_CONST
    {
        return None;
    }
    let result_const = code.read_leb128_i32().ok()? as u32;
    while !code.is_empty() {
        if Opcode::try_from(code.read_u8().ok()?).ok()? != Opcode::END {
            return None;
        }
    }

    let (caller_tag_idx, _tag_handle, tag_arity) =
        inline_throw_tag(compile, owner_store, raw_tag_idx)?;
    if tag_arity != 0 {
        return None;
    }
    Some(InlineConditionalThrowWrapper {
        tag_idx: caller_tag_idx?,
        compare_const,
        result_const,
    })
}

fn simple_inline_conditional_throw_wrapper(
    compile: CompileContext<'_>,
    func_idx: u32,
) -> Option<InlineConditionalThrowWrapper> {
    with_inline_local_spec(compile, func_idx, |spec, owner_store| {
        analyze_simple_inline_conditional_throw_wrapper(compile, spec, owner_store)
    })
}

/// Decode-time semantic context.
struct DecodeContext<'a> {
    compile: CompileContext<'a>,
    builder: SemanticBuilder,
    control: collections::Vec<DecodeControlFrame>,
    height: usize,
    max_height: usize,
    unreachable: bool,
    type_stack: collections::Vec<ValueType>,
    exn_tag_stack: collections::Vec<Option<u32>>,
    local_exn_tags: collections::Vec<Option<u32>>,
    last_static_exn_tag: Option<u32>,
    /// Per-op result types for calls and typed blocks.
    op_result_types: BTreeMap<usize, collections::Vec<ValueType>>,
}

impl<'a> DecodeContext<'a> {
    fn new(compile: CompileContext<'a>) -> Self {
        Self {
            compile,
            builder: SemanticBuilder::default(),
            control: collections::vec![DecodeControlFrame {
                kind: DecodeBlockKind::Function,
                start_height: 0,
                param_count: 0,
                result_count: compile.results,
                param_types: collections::Vec::new(),
                result_types: compile.result_types.to_vec().into(),
                start_inst_idx: SemanticTarget::new(0),
                if_inst_idx: None,
                pending_fixups: collections::Vec::new(),
                try_catches: collections::Vec::new(),
            }],
            height: 0,
            max_height: 0,
            unreachable: false,
            type_stack: collections::Vec::new(),
            exn_tag_stack: collections::Vec::new(),
            local_exn_tags: collections::vec![None; compile.local_count as usize],
            last_static_exn_tag: None,
            op_result_types: BTreeMap::new(),
        }
    }

    #[inline]
    fn current_index(&self) -> SemanticIndex {
        self.builder.current_index()
    }

    #[inline]
    fn push_op(&mut self, kind: impl Into<SemanticOpKind>) -> SemanticIndex {
        self.builder.push(kind)
    }

    #[inline]
    fn patch_target(&mut self, idx: SemanticIndex, target: SemanticTarget) {
        self.builder.patch_target(idx, target);
    }

    #[inline]
    fn patch_br_table_target(
        &mut self,
        idx: SemanticIndex,
        entry_idx: usize,
        target: SemanticTarget,
    ) {
        self.builder.patch_br_table_target(idx, entry_idx, target);
    }

    #[inline]
    fn push_value(&mut self) {
        if !self.unreachable {
            self.height += 1;
            self.max_height = self.max_height.max(self.height);
            self.exn_tag_stack.push(None);
        }
    }

    #[inline]
    fn push_values(&mut self, count: usize) {
        if !self.unreachable {
            self.height += count;
            self.max_height = self.max_height.max(self.height);
            self.exn_tag_stack.extend((0..count).map(|_| None));
        }
    }

    #[inline]
    fn pop_values(&mut self, count: usize) {
        if !self.unreachable {
            self.height = self.height.saturating_sub(count);
            let new_len = self.exn_tag_stack.len().saturating_sub(count);
            self.exn_tag_stack.truncate(new_len);
        }
    }

    #[inline]
    fn push_typed_value(&mut self, ty: ValueType) {
        self.push_typed_value_with_exn_tag(ty, None);
    }

    #[inline]
    fn push_typed_value_with_exn_tag(&mut self, ty: ValueType, exn_tag: Option<u32>) {
        if !self.unreachable {
            self.type_stack.push(ty);
        }
        self.push_value();
        if !self.unreachable {
            if let Some(top) = self.exn_tag_stack.last_mut() {
                *top = exn_tag.filter(|_| is_exn_ref_type(ty));
            }
        }
    }

    #[inline]
    fn push_typed_values(&mut self, tys: &[ValueType]) {
        for ty in tys {
            self.push_typed_value(*ty);
        }
    }

    #[inline]
    fn pop_typed_values(&mut self, count: usize) {
        if !self.unreachable {
            let new_len = self.type_stack.len().saturating_sub(count);
            self.type_stack.truncate(new_len);
        }
        self.pop_values(count);
    }

    #[inline]
    fn pop_typed_value(&mut self) -> ValueType {
        if self.unreachable {
            self.pop_values(1);
            return ValueType::Unknown;
        }
        let ty = self.type_stack.pop().unwrap_or(ValueType::Unknown);
        self.pop_values(1);
        ty
    }

    #[inline]
    fn top_type(&self) -> ValueType {
        if self.unreachable {
            ValueType::Unknown
        } else {
            self.type_stack
                .last()
                .copied()
                .unwrap_or(ValueType::Unknown)
        }
    }

    #[inline]
    fn set_unreachable(&mut self) {
        if let Some(frame) = self.control.last() {
            self.type_stack.truncate(frame.start_height);
            self.exn_tag_stack.truncate(frame.start_height);
        }
        self.unreachable = true;
    }

    fn types_at(&self, start: usize, count: u16) -> collections::Vec<ValueType> {
        let end = start.saturating_add(count as usize);
        if end <= self.type_stack.len() {
            self.type_stack[start..end].to_vec().into()
        } else {
            (0..count as usize)
                .map(|idx| {
                    self.type_stack
                        .get(start + idx)
                        .copied()
                        .unwrap_or(ValueType::Unknown)
                })
                .collect()
        }
    }

    fn enter_block(
        &mut self,
        kind: DecodeBlockKind,
        params: u16,
        results: u16,
        result_types: collections::Vec<ValueType>,
        start_inst_idx: SemanticTarget,
    ) {
        let start_height = self.height.saturating_sub(params as usize);
        let param_types = if self.unreachable {
            collections::Vec::new()
        } else {
            self.types_at(start_height, params)
        };
        self.control.push(DecodeControlFrame {
            kind,
            start_height,
            param_count: params,
            result_count: results,
            param_types,
            result_types,
            start_inst_idx,
            if_inst_idx: None,
            pending_fixups: collections::Vec::new(),
            try_catches: collections::Vec::new(),
        });
    }

    fn set_if_inst(&mut self, idx: SemanticIndex) {
        if let Some(frame) = self.control.last_mut() {
            frame.if_inst_idx = Some(idx);
        }
    }

    fn enter_else(&mut self) {
        if let Some(frame) = self.control.last() {
            self.height = frame.start_height + frame.param_count as usize;
            self.type_stack.truncate(frame.start_height);
            self.type_stack.extend_from_slice(&frame.param_types);
            self.exn_tag_stack.truncate(frame.start_height);
            self.exn_tag_stack
                .extend((0..frame.param_count as usize).map(|_| None));
            self.unreachable = false;
        }
    }

    fn exit_block(&mut self) -> Option<DecodeControlFrame> {
        let frame = self.control.pop()?;
        let block_has_exn_result = frame.result_types.iter().any(|ty| is_exn_ref_type(*ty));
        let static_exn_tag = if block_has_exn_result {
            self.last_static_exn_tag.take()
        } else {
            None
        };
        self.height = frame.start_height + frame.result_count as usize;
        self.type_stack.truncate(frame.start_height);
        self.type_stack.extend_from_slice(&frame.result_types);
        self.exn_tag_stack.truncate(frame.start_height);
        for ty in &frame.result_types {
            self.exn_tag_stack
                .push(static_exn_tag.filter(|_| is_exn_ref_type(*ty)));
        }
        self.unreachable = false;
        Some(frame)
    }

    fn handle_end(&mut self) {
        let end_idx = self.push_op(SemanticOpKind::End);
        let end_target = SemanticTarget::new(end_idx.as_usize());
        if let Some(frame) = self.exit_block() {
            if frame.kind == DecodeBlockKind::If && frame.if_inst_idx.is_some() {
                if let Some(if_idx) = frame.if_inst_idx {
                    if self.builder.ops.get(if_idx.as_usize()).is_some_and(|op| {
                        matches!(
                            op.kind,
                            SemanticOpKind::If { else_target, .. } if else_target.is_pending()
                        )
                    }) {
                        self.patch_target(if_idx, end_target);
                    }
                }
            }
            for fixup in frame.pending_fixups {
                match fixup.site {
                    PendingFixupSite::Target => {
                        self.patch_target(fixup.inst_idx, end_target);
                    }
                    PendingFixupSite::BrTableEntry(entry_idx) => {
                        self.patch_br_table_target(fixup.inst_idx, entry_idx, end_target);
                    }
                    PendingFixupSite::TryTableCatch(catch_idx) => {
                        self.builder.patch_try_table_catch_target(
                            fixup.inst_idx,
                            catch_idx,
                            end_target,
                        );
                    }
                }
            }
        }
    }

    fn frame_at_depth(&self, depth: u32) -> Option<&DecodeControlFrame> {
        let idx = self.control.len().checked_sub(depth as usize + 1)?;
        self.control.get(idx)
    }

    fn frame_at_depth_mut(&mut self, depth: u32) -> Option<&mut DecodeControlFrame> {
        let idx = self.control.len().checked_sub(depth as usize + 1)?;
        self.control.get_mut(idx)
    }

    fn branch_arity(&self, depth: u32) -> u16 {
        self.frame_at_depth(depth)
            .map(|frame| match frame.kind {
                DecodeBlockKind::Loop => frame.param_count,
                DecodeBlockKind::Function
                | DecodeBlockKind::Block
                | DecodeBlockKind::If
                | DecodeBlockKind::TryTable => frame.result_count,
            })
            .unwrap_or(0)
    }

    fn branch_types(&self, depth: u32) -> collections::Vec<ValueType> {
        self.frame_at_depth(depth)
            .map(|frame| match frame.kind {
                DecodeBlockKind::Loop => frame.param_types.clone(),
                DecodeBlockKind::Function
                | DecodeBlockKind::Block
                | DecodeBlockKind::If
                | DecodeBlockKind::TryTable => frame.result_types.clone(),
            })
            .unwrap_or_else(collections::Vec::new)
    }

    fn branch_info(&self, depth: u32) -> (u32, Option<SemanticTarget>) {
        let Some(frame) = self.frame_at_depth(depth) else {
            return (0, None);
        };

        let arity = self.branch_arity(depth) as usize;
        let stack_drop =
            self.height
                .saturating_sub(frame.start_height.saturating_add(arity)) as u32;
        let target = match frame.kind {
            DecodeBlockKind::Loop => Some(frame.start_inst_idx),
            DecodeBlockKind::Function
            | DecodeBlockKind::Block
            | DecodeBlockKind::If
            | DecodeBlockKind::TryTable => None,
        };
        (stack_drop, target)
    }

    fn register_forward_branch(
        &mut self,
        depth: u32,
        inst_idx: SemanticIndex,
        site: PendingFixupSite,
    ) {
        if let Some(frame) = self.frame_at_depth_mut(depth) {
            frame
                .pending_fixups
                .push(PendingBranchFixup { inst_idx, site });
        }
    }

    /// Walk enclosing control frames from inner-most outward looking for a
    /// `try_table` whose catch list accepts `tag_idx`. Return
    /// the `Br` lowering parameters for the first matching catch.
    /// Returns `None` if no match (throw escapes via runtime-call).
    ///
    /// The decoder runs after instantiation, so `TagInst.handle` values
    /// already reflect runtime identities — we compare by `TagIdentity`
    /// rather than `tag_idx` to correctly resolve aliasing imports (e.g.
    /// two module-local imports of the same source tag produce distinct
    /// indices but equal handles).
    fn match_throw_handle_to_catch(
        &self,
        throw_tag_identity: TagIdentity,
    ) -> Option<StaticCatchMatch> {
        for depth in 0..self.control.len() as u32 {
            let frame = self.frame_at_depth(depth)?;
            if frame.kind != DecodeBlockKind::TryTable {
                continue;
            }
            for catch in &frame.try_catches {
                match catch.kind {
                    CatchClauseKind::Catch | CatchClauseKind::CatchRef => {
                        let Some(catch_tag_idx) = catch.tag_idx else {
                            continue;
                        };
                        let Some(catch_tag_inst) =
                            self.compile.store.module().tags.get(catch_tag_idx as usize)
                        else {
                            continue;
                        };
                        if catch_tag_inst.handle != throw_tag_identity {
                            continue;
                        }
                    }
                    CatchClauseKind::CatchAll | CatchClauseKind::CatchAllRef => {}
                }

                // Everything above `handler_baseline + payload_arity` must
                // be dropped. The semantic `Br` still sees the tag payload on
                // the stack, so payload values are counted as the branch
                // arity rather than being removed by `throw`.
                let handler_baseline = catch.target_start_height as usize;
                let payload_arity = catch.payload_arity;
                let stack_drop = self
                    .height
                    .saturating_sub(handler_baseline.saturating_add(payload_arity as usize))
                    as u32;

                // If the catch's label target is still pending (forward
                // branch to a block whose End hasn't been seen yet), we
                // need to register a fixup. The depth for the fixup is
                // the handler's frame depth from the throw's position,
                // which is `depth + catch.label_idx + 1` (try_table's
                // frame is at `depth` from top; the catch's label was
                // resolved relative to outside the try_table, so +1 to
                // cross the try_table frame itself).
                let pending_fixup_depth = if catch.target.is_pending() {
                    Some(depth.saturating_add(catch.label_idx).saturating_add(1))
                } else {
                    None
                };

                return Some(StaticCatchMatch {
                    stack_drop,
                    arity: payload_arity
                        + u16::from(matches!(
                            catch.kind,
                            CatchClauseKind::CatchRef | CatchClauseKind::CatchAllRef
                        )),
                    target: catch.target,
                    forwards_exn: matches!(
                        catch.kind,
                        CatchClauseKind::CatchRef | CatchClauseKind::CatchAllRef
                    ),
                    pending_fixup_depth,
                });
            }
        }
        None
    }

    fn match_throw_to_catch(&self, tag_idx: u32) -> Option<StaticCatchMatch> {
        let throw_tag_identity = self
            .compile
            .store
            .module()
            .tags
            .get(tag_idx as usize)?
            .handle;
        self.match_throw_handle_to_catch(throw_tag_identity)
    }

    fn record_result_types(&mut self, idx: SemanticIndex, tys: &[ValueType]) {
        if !tys.is_empty() {
            self.op_result_types
                .insert(idx.as_usize(), tys.to_vec().into());
        }
    }

    fn handle_primitive(&mut self, kind: PrimitiveOpKind) {
        let (pops, pushes) = primitive_op::stack_effect(&kind);
        let result_ty = if matches!(kind, PrimitiveOpKind::Select) && pops >= 2 {
            self.type_stack
                .len()
                .checked_sub(2)
                .and_then(|idx| self.type_stack.get(idx).copied())
                .unwrap_or(ValueType::I64)
        } else if let Some(ty) = primitive_op::result_type(&kind) {
            ty
        } else {
            ValueType::I64
        };
        self.pop_typed_values(pops as usize);
        self.push_op(kind.clone());
        if pushes != 0 {
            self.push_typed_values(&collections::vec![result_ty; pushes as usize]);
        }
        if matches!(kind, PrimitiveOpKind::Unreachable) {
            self.set_unreachable();
        }
    }

    fn emit_return(&mut self) {
        let arity = self.compile.results;
        match arity {
            0 => {
                self.push_op(SemanticOpKind::ReturnVoid);
            }
            1 => {
                self.push_op(SemanticOpKind::ReturnOne);
            }
            _ => {
                self.push_op(SemanticOpKind::Return { arity });
            }
        }
        self.set_unreachable();
    }

    fn handle_call(&mut self, func_idx: u32) {
        if let Some(wrapper) = simple_inline_throw_wrapper(self.compile, func_idx) {
            if self.handle_inline_throw_wrapper(wrapper) {
                return;
            }
        }
        if let Some(wrapper) = simple_inline_conditional_throw_wrapper(self.compile, func_idx) {
            self.handle_inline_conditional_throw_wrapper(wrapper);
            return;
        }

        let (params, results) = self.compile.resolve_func_type(func_idx);
        let kind = SemanticOpKind::CallDirect {
            callee: func_idx,
            params,
            results,
        };
        let idx = self.push_op(kind);
        if results > 0 {
            let func = self.compile.store.function(func_idx as usize);
            self.record_result_types(idx, func.func_type().results());
        }
        self.pop_typed_values(params as usize);
        let func = self.compile.store.function(func_idx as usize);
        self.push_typed_values(func.func_type().results());
    }

    fn handle_inline_throw_wrapper(&mut self, wrapper: InlineThrowWrapper) -> bool {
        if let Some(tag_idx) = wrapper.tag_idx {
            for payload_op in wrapper.payload_ops {
                self.handle_inline_throw_payload_op(payload_op);
            }
            self.handle_throw(tag_idx);
            return true;
        }

        if !wrapper.payload_ops.is_empty() {
            return false;
        }

        let Some(resolved) = self.match_throw_handle_to_catch(wrapper.tag_identity) else {
            return false;
        };
        if resolved.forwards_exn {
            return false;
        }

        self.pop_typed_values(resolved.arity as usize);
        let br_idx = self.push_op(SemanticOpKind::Br {
            stack_drop: resolved.stack_drop,
            arity: resolved.arity,
            target: resolved.target,
        });
        if let Some(depth) = resolved.pending_fixup_depth {
            self.register_forward_branch(depth, br_idx, PendingFixupSite::Target);
        }
        self.set_unreachable();
        true
    }

    fn handle_inline_conditional_throw_wrapper(&mut self, wrapper: InlineConditionalThrowWrapper) {
        self.handle_primitive(PrimitiveOpKind::I32Const {
            value: wrapper.compare_const,
        });
        self.handle_primitive(PrimitiveOpKind::I32Ne);
        let idx = self.push_op(SemanticOpKind::If {
            params: 0,
            results: 0,
            else_target: SemanticTarget::pending(),
        });
        self.record_result_types(idx, &[]);
        let target = SemanticTarget::new(self.current_index().as_usize());
        self.enter_block(DecodeBlockKind::If, 0, 0, collections::Vec::new(), target);
        self.set_if_inst(idx);
        self.handle_throw(wrapper.tag_idx);
        self.handle_end();
        self.handle_primitive(PrimitiveOpKind::I32Const {
            value: wrapper.result_const,
        });
    }

    fn handle_inline_throw_payload_op(&mut self, payload_op: InlineThrowPayloadOp) {
        match payload_op {
            InlineThrowPayloadOp::I32Const(value) => {
                self.handle_primitive(PrimitiveOpKind::I32Const { value });
            }
            InlineThrowPayloadOp::I64Const(value) => {
                self.handle_primitive(PrimitiveOpKind::I64Const { value });
            }
            InlineThrowPayloadOp::F32Const(value) => {
                self.handle_primitive(PrimitiveOpKind::F32Const { value });
            }
            InlineThrowPayloadOp::F64Const(value) => {
                self.handle_primitive(PrimitiveOpKind::F64Const { value });
            }
            InlineThrowPayloadOp::RefFunc(func_idx) => {
                let op_idx = self.current_index();
                self.handle_primitive(PrimitiveOpKind::RefFunc { func_idx });
                let func_type_idx = self.compile.store.function(func_idx as usize).type_index();
                let ty = ValueType::Ref(RefType::new(false, HeapType::Concrete(func_type_idx)));
                self.record_result_types(op_idx, &[ty]);
                if !self.unreachable {
                    if let Some(last) = self.type_stack.last_mut() {
                        *last = ty;
                    }
                }
            }
        }
    }

    fn handle_throw(&mut self, tag_idx: u32) {
        let (arity, _results) = self.compile.resolve_tag_type(tag_idx);
        if let Some(resolved) = self.match_throw_to_catch(tag_idx) {
            if resolved.forwards_exn {
                if arity == 0 {
                    let op_idx = self.push_op(SemanticOpKind::AllocExnRef { tag_idx });
                    self.record_result_types(op_idx, &[ValueType::ref_exn()]);
                    if !self.unreachable {
                        self.push_typed_value_with_exn_tag(ValueType::ref_exn(), Some(tag_idx));
                    }
                } else {
                    let op_idx = self.current_index();
                    self.push_op(PrimitiveOpKind::RefNull);
                    self.record_result_types(op_idx, &[ValueType::ref_exn()]);
                    if !self.unreachable {
                        self.push_typed_value_with_exn_tag(ValueType::ref_exn(), Some(tag_idx));
                    }
                }
            }
            self.pop_typed_values(resolved.arity as usize);
            let br_idx = self.push_op(SemanticOpKind::Br {
                stack_drop: resolved.stack_drop,
                arity: resolved.arity,
                target: resolved.target,
            });
            if let Some(fixup_depth) = resolved.pending_fixup_depth {
                self.register_forward_branch(fixup_depth, br_idx, PendingFixupSite::Target);
            }
            if resolved.forwards_exn {
                self.last_static_exn_tag = Some(tag_idx);
            }
            self.set_unreachable();
        } else {
            self.pop_typed_values(arity as usize);
            self.push_op(SemanticOpKind::Throw { tag_idx, arity });
            self.set_unreachable();
        }
    }

    fn handle_call_indirect(&mut self, type_idx: u32, table_idx: u32) {
        let (params, results) = self.compile.resolve_type_index(type_idx);
        self.pop_typed_values(1);
        let idx = self.push_op(SemanticOpKind::CallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        });
        if results > 0 {
            if let Some(ty) = self.compile.types.get_function_type(type_idx) {
                self.record_result_types(idx, ty.results());
            }
        }
        self.pop_typed_values(params as usize);
        if let Some(ty) = self.compile.types.get_function_type(type_idx) {
            self.push_typed_values(ty.results());
        } else {
            self.push_values(results as usize);
        }
    }

    fn handle_return_call(&mut self, func_idx: u32) {
        let (params, results) = self.compile.resolve_func_type(func_idx);
        self.push_op(SemanticOpKind::ReturnCallDirect {
            callee: func_idx,
            params,
            results,
        });
        self.pop_typed_values(params as usize);
        self.set_unreachable();
    }

    fn handle_return_call_indirect(&mut self, type_idx: u32, table_idx: u32) {
        let (params, results) = self.compile.resolve_type_index(type_idx);
        self.pop_typed_values(1);
        self.push_op(SemanticOpKind::ReturnCallIndirect {
            type_idx,
            table_idx,
            params,
            results,
        });
        self.pop_typed_values(params as usize);
        self.set_unreachable();
    }

    fn handle_return_call_ref(&mut self, type_idx: u32) {
        let (params, results) = self.compile.resolve_type_index(type_idx);
        self.push_op(SemanticOpKind::ReturnCallRef {
            type_idx,
            params,
            results,
        });
        self.pop_typed_values(params.saturating_add(1) as usize);
        self.set_unreachable();
    }

    fn handle_call_ref(&mut self, type_idx: u32) {
        let (params, results) = self.compile.resolve_type_index(type_idx);
        let idx = self.push_op(SemanticOpKind::CallRef {
            type_idx,
            params,
            results,
        });
        if results > 0 {
            if let Some(ty) = self.compile.types.get_function_type(type_idx) {
                self.record_result_types(idx, ty.results());
            }
        }
        self.pop_typed_values(params.saturating_add(1) as usize);
        if let Some(ty) = self.compile.types.get_function_type(type_idx) {
            self.push_typed_values(ty.results());
        } else {
            self.push_values(results as usize);
        }
    }

    fn finish(self) -> SemanticProgram {
        self.builder.finish(
            self.compile.params,
            self.compile.results,
            self.compile.local_count,
            self.max_height as u16,
            self.compile.local_types.to_vec().into(),
            self.compile.result_types.to_vec().into(),
            self.op_result_types,
        )
    }

    fn dispatch(&mut self, wasm_op: WasmOpcode, imm: &Immediate) -> Result<(), WasmError> {
        use Opcode::*;
        use WasmOpcode::{FB, FC, FD, OP};

        match wasm_op {
            OP(LOCAL_GET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    self.push_op(SemanticOpKind::LocalGet { idx: *idx as u16 });
                    let local_ty = self
                        .compile
                        .local_types
                        .get(*idx as usize)
                        .copied()
                        .unwrap_or(ValueType::I64);
                    let exn_tag = self.local_exn_tags.get(*idx as usize).copied().flatten();
                    self.push_typed_value_with_exn_tag(local_ty, exn_tag);
                }
            }
            OP(LOCAL_SET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    let exn_tag = self.exn_tag_stack.last().copied().flatten();
                    if let Some(slot) = self.local_exn_tags.get_mut(*idx as usize) {
                        *slot = exn_tag;
                    }
                    self.pop_typed_values(1);
                    self.push_op(SemanticOpKind::LocalSet { idx: *idx as u16 });
                }
            }
            OP(LOCAL_TEE) => {
                if let Immediate::LocalIndex(idx) = imm {
                    let exn_tag = self.exn_tag_stack.last().copied().flatten();
                    if let Some(slot) = self.local_exn_tags.get_mut(*idx as usize) {
                        *slot = exn_tag;
                    }
                    self.push_op(SemanticOpKind::LocalTee { idx: *idx as u16 });
                }
            }

            OP(I32_CONST) => {
                if let Immediate::I32(value) = imm {
                    self.handle_primitive(PrimitiveOpKind::I32Const {
                        value: *value as u32,
                    });
                }
            }
            OP(I64_CONST) => {
                if let Immediate::I64(value) = imm {
                    self.handle_primitive(PrimitiveOpKind::I64Const {
                        value: *value as u64,
                    });
                }
            }
            OP(F32_CONST) => {
                if let Immediate::F32(value) = imm {
                    self.handle_primitive(PrimitiveOpKind::F32Const {
                        value: value.to_bits(),
                    });
                }
            }
            OP(F64_CONST) => {
                if let Immediate::F64(value) = imm {
                    self.handle_primitive(PrimitiveOpKind::F64Const {
                        value: value.to_bits(),
                    });
                }
            }

            OP(I32_ADD) => self.handle_primitive(PrimitiveOpKind::I32Add),
            OP(I32_SUB) => self.handle_primitive(PrimitiveOpKind::I32Sub),
            OP(I32_MUL) => self.handle_primitive(PrimitiveOpKind::I32Mul),
            OP(I32_DIV_S) => self.handle_primitive(PrimitiveOpKind::I32DivS),
            OP(I32_DIV_U) => self.handle_primitive(PrimitiveOpKind::I32DivU),
            OP(I32_REM_S) => self.handle_primitive(PrimitiveOpKind::I32RemS),
            OP(I32_REM_U) => self.handle_primitive(PrimitiveOpKind::I32RemU),
            OP(I32_AND) => self.handle_primitive(PrimitiveOpKind::I32And),
            OP(I32_OR) => self.handle_primitive(PrimitiveOpKind::I32Or),
            OP(I32_XOR) => self.handle_primitive(PrimitiveOpKind::I32Xor),
            OP(I32_SHL) => self.handle_primitive(PrimitiveOpKind::I32Shl),
            OP(I32_SHR_S) => self.handle_primitive(PrimitiveOpKind::I32ShrS),
            OP(I32_SHR_U) => self.handle_primitive(PrimitiveOpKind::I32ShrU),
            OP(I32_ROTL) => self.handle_primitive(PrimitiveOpKind::I32Rotl),
            OP(I32_ROTR) => self.handle_primitive(PrimitiveOpKind::I32Rotr),

            OP(I64_ADD) => self.handle_primitive(PrimitiveOpKind::I64Add),
            OP(I64_SUB) => self.handle_primitive(PrimitiveOpKind::I64Sub),
            OP(I64_MUL) => self.handle_primitive(PrimitiveOpKind::I64Mul),
            OP(I64_DIV_S) => self.handle_primitive(PrimitiveOpKind::I64DivS),
            OP(I64_DIV_U) => self.handle_primitive(PrimitiveOpKind::I64DivU),
            OP(I64_REM_S) => self.handle_primitive(PrimitiveOpKind::I64RemS),
            OP(I64_REM_U) => self.handle_primitive(PrimitiveOpKind::I64RemU),
            OP(I64_AND) => self.handle_primitive(PrimitiveOpKind::I64And),
            OP(I64_OR) => self.handle_primitive(PrimitiveOpKind::I64Or),
            OP(I64_XOR) => self.handle_primitive(PrimitiveOpKind::I64Xor),
            OP(I64_SHL) => self.handle_primitive(PrimitiveOpKind::I64Shl),
            OP(I64_SHR_S) => self.handle_primitive(PrimitiveOpKind::I64ShrS),
            OP(I64_SHR_U) => self.handle_primitive(PrimitiveOpKind::I64ShrU),
            OP(I64_ROTL) => self.handle_primitive(PrimitiveOpKind::I64Rotl),
            OP(I64_ROTR) => self.handle_primitive(PrimitiveOpKind::I64Rotr),

            OP(F32_ADD) => self.handle_primitive(PrimitiveOpKind::F32Add),
            OP(F32_SUB) => self.handle_primitive(PrimitiveOpKind::F32Sub),
            OP(F32_MUL) => self.handle_primitive(PrimitiveOpKind::F32Mul),
            OP(F32_DIV) => self.handle_primitive(PrimitiveOpKind::F32Div),
            OP(F32_MIN) => self.handle_primitive(PrimitiveOpKind::F32Min),
            OP(F32_MAX) => self.handle_primitive(PrimitiveOpKind::F32Max),
            OP(F32_COPYSIGN) => self.handle_primitive(PrimitiveOpKind::F32Copysign),

            OP(F64_ADD) => self.handle_primitive(PrimitiveOpKind::F64Add),
            OP(F64_SUB) => self.handle_primitive(PrimitiveOpKind::F64Sub),
            OP(F64_MUL) => self.handle_primitive(PrimitiveOpKind::F64Mul),
            OP(F64_DIV) => self.handle_primitive(PrimitiveOpKind::F64Div),
            OP(F64_MIN) => self.handle_primitive(PrimitiveOpKind::F64Min),
            OP(F64_MAX) => self.handle_primitive(PrimitiveOpKind::F64Max),
            OP(F64_COPYSIGN) => self.handle_primitive(PrimitiveOpKind::F64Copysign),

            OP(I32_EQ) => self.handle_primitive(PrimitiveOpKind::I32Eq),
            OP(I32_NE) => self.handle_primitive(PrimitiveOpKind::I32Ne),
            OP(I32_LT_S) => self.handle_primitive(PrimitiveOpKind::I32LtS),
            OP(I32_LT_U) => self.handle_primitive(PrimitiveOpKind::I32LtU),
            OP(I32_GT_S) => self.handle_primitive(PrimitiveOpKind::I32GtS),
            OP(I32_GT_U) => self.handle_primitive(PrimitiveOpKind::I32GtU),
            OP(I32_LE_S) => self.handle_primitive(PrimitiveOpKind::I32LeS),
            OP(I32_LE_U) => self.handle_primitive(PrimitiveOpKind::I32LeU),
            OP(I32_GE_S) => self.handle_primitive(PrimitiveOpKind::I32GeS),
            OP(I32_GE_U) => self.handle_primitive(PrimitiveOpKind::I32GeU),

            OP(I64_EQ) => self.handle_primitive(PrimitiveOpKind::I64Eq),
            OP(I64_NE) => self.handle_primitive(PrimitiveOpKind::I64Ne),
            OP(I64_LT_S) => self.handle_primitive(PrimitiveOpKind::I64LtS),
            OP(I64_LT_U) => self.handle_primitive(PrimitiveOpKind::I64LtU),
            OP(I64_GT_S) => self.handle_primitive(PrimitiveOpKind::I64GtS),
            OP(I64_GT_U) => self.handle_primitive(PrimitiveOpKind::I64GtU),
            OP(I64_LE_S) => self.handle_primitive(PrimitiveOpKind::I64LeS),
            OP(I64_LE_U) => self.handle_primitive(PrimitiveOpKind::I64LeU),
            OP(I64_GE_S) => self.handle_primitive(PrimitiveOpKind::I64GeS),
            OP(I64_GE_U) => self.handle_primitive(PrimitiveOpKind::I64GeU),

            OP(F32_EQ) => self.handle_primitive(PrimitiveOpKind::F32Eq),
            OP(F32_NE) => self.handle_primitive(PrimitiveOpKind::F32Ne),
            OP(F32_LT) => self.handle_primitive(PrimitiveOpKind::F32Lt),
            OP(F32_GT) => self.handle_primitive(PrimitiveOpKind::F32Gt),
            OP(F32_LE) => self.handle_primitive(PrimitiveOpKind::F32Le),
            OP(F32_GE) => self.handle_primitive(PrimitiveOpKind::F32Ge),

            OP(F64_EQ) => self.handle_primitive(PrimitiveOpKind::F64Eq),
            OP(F64_NE) => self.handle_primitive(PrimitiveOpKind::F64Ne),
            OP(F64_LT) => self.handle_primitive(PrimitiveOpKind::F64Lt),
            OP(F64_GT) => self.handle_primitive(PrimitiveOpKind::F64Gt),
            OP(F64_LE) => self.handle_primitive(PrimitiveOpKind::F64Le),
            OP(F64_GE) => self.handle_primitive(PrimitiveOpKind::F64Ge),

            OP(I32_EQZ) => self.handle_primitive(PrimitiveOpKind::I32Eqz),
            OP(I32_CLZ) => self.handle_primitive(PrimitiveOpKind::I32Clz),
            OP(I32_CTZ) => self.handle_primitive(PrimitiveOpKind::I32Ctz),
            OP(I32_POPCNT) => self.handle_primitive(PrimitiveOpKind::I32Popcnt),
            OP(I64_EQZ) => self.handle_primitive(PrimitiveOpKind::I64Eqz),
            OP(I64_CLZ) => self.handle_primitive(PrimitiveOpKind::I64Clz),
            OP(I64_CTZ) => self.handle_primitive(PrimitiveOpKind::I64Ctz),
            OP(I64_POPCNT) => self.handle_primitive(PrimitiveOpKind::I64Popcnt),
            OP(F32_ABS) => self.handle_primitive(PrimitiveOpKind::F32Abs),
            OP(F32_NEG) => self.handle_primitive(PrimitiveOpKind::F32Neg),
            OP(F32_CEIL) => self.handle_primitive(PrimitiveOpKind::F32Ceil),
            OP(F32_FLOOR) => self.handle_primitive(PrimitiveOpKind::F32Floor),
            OP(F32_TRUNC) => self.handle_primitive(PrimitiveOpKind::F32Trunc),
            OP(F32_NEAREST) => self.handle_primitive(PrimitiveOpKind::F32Nearest),
            OP(F32_SQRT) => self.handle_primitive(PrimitiveOpKind::F32Sqrt),
            OP(F64_ABS) => self.handle_primitive(PrimitiveOpKind::F64Abs),
            OP(F64_NEG) => self.handle_primitive(PrimitiveOpKind::F64Neg),
            OP(F64_CEIL) => self.handle_primitive(PrimitiveOpKind::F64Ceil),
            OP(F64_FLOOR) => self.handle_primitive(PrimitiveOpKind::F64Floor),
            OP(F64_TRUNC) => self.handle_primitive(PrimitiveOpKind::F64Trunc),
            OP(F64_NEAREST) => self.handle_primitive(PrimitiveOpKind::F64Nearest),
            OP(F64_SQRT) => self.handle_primitive(PrimitiveOpKind::F64Sqrt),

            OP(I32_WRAP_I64) => self.handle_primitive(PrimitiveOpKind::I32WrapI64),
            OP(I32_TRUNC_F32_S) => self.handle_primitive(PrimitiveOpKind::I32TruncF32S),
            OP(I32_TRUNC_F32_U) => self.handle_primitive(PrimitiveOpKind::I32TruncF32U),
            OP(I32_TRUNC_F64_S) => self.handle_primitive(PrimitiveOpKind::I32TruncF64S),
            OP(I32_TRUNC_F64_U) => self.handle_primitive(PrimitiveOpKind::I32TruncF64U),
            OP(I64_EXTEND_I32_S) => self.handle_primitive(PrimitiveOpKind::I64ExtendI32S),
            OP(I64_EXTEND_I32_U) => self.handle_primitive(PrimitiveOpKind::I64ExtendI32U),
            OP(I64_TRUNC_F32_S) => self.handle_primitive(PrimitiveOpKind::I64TruncF32S),
            OP(I64_TRUNC_F32_U) => self.handle_primitive(PrimitiveOpKind::I64TruncF32U),
            OP(I64_TRUNC_F64_S) => self.handle_primitive(PrimitiveOpKind::I64TruncF64S),
            OP(I64_TRUNC_F64_U) => self.handle_primitive(PrimitiveOpKind::I64TruncF64U),
            OP(F32_CONVERT_I32_S) => self.handle_primitive(PrimitiveOpKind::F32ConvertI32S),
            OP(F32_CONVERT_I32_U) => self.handle_primitive(PrimitiveOpKind::F32ConvertI32U),
            OP(F32_CONVERT_I64_S) => self.handle_primitive(PrimitiveOpKind::F32ConvertI64S),
            OP(F32_CONVERT_I64_U) => self.handle_primitive(PrimitiveOpKind::F32ConvertI64U),
            OP(F32_DEMOTE_F64) => self.handle_primitive(PrimitiveOpKind::F32DemoteF64),
            OP(F64_CONVERT_I32_S) => self.handle_primitive(PrimitiveOpKind::F64ConvertI32S),
            OP(F64_CONVERT_I32_U) => self.handle_primitive(PrimitiveOpKind::F64ConvertI32U),
            OP(F64_CONVERT_I64_S) => self.handle_primitive(PrimitiveOpKind::F64ConvertI64S),
            OP(F64_CONVERT_I64_U) => self.handle_primitive(PrimitiveOpKind::F64ConvertI64U),
            OP(F64_PROMOTE_F32) => self.handle_primitive(PrimitiveOpKind::F64PromoteF32),
            OP(I32_REINTERPRET_F32) => self.handle_primitive(PrimitiveOpKind::I32ReinterpretF32),
            OP(I64_REINTERPRET_F64) => self.handle_primitive(PrimitiveOpKind::I64ReinterpretF64),
            OP(F32_REINTERPRET_I32) => self.handle_primitive(PrimitiveOpKind::F32ReinterpretI32),
            OP(F64_REINTERPRET_I64) => self.handle_primitive(PrimitiveOpKind::F64ReinterpretI64),
            OP(I32_EXTEND8_S) => self.handle_primitive(PrimitiveOpKind::I32Extend8S),
            OP(I32_EXTEND16_S) => self.handle_primitive(PrimitiveOpKind::I32Extend16S),
            OP(I64_EXTEND8_S) => self.handle_primitive(PrimitiveOpKind::I64Extend8S),
            OP(I64_EXTEND16_S) => self.handle_primitive(PrimitiveOpKind::I64Extend16S),
            OP(I64_EXTEND32_S) => self.handle_primitive(PrimitiveOpKind::I64Extend32S),

            FC(OpcodeFC::I32_TRUNC_SAT_F32_S) => {
                self.handle_primitive(PrimitiveOpKind::I32TruncSatF32S)
            }
            FC(OpcodeFC::I32_TRUNC_SAT_F32_U) => {
                self.handle_primitive(PrimitiveOpKind::I32TruncSatF32U)
            }
            FC(OpcodeFC::I32_TRUNC_SAT_F64_S) => {
                self.handle_primitive(PrimitiveOpKind::I32TruncSatF64S)
            }
            FC(OpcodeFC::I32_TRUNC_SAT_F64_U) => {
                self.handle_primitive(PrimitiveOpKind::I32TruncSatF64U)
            }
            FC(OpcodeFC::I64_TRUNC_SAT_F32_S) => {
                self.handle_primitive(PrimitiveOpKind::I64TruncSatF32S)
            }
            FC(OpcodeFC::I64_TRUNC_SAT_F32_U) => {
                self.handle_primitive(PrimitiveOpKind::I64TruncSatF32U)
            }
            FC(OpcodeFC::I64_TRUNC_SAT_F64_S) => {
                self.handle_primitive(PrimitiveOpKind::I64TruncSatF64S)
            }
            FC(OpcodeFC::I64_TRUNC_SAT_F64_U) => {
                self.handle_primitive(PrimitiveOpKind::I64TruncSatF64U)
            }

            OP(I32_LOAD) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I32Load {
                offset,
                memidx,
            }),
            OP(I64_LOAD) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I64Load {
                offset,
                memidx,
            }),
            OP(F32_LOAD) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::F32Load {
                offset,
                memidx,
            }),
            OP(F64_LOAD) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::F64Load {
                offset,
                memidx,
            }),
            OP(I32_LOAD8_S) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I32Load8S {
                offset,
                memidx,
            }),
            OP(I32_LOAD8_U) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I32Load8U {
                offset,
                memidx,
            }),
            OP(I32_LOAD16_S) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I32Load16S { offset, memidx }
            }),
            OP(I32_LOAD16_U) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I32Load16U { offset, memidx }
            }),
            OP(I64_LOAD8_S) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I64Load8S {
                offset,
                memidx,
            }),
            OP(I64_LOAD8_U) => self.handle_load(imm, |offset, memidx| PrimitiveOpKind::I64Load8U {
                offset,
                memidx,
            }),
            OP(I64_LOAD16_S) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I64Load16S { offset, memidx }
            }),
            OP(I64_LOAD16_U) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I64Load16U { offset, memidx }
            }),
            OP(I64_LOAD32_S) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I64Load32S { offset, memidx }
            }),
            OP(I64_LOAD32_U) => self.handle_load(imm, |offset, memidx| {
                PrimitiveOpKind::I64Load32U { offset, memidx }
            }),

            OP(I32_STORE) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::I32Store {
                offset,
                memidx,
            }),
            OP(I64_STORE) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::I64Store {
                offset,
                memidx,
            }),
            OP(F32_STORE) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::F32Store {
                offset,
                memidx,
            }),
            OP(F64_STORE) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::F64Store {
                offset,
                memidx,
            }),
            OP(I32_STORE8) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::I32Store8 {
                offset,
                memidx,
            }),
            OP(I32_STORE16) => self.handle_store(imm, |offset, memidx| {
                PrimitiveOpKind::I32Store16 { offset, memidx }
            }),
            OP(I64_STORE8) => self.handle_store(imm, |offset, memidx| PrimitiveOpKind::I64Store8 {
                offset,
                memidx,
            }),
            OP(I64_STORE16) => self.handle_store(imm, |offset, memidx| {
                PrimitiveOpKind::I64Store16 { offset, memidx }
            }),
            OP(I64_STORE32) => self.handle_store(imm, |offset, memidx| {
                PrimitiveOpKind::I64Store32 { offset, memidx }
            }),

            OP(MEMORY_SIZE) => {
                if let Immediate::MemoryIndex(mem_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::MemorySize { mem_idx: *mem_idx });
                    let is_mem64 = self.compile.store.memory(*mem_idx as usize).limits.is64;
                    self.record_result_types(
                        op_idx,
                        &[if is_mem64 {
                            ValueType::I64
                        } else {
                            ValueType::I32
                        }],
                    );
                }
            }
            OP(MEMORY_GROW) => {
                if let Immediate::MemoryIndex(mem_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::MemoryGrow { mem_idx: *mem_idx });
                    let is_mem64 = self.compile.store.memory(*mem_idx as usize).limits.is64;
                    self.record_result_types(
                        op_idx,
                        &[if is_mem64 {
                            ValueType::I64
                        } else {
                            ValueType::I32
                        }],
                    );
                }
            }
            FC(OpcodeFC::MEMORY_FILL) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::MemoryFill { imm0, imm1 });
            }
            FC(OpcodeFC::MEMORY_COPY) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::MemoryCopy { imm0, imm1 });
            }
            FC(OpcodeFC::MEMORY_INIT) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::MemoryInit { imm0, imm1 });
            }
            FC(OpcodeFC::DATA_DROP) => {
                if let Immediate::DataIndex(data_idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::DataDrop {
                        data_idx: *data_idx,
                    });
                }
            }

            OP(GLOBAL_GET) => {
                if let Immediate::GlobalIndex(idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::GlobalGet { idx: *idx });
                    let gty = self.compile.store.global(*idx as usize).value_type;
                    self.record_result_types(op_idx, &[gty]);
                }
            }
            OP(GLOBAL_SET) => {
                if let Immediate::GlobalIndex(idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::GlobalSet {
                        idx: *idx,
                        retag: self
                            .compile
                            .store
                            .module()
                            .global_needs_funcref_retag(*idx as usize),
                    });
                }
            }

            OP(TABLE_GET) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::TableGet {
                        table_idx: *table_idx,
                    });
                    let tty = self.compile.store.table(*table_idx as usize).value_type;
                    self.record_result_types(op_idx, &[tty]);
                }
            }
            OP(TABLE_SET) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::TableSet {
                        table_idx: *table_idx,
                        retag: self
                            .compile
                            .store
                            .module()
                            .table_is_reachable(*table_idx as usize),
                    });
                }
            }
            FC(OpcodeFC::TABLE_SIZE) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::TableSize {
                        table_idx: *table_idx,
                    });
                    let is_table64 = self.compile.store.table(*table_idx as usize).limits.is64;
                    self.record_result_types(
                        op_idx,
                        &[if is_table64 {
                            ValueType::I64
                        } else {
                            ValueType::I32
                        }],
                    );
                }
            }
            FC(OpcodeFC::TABLE_GROW) => {
                if let Immediate::TableIndex(table_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::TableGrow {
                        table_idx: *table_idx,
                    });
                    let is_table64 = self.compile.store.table(*table_idx as usize).limits.is64;
                    self.record_result_types(
                        op_idx,
                        &[if is_table64 {
                            ValueType::I64
                        } else {
                            ValueType::I32
                        }],
                    );
                }
            }
            FC(OpcodeFC::TABLE_FILL) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::TableFill { imm0, imm1 });
            }
            FC(OpcodeFC::TABLE_COPY) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::TableCopy { imm0, imm1 });
            }
            FC(OpcodeFC::TABLE_INIT) => {
                let (imm0, imm1) = extract_imm01(imm);
                self.handle_primitive(PrimitiveOpKind::TableInit { imm0, imm1 });
            }
            FC(OpcodeFC::ELEM_DROP) => {
                if let Immediate::ElementIndex(elem_idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::ElemDrop {
                        elem_idx: *elem_idx,
                    });
                }
            }

            OP(REF_NULL) => {
                let op_idx = self.current_index();
                self.handle_primitive(PrimitiveOpKind::RefNull);
                if let Immediate::RefType(vt) = imm {
                    self.record_result_types(op_idx, &[*vt]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = *vt;
                        }
                    }
                }
            }
            OP(REF_IS_NULL) => self.handle_primitive(PrimitiveOpKind::RefIsNull),
            OP(REF_EQ) => self.handle_primitive(PrimitiveOpKind::RefEq),
            OP(REF_AS_NON_NULL) => {
                let result_ty = self.top_type().to_non_nullable();
                let op_idx = self.current_index();
                self.handle_primitive(PrimitiveOpKind::RefAsNonNull);
                self.record_result_types(op_idx, &[result_ty]);
                if !self.unreachable {
                    if let Some(last) = self.type_stack.last_mut() {
                        *last = result_ty;
                    }
                }
            }
            OP(REF_FUNC) => {
                if let Immediate::FunctionIndex(func_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::RefFunc {
                        func_idx: *func_idx,
                    });
                    let func_type_idx =
                        self.compile.store.function(*func_idx as usize).type_index();
                    let ty = ValueType::Ref(RefType::new(false, HeapType::Concrete(func_type_idx)));
                    self.record_result_types(op_idx, &[ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = ty;
                        }
                    }
                }
            }

            OP(DROP) => self.handle_primitive(PrimitiveOpKind::Drop),
            OP(SELECT) | OP(SELECT_T) => self.handle_primitive(PrimitiveOpKind::Select),

            OP(BLOCK) => {
                let (params, results) = self.compile.resolve_block_type_from_imm(imm);
                let result_types = self.compile.resolve_block_result_types_from_imm(imm);
                let idx = self.push_op(SemanticOpKind::Block { params, results });
                self.record_result_types(idx, &result_types);
                let target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(
                    DecodeBlockKind::Block,
                    params,
                    results,
                    result_types,
                    target,
                );
            }
            OP(LOOP) => {
                let (params, results) = self.compile.resolve_block_type_from_imm(imm);
                let result_types = self.compile.resolve_block_result_types_from_imm(imm);
                let idx = self.push_op(SemanticOpKind::Loop { params, results });
                self.record_result_types(idx, &result_types);
                let target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(DecodeBlockKind::Loop, params, results, result_types, target);
            }
            OP(IF) => {
                self.pop_typed_values(1);
                let (params, results) = self.compile.resolve_block_type_from_imm(imm);
                let result_types = self.compile.resolve_block_result_types_from_imm(imm);
                let idx = self.push_op(SemanticOpKind::If {
                    params,
                    results,
                    else_target: SemanticTarget::pending(),
                });
                self.record_result_types(idx, &result_types);
                let target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(DecodeBlockKind::If, params, results, result_types, target);
                self.set_if_inst(idx);
            }
            OP(ELSE) => {
                let else_idx = self.push_op(SemanticOpKind::Else {
                    end_target: SemanticTarget::pending(),
                });
                let else_body_start = SemanticTarget::new(self.current_index().as_usize());
                if let Some(frame) = self.control.last() {
                    if let Some(if_idx) = frame.if_inst_idx {
                        self.patch_target(if_idx, else_body_start);
                    }
                }
                if let Some(frame) = self.control.last_mut() {
                    frame.pending_fixups.push(PendingBranchFixup {
                        inst_idx: else_idx,
                        site: PendingFixupSite::Target,
                    });
                }
                self.enter_else();
            }
            OP(END) => {
                self.handle_end();
            }
            OP(TRY_TABLE) => {
                let Immediate::TryTable { catches, .. } = imm else {
                    return Err(WasmError::internal("decoder expected try_table immediate"));
                };
                let (params, results) = self.compile.resolve_try_table_block_type(imm);
                let result_types = self.compile.resolve_try_table_result_types(imm);

                // Clause label indices resolve against the outer control
                // stack; the try_table frame is not yet in scope.
                let mut sem_catches: collections::Vec<SemanticCatchClause> =
                    collections::Vec::with_capacity(catches.len());
                for clause in catches {
                    let payload_arity = match clause.kind {
                        CatchClauseKind::Catch | CatchClauseKind::CatchRef => clause
                            .tag_idx
                            .map(|tag| self.compile.resolve_tag_type(tag).0)
                            .unwrap_or(0),
                        CatchClauseKind::CatchAll | CatchClauseKind::CatchAllRef => 0,
                    };
                    let forwards_exn = matches!(
                        clause.kind,
                        CatchClauseKind::CatchRef | CatchClauseKind::CatchAllRef
                    );
                    let (stack_drop, target) = self.branch_info(clause.label_idx);
                    sem_catches.push(SemanticCatchClause {
                        tag_idx: clause.tag_idx,
                        payload_arity,
                        forwards_exn,
                        stack_drop,
                        target: target.unwrap_or_else(SemanticTarget::pending),
                    });
                }

                let mut decode_catches: collections::Vec<DecodeTryCatch> =
                    collections::Vec::with_capacity(catches.len());
                for (clause, sem) in catches.iter().zip(sem_catches.iter()) {
                    let target_start_height = self
                        .frame_at_depth(clause.label_idx)
                        .map(|f| f.start_height as u32)
                        .unwrap_or(0);
                    decode_catches.push(DecodeTryCatch {
                        tag_idx: clause.tag_idx,
                        kind: clause.kind,
                        payload_arity: sem.payload_arity,
                        target: sem.target,
                        target_start_height,
                        label_idx: clause.label_idx,
                    });
                }

                self.pop_typed_values(params as usize);
                let idx = self.push_op(SemanticOpKind::TryTable {
                    params,
                    results,
                    catches: sem_catches,
                });
                self.record_result_types(idx, &result_types);
                for (catch_idx, clause) in catches.iter().enumerate() {
                    let (_stack_drop, target) = self.branch_info(clause.label_idx);
                    if target.is_none() {
                        self.register_forward_branch(
                            clause.label_idx,
                            idx,
                            PendingFixupSite::TryTableCatch(catch_idx),
                        );
                    }
                }

                let try_target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(
                    DecodeBlockKind::TryTable,
                    params,
                    results,
                    result_types,
                    try_target,
                );
                if let Some(frame) = self.control.last_mut() {
                    frame.try_catches = decode_catches;
                }
            }
            OP(THROW) => {
                let Immediate::TagIndex(tag_idx) = imm else {
                    return Err(WasmError::internal("decoder expected tag index immediate"));
                };
                self.handle_throw(*tag_idx);
            }
            OP(THROW_REF) => {
                let exn_tag = self.exn_tag_stack.last().copied().flatten();
                if let Some(tag_idx) = exn_tag {
                    if let Some(mut resolved) = self.match_throw_to_catch(tag_idx) {
                        if resolved.forwards_exn {
                            resolved.stack_drop = resolved.stack_drop.saturating_sub(1);
                        }
                        self.pop_typed_values(1);
                        let br_idx = self.push_op(SemanticOpKind::Br {
                            stack_drop: resolved.stack_drop,
                            arity: resolved.arity,
                            target: resolved.target,
                        });
                        if let Some(fixup_depth) = resolved.pending_fixup_depth {
                            self.register_forward_branch(
                                fixup_depth,
                                br_idx,
                                PendingFixupSite::Target,
                            );
                        }
                        if resolved.forwards_exn {
                            self.last_static_exn_tag = Some(tag_idx);
                        }
                        self.set_unreachable();
                    } else {
                        self.pop_typed_values(1);
                        self.push_op(SemanticOpKind::ThrowRef);
                        self.set_unreachable();
                    }
                } else {
                    self.pop_typed_values(1);
                    self.push_op(SemanticOpKind::ThrowRef);
                    self.set_unreachable();
                }
            }
            OP(BR) => {
                if let Immediate::LabelIndex(label) = imm {
                    let arity = self.branch_arity(*label);
                    let (stack_drop, target) = self.branch_info(*label);
                    let idx = self.push_op(SemanticOpKind::Br {
                        stack_drop,
                        arity,
                        target: target.unwrap_or_else(SemanticTarget::pending),
                    });
                    if let Some(target) = target {
                        self.patch_target(idx, target);
                    } else {
                        self.register_forward_branch(*label, idx, PendingFixupSite::Target);
                    }
                    self.set_unreachable();
                }
            }
            OP(BR_IF) => {
                if let Immediate::LabelIndex(label) = imm {
                    self.pop_typed_values(1);
                    let arity = self.branch_arity(*label);
                    let (stack_drop, target) = self.branch_info(*label);
                    let idx = self.push_op(SemanticOpKind::BrIf {
                        stack_drop,
                        arity,
                        target: target.unwrap_or_else(SemanticTarget::pending),
                    });
                    if let Some(target) = target {
                        self.patch_target(idx, target);
                    } else {
                        self.register_forward_branch(*label, idx, PendingFixupSite::Target);
                    }
                }
            }
            OP(BR_ON_NULL) => {
                if let Immediate::LabelIndex(label) = imm {
                    if self.unreachable {
                        self.handle_primitive(PrimitiveOpKind::Nop);
                    } else {
                        let ref_ty = self.pop_typed_value();
                        let refined_ty = ref_ty.to_non_nullable();
                        let arity = self.branch_arity(*label);
                        let (stack_drop, target) = self.branch_info(*label);
                        let idx = self.push_op(SemanticOpKind::BrOnNull {
                            stack_drop,
                            arity,
                            target: target.unwrap_or_else(SemanticTarget::pending),
                            ref_type: refined_ty,
                        });
                        if let Some(target) = target {
                            self.patch_target(idx, target);
                        } else {
                            self.register_forward_branch(*label, idx, PendingFixupSite::Target);
                        }
                        self.push_typed_value(refined_ty);
                    }
                }
            }
            OP(BR_ON_NON_NULL) => {
                if let Immediate::LabelIndex(label) = imm {
                    if self.unreachable {
                        self.handle_primitive(PrimitiveOpKind::Nop);
                    } else {
                        let ref_ty = self.pop_typed_value();
                        let label_types = self.branch_types(*label);
                        let branch_ref_ty = label_types
                            .last()
                            .copied()
                            .unwrap_or_else(|| ref_ty.to_non_nullable());
                        let arity = label_types.len() as u16;
                        let Some(frame) = self.frame_at_depth(*label) else {
                            return Err(WasmError::invalid("invalid frame index"));
                        };
                        let stack_drop = self
                            .height
                            .saturating_add(1)
                            .saturating_sub(frame.start_height.saturating_add(arity as usize))
                            as u32;
                        let target = match frame.kind {
                            DecodeBlockKind::Loop => Some(frame.start_inst_idx),
                            DecodeBlockKind::Function
                            | DecodeBlockKind::Block
                            | DecodeBlockKind::If
                            | DecodeBlockKind::TryTable => None,
                        };
                        let idx = self.push_op(SemanticOpKind::BrOnNonNull {
                            stack_drop,
                            arity,
                            target: target.unwrap_or_else(SemanticTarget::pending),
                            ref_type: branch_ref_ty,
                        });
                        if let Some(target) = target {
                            self.patch_target(idx, target);
                        } else {
                            self.register_forward_branch(*label, idx, PendingFixupSite::Target);
                        }
                    }
                }
            }
            FB(OpcodeFB::BR_ON_CAST) => {
                if let Immediate::BrOnCast {
                    label_idx,
                    rt1,
                    rt2,
                    ..
                } = imm
                {
                    if self.unreachable {
                        self.handle_primitive(PrimitiveOpKind::Nop);
                    } else {
                        let _actual_ref_ty = self.pop_typed_value();
                        let fail_type = match (*rt1, *rt2) {
                            (ValueType::Ref(from_ref), ValueType::Ref(to_ref)) => {
                                ValueType::Ref(RefType::difference(from_ref, to_ref))
                            }
                            _ => ValueType::Unknown,
                        };
                        let label_types = self.branch_types(*label_idx);
                        let arity = label_types.len() as u16;
                        let Some(frame) = self.frame_at_depth(*label_idx) else {
                            return Err(WasmError::invalid("invalid frame index"));
                        };
                        let stack_drop = self
                            .height
                            .saturating_add(1)
                            .saturating_sub(frame.start_height.saturating_add(arity as usize))
                            as u32;
                        let target = match frame.kind {
                            DecodeBlockKind::Loop => Some(frame.start_inst_idx),
                            DecodeBlockKind::Function
                            | DecodeBlockKind::Block
                            | DecodeBlockKind::If
                            | DecodeBlockKind::TryTable => None,
                        };
                        let idx = self.push_op(SemanticOpKind::BrOnCast {
                            stack_drop,
                            arity,
                            target: target.unwrap_or_else(SemanticTarget::pending),
                            fail_type,
                            cast_type: *rt2,
                        });
                        if let Some(target) = target {
                            self.patch_target(idx, target);
                        } else {
                            self.register_forward_branch(*label_idx, idx, PendingFixupSite::Target);
                        }
                        self.push_typed_value(fail_type);
                    }
                }
            }
            FB(OpcodeFB::BR_ON_CAST_FAIL) => {
                if let Immediate::BrOnCast {
                    label_idx,
                    rt1,
                    rt2,
                    ..
                } = imm
                {
                    if self.unreachable {
                        self.handle_primitive(PrimitiveOpKind::Nop);
                    } else {
                        let _actual_ref_ty = self.pop_typed_value();
                        let fail_type = match (*rt1, *rt2) {
                            (ValueType::Ref(from_ref), ValueType::Ref(to_ref)) => {
                                ValueType::Ref(RefType::difference(from_ref, to_ref))
                            }
                            _ => ValueType::Unknown,
                        };
                        let label_types = self.branch_types(*label_idx);
                        let arity = label_types.len() as u16;
                        let Some(frame) = self.frame_at_depth(*label_idx) else {
                            return Err(WasmError::invalid("invalid frame index"));
                        };
                        let stack_drop = self
                            .height
                            .saturating_add(1)
                            .saturating_sub(frame.start_height.saturating_add(arity as usize))
                            as u32;
                        let target = match frame.kind {
                            DecodeBlockKind::Loop => Some(frame.start_inst_idx),
                            DecodeBlockKind::Function
                            | DecodeBlockKind::Block
                            | DecodeBlockKind::If
                            | DecodeBlockKind::TryTable => None,
                        };
                        let idx = self.push_op(SemanticOpKind::BrOnCastFail {
                            stack_drop,
                            arity,
                            target: target.unwrap_or_else(SemanticTarget::pending),
                            fail_type,
                            cast_type: *rt2,
                        });
                        if let Some(target) = target {
                            self.patch_target(idx, target);
                        } else {
                            self.register_forward_branch(*label_idx, idx, PendingFixupSite::Target);
                        }
                        self.push_typed_value(*rt2);
                    }
                }
            }
            OP(BR_TABLE) => {
                if let Immediate::BrLabels(labels, default_label) = imm {
                    self.pop_typed_values(1);
                    let all_labels = labels
                        .iter()
                        .copied()
                        .chain(core::iter::once(*default_label))
                        .collect::<collections::Vec<_>>();
                    let mut entries = collections::Vec::with_capacity(all_labels.len());
                    for label in &all_labels {
                        let arity = self.branch_arity(*label);
                        let (stack_drop, target) = self.branch_info(*label);
                        entries.push(BrTableEntry {
                            target: target.unwrap_or_else(SemanticTarget::pending),
                            stack_drop,
                            arity,
                        });
                    }
                    let inst_idx = self.push_op(SemanticOpKind::BrTable { entries });
                    for (entry_idx, label) in all_labels.iter().copied().enumerate() {
                        let (_, target) = self.branch_info(label);
                        if target.is_none() {
                            self.register_forward_branch(
                                label,
                                inst_idx,
                                PendingFixupSite::BrTableEntry(entry_idx),
                            );
                        }
                    }
                    self.set_unreachable();
                }
            }

            OP(RETURN) => self.emit_return(),
            OP(UNREACHABLE) => self.handle_primitive(PrimitiveOpKind::Unreachable),

            OP(CALL) => self.handle_call(expect_function_index_immediate(imm)?),
            OP(RETURN_CALL) => {
                let func_idx = expect_function_index_immediate(imm)?;
                if self.unreachable {
                    self.handle_primitive(PrimitiveOpKind::Nop);
                } else {
                    self.handle_return_call(func_idx);
                }
            }
            OP(CALL_INDIRECT) => {
                let (typeidx, tableidx) = expect_call_indirect_immediate(imm)?;
                self.handle_call_indirect(typeidx, tableidx);
            }
            OP(RETURN_CALL_INDIRECT) => {
                let (typeidx, tableidx) = expect_call_indirect_immediate(imm)?;
                if self.unreachable {
                    self.handle_primitive(PrimitiveOpKind::Nop);
                } else {
                    self.handle_return_call_indirect(typeidx, tableidx);
                }
            }
            OP(CALL_REF) => {
                let typeidx = expect_type_index_immediate(imm)?;
                if self.unreachable {
                    self.handle_primitive(PrimitiveOpKind::Nop);
                } else {
                    self.handle_call_ref(typeidx);
                }
            }
            OP(RETURN_CALL_REF) => {
                let typeidx = expect_type_index_immediate(imm)?;
                if self.unreachable {
                    self.handle_primitive(PrimitiveOpKind::Nop);
                } else {
                    self.handle_return_call_ref(typeidx);
                }
            }

            OP(NOP) => self.handle_primitive(PrimitiveOpKind::Nop),

            FD(OpcodeFD::V128_CONST) => self.handle_primitive(PrimitiveOpKind::V128Const {
                value: expect_v128_immediate(imm)?,
            }),
            FD(OpcodeFD::V128_LOAD) => {
                self.handle_load(imm, |offset, memidx| PrimitiveOpKind::V128Load {
                    offset,
                    memidx,
                });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::V128_LOAD8X8_S
                        | OpcodeFD::V128_LOAD8X8_U
                        | OpcodeFD::V128_LOAD16X4_S
                        | OpcodeFD::V128_LOAD16X4_U
                        | OpcodeFD::V128_LOAD32X2_S
                        | OpcodeFD::V128_LOAD32X2_U
                        | OpcodeFD::V128_LOAD8_SPLAT
                        | OpcodeFD::V128_LOAD16_SPLAT
                        | OpcodeFD::V128_LOAD32_SPLAT
                        | OpcodeFD::V128_LOAD64_SPLAT
                        | OpcodeFD::V128_LOAD32_ZERO
                        | OpcodeFD::V128_LOAD64_ZERO
                ) =>
            {
                self.handle_load(imm, |offset, memidx| PrimitiveOpKind::SimdMemLoadV128 {
                    opcode: op_ext as u32,
                    offset,
                    memidx,
                });
            }
            FD(OpcodeFD::V128_STORE) => {
                self.handle_store(imm, |offset, memidx| PrimitiveOpKind::V128Store {
                    offset,
                    memidx,
                });
            }
            FD(OpcodeFD::V128_NOT) => self.handle_primitive(PrimitiveOpKind::V128Not),
            FD(OpcodeFD::V128_AND) => self.handle_primitive(PrimitiveOpKind::V128And),
            FD(OpcodeFD::V128_ANDNOT) => self.handle_primitive(PrimitiveOpKind::V128AndNot),
            FD(OpcodeFD::V128_OR) => self.handle_primitive(PrimitiveOpKind::V128Or),
            FD(OpcodeFD::V128_XOR) => self.handle_primitive(PrimitiveOpKind::V128Xor),
            FD(OpcodeFD::V128_BITSELECT) => self.handle_primitive(PrimitiveOpKind::V128Bitselect),
            FD(OpcodeFD::V128_ANY_TRUE) => self.handle_primitive(PrimitiveOpKind::V128AnyTrue),
            FD(OpcodeFD::I8X16_ALL_TRUE) => self.handle_primitive(PrimitiveOpKind::I8x16AllTrue),
            FD(OpcodeFD::I16X8_ALL_TRUE) => self.handle_primitive(PrimitiveOpKind::I16x8AllTrue),
            FD(OpcodeFD::I32X4_ALL_TRUE) => self.handle_primitive(PrimitiveOpKind::I32x4AllTrue),
            FD(OpcodeFD::I64X2_ALL_TRUE) => self.handle_primitive(PrimitiveOpKind::I64x2AllTrue),
            FD(OpcodeFD::I8X16_BITMASK) => self.handle_primitive(PrimitiveOpKind::I8x16Bitmask),
            FD(OpcodeFD::I16X8_BITMASK) => self.handle_primitive(PrimitiveOpKind::I16x8Bitmask),
            FD(OpcodeFD::I32X4_BITMASK) => self.handle_primitive(PrimitiveOpKind::I32x4Bitmask),
            FD(OpcodeFD::I64X2_BITMASK) => self.handle_primitive(PrimitiveOpKind::I64x2Bitmask),
            FD(OpcodeFD::I8X16_SPLAT) => self.handle_primitive(PrimitiveOpKind::I8x16Splat),
            FD(OpcodeFD::I16X8_SPLAT) => self.handle_primitive(PrimitiveOpKind::I16x8Splat),
            FD(OpcodeFD::I32X4_SPLAT) => self.handle_primitive(PrimitiveOpKind::I32x4Splat),
            FD(OpcodeFD::I64X2_SPLAT) => self.handle_primitive(PrimitiveOpKind::I64x2Splat),
            FD(OpcodeFD::F32X4_SPLAT) => self.handle_primitive(PrimitiveOpKind::F32x4Splat),
            FD(OpcodeFD::F64X2_SPLAT) => self.handle_primitive(PrimitiveOpKind::F64x2Splat),
            FD(OpcodeFD::I8X16_EXTRACT_LANE_S) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16ExtractLaneS { lane });
            }
            FD(OpcodeFD::I8X16_EXTRACT_LANE_U) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16ExtractLaneU { lane });
            }
            FD(OpcodeFD::I16X8_EXTRACT_LANE_S) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I16x8ExtractLaneS { lane });
            }
            FD(OpcodeFD::I16X8_EXTRACT_LANE_U) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I16x8ExtractLaneU { lane });
            }
            FD(OpcodeFD::I32X4_EXTRACT_LANE) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I32x4ExtractLane { lane });
            }
            FD(OpcodeFD::I64X2_EXTRACT_LANE) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I64x2ExtractLane { lane });
            }
            FD(OpcodeFD::F32X4_EXTRACT_LANE) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::F32x4ExtractLane { lane });
            }
            FD(OpcodeFD::F64X2_EXTRACT_LANE) => {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::F64x2ExtractLane { lane });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_REPLACE_LANE
                        | OpcodeFD::I16X8_REPLACE_LANE
                        | OpcodeFD::I32X4_REPLACE_LANE
                        | OpcodeFD::I64X2_REPLACE_LANE
                        | OpcodeFD::F32X4_REPLACE_LANE
                        | OpcodeFD::F64X2_REPLACE_LANE
                ) =>
            {
                let lane = expect_lane_index_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdReplaceLane {
                    opcode: op_ext as u32,
                    lane,
                });
            }
            FD(OpcodeFD::I8X16_SHUFFLE) => {
                let lanes = expect_shuffle_mask_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::I8x16Shuffle { lanes });
            }
            FD(OpcodeFD::I8X16_RELAXED_SWIZZLE) => {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: OpcodeFD::I8X16_SWIZZLE as u32,
                });
            }
            FD(OpcodeFD::I32X4_RELAXED_TRUNC_F32X4_S) => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F32X4_S as u32,
                });
            }
            FD(OpcodeFD::I32X4_RELAXED_TRUNC_F32X4_U) => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F32X4_U as u32,
                });
            }
            FD(OpcodeFD::I32X4_RELAXED_TRUNC_F64X2_S_ZERO) => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO as u32,
                });
            }
            FD(OpcodeFD::I32X4_RELAXED_TRUNC_F64X2_U_ZERO) => {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO as u32,
                });
            }
            FD(OpcodeFD::I16X8_RELAXED_Q15MULR_S) => {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: OpcodeFD::I16X8_Q15MULR_SAT_S as u32,
                });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_RELAXED_LANESELECT
                        | OpcodeFD::I16X8_RELAXED_LANESELECT
                        | OpcodeFD::I32X4_RELAXED_LANESELECT
                        | OpcodeFD::I64X2_RELAXED_LANESELECT
                        | OpcodeFD::F32X4_RELAXED_MADD
                        | OpcodeFD::F32X4_RELAXED_NMADD
                        | OpcodeFD::F64X2_RELAXED_MADD
                        | OpcodeFD::F64X2_RELAXED_NMADD
                        | OpcodeFD::I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdTernaryV128 {
                    opcode: op_ext as u32,
                });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::F32X4_RELAXED_MIN
                        | OpcodeFD::F32X4_RELAXED_MAX
                        | OpcodeFD::F64X2_RELAXED_MIN
                        | OpcodeFD::F64X2_RELAXED_MAX
                        | OpcodeFD::I16X8_RELAXED_DOT_I8X16_I7X16_S
                ) =>
            {
                let opcode = match op_ext {
                    OpcodeFD::F32X4_RELAXED_MIN => OpcodeFD::F32X4_MIN as u32,
                    OpcodeFD::F32X4_RELAXED_MAX => OpcodeFD::F32X4_MAX as u32,
                    OpcodeFD::F64X2_RELAXED_MIN => OpcodeFD::F64X2_MIN as u32,
                    OpcodeFD::F64X2_RELAXED_MAX => OpcodeFD::F64X2_MAX as u32,
                    _ => op_ext as u32,
                };
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 { opcode });
            }
            FD(op_ext) if is_relaxed_simd_opcode(op_ext) => {
                return Err(WasmError::invalid("unhandled relaxed SIMD opcode"));
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::V128_LOAD8_LANE
                        | OpcodeFD::V128_LOAD16_LANE
                        | OpcodeFD::V128_LOAD32_LANE
                        | OpcodeFD::V128_LOAD64_LANE
                ) =>
            {
                let (offset, memidx, lane) = expect_memarg_lane_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdMemLoadLaneV128 {
                    opcode: op_ext as u32,
                    lane,
                    offset: u32::try_from(offset)
                        .map_err(|_| WasmError::invalid("simd memarg offset exceeds u32"))?,
                    memidx,
                });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::V128_STORE8_LANE
                        | OpcodeFD::V128_STORE16_LANE
                        | OpcodeFD::V128_STORE32_LANE
                        | OpcodeFD::V128_STORE64_LANE
                ) =>
            {
                let (offset, memidx, lane) = expect_memarg_lane_immediate(imm)?;
                self.handle_primitive(PrimitiveOpKind::SimdMemStoreLaneV128 {
                    opcode: op_ext as u32,
                    lane,
                    offset: u32::try_from(offset)
                        .map_err(|_| WasmError::invalid("simd memarg offset exceeds u32"))?,
                    memidx,
                });
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_SWIZZLE
                        | OpcodeFD::I8X16_NARROW_I16X8_S
                        | OpcodeFD::I8X16_NARROW_I16X8_U
                        | OpcodeFD::I8X16_MIN_S
                        | OpcodeFD::I8X16_MIN_U
                        | OpcodeFD::I8X16_MAX_S
                        | OpcodeFD::I8X16_MAX_U
                        | OpcodeFD::I8X16_AVGR_U
                        | OpcodeFD::I8X16_ADD
                        | OpcodeFD::I8X16_SUB
                        | OpcodeFD::I16X8_MIN_S
                        | OpcodeFD::I16X8_MIN_U
                        | OpcodeFD::I16X8_MAX_S
                        | OpcodeFD::I16X8_MAX_U
                        | OpcodeFD::I16X8_AVGR_U
                        | OpcodeFD::I16X8_Q15MULR_SAT_S
                        | OpcodeFD::I16X8_NARROW_I32X4_S
                        | OpcodeFD::I16X8_NARROW_I32X4_U
                        | OpcodeFD::I16X8_ADD
                        | OpcodeFD::I16X8_SUB
                        | OpcodeFD::I16X8_MUL
                        | OpcodeFD::I16X8_EXTMUL_LOW_I8X16_S
                        | OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_S
                        | OpcodeFD::I16X8_EXTMUL_LOW_I8X16_U
                        | OpcodeFD::I16X8_EXTMUL_HIGH_I8X16_U
                        | OpcodeFD::I32X4_MIN_S
                        | OpcodeFD::I32X4_MIN_U
                        | OpcodeFD::I32X4_MAX_S
                        | OpcodeFD::I32X4_MAX_U
                        | OpcodeFD::I32X4_SUB
                        | OpcodeFD::I32X4_MUL
                        | OpcodeFD::I32X4_DOT_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_LOW_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_S
                        | OpcodeFD::I32X4_EXTMUL_LOW_I16X8_U
                        | OpcodeFD::I32X4_EXTMUL_HIGH_I16X8_U
                        | OpcodeFD::I64X2_SUB
                        | OpcodeFD::I64X2_MUL
                        | OpcodeFD::I64X2_EXTMUL_LOW_I32X4_S
                        | OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_S
                        | OpcodeFD::I64X2_EXTMUL_LOW_I32X4_U
                        | OpcodeFD::I64X2_EXTMUL_HIGH_I32X4_U
                        | OpcodeFD::F64X2_ADD
                        | OpcodeFD::F64X2_SUB
                        | OpcodeFD::F64X2_MUL
                        | OpcodeFD::I8X16_ADD_SAT_S
                        | OpcodeFD::I8X16_ADD_SAT_U
                        | OpcodeFD::I16X8_ADD_SAT_S
                        | OpcodeFD::I16X8_ADD_SAT_U
                        | OpcodeFD::I8X16_SUB_SAT_S
                        | OpcodeFD::I8X16_SUB_SAT_U
                        | OpcodeFD::I16X8_SUB_SAT_S
                        | OpcodeFD::I16X8_SUB_SAT_U
                        | OpcodeFD::I8X16_EQ
                        | OpcodeFD::I8X16_NE
                        | OpcodeFD::I8X16_LT_S
                        | OpcodeFD::I8X16_LT_U
                        | OpcodeFD::I8X16_GT_S
                        | OpcodeFD::I8X16_GT_U
                        | OpcodeFD::I8X16_LE_S
                        | OpcodeFD::I8X16_LE_U
                        | OpcodeFD::I8X16_GE_S
                        | OpcodeFD::I8X16_GE_U
                        | OpcodeFD::I16X8_EQ
                        | OpcodeFD::I16X8_NE
                        | OpcodeFD::I16X8_LT_S
                        | OpcodeFD::I16X8_LT_U
                        | OpcodeFD::I16X8_GT_S
                        | OpcodeFD::I16X8_GT_U
                        | OpcodeFD::I16X8_LE_S
                        | OpcodeFD::I16X8_LE_U
                        | OpcodeFD::I16X8_GE_S
                        | OpcodeFD::I16X8_GE_U
                        | OpcodeFD::I32X4_EQ
                        | OpcodeFD::I32X4_NE
                        | OpcodeFD::I32X4_LT_S
                        | OpcodeFD::I32X4_LT_U
                        | OpcodeFD::I32X4_GT_S
                        | OpcodeFD::I32X4_GT_U
                        | OpcodeFD::I32X4_LE_S
                        | OpcodeFD::I32X4_LE_U
                        | OpcodeFD::I32X4_GE_S
                        | OpcodeFD::I32X4_GE_U
                        | OpcodeFD::F32X4_EQ
                        | OpcodeFD::F32X4_NE
                        | OpcodeFD::F32X4_LT
                        | OpcodeFD::F32X4_GT
                        | OpcodeFD::F32X4_LE
                        | OpcodeFD::F32X4_GE
                        | OpcodeFD::F32X4_ADD
                        | OpcodeFD::F32X4_SUB
                        | OpcodeFD::F32X4_MUL
                        | OpcodeFD::F32X4_MAX
                        | OpcodeFD::F32X4_PMIN
                        | OpcodeFD::F32X4_PMAX
                        | OpcodeFD::I64X2_EQ
                        | OpcodeFD::I64X2_NE
                        | OpcodeFD::I64X2_LT_S
                        | OpcodeFD::I64X2_GT_S
                        | OpcodeFD::I64X2_LE_S
                        | OpcodeFD::I64X2_GE_S
                        | OpcodeFD::F64X2_EQ
                        | OpcodeFD::F64X2_NE
                        | OpcodeFD::F64X2_LT
                        | OpcodeFD::F64X2_GT
                        | OpcodeFD::F64X2_LE
                        | OpcodeFD::F64X2_GE
                        | OpcodeFD::F32X4_MIN
                        | OpcodeFD::F64X2_PMIN
                        | OpcodeFD::F64X2_PMAX
                        | OpcodeFD::F64X2_MIN
                        | OpcodeFD::F64X2_MAX
                        | OpcodeFD::F32X4_DIV
                        | OpcodeFD::F64X2_DIV
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdBinaryV128 {
                    opcode: op_ext as u32,
                })
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_SHL
                        | OpcodeFD::I8X16_SHR_S
                        | OpcodeFD::I8X16_SHR_U
                        | OpcodeFD::I16X8_SHL
                        | OpcodeFD::I16X8_SHR_S
                        | OpcodeFD::I16X8_SHR_U
                        | OpcodeFD::I32X4_SHL
                        | OpcodeFD::I32X4_SHR_S
                        | OpcodeFD::I32X4_SHR_U
                        | OpcodeFD::I64X2_SHL
                        | OpcodeFD::I64X2_SHR_S
                        | OpcodeFD::I64X2_SHR_U
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdShiftV128 {
                    opcode: op_ext as u32,
                })
            }
            FD(op_ext)
                if matches!(
                    op_ext,
                    OpcodeFD::I8X16_ABS
                        | OpcodeFD::I8X16_NEG
                        | OpcodeFD::I8X16_POPCNT
                        | OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_S
                        | OpcodeFD::I16X8_EXTADD_PAIRWISE_I8X16_U
                        | OpcodeFD::I16X8_ABS
                        | OpcodeFD::I16X8_NEG
                        | OpcodeFD::I16X8_EXTEND_LOW_I8X16_S
                        | OpcodeFD::I16X8_EXTEND_HIGH_I8X16_S
                        | OpcodeFD::I16X8_EXTEND_LOW_I8X16_U
                        | OpcodeFD::I16X8_EXTEND_HIGH_I8X16_U
                        | OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_S
                        | OpcodeFD::I32X4_EXTADD_PAIRWISE_I16X8_U
                        | OpcodeFD::I32X4_ABS
                        | OpcodeFD::I32X4_NEG
                        | OpcodeFD::I32X4_EXTEND_LOW_I16X8_S
                        | OpcodeFD::I32X4_EXTEND_HIGH_I16X8_S
                        | OpcodeFD::I32X4_EXTEND_LOW_I16X8_U
                        | OpcodeFD::I32X4_EXTEND_HIGH_I16X8_U
                        | OpcodeFD::I64X2_ABS
                        | OpcodeFD::I64X2_NEG
                        | OpcodeFD::I64X2_EXTEND_LOW_I32X4_S
                        | OpcodeFD::I64X2_EXTEND_HIGH_I32X4_S
                        | OpcodeFD::I64X2_EXTEND_LOW_I32X4_U
                        | OpcodeFD::I64X2_EXTEND_HIGH_I32X4_U
                        | OpcodeFD::F32X4_NEG
                        | OpcodeFD::F32X4_DEMOTE_F64X2_ZERO
                        | OpcodeFD::F32X4_SQRT
                        | OpcodeFD::F32X4_CEIL
                        | OpcodeFD::F32X4_FLOOR
                        | OpcodeFD::F32X4_TRUNC
                        | OpcodeFD::F32X4_NEAREST
                        | OpcodeFD::F64X2_ABS
                        | OpcodeFD::F64X2_NEG
                        | OpcodeFD::F64X2_SQRT
                        | OpcodeFD::F64X2_CEIL
                        | OpcodeFD::F64X2_FLOOR
                        | OpcodeFD::F64X2_TRUNC
                        | OpcodeFD::F64X2_NEAREST
                        | OpcodeFD::F32X4_ABS
                        | OpcodeFD::F32X4_CONVERT_I32X4_S
                        | OpcodeFD::F32X4_CONVERT_I32X4_U
                        | OpcodeFD::F64X2_PROMOTE_LOW_F32X4
                        | OpcodeFD::F64X2_CONVERT_LOW_I32X4_S
                        | OpcodeFD::F64X2_CONVERT_LOW_I32X4_U
                        | OpcodeFD::I32X4_TRUNC_SAT_F32X4_S
                        | OpcodeFD::I32X4_TRUNC_SAT_F32X4_U
                        | OpcodeFD::I32X4_TRUNC_SAT_F64X2_S_ZERO
                        | OpcodeFD::I32X4_TRUNC_SAT_F64X2_U_ZERO
                ) =>
            {
                self.handle_primitive(PrimitiveOpKind::SimdUnaryV128 {
                    opcode: op_ext as u32,
                })
            }
            FD(OpcodeFD::I32X4_ADD) => self.handle_primitive(PrimitiveOpKind::I32x4Add),
            FD(OpcodeFD::I64X2_ADD) => self.handle_primitive(PrimitiveOpKind::I64x2Add),

            FB(OpcodeFB::REF_I31) => self.handle_primitive(PrimitiveOpKind::RefI31),
            FB(OpcodeFB::I31_GET_S) => self.handle_primitive(PrimitiveOpKind::I31GetS),
            FB(OpcodeFB::I31_GET_U) => self.handle_primitive(PrimitiveOpKind::I31GetU),
            FB(OpcodeFB::ANY_CONVERT_EXTERN) => {
                self.handle_primitive(PrimitiveOpKind::AnyConvertExtern)
            }
            FB(OpcodeFB::EXTERN_CONVERT_ANY) => {
                self.handle_primitive(PrimitiveOpKind::ExternConvertAny)
            }
            FB(OpcodeFB::REF_TEST) | FB(OpcodeFB::REF_TEST_NULL) => {
                if let Immediate::RefType(ValueType::Ref(ref_type)) = imm {
                    self.handle_primitive(PrimitiveOpKind::RefTest {
                        ref_type: *ref_type,
                    });
                }
            }
            FB(OpcodeFB::REF_CAST) | FB(OpcodeFB::REF_CAST_NULL) => {
                if let Immediate::RefType(ValueType::Ref(ref_type)) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::RefCast {
                        ref_type: *ref_type,
                    });
                    self.record_result_types(op_idx, &[ValueType::Ref(*ref_type)]);
                }
            }
            FB(OpcodeFB::STRUCT_NEW) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let def_type =
                        self.compile.types.get(*type_idx).ok_or_else(|| {
                            WasmError::invalid("struct.new type index out of bounds")
                        })?;
                    let struct_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Struct(struct_type) => struct_type,
                        _ => return Err(WasmError::invalid("struct.new expected struct type")),
                    };
                    let field_count = u32::try_from(struct_type.fields.len()).map_err(|_| {
                        WasmError::invalid("struct.new field count exceeds decoder limit")
                    })?;
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*type_idx)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructNew {
                        type_idx: *type_idx,
                        field_count,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::STRUCT_NEW_DEFAULT) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*type_idx)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructNewDefault {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::STRUCT_GET) => {
                if let Immediate::StructFieldArgs { typeidx, fieldidx } = imm {
                    let def_type =
                        self.compile.types.get(*typeidx).ok_or_else(|| {
                            WasmError::invalid("struct.get type index out of bounds")
                        })?;
                    let struct_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Struct(struct_type) => struct_type,
                        _ => return Err(WasmError::invalid("struct.get expected struct type")),
                    };
                    let field_type = struct_type
                        .fields
                        .get(*fieldidx as usize)
                        .map(|field| field.storage.to_valtype())
                        .ok_or_else(|| {
                            WasmError::invalid("struct.get field index out of bounds")
                        })?;
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructGet {
                        type_idx: *typeidx,
                        field_idx: *fieldidx,
                    });
                    self.record_result_types(op_idx, &[field_type]);
                }
            }
            FB(OpcodeFB::STRUCT_GET_S) => {
                if let Immediate::StructFieldArgs { typeidx, fieldidx } = imm {
                    let def_type = self.compile.types.get(*typeidx).ok_or_else(|| {
                        WasmError::invalid("struct.get_s type index out of bounds")
                    })?;
                    let struct_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Struct(struct_type) => struct_type,
                        _ => return Err(WasmError::invalid("struct.get_s expected struct type")),
                    };
                    let field_type = struct_type
                        .fields
                        .get(*fieldidx as usize)
                        .map(|field| field.storage.to_valtype())
                        .ok_or_else(|| {
                            WasmError::invalid("struct.get_s field index out of bounds")
                        })?;
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructGetS {
                        type_idx: *typeidx,
                        field_idx: *fieldidx,
                    });
                    self.record_result_types(op_idx, &[field_type]);
                }
            }
            FB(OpcodeFB::STRUCT_GET_U) => {
                if let Immediate::StructFieldArgs { typeidx, fieldidx } = imm {
                    let def_type = self.compile.types.get(*typeidx).ok_or_else(|| {
                        WasmError::invalid("struct.get_u type index out of bounds")
                    })?;
                    let struct_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Struct(struct_type) => struct_type,
                        _ => return Err(WasmError::invalid("struct.get_u expected struct type")),
                    };
                    let field_type = struct_type
                        .fields
                        .get(*fieldidx as usize)
                        .map(|field| field.storage.to_valtype())
                        .ok_or_else(|| {
                            WasmError::invalid("struct.get_u field index out of bounds")
                        })?;
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructGetU {
                        type_idx: *typeidx,
                        field_idx: *fieldidx,
                    });
                    self.record_result_types(op_idx, &[field_type]);
                }
            }
            FB(OpcodeFB::STRUCT_SET) => {
                if let Immediate::StructFieldArgs { typeidx, fieldidx } = imm {
                    self.handle_primitive(PrimitiveOpKind::StructSet {
                        type_idx: *typeidx,
                        field_idx: *fieldidx,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_NEW) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*type_idx)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNew {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_NEW_DEFAULT) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*type_idx)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNewDefault {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_GET) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let def_type =
                        self.compile.types.get(*type_idx).ok_or_else(|| {
                            WasmError::invalid("array.get type index out of bounds")
                        })?;
                    let array_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Array(array_type) => array_type,
                        _ => return Err(WasmError::invalid("array.get expected array type")),
                    };
                    if matches!(
                        array_type.element.storage,
                        crate::module::type_defs::StorageType::Packed(_)
                    ) {
                        return Err(WasmError::invalid(
                            "array.get requires an unpacked array element type",
                        ));
                    }
                    let result_ty = array_type.element.storage.to_valtype();
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayGet {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_GET_S) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let def_type = self.compile.types.get(*type_idx).ok_or_else(|| {
                        WasmError::invalid("array.get_s type index out of bounds")
                    })?;
                    let array_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Array(array_type) => array_type,
                        _ => return Err(WasmError::invalid("array.get_s expected array type")),
                    };
                    if matches!(
                        array_type.element.storage,
                        crate::module::type_defs::StorageType::Val(_)
                    ) {
                        return Err(WasmError::invalid(
                            "array.get_s requires a packed array element type",
                        ));
                    }
                    let result_ty = array_type.element.storage.to_valtype();
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayGetS {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_GET_U) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let def_type = self.compile.types.get(*type_idx).ok_or_else(|| {
                        WasmError::invalid("array.get_u type index out of bounds")
                    })?;
                    let array_type = match &def_type.composite {
                        crate::module::type_defs::CompositeType::Array(array_type) => array_type,
                        _ => return Err(WasmError::invalid("array.get_u expected array type")),
                    };
                    if matches!(
                        array_type.element.storage,
                        crate::module::type_defs::StorageType::Val(_)
                    ) {
                        return Err(WasmError::invalid(
                            "array.get_u requires a packed array element type",
                        ));
                    }
                    let result_ty = array_type.element.storage.to_valtype();
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayGetU {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_LEN) => {
                self.handle_primitive(PrimitiveOpKind::ArrayLen);
            }
            FB(OpcodeFB::ARRAY_SET) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::ArraySet {
                        type_idx: *type_idx,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_FILL) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    self.handle_primitive(PrimitiveOpKind::ArrayFill {
                        type_idx: *type_idx,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_COPY) => {
                if let Immediate::TwoTypeIndices { type1, type2 } = imm {
                    self.handle_primitive(PrimitiveOpKind::ArrayCopy {
                        dst_type_idx: *type1,
                        src_type_idx: *type2,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_INIT_DATA) => {
                if let Immediate::TwoU32s { value1, value2 } = imm {
                    self.handle_primitive(PrimitiveOpKind::ArrayInitData {
                        type_idx: *value1,
                        data_idx: *value2,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_INIT_ELEM) => {
                if let Immediate::TwoU32s { value1, value2 } = imm {
                    self.handle_primitive(PrimitiveOpKind::ArrayInitElem {
                        type_idx: *value1,
                        elem_idx: *value2,
                    });
                }
            }
            FB(OpcodeFB::ARRAY_NEW_FIXED) => {
                if let Immediate::ArrayNewFixed { typeidx, n } = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*typeidx)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNewFixed {
                        type_idx: *typeidx,
                        count: *n,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_NEW_DATA) => {
                if let Immediate::TwoU32s { value1, value2 } = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*value1)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNewData {
                        type_idx: *value1,
                        data_idx: *value2,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }
            FB(OpcodeFB::ARRAY_NEW_ELEM) => {
                if let Immediate::TwoU32s { value1, value2 } = imm {
                    let result_ty =
                        ValueType::Ref(RefType::new(false, HeapType::Concrete(*value1)));
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNewElem {
                        type_idx: *value1,
                        elem_idx: *value2,
                    });
                    self.record_result_types(op_idx, &[result_ty]);
                    if !self.unreachable {
                        if let Some(last) = self.type_stack.last_mut() {
                            *last = result_ty;
                        }
                    }
                }
            }

            _ => {
                return Err(WasmError::invalid("unsupported semantic decode opcode"));
            }
        }

        Ok(())
    }

    fn handle_load(
        &mut self,
        imm: &Immediate,
        make_kind: impl FnOnce(u32, u32) -> PrimitiveOpKind,
    ) {
        if let Immediate::MemArg { memidx, offset, .. } = imm {
            self.handle_primitive(make_kind(*offset as u32, *memidx));
        }
    }

    fn handle_store(
        &mut self,
        imm: &Immediate,
        make_kind: impl FnOnce(u32, u32) -> PrimitiveOpKind,
    ) {
        if let Immediate::MemArg { memidx, offset, .. } = imm {
            self.handle_primitive(make_kind(*offset as u32, *memidx));
        }
    }
}

pub(crate) fn decode_to_semantic_ir(
    code: &[u8],
    compile: CompileContext<'_>,
) -> Result<SemanticProgram, WasmError> {
    let mut cx = DecodeContext::new(compile);
    decode_function_body(&mut cx, code)?;
    Ok(cx.finish())
}

fn decode_function_body(cx: &mut DecodeContext<'_>, code: &[u8]) -> Result<(), WasmError> {
    let mut handler = SemanticDecodeHandler { cx };
    let mut decoder = Decoder::new(code);
    decoder.add_handler(&mut handler);
    decoder.decode_function()
}

struct SemanticDecodeHandler<'a, 'b> {
    cx: &'a mut DecodeContext<'b>,
}

impl<'a, 'b> OpcodeHandler for SemanticDecodeHandler<'a, 'b> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            self.cx.dispatch(decoded.wasm_op, &decoded.imm)?;
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        // Wasm functions implicitly return the final stack results on
        // fallthrough. The closing `end` remains a semantic marker; the actual
        // function return is synthesized here after the body is fully decoded.
        if !self.cx.unreachable {
            self.cx.emit_return();
        }
        Ok(())
    }
}

fn extract_imm01(imm: &Immediate) -> (u32, u32) {
    match imm {
        Immediate::MemoryIndex(idx) => (*idx, 0),
        Immediate::TableIndex(idx) => (*idx, 0),
        Immediate::MemoryInitArgs { dataidx, memidx } => (*memidx, *dataidx),
        Immediate::MemoryCopyArgs { dstidx, srcidx } => (*dstidx, *srcidx),
        Immediate::TableInitArgs { elemidx, tableidx } => (*tableidx, *elemidx),
        Immediate::TableCopyArgs { dstidx, srcidx } => (*dstidx, *srcidx),
        _ => (0, 0),
    }
}
