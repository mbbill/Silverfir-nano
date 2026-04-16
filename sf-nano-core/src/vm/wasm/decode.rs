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
    op_decoder::{Decoder, Immediate, OpStream, OpcodeHandler},
    opcodes::{Opcode, OpcodeFB, OpcodeFC, WasmOpcode},
    value_type::{HeapType, RefType, ValueType},
};

use super::{
    common::{BrTableEntry, SemanticIndex, SemanticTarget},
    context::CompileContext,
    primitive_op::{self, PrimitiveOpKind},
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

/// Semantic IR builder used by Wasm decode.
#[derive(Default)]
pub(crate) struct SemanticBuilder {
    ops: collections::Vec<SemanticOp>,
}

impl SemanticBuilder {
    #[inline]
    pub(crate) fn current_index(&self) -> SemanticIndex {
        SemanticIndex::new(self.ops.len())
    }

    pub(crate) fn push(&mut self, kind: impl Into<SemanticOpKind>) -> SemanticIndex {
        let idx = self.current_index();
        self.ops.push(SemanticOp { kind: kind.into() });
        idx
    }

    pub(crate) fn patch_target(&mut self, idx: SemanticIndex, target: SemanticTarget) {
        if let Some(op) = self.ops.get_mut(idx.as_usize()) {
            match &mut op.kind {
                SemanticOpKind::If { else_target, .. } => *else_target = target,
                SemanticOpKind::Else { end_target } => *end_target = target,
                SemanticOpKind::Br { target: branch, .. }
                | SemanticOpKind::BrIf { target: branch, .. } => *branch = target,
                _ => {}
            }
        }
    }

    pub(crate) fn patch_br_table_target(
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

    pub(crate) fn finish(
        self,
        params: u16,
        results: u16,
        local_count: u16,
        max_stack_height: u16,
        local_types: collections::Vec<ValueType>,
        result_types: collections::Vec<ValueType>,
        op_result_types: BTreeMap<usize, collections::Vec<ValueType>>,
    ) -> SemanticProgram {
        SemanticProgram {
            params,
            results,
            local_count,
            max_stack_height,
            ops: self.ops,
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
}

#[derive(Clone, Debug)]
struct PendingBranchFixup {
    inst_idx: SemanticIndex,
    br_table_entry: Option<usize>,
}

#[derive(Clone, Debug)]
struct DecodeControlFrame {
    kind: DecodeBlockKind,
    start_height: usize,
    param_count: u16,
    result_count: u16,
    start_inst_idx: SemanticTarget,
    if_inst_idx: Option<SemanticIndex>,
    pending_fixups: collections::Vec<PendingBranchFixup>,
}

/// Decode-time semantic context.
pub(crate) struct DecodeContext<'a> {
    compile: CompileContext<'a>,
    builder: SemanticBuilder,
    control: collections::Vec<DecodeControlFrame>,
    height: usize,
    max_height: usize,
    unreachable: bool,
    /// Per-op result types for calls and typed blocks.
    op_result_types: BTreeMap<usize, collections::Vec<ValueType>>,
}

impl<'a> DecodeContext<'a> {
    pub(crate) fn new(compile: CompileContext<'a>) -> Self {
        Self {
            compile,
            builder: SemanticBuilder::default(),
            control: collections::vec![DecodeControlFrame {
                kind: DecodeBlockKind::Function,
                start_height: 0,
                param_count: 0,
                result_count: compile.results,
                start_inst_idx: SemanticTarget::new(0),
                if_inst_idx: None,
                pending_fixups: collections::Vec::new(),
            }],
            height: 0,
            max_height: 0,
            unreachable: false,
            op_result_types: BTreeMap::new(),
        }
    }

    #[inline]
    pub(crate) fn current_index(&self) -> SemanticIndex {
        self.builder.current_index()
    }

    #[inline]
    pub(crate) fn push_op(&mut self, kind: impl Into<SemanticOpKind>) -> SemanticIndex {
        self.builder.push(kind)
    }

    #[inline]
    pub(crate) fn patch_target(&mut self, idx: SemanticIndex, target: SemanticTarget) {
        self.builder.patch_target(idx, target);
    }

    #[inline]
    pub(crate) fn patch_br_table_target(
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
        }
    }

    #[inline]
    fn push_values(&mut self, count: usize) {
        if !self.unreachable {
            self.height += count;
            self.max_height = self.max_height.max(self.height);
        }
    }

    #[inline]
    fn pop_values(&mut self, count: usize) {
        if !self.unreachable {
            self.height = self.height.saturating_sub(count);
        }
    }

    #[inline]
    fn set_unreachable(&mut self) {
        self.unreachable = true;
    }

    fn enter_block(
        &mut self,
        kind: DecodeBlockKind,
        params: u16,
        results: u16,
        start_inst_idx: SemanticTarget,
    ) {
        self.control.push(DecodeControlFrame {
            kind,
            start_height: self.height.saturating_sub(params as usize),
            param_count: params,
            result_count: results,
            start_inst_idx,
            if_inst_idx: None,
            pending_fixups: collections::Vec::new(),
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
            self.unreachable = false;
        }
    }

    fn exit_block(&mut self) -> Option<DecodeControlFrame> {
        let frame = self.control.pop()?;
        self.height = frame.start_height + frame.result_count as usize;
        self.unreachable = false;
        Some(frame)
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
                DecodeBlockKind::Function | DecodeBlockKind::Block | DecodeBlockKind::If => {
                    frame.result_count
                }
            })
            .unwrap_or(0)
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
            DecodeBlockKind::Function | DecodeBlockKind::Block | DecodeBlockKind::If => None,
        };
        (stack_drop, target)
    }

    fn register_forward_branch(
        &mut self,
        depth: u32,
        inst_idx: SemanticIndex,
        br_table_entry: Option<usize>,
    ) {
        if let Some(frame) = self.frame_at_depth_mut(depth) {
            frame.pending_fixups.push(PendingBranchFixup {
                inst_idx,
                br_table_entry,
            });
        }
    }

    fn record_result_types(&mut self, idx: SemanticIndex, tys: &[ValueType]) {
        if !tys.is_empty() {
            self.op_result_types
                .insert(idx.as_usize(), tys.to_vec().into());
        }
    }

    fn handle_primitive(&mut self, kind: PrimitiveOpKind) {
        let (pops, pushes) = primitive_op::stack_effect(&kind);
        self.pop_values(pops as usize);
        self.push_op(kind.clone());
        self.push_values(pushes as usize);
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
        self.pop_values(params as usize);
        self.push_values(results as usize);
    }

    fn handle_call_indirect(&mut self, type_idx: u32, table_idx: u32) {
        let (params, results) = self.compile.resolve_type_index(type_idx);
        self.pop_values(1);
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
        self.pop_values(params as usize);
        self.push_values(results as usize);
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
        self.pop_values(params.saturating_add(1) as usize);
        self.push_values(results as usize);
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
        use WasmOpcode::{FB, FC, OP};

        match wasm_op {
            OP(LOCAL_GET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    self.push_op(SemanticOpKind::LocalGet { idx: *idx as u16 });
                    self.push_value();
                }
            }
            OP(LOCAL_SET) => {
                if let Immediate::LocalIndex(idx) = imm {
                    self.pop_values(1);
                    self.push_op(SemanticOpKind::LocalSet { idx: *idx as u16 });
                }
            }
            OP(LOCAL_TEE) => {
                if let Immediate::LocalIndex(idx) = imm {
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
                    self.handle_primitive(PrimitiveOpKind::GlobalSet { idx: *idx });
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
                }
            }
            OP(REF_IS_NULL) => self.handle_primitive(PrimitiveOpKind::RefIsNull),
            OP(REF_EQ) => self.handle_primitive(PrimitiveOpKind::RefEq),
            OP(REF_AS_NON_NULL) => self.handle_primitive(PrimitiveOpKind::RefAsNonNull),
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
                self.enter_block(DecodeBlockKind::Block, params, results, target);
            }
            OP(LOOP) => {
                let (params, results) = self.compile.resolve_block_type_from_imm(imm);
                let result_types = self.compile.resolve_block_result_types_from_imm(imm);
                let idx = self.push_op(SemanticOpKind::Loop { params, results });
                self.record_result_types(idx, &result_types);
                let target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(DecodeBlockKind::Loop, params, results, target);
            }
            OP(IF) => {
                self.pop_values(1);
                let (params, results) = self.compile.resolve_block_type_from_imm(imm);
                let result_types = self.compile.resolve_block_result_types_from_imm(imm);
                let idx = self.push_op(SemanticOpKind::If {
                    params,
                    results,
                    else_target: SemanticTarget::pending(),
                });
                self.record_result_types(idx, &result_types);
                let target = SemanticTarget::new(self.current_index().as_usize());
                self.enter_block(DecodeBlockKind::If, params, results, target);
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
                        br_table_entry: None,
                    });
                }
                self.enter_else();
            }
            OP(END) => {
                let end_idx = self.push_op(SemanticOpKind::End);
                let end_target = SemanticTarget::new(end_idx.as_usize());
                if let Some(frame) = self.exit_block() {
                    if frame.kind == DecodeBlockKind::If && frame.if_inst_idx.is_some() {
                        if let Some(if_idx) = frame.if_inst_idx {
                            if self
                                .builder
                                .ops
                                .get(if_idx.as_usize())
                                .is_some_and(|op| {
                                    matches!(
                                        op.kind,
                                        SemanticOpKind::If { else_target, .. } if else_target.is_pending()
                                    )
                                })
                            {
                                self.patch_target(if_idx, end_target);
                            }
                        }
                    }
                    for fixup in frame.pending_fixups {
                        if let Some(entry_idx) = fixup.br_table_entry {
                            self.patch_br_table_target(fixup.inst_idx, entry_idx, end_target);
                        } else {
                            self.patch_target(fixup.inst_idx, end_target);
                        }
                    }
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
                        self.register_forward_branch(*label, idx, None);
                    }
                    self.set_unreachable();
                }
            }
            OP(BR_IF) => {
                if let Immediate::LabelIndex(label) = imm {
                    self.pop_values(1);
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
                        self.register_forward_branch(*label, idx, None);
                    }
                }
            }
            OP(BR_TABLE) => {
                if let Immediate::BrLabels(labels, default_label) = imm {
                    self.pop_values(1);
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
                            self.register_forward_branch(label, inst_idx, Some(entry_idx));
                        }
                    }
                    self.set_unreachable();
                }
            }

            OP(RETURN) => self.emit_return(),
            OP(UNREACHABLE) => self.handle_primitive(PrimitiveOpKind::Unreachable),

            OP(CALL) => {
                if let Immediate::FunctionIndex(func_idx) = imm {
                    self.handle_call(*func_idx);
                }
            }
            OP(CALL_INDIRECT) => {
                if let Immediate::CallIndirectArgs { typeidx, tableidx } = imm {
                    self.handle_call_indirect(*typeidx, *tableidx);
                }
            }
            OP(CALL_REF) => {
                if let Immediate::TypeIndex(typeidx) = imm {
                    if self.unreachable {
                        self.handle_primitive(PrimitiveOpKind::Nop);
                    } else {
                        self.handle_call_ref(*typeidx);
                    }
                }
            }

            OP(NOP) => self.handle_primitive(PrimitiveOpKind::Nop),

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
            FB(OpcodeFB::STRUCT_NEW_DEFAULT) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::StructNewDefault {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(
                        op_idx,
                        &[ValueType::Ref(RefType::new(
                            false,
                            HeapType::Concrete(*type_idx),
                        ))],
                    );
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
            FB(OpcodeFB::ARRAY_NEW_DEFAULT) => {
                if let Immediate::TypeIndex(type_idx) = imm {
                    let op_idx = self.current_index();
                    self.handle_primitive(PrimitiveOpKind::ArrayNewDefault {
                        type_idx: *type_idx,
                    });
                    self.record_result_types(
                        op_idx,
                        &[ValueType::Ref(RefType::new(
                            false,
                            HeapType::Concrete(*type_idx),
                        ))],
                    );
                }
            }
            FB(_op) => {
                return Err(WasmError::invalid("unsupported semantic decode GC opcode"));
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
