use crate::collections;

use crate::{
    error::WasmError,
    module::{
        entities::{FunctionSpec, FunctionType},
        type_context::TypeContext,
        type_defs::{ArrayType, CompositeType, StorageType},
        Module,
    },
    op_decoder::{BlockType, CatchClause, CatchClauseKind, Immediate, OpStream, OpcodeHandler},
    opcodes::{Opcode, OpcodeFB, OpcodeFC, WasmOpcode},
    utils::limits::Limitable,
    value_type::{HeapType, RefType, ValueType},
};
use tracked_alloc::rc::Rc;

#[cfg(not(sf_has_simd))]
use crate::op_decoder::simd_opcode_error;
#[cfg(sf_has_simd)]
use crate::opcodes::OpcodeFD;

#[inline]
fn validator_immediate_mismatch(message: &'static str) -> WasmError {
    WasmError::internal(message)
}

#[inline]
fn expect_block_immediate(imm: &Immediate) -> Result<&BlockType, WasmError> {
    match imm {
        Immediate::Block(block_type) => Ok(block_type),
        _ => Err(validator_immediate_mismatch(
            "validator expected block immediate",
        )),
    }
}

#[inline]
fn expect_label_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::LabelIndex(label_index) => Ok(*label_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected label index immediate",
        )),
    }
}

#[inline]
fn expect_br_labels_immediate(imm: &Immediate) -> Result<(&[u32], u32), WasmError> {
    match imm {
        Immediate::BrLabels(labels, default) => Ok((labels, *default)),
        _ => Err(validator_immediate_mismatch(
            "validator expected br_table immediate",
        )),
    }
}

#[inline]
fn expect_function_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::FunctionIndex(function_index) => Ok(*function_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected function index immediate",
        )),
    }
}

/// Accept any label slot type that a non-null `(ref exn)` can flow into.
/// That covers both the non-null `(ref exn)` and the nullable `exnref`
/// forms — the caught exception is always live, so flowing into a
/// nullable slot is a subtype relationship, not a coercion.
#[inline]
fn is_exn_ref_sink(ty: ValueType) -> bool {
    let ValueType::Ref(ref_ty) = ty else {
        return false;
    };
    matches!(
        ref_ty.heap_type,
        HeapType::Abstract(crate::value_type::AbstractHeapType::Exn)
    )
}

#[inline]
fn expect_tag_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::TagIndex(tag_index) => Ok(*tag_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected tag index immediate",
        )),
    }
}

#[inline]
fn expect_try_table_immediate(imm: &Immediate) -> Result<(&BlockType, &[CatchClause]), WasmError> {
    match imm {
        Immediate::TryTable {
            block_type,
            catches,
        } => Ok((block_type, catches)),
        _ => Err(validator_immediate_mismatch(
            "validator expected try_table immediate",
        )),
    }
}

#[inline]
fn expect_local_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::LocalIndex(local_index) => Ok(*local_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected local index immediate",
        )),
    }
}

#[inline]
fn expect_global_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::GlobalIndex(global_index) => Ok(*global_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected global index immediate",
        )),
    }
}

#[inline]
fn expect_table_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::TableIndex(table_index) => Ok(*table_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected table index immediate",
        )),
    }
}

#[inline]
fn expect_memory_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::MemoryIndex(memory_index) => Ok(*memory_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected memory index immediate",
        )),
    }
}

#[inline]
fn expect_type_index_immediate(imm: &Immediate) -> Result<u32, WasmError> {
    match imm {
        Immediate::TypeIndex(type_index) => Ok(*type_index),
        _ => Err(validator_immediate_mismatch(
            "validator expected type index immediate",
        )),
    }
}

#[inline]
fn expect_ref_type_immediate(imm: &Immediate) -> Result<ValueType, WasmError> {
    match imm {
        Immediate::RefType(ref_type) => Ok(*ref_type),
        _ => Err(validator_immediate_mismatch(
            "validator expected ref type immediate",
        )),
    }
}

#[inline]
fn expect_select_types_immediate(imm: &Immediate) -> Result<&[ValueType], WasmError> {
    match imm {
        Immediate::SelectTypes(select_types) => Ok(select_types),
        _ => Err(validator_immediate_mismatch(
            "validator expected select types immediate",
        )),
    }
}

#[inline]
fn expect_call_indirect_immediate(imm: &Immediate) -> Result<(u32, u32), WasmError> {
    match imm {
        Immediate::CallIndirectArgs { typeidx, tableidx } => Ok((*typeidx, *tableidx)),
        _ => Err(validator_immediate_mismatch(
            "validator expected call_indirect immediate",
        )),
    }
}

#[inline]
fn expect_br_on_cast_immediate(imm: &Immediate) -> Result<(u32, ValueType, ValueType), WasmError> {
    match imm {
        Immediate::BrOnCast {
            label_idx,
            rt1,
            rt2,
            ..
        } => Ok((*label_idx, *rt1, *rt2)),
        _ => Err(validator_immediate_mismatch(
            "validator expected br_on_cast immediate",
        )),
    }
}

#[cfg(sf_has_simd)]
#[inline]
fn expect_simd_lane_immediate(imm: &Immediate) -> Result<u8, WasmError> {
    match imm {
        Immediate::LaneIndex(lane) => Ok(*lane),
        _ => Err(validator_immediate_mismatch(
            "validator expected SIMD lane immediate",
        )),
    }
}

#[cfg(sf_has_simd)]
#[inline]
fn expect_simd_shuffle_immediate(imm: &Immediate) -> Result<[u8; 16], WasmError> {
    match imm {
        Immediate::ShuffleMask(lanes) => Ok(*lanes),
        _ => Err(validator_immediate_mismatch(
            "validator expected SIMD shuffle immediate",
        )),
    }
}

pub(crate) struct FunctionValidator<'a> {
    module: &'a Module,
    function: &'a FunctionSpec,
    declared_functions: &'a [bool],
    context: Context,
}

impl<'a> OpcodeHandler for FunctionValidator<'a> {
    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            self.validate_decoded(decoded)?;
        }
        Ok(())
    }

    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        self.finish()
    }
}

impl<'a> FunctionValidator<'a> {
    pub(crate) fn validate_decoded(
        &mut self,
        decoded: &crate::op_decoder::DecodedOp,
    ) -> Result<(), WasmError> {
        match decoded.wasm_op {
            WasmOpcode::OP(op) => {
                self.on_op(op, decoded.op_offset, decoded.next_op_offset, &decoded.imm)
            }
            WasmOpcode::FC(op) => {
                self.on_op_fc(op, decoded.op_offset, decoded.next_op_offset, &decoded.imm)
            }
            WasmOpcode::FB(op) => {
                self.on_op_fb(op, decoded.op_offset, decoded.next_op_offset, &decoded.imm)
            }
            #[cfg(sf_has_simd)]
            WasmOpcode::FD(op) => self.on_op_fd(op, &decoded.imm),
            #[cfg(not(sf_has_simd))]
            WasmOpcode::FD(_) => self.on_op_fd(),
        }
    }

    pub(crate) fn finish(&self) -> Result<(), WasmError> {
        if !self.context.control_frames.is_empty() {
            return Err(WasmError::invalid(
                "function parsing ended with unclosed control frames",
            ));
        }

        let func_type = self.function.func_type();
        let actual_results = self.context.val_stack.as_slice();
        let expected_results = func_type.results();

        if actual_results.len() != expected_results.len() {
            return Err(WasmError::invalid("function return arity mismatch"));
        }

        for (actual, expected) in actual_results.iter().zip(expected_results.iter()) {
            if !actual.is_subtype_of(expected, self.module.types()) {
                return Err(WasmError::invalid(
                    "function return type mismatch, expected: , actual",
                ));
            }
        }

        Ok(())
    }

    /// Current validated operand stack after the most recent decoded op.
    /// Callers may copy this slice immediately, but must not retain it across
    /// another [`Self::validate_decoded`] mutation.
    #[cfg(test)]
    pub(crate) fn operand_types(&self) -> &[ValueType] {
        self.context.val_stack.as_slice()
    }

    pub(crate) fn new(
        module: &'a Module,
        function: &'a FunctionSpec,
        declared_functions: &'a [bool],
    ) -> Result<Self, WasmError> {
        let mut context = Context::new(
            module.types().clone(),
            function.func_type().params(),
            function.locals(),
        );

        context.push_ctrl(FrameType::Function, function.func_type_rc())?;

        Ok(FunctionValidator {
            module,
            function,
            declared_functions,
            context,
        })
    }

    fn get_block_type(&self, block_type: &BlockType) -> Result<Rc<FunctionType>, WasmError> {
        match *block_type {
            BlockType::Empty => Ok(Rc::new(FunctionType::new(
                collections::Vec::new(),
                collections::Vec::new(),
            ))),
            BlockType::ValueType(value_type) => {
                if let ValueType::Ref(ref_type) = value_type {
                    if let HeapType::Concrete(idx) = ref_type.heap_type {
                        if idx as usize >= self.module.types().len() {
                            return Err(WasmError::invalid("unknown type"));
                        }
                    }
                }
                Ok(Rc::new(FunctionType::new(
                    collections::Vec::new(),
                    collections::vec![value_type],
                )))
            }
            BlockType::TypeIndex(type_index) => self
                .module
                .types()
                .get_function_type(type_index as u32)
                .cloned()
                .ok_or_else(|| WasmError::malformed("block type index out of range")),
        }
    }

    fn get_local_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let local_index = expect_local_index_immediate(imm)?;
        let local_type = self
            .context
            .all_locals
            .get(local_index as usize)
            .ok_or_else(|| WasmError::invalid("local index out of range"))?;
        Ok(*local_type)
    }

    fn get_global_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let global_index = expect_global_index_immediate(imm)? as usize;
        let global = self
            .module
            .globals()
            .get(global_index)
            .ok_or_else(|| WasmError::invalid("global index out of range"))?;
        Ok(global.value_type())
    }

    fn get_table_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let table_index = expect_table_index_immediate(imm)? as usize;
        let table = self
            .module
            .tables()
            .get(table_index)
            .ok_or_else(|| WasmError::invalid("table index out of range"))?;
        Ok(table.value_type())
    }

    fn get_table_index_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let table_index = expect_table_index_immediate(imm)? as usize;
        let table = self
            .module
            .tables()
            .get(table_index)
            .ok_or_else(|| WasmError::invalid("table index out of range"))?;
        let is_table64 = table.spec().limits().is64;
        Ok(if is_table64 {
            ValueType::I64
        } else {
            ValueType::I32
        })
    }

    fn get_array_type(&self, typeidx: u32) -> Result<&ArrayType, WasmError> {
        let def_type = self
            .module
            .types()
            .get(typeidx)
            .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
        match &def_type.composite {
            CompositeType::Array(array_type) => Ok(array_type),
            _ => Err(WasmError::invalid("Expected array type")),
        }
    }

    fn storage_matches_for_array_copy(&self, src: StorageType, dst: StorageType) -> bool {
        match (src, dst) {
            (StorageType::Packed(src), StorageType::Packed(dst)) => src == dst,
            (StorageType::Val(src), StorageType::Val(dst)) => {
                src.is_subtype_of(&dst, &self.context.types)
            }
            _ => false,
        }
    }

    fn storage_is_data_segment_compatible(storage: StorageType) -> bool {
        matches!(
            storage,
            StorageType::Packed(_)
                | StorageType::Val(
                    ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
                )
        )
    }

    fn storage_ref_type(storage: StorageType) -> Option<RefType> {
        match storage {
            StorageType::Val(ValueType::Ref(ref_type)) => Some(ref_type),
            _ => None,
        }
    }

    fn handle_load<T: Sized>(
        &mut self,
        imm: &Immediate,
        val_type: ValueType,
    ) -> Result<(), WasmError> {
        use ValueType::*;
        let (align, memidx, offset) = match imm {
            &Immediate::MemArg {
                align,
                memidx,
                offset,
            } => (align, memidx, offset),
            _ => return Err(WasmError::internal("validator expected memarg immediate")),
        };
        if align > 63 {
            return Err(WasmError::invalid("invalid mem load alignment"));
        }
        if 2usize.pow(align) > core::mem::size_of::<T>() {
            return Err(WasmError::invalid("invalid mem load alignment"));
        }
        if memidx as usize >= self.module.memories().len() {
            return Err(WasmError::invalid("unknown memory"));
        }
        let mem = &self.module.memories()[memidx as usize];
        let is_mem64 = mem.spec().limits().is64;
        if !is_mem64 && offset > u32::MAX as u64 {
            return Err(WasmError::invalid("offset out of range"));
        }
        let index_type = if is_mem64 { I64 } else { I32 };
        self.context.pop_val(Some(index_type))?;
        self.context.push_val(val_type)
    }

    fn handle_store<T: Sized>(
        &mut self,
        imm: &Immediate,
        val_type: ValueType,
    ) -> Result<(), WasmError> {
        use ValueType::*;
        let (align, memidx, offset) = match imm {
            &Immediate::MemArg {
                align,
                memidx,
                offset,
            } => (align, memidx, offset),
            _ => return Err(WasmError::internal("validator expected memarg immediate")),
        };
        if align > 63 {
            return Err(WasmError::invalid("invalid mem store alignment"));
        }
        if 2usize.pow(align) > core::mem::size_of::<T>() {
            return Err(WasmError::invalid("invalid mem store alignment"));
        }
        if memidx as usize >= self.module.memories().len() {
            return Err(WasmError::invalid("unknown memory"));
        }
        let mem = &self.module.memories()[memidx as usize];
        let is_mem64 = mem.spec().limits().is64;
        if !is_mem64 && offset > u32::MAX as u64 {
            return Err(WasmError::invalid("offset out of range"));
        }
        let index_type = if is_mem64 { I64 } else { I32 };
        self.context.pop_val(Some(val_type))?;
        self.context.pop_val(Some(index_type))?;
        Ok(())
    }

    #[cfg(sf_has_simd)]
    fn validate_simd_lane(&self, lane: u8, limit: u8) -> Result<(), WasmError> {
        if lane >= limit {
            return Err(WasmError::invalid("SIMD lane index out of range"));
        }
        Ok(())
    }

    #[cfg(sf_has_simd)]
    fn handle_simd_mem_lane(
        &mut self,
        imm: &Immediate,
        elem_bytes: usize,
        lane_limit: u8,
        is_store: bool,
    ) -> Result<(), WasmError> {
        use ValueType::*;
        let (align, memidx, offset, lane) = match imm {
            &Immediate::MemArgLane {
                align,
                memidx,
                offset,
                lane,
            } => (align, memidx, offset, lane),
            _ => {
                return Err(WasmError::internal(
                    "validator expected SIMD memarg-lane immediate",
                ));
            }
        };
        self.validate_simd_lane(lane, lane_limit)?;
        if align > 63 {
            return Err(WasmError::invalid("invalid mem load alignment"));
        }
        if 2usize.pow(align) > elem_bytes {
            return Err(WasmError::invalid("invalid mem load alignment"));
        }
        if memidx as usize >= self.module.memories().len() {
            return Err(WasmError::invalid("unknown memory"));
        }
        let mem = &self.module.memories()[memidx as usize];
        let is_mem64 = mem.spec().limits().is64;
        if !is_mem64 && offset > u32::MAX as u64 {
            return Err(WasmError::invalid("offset out of range"));
        }
        let index_type = if is_mem64 { I64 } else { I32 };
        self.context.pop_val(Some(V128))?;
        self.context.pop_val(Some(index_type))?;
        if !is_store {
            self.context.push_val(V128)?;
        }
        Ok(())
    }

    fn on_op(
        &mut self,
        op: Opcode,
        _op_offset: usize,
        _next_op_offset: usize,
        imm: &Immediate,
    ) -> Result<(), WasmError> {
        use Opcode::*;
        use ValueType::*;
        match op {
            NOP | PREFIX_FB | PREFIX_FC | PREFIX_FD => Ok(()),
            UNREACHABLE => self.context.mark_unreachable(),
            BLOCK => {
                let block_type = expect_block_immediate(imm)?;
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::Block, function_type)?;
                Ok(())
            }
            LOOP => {
                let block_type = expect_block_immediate(imm)?;
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::Loop, function_type)?;
                Ok(())
            }
            IF => {
                let block_type = expect_block_immediate(imm)?;
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::If, function_type)?;
                Ok(())
            }
            TRY_TABLE => {
                let (block_type, catches) = expect_try_table_immediate(imm)?;
                let function_type = self.get_block_type(block_type)?;

                // Clause label indices refer to the *outer* control stack, so
                // validate catches before pushing the try_table frame.
                for catch in catches {
                    let label_types = self.context.frame_at(catch.label_idx)?.label_types();
                    match catch.kind {
                        CatchClauseKind::Catch | CatchClauseKind::CatchRef => {
                            let tag_idx = catch.tag_idx.ok_or_else(|| {
                                WasmError::invalid("catch/catch_ref requires a tag index")
                            })?;
                            let tag = self
                                .module
                                .tags()
                                .get(tag_idx as usize)
                                .ok_or_else(|| WasmError::invalid("unknown tag"))?;
                            let tag_ft = tag.func_type();
                            let expected_arity = tag_ft.params().len()
                                + if catch.kind == CatchClauseKind::CatchRef {
                                    1
                                } else {
                                    0
                                };
                            if label_types.len() != expected_arity {
                                return Err(WasmError::invalid(
                                    "catch clause label arity mismatch",
                                ));
                            }
                            if !tag_ft.params().iter().zip(label_types.iter()).all(
                                |(actual, expected)| {
                                    actual.is_subtype_of(expected, self.module.types())
                                },
                            ) {
                                return Err(WasmError::invalid(
                                    "catch clause label types mismatch",
                                ));
                            }
                            if catch.kind == CatchClauseKind::CatchRef
                                && !is_exn_ref_sink(label_types[tag_ft.params().len()])
                            {
                                return Err(WasmError::invalid(
                                    "catch_ref label must end with an exn ref",
                                ));
                            }
                        }
                        CatchClauseKind::CatchAll => {
                            if !label_types.is_empty() {
                                return Err(WasmError::invalid(
                                    "catch_all label must take no values",
                                ));
                            }
                        }
                        CatchClauseKind::CatchAllRef => {
                            if label_types.len() != 1 || !is_exn_ref_sink(label_types[0]) {
                                return Err(WasmError::invalid(
                                    "catch_all_ref label must take a single exn ref",
                                ));
                            }
                        }
                    }
                }

                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::TryTable, function_type)?;
                Ok(())
            }
            THROW => {
                let tag_idx = expect_tag_index_immediate(imm)?;
                let tag = self
                    .module
                    .tags()
                    .get(tag_idx as usize)
                    .ok_or_else(|| WasmError::invalid("unknown tag"))?;
                let tag_ft = tag.func_type();
                self.context.pop_vals(tag_ft.params())?;
                self.context.mark_unreachable()
            }
            THROW_REF => {
                self.context.pop_val(Some(ValueType::exnref()))?;
                self.context.mark_unreachable()
            }
            ELSE => {
                if self.context.frame_at(0)?.frame_type() != FrameType::If {
                    return Err(WasmError::invalid("invalid else"));
                }
                let if_frame = self.context.pop_ctrl()?;
                self.context
                    .push_ctrl(FrameType::Else, if_frame.function_type().clone())?;
                Ok(())
            }
            END => {
                let current_frame = self.context.pop_ctrl()?;
                let frame_type = current_frame.function_type();
                if current_frame.frame_type() == FrameType::If
                    && frame_type.params() != frame_type.results()
                {
                    return Err(WasmError::invalid(
                        "if without else should keep the stack consistent".into(),
                    ));
                }
                self.context.push_vals(frame_type.results())?;
                Ok(())
            }
            BR | BR_IF => {
                let label_index = expect_label_index_immediate(imm)?;
                if op == BR_IF {
                    self.context.pop_val(Some(I32))?;
                }
                let label_types = self.context.frame_at(label_index)?.label_types();
                self.context.pop_vals(&label_types)?;
                if op == BR {
                    self.context.mark_unreachable()
                } else {
                    self.context.push_vals(&label_types)
                }
            }
            BR_TABLE => {
                let (labels, default) = expect_br_labels_immediate(imm)?;
                self.context.pop_val(Some(I32))?;
                let default_label_types = self.context.frame_at(default)?.label_types();
                let default_arity = default_label_types.len();
                labels
                    .iter()
                    .copied()
                    .chain(core::iter::once(default))
                    .try_for_each(|label| {
                        let label_types = self.context.frame_at(label)?.label_types();
                        let arity = label_types.len();
                        if arity != default_arity {
                            return Err(WasmError::invalid("invalid br_table arity"));
                        }
                        let popped = self.context.pop_vals(&label_types)?;
                        self.context.push_vals(&popped)
                    })?;
                self.context.pop_vals(&default_label_types)?;
                self.context.mark_unreachable()?;
                Ok(())
            }
            RETURN => {
                let func_label_types = self.context.frame_last()?.label_types();
                self.context.pop_vals(&func_label_types)?;
                self.context.mark_unreachable()?;
                Ok(())
            }
            CALL => {
                let function_index = expect_function_index_immediate(imm)?;
                let function = self
                    .module
                    .functions()
                    .get(function_index as usize)
                    .ok_or_else(|| WasmError::invalid("function index out of range"))?;
                let function_type = function.func_type();
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())
            }
            CALL_INDIRECT => {
                let (typeidx, tableidx) = expect_call_indirect_immediate(imm)?;
                let table = self
                    .module
                    .tables()
                    .get(tableidx as usize)
                    .ok_or_else(|| WasmError::invalid("invalid table index"))?;
                let is_table64 = table.spec().limits().is64;
                let idx_type = if is_table64 { I64 } else { I32 };
                self.context.pop_val(Some(idx_type))?;
                let function_type = self
                    .module
                    .types()
                    .get_function_type(typeidx)
                    .cloned()
                    .ok_or_else(|| WasmError::invalid("invalid function type index"))?;
                let table_type = table.value_type();
                if !table_type.is_funcref() {
                    return Err(WasmError::invalid(
                        "call_indirect requires funcref table, got",
                    ));
                }
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())
            }
            CALL_REF => {
                let type_idx = expect_type_index_immediate(imm)?;
                let function_type = self
                    .module
                    .types()
                    .get_function_type(type_idx)
                    .cloned()
                    .ok_or_else(|| WasmError::invalid("invalid function type index"))?;
                self.context
                    .pop_val(Some(ValueType::Ref(RefType::nullable_concrete(type_idx))))?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())
            }
            RETURN_CALL => {
                let function_index = expect_function_index_immediate(imm)?;
                let function = self
                    .module
                    .functions()
                    .get(function_index as usize)
                    .ok_or_else(|| WasmError::invalid("function index out of range"))?;
                let function_type = function.func_type();
                let func_label_types = self.context.frame_last()?.label_types();
                if function_type.results().len() != func_label_types.len() {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())?;
                self.context.pop_vals(&func_label_types)?;
                self.context.mark_unreachable()?;
                Ok(())
            }
            RETURN_CALL_INDIRECT => {
                let (typeidx, tableidx) = expect_call_indirect_immediate(imm)?;
                let table = self
                    .module
                    .tables()
                    .get(tableidx as usize)
                    .ok_or_else(|| WasmError::invalid("invalid table index"))?;
                let is_table64 = table.spec().limits().is64;
                let idx_type = if is_table64 { I64 } else { I32 };
                self.context.pop_val(Some(idx_type))?;
                let function_type = self
                    .module
                    .types()
                    .get_function_type(typeidx)
                    .cloned()
                    .ok_or_else(|| WasmError::invalid("invalid function type index"))?;
                let table_type = table.value_type();
                if !table_type.is_funcref() {
                    return Err(WasmError::invalid(
                        "call_indirect requires funcref table, got",
                    ));
                }
                let func_label_types = self.context.frame_last()?.label_types();
                if function_type.results().len() != func_label_types.len() {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())?;
                self.context.pop_vals(&func_label_types)?;
                self.context.mark_unreachable()?;
                Ok(())
            }
            RETURN_CALL_REF => {
                let type_idx = expect_type_index_immediate(imm)?;
                let function_type = self
                    .module
                    .types()
                    .get_function_type(type_idx)
                    .cloned()
                    .ok_or_else(|| WasmError::invalid("invalid function type index"))?;
                let func_label_types = self.context.frame_last()?.label_types();
                if function_type.results().len() != func_label_types.len() {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context
                    .pop_val(Some(ValueType::Ref(RefType::nullable_concrete(type_idx))))?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_vals(function_type.results())?;
                self.context.pop_vals(&func_label_types)?;
                self.context.mark_unreachable()?;
                Ok(())
            }
            DROP => {
                self.context.pop_val(None)?;
                Ok(())
            }
            SELECT => {
                self.context.pop_val(Some(I32))?;
                let t1 = self.context.pop_val(None)?;
                let t2 = self.context.pop_val(None)?;
                let same_category = (t1.is_num() && t2.is_num()) || (t1.is_vec() && t2.is_vec());
                if !same_category {
                    return Err(WasmError::invalid(
                        "SELECT requires both operands to be numbers or both be vectors".into(),
                    ));
                }
                if !t1.is_compatible_with(&t2) && t1 != Unknown && t2 != Unknown {
                    return Err(WasmError::invalid("SELECT type mismatch: vs"));
                }
                if t1 == Unknown {
                    self.context.push_val(t2)
                } else {
                    self.context.push_val(t1)
                }
            }
            SELECT_T => {
                let select_types = expect_select_types_immediate(imm)?;
                if select_types.len() != 1 {
                    return Err(WasmError::invalid("invalid select types size"));
                }
                let select_type = select_types[0];
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(select_type))?;
                self.context.pop_val(Some(select_type))?;
                self.context.push_val(select_type)
            }
            LOCAL_GET => {
                let local_index = expect_local_index_immediate(imm)?;
                let local_type = self.get_local_type(imm)?;
                if local_index as usize >= self.context.locals_init.len() {
                    return Err(WasmError::invalid("local index out of range"));
                }
                if !self.context.locals_init[local_index as usize] {
                    return Err(WasmError::invalid("uninitialized local"));
                }
                self.context.push_val(local_type)
            }
            LOCAL_SET => {
                let local_index = expect_local_index_immediate(imm)?;
                let local_type = self.get_local_type(imm)?;
                self.context.pop_val(Some(local_type))?;
                self.context.set_local_initialized(local_index as usize);
                Ok(())
            }
            LOCAL_TEE => {
                let local_index = expect_local_index_immediate(imm)?;
                let local_type = self.get_local_type(imm)?;
                self.context.pop_val(Some(local_type))?;
                self.context.set_local_initialized(local_index as usize);
                self.context.push_val(local_type)
            }
            GLOBAL_GET => {
                let global_type = self.get_global_type(imm)?;
                self.context.push_val(global_type)
            }
            GLOBAL_SET => {
                let global_index = expect_global_index_immediate(imm)? as usize;
                let global = self
                    .module
                    .globals()
                    .get(global_index)
                    .ok_or_else(|| WasmError::invalid("global not found"))?;
                if !global.mutable() {
                    return Err(WasmError::invalid("Global is immutable"));
                }
                let global_type = self.get_global_type(imm)?;
                self.context.pop_val(Some(global_type))?;
                Ok(())
            }
            TABLE_GET => {
                let table_type = self.get_table_type(imm)?;
                let index_type = self.get_table_index_type(imm)?;
                self.context.pop_val(Some(index_type))?;
                self.context.push_val(table_type)
            }
            TABLE_SET => {
                let table_type = self.get_table_type(imm)?;
                let index_type = self.get_table_index_type(imm)?;
                self.context.pop_val(Some(table_type))?;
                self.context.pop_val(Some(index_type))?;
                Ok(())
            }
            I32_LOAD => self.handle_load::<i32>(imm, I32),
            I32_LOAD8_S | I32_LOAD8_U => self.handle_load::<i8>(imm, I32),
            I32_LOAD16_S | I32_LOAD16_U => self.handle_load::<i16>(imm, I32),
            I64_LOAD => self.handle_load::<i64>(imm, I64),
            I64_LOAD8_S | I64_LOAD8_U => self.handle_load::<i8>(imm, I64),
            I64_LOAD16_S | I64_LOAD16_U => self.handle_load::<i16>(imm, I64),
            I64_LOAD32_S | I64_LOAD32_U => self.handle_load::<i32>(imm, I64),
            F32_LOAD => self.handle_load::<f32>(imm, F32),
            F64_LOAD => self.handle_load::<f64>(imm, F64),
            I32_STORE => self.handle_store::<i32>(imm, I32),
            I32_STORE8 => self.handle_store::<i8>(imm, I32),
            I32_STORE16 => self.handle_store::<i16>(imm, I32),
            I64_STORE => self.handle_store::<i64>(imm, I64),
            I64_STORE8 => self.handle_store::<i8>(imm, I64),
            I64_STORE16 => self.handle_store::<i16>(imm, I64),
            I64_STORE32 => self.handle_store::<i32>(imm, I64),
            F32_STORE => self.handle_store::<f32>(imm, F32),
            F64_STORE => self.handle_store::<f64>(imm, F64),
            MEMORY_SIZE => {
                let memidx = expect_memory_index_immediate(imm)? as usize;
                if memidx >= self.module.memories().len() {
                    return Err(WasmError::invalid("unknown memory"));
                }
                let mem = &self.module.memories()[memidx];
                let is_mem64 = mem.spec().limits().is64;
                let size_type = if is_mem64 { I64 } else { I32 };
                self.context.push_val(size_type)
            }
            MEMORY_GROW => {
                let memidx = expect_memory_index_immediate(imm)? as usize;
                if memidx >= self.module.memories().len() {
                    return Err(WasmError::invalid("unknown memory"));
                }
                let mem = &self.module.memories()[memidx];
                let is_mem64 = mem.spec().limits().is64;
                let size_type = if is_mem64 { I64 } else { I32 };
                self.context.pop_val(Some(size_type))?;
                self.context.push_val(size_type)
            }
            REF_NULL => {
                let ref_type = expect_ref_type_immediate(imm)?;
                if !ref_type.is_ref() {
                    return Err(WasmError::invalid("invalid ref type"));
                }
                self.context.push_val(ref_type)
            }
            REF_IS_NULL => {
                self.context.pop_ref_type()?;
                self.context.push_val(I32)?;
                Ok(())
            }
            REF_EQ => {
                let ref1 = self.context.pop_ref_type()?;
                let ref2 = self.context.pop_ref_type()?;
                if !ref1.is_subtype_of_eqref() || !ref2.is_subtype_of_eqref() {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.push_val(I32)?;
                Ok(())
            }
            REF_AS_NON_NULL => {
                let ref_type = self.context.pop_ref_type()?;
                self.context.push_val(ref_type.to_non_nullable())?;
                Ok(())
            }
            BR_ON_NULL => {
                let ref_type = self.context.pop_ref_type()?;
                let label_index = expect_label_index_immediate(imm)?;
                let label_types = self.context.frame_at(label_index)?.label_types();
                self.context.pop_vals(&label_types)?;
                self.context.push_vals(&label_types)?;
                self.context.push_val(ref_type.to_non_nullable())?;
                Ok(())
            }
            BR_ON_NON_NULL => {
                let ref_type = self.context.pop_ref_type()?;
                let label_index = expect_label_index_immediate(imm)?;
                let label_types = self.context.frame_at(label_index)?.label_types();
                let (branch_ref, prefix) = label_types
                    .split_last()
                    .ok_or_else(|| WasmError::invalid("br_on_non_null requires label result"))?;
                if !branch_ref.is_ref() {
                    return Err(WasmError::invalid(
                        "br_on_non_null requires reference label type",
                    ));
                }
                let refined = ref_type.to_non_nullable();
                if !refined.is_subtype_of(branch_ref, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.pop_vals(prefix)?;
                self.context.push_vals(prefix)?;
                Ok(())
            }
            REF_FUNC => {
                let function_index = expect_function_index_immediate(imm)?;
                if function_index as usize >= self.module.functions().len() {
                    return Err(WasmError::invalid("function index out of range"));
                }
                if !self.declared_functions[function_index as usize] {
                    return Err(WasmError::invalid("undeclared function reference"));
                }

                let type_idx = self.module.functions()[function_index as usize].type_index();
                let heap_type = HeapType::Concrete(type_idx);
                let ref_type = RefType::new(false, heap_type);
                let value_type = ValueType::Ref(ref_type);
                self.context.push_val(value_type)
            }
            I32_CONST => self.context.push_val(I32),
            I64_CONST => self.context.push_val(I64),
            F32_CONST => self.context.push_val(F32),
            F64_CONST => self.context.push_val(F64),
            I32_EQZ => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I32)
            }
            I64_EQZ => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I32)
            }
            I32_CLZ | I32_CTZ | I32_POPCNT => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I32)
            }
            I64_CLZ | I64_CTZ | I64_POPCNT => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I64)
            }
            I32_EQ | I32_NE | I32_LT_S | I32_LT_U | I32_GT_S | I32_GT_U | I32_LE_S | I32_LE_U
            | I32_GE_S | I32_GE_U => {
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I32)
            }
            I64_EQ | I64_NE | I64_LT_S | I64_LT_U | I64_GT_S | I64_GT_U | I64_LE_S | I64_LE_U
            | I64_GE_S | I64_GE_U => {
                self.context.pop_val(Some(I64))?;
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I32)
            }
            I32_ADD | I32_SUB | I32_MUL | I32_DIV_S | I32_DIV_U | I32_REM_S | I32_REM_U
            | I32_AND | I32_OR | I32_XOR | I32_SHL | I32_SHR_S | I32_SHR_U | I32_ROTL
            | I32_ROTR => {
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I32)
            }
            I64_ADD | I64_SUB | I64_MUL | I64_DIV_S | I64_DIV_U | I64_REM_S | I64_REM_U
            | I64_AND | I64_OR | I64_XOR | I64_SHL | I64_SHR_S | I64_SHR_U | I64_ROTL
            | I64_ROTR => {
                self.context.pop_val(Some(I64))?;
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I64)
            }
            F32_ABS | F32_NEG | F32_CEIL | F32_FLOOR | F32_TRUNC | F32_NEAREST | F32_SQRT => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(F32)
            }
            F64_ABS | F64_NEG | F64_CEIL | F64_FLOOR | F64_TRUNC | F64_NEAREST | F64_SQRT => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(F64)
            }
            F32_EQ | F32_NE | F32_LT | F32_GT | F32_LE | F32_GE => {
                self.context.pop_val(Some(F32))?;
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I32)
            }
            F64_EQ | F64_NE | F64_LT | F64_GT | F64_LE | F64_GE => {
                self.context.pop_val(Some(F64))?;
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I32)
            }
            F32_ADD | F32_SUB | F32_MUL | F32_DIV | F32_MIN | F32_MAX | F32_COPYSIGN => {
                self.context.pop_val(Some(F32))?;
                self.context.pop_val(Some(F32))?;
                self.context.push_val(F32)
            }
            F64_ADD | F64_SUB | F64_MUL | F64_DIV | F64_MIN | F64_MAX | F64_COPYSIGN => {
                self.context.pop_val(Some(F64))?;
                self.context.pop_val(Some(F64))?;
                self.context.push_val(F64)
            }
            I32_WRAP_I64 => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I32)
            }
            I32_TRUNC_F32_S | I32_TRUNC_F32_U => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I32)
            }
            I32_TRUNC_F64_S | I32_TRUNC_F64_U => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I32)
            }
            I64_TRUNC_F32_S | I64_TRUNC_F32_U => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I64)
            }
            I64_TRUNC_F64_S | I64_TRUNC_F64_U => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I64)
            }
            F32_CONVERT_I32_S | F32_CONVERT_I32_U => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(F32)
            }
            F32_CONVERT_I64_S | F32_CONVERT_I64_U => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(F32)
            }
            F64_CONVERT_I32_S | F64_CONVERT_I32_U => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(F64)
            }
            F64_CONVERT_I64_S | F64_CONVERT_I64_U => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(F64)
            }
            F32_DEMOTE_F64 => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(F32)
            }
            F64_PROMOTE_F32 => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(F64)
            }
            I32_REINTERPRET_F32 => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I32)
            }
            I64_REINTERPRET_F64 => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I64)
            }
            F32_REINTERPRET_I32 => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(F32)
            }
            F64_REINTERPRET_I64 => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(F64)
            }
            I64_EXTEND_I32_S | I64_EXTEND_I32_U => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I64)
            }
            I32_EXTEND8_S | I32_EXTEND16_S => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(I32)
            }
            I64_EXTEND8_S | I64_EXTEND16_S | I64_EXTEND32_S => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(I64)
            }
        }
    }

    fn on_op_fc(
        &mut self,
        op: OpcodeFC,
        _op_offset: usize,
        _next_op_offset: usize,
        imm: &Immediate,
    ) -> Result<(), WasmError> {
        use OpcodeFC::*;
        use ValueType::*;
        match op {
            I32_TRUNC_SAT_F32_S | I32_TRUNC_SAT_F32_U => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I32)
            }
            I32_TRUNC_SAT_F64_S | I32_TRUNC_SAT_F64_U => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I32)
            }
            I64_TRUNC_SAT_F32_S | I64_TRUNC_SAT_F32_U => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(I64)
            }
            I64_TRUNC_SAT_F64_S | I64_TRUNC_SAT_F64_U => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(I64)
            }
            MEMORY_INIT => {
                let (dataidx, memidx) = match imm {
                    &Immediate::MemoryInitArgs { dataidx, memidx } => (dataidx, memidx),
                    _ => return Err(WasmError::invalid("invalid memory init arguments")),
                };
                if dataidx as usize >= self.module.data().len() {
                    return Err(WasmError::invalid("invalid memory init data index"));
                }
                if memidx as usize >= self.module.memories().len() {
                    return Err(WasmError::invalid(
                        "invalid memory init memory index".into(),
                    ));
                }
                if self.module.data_count().is_none() {
                    return Err(WasmError::malformed(
                        "memory.init requires a datacount section".into(),
                    ));
                }
                let mem = &self.module.memories()[memidx as usize];
                let is_mem64 = mem.spec().limits().is64;
                let dest_type = if is_mem64 { I64 } else { I32 };
                self.context.pop_val(Some(I32))?; // size
                self.context.pop_val(Some(I32))?; // src offset
                self.context.pop_val(Some(dest_type))?; // dest
                Ok(())
            }
            MEMORY_COPY => {
                let (dstidx, srcidx) = match imm {
                    &Immediate::MemoryCopyArgs { dstidx, srcidx } => (dstidx, srcidx),
                    _ => return Err(WasmError::invalid("invalid memory copy arguments")),
                };
                if dstidx as usize >= self.module.memories().len() {
                    return Err(WasmError::invalid(
                        "invalid memory copy destination memory index".into(),
                    ));
                }
                if srcidx as usize >= self.module.memories().len() {
                    return Err(WasmError::invalid(
                        "invalid memory copy source memory index".into(),
                    ));
                }
                let dst_is_64 = self.module.memories()[dstidx as usize].spec().limits().is64;
                let src_is_64 = self.module.memories()[srcidx as usize].spec().limits().is64;
                let dst_idx_type = if dst_is_64 { I64 } else { I32 };
                let src_idx_type = if src_is_64 { I64 } else { I32 };
                let size_type = if dst_is_64 && src_is_64 { I64 } else { I32 };
                self.context.pop_val(Some(size_type))?;
                self.context.pop_val(Some(src_idx_type))?;
                self.context.pop_val(Some(dst_idx_type))?;
                Ok(())
            }
            MEMORY_FILL => {
                let memidx = match imm {
                    &Immediate::MemoryIndex(memidx) => memidx,
                    _ => return Err(WasmError::invalid("invalid memory fill arguments")),
                };
                if memidx as usize >= self.module.memories().len() {
                    return Err(WasmError::invalid(
                        "invalid memory fill memory index".into(),
                    ));
                }
                let mem = &self.module.memories()[memidx as usize];
                let is_mem64 = mem.spec().limits().is64;
                let idx_type = if is_mem64 { I64 } else { I32 };
                self.context.pop_val(Some(idx_type))?; // size
                self.context.pop_val(Some(I32))?; // value
                self.context.pop_val(Some(idx_type))?; // dest
                Ok(())
            }
            DATA_DROP => {
                if self.module.data().is_empty() {
                    return Err(WasmError::invalid("unknown data segment"));
                }
                if self.module.data_count().is_none() {
                    return Err(WasmError::malformed(
                        "data.drop requires a datacount section".into(),
                    ));
                }
                let dataidx = match imm {
                    &Immediate::DataIndex(dataidx) => dataidx,
                    _ => return Err(WasmError::invalid("invalid data index")),
                };
                if dataidx as usize >= self.module.data().len() {
                    return Err(WasmError::invalid("invalid data index"));
                }
                Ok(())
            }
            TABLE_INIT => {
                let (elemidx, tableidx) = match imm {
                    &Immediate::TableInitArgs { elemidx, tableidx } => (elemidx, tableidx),
                    _ => return Err(WasmError::invalid("invalid table init arguments")),
                };
                if elemidx as usize >= self.module.elements().len() {
                    return Err(WasmError::invalid(
                        "invalid table init element index".into(),
                    ));
                }
                if tableidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table init table index"));
                }
                // Check element-table type compatibility
                let elem = &self.module.elements()[elemidx as usize];
                let table = &self.module.tables()[tableidx as usize];
                let elem_type = elem.value_type();
                let table_type = table.value_type();
                if !elem_type.is_subtype_of(&table_type, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                let table = &self.module.tables()[tableidx as usize];
                let is_table64 = table.spec().limits().is64;
                let dest_type = if is_table64 { I64 } else { I32 };
                self.context.pop_val(Some(I32))?; // size
                self.context.pop_val(Some(I32))?; // src offset
                self.context.pop_val(Some(dest_type))?; // dest
                Ok(())
            }
            ELEM_DROP => {
                let elemidx = match imm {
                    &Immediate::ElementIndex(elemidx) => elemidx,
                    _ => return Err(WasmError::invalid("invalid element index")),
                };
                if elemidx as usize >= self.module.elements().len() {
                    return Err(WasmError::invalid("invalid element index"));
                }
                Ok(())
            }
            TABLE_COPY => {
                let (dstidx, srcidx) = match imm {
                    &Immediate::TableCopyArgs { dstidx, srcidx } => (dstidx, srcidx),
                    _ => return Err(WasmError::invalid("invalid table copy arguments")),
                };
                if dstidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table copy dst index"));
                }
                if srcidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table copy src index"));
                }
                let dst_table = &self.module.tables()[dstidx as usize];
                let src_table = &self.module.tables()[srcidx as usize];
                let dst_type = dst_table.value_type();
                let src_type = src_table.value_type();
                if !src_type.is_subtype_of(&dst_type, self.module.types()) {
                    return Err(WasmError::invalid("table copy type mismatch"));
                }

                let dst_is_64 = dst_table.spec().limits().is64;
                let src_is_64 = src_table.spec().limits().is64;
                let dst_index_type = if dst_is_64 { I64 } else { I32 };
                let src_index_type = if src_is_64 { I64 } else { I32 };
                let size_type = if dst_is_64 && src_is_64 { I64 } else { I32 };
                self.context.pop_val(Some(size_type))?;
                self.context.pop_val(Some(src_index_type))?;
                self.context.pop_val(Some(dst_index_type))?;
                Ok(())
            }
            TABLE_GROW => {
                let tableidx = match imm {
                    &Immediate::TableIndex(tableidx) => tableidx,
                    _ => return Err(WasmError::invalid("invalid table index")),
                };
                if tableidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table index"));
                }
                let table = &self.module.tables()[tableidx as usize];
                let table_type = table.value_type();
                let is_table64 = table.spec().limits().is64;
                let size_type = if is_table64 { I64 } else { I32 };
                self.context.pop_val(Some(size_type))?;
                self.context.pop_val(Some(table_type))?;
                self.context.push_val(size_type)
            }
            TABLE_SIZE => {
                let tableidx = match imm {
                    &Immediate::TableIndex(tableidx) => tableidx,
                    _ => return Err(WasmError::invalid("invalid table index")),
                };
                if tableidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table index"));
                }
                let table = &self.module.tables()[tableidx as usize];
                let is_table64 = table.spec().limits().is64;
                let size_type = if is_table64 { I64 } else { I32 };
                self.context.push_val(size_type)
            }
            TABLE_FILL => {
                let tableidx = match imm {
                    &Immediate::TableIndex(tableidx) => tableidx,
                    _ => return Err(WasmError::invalid("invalid table index")),
                };
                if tableidx as usize >= self.module.tables().len() {
                    return Err(WasmError::invalid("invalid table index"));
                }
                let table = &self.module.tables()[tableidx as usize];
                let table_type = table.value_type();
                let is_table64 = table.spec().limits().is64;
                let idx_type = if is_table64 { I64 } else { I32 };
                self.context.pop_val(Some(idx_type))?; // n (size)
                let ref_val = self.context.pop_ref_type()?; // value
                if !ref_val.is_subtype_of(&table_type, self.module.types()) {
                    return Err(WasmError::invalid(
                        "table fill type mismatch: expected , got",
                    ));
                }
                self.context.pop_val(Some(idx_type))?; // dest
                Ok(())
            }
        }
    }

    fn on_op_fb(
        &mut self,
        op: OpcodeFB,
        _op_offset: usize,
        _next_op_offset: usize,
        imm: &Immediate,
    ) -> Result<(), WasmError> {
        use crate::{
            module::type_defs::CompositeType,
            value_type::{AbstractHeapType, HeapType, RefType},
        };
        use OpcodeFB::*;
        use ValueType::*;

        match op {
            STRUCT_NEW => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let struct_type = match &def_type.composite {
                    CompositeType::Struct(s) => s,
                    _ => return Err(WasmError::invalid("Expected struct type")),
                };
                for field in struct_type.fields.iter().rev() {
                    let field_type = field.storage.to_valtype();
                    self.context.pop_val(Some(field_type))?;
                }
                let struct_ref =
                    ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx as u32)));
                self.context.push_val(struct_ref)?;
                Ok(())
            }
            STRUCT_NEW_DEFAULT => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                match &def_type.composite {
                    CompositeType::Struct(_) => {}
                    _ => return Err(WasmError::invalid("Expected struct type")),
                }
                let struct_ref =
                    ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx as u32)));
                self.context.push_val(struct_ref)?;
                Ok(())
            }
            STRUCT_GET | STRUCT_GET_S | STRUCT_GET_U => {
                let (typeidx, fieldidx) = match imm {
                    &Immediate::StructFieldArgs { typeidx, fieldidx } => (typeidx, fieldidx),
                    _ => return Err(WasmError::invalid("Invalid immediate for struct.get")),
                };
                let def_type = self
                    .module
                    .types()
                    .get(typeidx)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let struct_type = match &def_type.composite {
                    CompositeType::Struct(s) => s,
                    _ => return Err(WasmError::invalid("Expected struct type")),
                };
                if fieldidx as usize >= struct_type.fields.len() {
                    return Err(WasmError::invalid("Field index out of bounds"));
                }
                let structref = self.context.pop_val(None)?;
                let expected_struct_ref =
                    ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx)));
                if !structref.is_subtype_of(&expected_struct_ref, &self.context.types)
                    && structref != ValueType::Unknown
                {
                    return Err(WasmError::invalid("struct.get requires struct reference"));
                }
                let field_type = struct_type.fields[fieldidx as usize].storage.to_valtype();
                self.context.push_val(field_type)?;
                Ok(())
            }
            STRUCT_SET => {
                let (typeidx, fieldidx) = match imm {
                    &Immediate::StructFieldArgs { typeidx, fieldidx } => (typeidx, fieldidx),
                    _ => return Err(WasmError::invalid("Invalid immediate for struct.set")),
                };
                let def_type = self
                    .module
                    .types()
                    .get(typeidx)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let struct_type = match &def_type.composite {
                    CompositeType::Struct(s) => s,
                    _ => return Err(WasmError::invalid("Expected struct type")),
                };
                if fieldidx as usize >= struct_type.fields.len() {
                    return Err(WasmError::invalid("Field index out of bounds"));
                }
                if !struct_type.fields[fieldidx as usize].mutable {
                    return Err(WasmError::invalid("Cannot set immutable field"));
                }
                let field_type = struct_type.fields[fieldidx as usize].storage.to_valtype();
                let value = self.context.pop_val(None)?;
                let structref = self.context.pop_val(None)?;
                if !value.is_subtype_of(&field_type, &self.context.types) {
                    return Err(WasmError::invalid("struct.set value type mismatch"));
                }
                let expected_struct_ref =
                    ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx)));
                if !structref.is_subtype_of(&expected_struct_ref, &self.context.types)
                    && structref != ValueType::Unknown
                {
                    return Err(WasmError::invalid("struct.set requires struct reference"));
                }
                Ok(())
            }
            ARRAY_NEW => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let array_type = match &def_type.composite {
                    CompositeType::Array(a) => a,
                    _ => return Err(WasmError::invalid("Expected array type")),
                };
                self.context.pop_val(Some(I32))?;
                let elem_type = array_type.element.storage.to_valtype();
                self.context.pop_val(Some(elem_type))?;
                let array_ref =
                    ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx as u32)));
                self.context.push_val(array_ref)?;
                Ok(())
            }
            ARRAY_NEW_DEFAULT => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                match &def_type.composite {
                    CompositeType::Array(_) => {}
                    _ => return Err(WasmError::invalid("Expected array type")),
                }
                self.context.pop_val(Some(I32))?;
                let array_ref =
                    ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx as u32)));
                self.context.push_val(array_ref)?;
                Ok(())
            }
            ARRAY_NEW_FIXED => {
                let (typeidx, n) = match imm {
                    &Immediate::ArrayNewFixed { typeidx, n } => (typeidx, n),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.new_fixed")),
                };
                let def_type = self
                    .module
                    .types()
                    .get(typeidx)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let array_type = match &def_type.composite {
                    CompositeType::Array(a) => a,
                    _ => return Err(WasmError::invalid("Expected array type")),
                };
                let elem_type = array_type.element.storage.to_valtype();
                for _ in 0..n {
                    self.context.pop_val(Some(elem_type))?;
                }
                let array_ref = ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                self.context.push_val(array_ref)?;
                Ok(())
            }
            ARRAY_GET => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let array_type = match &def_type.composite {
                    CompositeType::Array(a) => a,
                    _ => return Err(WasmError::invalid("Expected array type")),
                };
                self.context.pop_val(Some(I32))?;
                let array_ref =
                    ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx as u32)));
                self.context.pop_val(Some(array_ref))?;
                let elem_type = match array_type.element.storage {
                    StorageType::Packed(_) => {
                        return Err(WasmError::invalid(
                            "array.get requires an unpacked array element type",
                        ));
                    }
                    _ => array_type.element.storage.to_valtype(),
                };
                self.context.push_val(elem_type)?;
                Ok(())
            }
            ARRAY_GET_S | ARRAY_GET_U => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let array_type = match &def_type.composite {
                    CompositeType::Array(a) => a,
                    _ => return Err(WasmError::invalid("Expected array type")),
                };
                self.context.pop_val(Some(I32))?;
                let array_ref =
                    ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx as u32)));
                self.context.pop_val(Some(array_ref))?;
                let elem_type = match array_type.element.storage {
                    StorageType::Val(_) => {
                        return Err(WasmError::invalid(
                            "array.get_s/u requires a packed array element type",
                        ));
                    }
                    _ => array_type.element.storage.to_valtype(),
                };
                self.context.push_val(elem_type)?;
                Ok(())
            }
            ARRAY_SET => {
                let typeidx = expect_type_index_immediate(imm)? as usize;
                let def_type = self
                    .module
                    .types()
                    .get(typeidx as u32)
                    .ok_or_else(|| WasmError::invalid("Type index out of bounds"))?;
                let array_type = match &def_type.composite {
                    CompositeType::Array(a) => a,
                    _ => return Err(WasmError::invalid("Expected array type")),
                };
                if !array_type.element.mutable {
                    return Err(WasmError::invalid("Cannot set immutable array element"));
                }
                let elem_type = array_type.element.storage.to_valtype();
                let value = self.context.pop_val(None)?;
                let index = self.context.pop_val(None)?;
                let arrayref = self.context.pop_val(None)?;
                if !value.is_subtype_of(&elem_type, &self.context.types) {
                    return Err(WasmError::invalid("array.set value type mismatch"));
                }
                if index != I32 && index != ValueType::Unknown {
                    return Err(WasmError::invalid("array.set index must be i32"));
                }
                if !arrayref.is_ref() && arrayref != ValueType::Unknown {
                    return Err(WasmError::invalid("array.set requires arrayref"));
                }
                Ok(())
            }
            ARRAY_LEN => {
                let array_ref = ValueType::Ref(RefType::new(
                    true,
                    HeapType::Abstract(AbstractHeapType::Array),
                ));
                self.context.pop_val(Some(array_ref))?;
                self.context.push_val(I32)?;
                Ok(())
            }
            REF_I31 => {
                self.context.pop_val(Some(I32))?;
                let i31_ref = ValueType::Ref(RefType::new(
                    false,
                    HeapType::Abstract(AbstractHeapType::I31),
                ));
                self.context.push_val(i31_ref)?;
                Ok(())
            }
            I31_GET_S | I31_GET_U => {
                let i31_ref = ValueType::Ref(RefType::new(
                    true,
                    HeapType::Abstract(AbstractHeapType::I31),
                ));
                self.context.pop_val(Some(i31_ref))?;
                self.context.push_val(I32)?;
                Ok(())
            }
            ARRAY_FILL => {
                let typeidx = expect_type_index_immediate(imm)?;
                let array_type = self.get_array_type(typeidx)?;
                if !array_type.element.mutable {
                    return Err(WasmError::invalid("Cannot fill immutable array"));
                }
                let elem_type = array_type.element.storage.to_valtype();
                self.context.pop_val(Some(elem_type))?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let array_ref = ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx)));
                self.context.pop_val(Some(array_ref))?;
                Ok(())
            }
            ARRAY_COPY => {
                let (dst_typeidx, src_typeidx) = match imm {
                    &Immediate::TwoTypeIndices { type1, type2 } => (type1, type2),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.copy")),
                };
                let dst_array = self.get_array_type(dst_typeidx)?;
                let src_array = self.get_array_type(src_typeidx)?;
                if !dst_array.element.mutable {
                    return Err(WasmError::invalid("Cannot copy into immutable array"));
                }
                if !self.storage_matches_for_array_copy(
                    src_array.element.storage,
                    dst_array.element.storage,
                ) {
                    return Err(WasmError::invalid("array.copy type mismatch"));
                }
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let src_ref = ValueType::Ref(RefType::new(true, HeapType::Concrete(src_typeidx)));
                self.context.pop_val(Some(src_ref))?;
                self.context.pop_val(Some(I32))?;
                let dst_ref = ValueType::Ref(RefType::new(true, HeapType::Concrete(dst_typeidx)));
                self.context.pop_val(Some(dst_ref))?;
                Ok(())
            }
            ARRAY_INIT_DATA => {
                let (typeidx, dataidx) = match imm {
                    &Immediate::TwoU32s { value1, value2 } => (value1, value2),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.init_data")),
                };
                let array_type = self.get_array_type(typeidx)?;
                if dataidx as usize >= self.module.data().len() {
                    return Err(WasmError::invalid("Data index out of bounds"));
                }
                if !array_type.element.mutable {
                    return Err(WasmError::invalid("Cannot initialize immutable array"));
                }
                if !Self::storage_is_data_segment_compatible(array_type.element.storage) {
                    return Err(WasmError::invalid(
                        "array.init_data requires numeric array element type",
                    ));
                }
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let array_ref = ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx)));
                self.context.pop_val(Some(array_ref))?;
                Ok(())
            }
            ARRAY_INIT_ELEM => {
                let (typeidx, elemidx) = match imm {
                    &Immediate::TwoU32s { value1, value2 } => (value1, value2),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.init_elem")),
                };
                let array_type = self.get_array_type(typeidx)?;
                if elemidx as usize >= self.module.elements().len() {
                    return Err(WasmError::invalid("Element index out of bounds"));
                }
                if !array_type.element.mutable {
                    return Err(WasmError::invalid("Cannot initialize immutable array"));
                }
                let Some(elem_ref_type) = Self::storage_ref_type(array_type.element.storage) else {
                    return Err(WasmError::invalid(
                        "array.init_elem requires reference array element type",
                    ));
                };
                let elem_type = self.module.elements()[elemidx as usize].value_type();
                if !elem_type.is_subtype_of(&ValueType::Ref(elem_ref_type), self.module.types()) {
                    return Err(WasmError::invalid("array.init_elem type mismatch"));
                }
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let array_ref = ValueType::Ref(RefType::new(true, HeapType::Concrete(typeidx)));
                self.context.pop_val(Some(array_ref))?;
                Ok(())
            }
            ARRAY_NEW_DATA => {
                let (typeidx, dataidx) = match imm {
                    &Immediate::TwoU32s { value1, value2 } => (value1, value2),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.new_data")),
                };
                let array_type = self.get_array_type(typeidx)?;
                if dataidx as usize >= self.module.data().len() {
                    return Err(WasmError::invalid("Data index out of bounds"));
                }
                if !Self::storage_is_data_segment_compatible(array_type.element.storage) {
                    return Err(WasmError::invalid(
                        "array.new_data requires numeric array element type",
                    ));
                }
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let array_ref = ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                self.context.push_val(array_ref)?;
                Ok(())
            }
            ARRAY_NEW_ELEM => {
                let (typeidx, elemidx) = match imm {
                    &Immediate::TwoU32s { value1, value2 } => (value1, value2),
                    _ => return Err(WasmError::invalid("Invalid immediate for array.new_elem")),
                };
                let array_type = self.get_array_type(typeidx)?;
                if elemidx as usize >= self.module.elements().len() {
                    return Err(WasmError::invalid("Element index out of bounds"));
                }
                let Some(elem_ref_type) = Self::storage_ref_type(array_type.element.storage) else {
                    return Err(WasmError::invalid(
                        "array.new_elem requires reference array element type",
                    ));
                };
                let elem_type = self.module.elements()[elemidx as usize].value_type();
                if !elem_type.is_subtype_of(&ValueType::Ref(elem_ref_type), self.module.types()) {
                    return Err(WasmError::invalid("array.new_elem type mismatch"));
                }
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(I32))?;
                let array_ref = ValueType::Ref(RefType::new(false, HeapType::Concrete(typeidx)));
                self.context.push_val(array_ref)?;
                Ok(())
            }
            REF_TEST | REF_TEST_NULL => {
                let _ref_type = expect_ref_type_immediate(imm)?;
                let _popped = self.context.pop_ref_type()?;
                self.context.push_val(I32)?;
                Ok(())
            }
            REF_CAST | REF_CAST_NULL => {
                let ref_type = expect_ref_type_immediate(imm)?;
                let _popped = self.context.pop_ref_type()?;
                self.context.push_val(ref_type)?;
                Ok(())
            }
            BR_ON_CAST => {
                let actual_ref = self.context.pop_ref_type()?;
                let (label_idx, rt1, rt2) = expect_br_on_cast_immediate(imm)?;
                if !rt2.is_subtype_of(&rt1, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                if !actual_ref.is_subtype_of(&rt1, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                let diff_type = match (rt1, rt2) {
                    (ValueType::Ref(from_ref), ValueType::Ref(to_ref)) => {
                        ValueType::Ref(RefType::difference(from_ref, to_ref))
                    }
                    _ => return Err(WasmError::invalid("type mismatch")),
                };
                let label_types = self.context.frame_at(label_idx)?.label_types();
                let (branch_ref, prefix) = label_types
                    .split_last()
                    .ok_or_else(|| WasmError::invalid("br_on_cast requires label result"))?;
                if !rt2.is_subtype_of(branch_ref, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.pop_vals(prefix)?;
                self.context.push_vals(prefix)?;
                self.context.push_val(diff_type)?;
                Ok(())
            }
            BR_ON_CAST_FAIL => {
                let actual_ref = self.context.pop_ref_type()?;
                let (label_idx, rt1, rt2) = expect_br_on_cast_immediate(imm)?;
                if !rt2.is_subtype_of(&rt1, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                if !actual_ref.is_subtype_of(&rt1, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                let diff_type = match (rt1, rt2) {
                    (ValueType::Ref(from_ref), ValueType::Ref(to_ref)) => {
                        ValueType::Ref(RefType::difference(from_ref, to_ref))
                    }
                    _ => return Err(WasmError::invalid("type mismatch")),
                };
                let label_types = self.context.frame_at(label_idx)?.label_types();
                let (branch_ref, prefix) = label_types
                    .split_last()
                    .ok_or_else(|| WasmError::invalid("br_on_cast_fail requires label result"))?;
                if !diff_type.is_subtype_of(branch_ref, self.module.types()) {
                    return Err(WasmError::invalid("type mismatch"));
                }
                self.context.pop_vals(prefix)?;
                self.context.push_vals(prefix)?;
                self.context.push_val(rt2)?;
                Ok(())
            }
            ANY_CONVERT_EXTERN => {
                let _popped = self.context.pop_ref_type()?;
                let anyref = ValueType::Ref(RefType::new(
                    true,
                    HeapType::Abstract(AbstractHeapType::Any),
                ));
                self.context.push_val(anyref)?;
                Ok(())
            }
            EXTERN_CONVERT_ANY => {
                let _popped = self.context.pop_ref_type()?;
                let externref = ValueType::Ref(RefType::new(
                    true,
                    HeapType::Abstract(AbstractHeapType::Extern),
                ));
                self.context.push_val(externref)?;
                Ok(())
            }
        }
    }

    #[cfg(not(sf_has_simd))]
    fn on_op_fd(&mut self) -> Result<(), WasmError> {
        Err(simd_opcode_error())
    }

    #[cfg(sf_has_simd)]
    fn on_op_fd(&mut self, op: OpcodeFD, imm: &Immediate) -> Result<(), WasmError> {
        use OpcodeFD::*;
        use ValueType::*;

        match op {
            V128_CONST => match imm {
                Immediate::V128(_) => self.context.push_val(V128),
                _ => Err(WasmError::internal("validator expected v128 immediate")),
            },
            V128_LOAD => self.handle_load::<[u8; 16]>(imm, V128),
            op if matches!(
                op,
                V128_LOAD8X8_S
                    | V128_LOAD8X8_U
                    | V128_LOAD16X4_S
                    | V128_LOAD16X4_U
                    | V128_LOAD32X2_S
                    | V128_LOAD32X2_U
            ) =>
            {
                self.handle_load::<u64>(imm, V128)
            }
            V128_LOAD8_SPLAT => self.handle_load::<u8>(imm, V128),
            V128_LOAD16_SPLAT => self.handle_load::<u16>(imm, V128),
            V128_LOAD32_SPLAT | V128_LOAD32_ZERO => self.handle_load::<u32>(imm, V128),
            V128_LOAD64_SPLAT | V128_LOAD64_ZERO => self.handle_load::<u64>(imm, V128),
            V128_STORE => self.handle_store::<[u8; 16]>(imm, V128),
            V128_LOAD8_LANE => self.handle_simd_mem_lane(imm, 1, 16, false),
            V128_LOAD16_LANE => self.handle_simd_mem_lane(imm, 2, 8, false),
            V128_LOAD32_LANE => self.handle_simd_mem_lane(imm, 4, 4, false),
            V128_LOAD64_LANE => self.handle_simd_mem_lane(imm, 8, 2, false),
            V128_STORE8_LANE => self.handle_simd_mem_lane(imm, 1, 16, true),
            V128_STORE16_LANE => self.handle_simd_mem_lane(imm, 2, 8, true),
            V128_STORE32_LANE => self.handle_simd_mem_lane(imm, 4, 4, true),
            V128_STORE64_LANE => self.handle_simd_mem_lane(imm, 8, 2, true),

            V128_ANY_TRUE | I8X16_ALL_TRUE | I16X8_ALL_TRUE | I32X4_ALL_TRUE | I64X2_ALL_TRUE
            | I8X16_BITMASK | I16X8_BITMASK | I32X4_BITMASK | I64X2_BITMASK => {
                self.context.pop_val(Some(V128))?;
                self.context.push_val(I32)
            }

            I8X16_SPLAT => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(V128)
            }
            I16X8_SPLAT => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(V128)
            }
            I32X4_SPLAT => {
                self.context.pop_val(Some(I32))?;
                self.context.push_val(V128)
            }
            I64X2_SPLAT => {
                self.context.pop_val(Some(I64))?;
                self.context.push_val(V128)
            }
            F32X4_SPLAT => {
                self.context.pop_val(Some(F32))?;
                self.context.push_val(V128)
            }
            F64X2_SPLAT => {
                self.context.pop_val(Some(F64))?;
                self.context.push_val(V128)
            }

            I8X16_EXTRACT_LANE_S | I8X16_EXTRACT_LANE_U => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 16)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(I32)
            }
            I16X8_EXTRACT_LANE_S | I16X8_EXTRACT_LANE_U => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 8)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(I32)
            }
            I32X4_EXTRACT_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 4)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(I32)
            }
            I64X2_EXTRACT_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 2)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(I64)
            }
            F32X4_EXTRACT_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 4)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(F32)
            }
            F64X2_EXTRACT_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 2)?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(F64)
            }

            I8X16_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 16)?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            I16X8_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 8)?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            I32X4_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 4)?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            I64X2_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 2)?;
                self.context.pop_val(Some(I64))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            F32X4_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 4)?;
                self.context.pop_val(Some(F32))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            F64X2_REPLACE_LANE => {
                let lane = expect_simd_lane_immediate(imm)?;
                self.validate_simd_lane(lane, 2)?;
                self.context.pop_val(Some(F64))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }

            I8X16_SHUFFLE => {
                let lanes = expect_simd_shuffle_immediate(imm)?;
                if lanes.iter().any(|lane| *lane >= 32) {
                    return Err(WasmError::invalid("SIMD shuffle lane index out of range"));
                }
                self.context.pop_val(Some(V128))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }

            op if matches!(
                op,
                V128_NOT
                    | I8X16_ABS
                    | I8X16_NEG
                    | I8X16_POPCNT
                    | I16X8_EXTADD_PAIRWISE_I8X16_S
                    | I16X8_EXTADD_PAIRWISE_I8X16_U
                    | I16X8_ABS
                    | I16X8_NEG
                    | I16X8_EXTEND_LOW_I8X16_S
                    | I16X8_EXTEND_HIGH_I8X16_S
                    | I16X8_EXTEND_LOW_I8X16_U
                    | I16X8_EXTEND_HIGH_I8X16_U
                    | I32X4_EXTADD_PAIRWISE_I16X8_S
                    | I32X4_EXTADD_PAIRWISE_I16X8_U
                    | I32X4_ABS
                    | I32X4_NEG
                    | I32X4_EXTEND_LOW_I16X8_S
                    | I32X4_EXTEND_HIGH_I16X8_S
                    | I32X4_EXTEND_LOW_I16X8_U
                    | I32X4_EXTEND_HIGH_I16X8_U
                    | I64X2_ABS
                    | I64X2_NEG
                    | I64X2_EXTEND_LOW_I32X4_S
                    | I64X2_EXTEND_HIGH_I32X4_S
                    | I64X2_EXTEND_LOW_I32X4_U
                    | I64X2_EXTEND_HIGH_I32X4_U
                    | F32X4_NEG
                    | F32X4_DEMOTE_F64X2_ZERO
                    | F32X4_SQRT
                    | F32X4_CEIL
                    | F32X4_FLOOR
                    | F32X4_TRUNC
                    | F32X4_NEAREST
                    | F64X2_ABS
                    | F64X2_NEG
                    | F64X2_SQRT
                    | F64X2_CEIL
                    | F64X2_FLOOR
                    | F64X2_TRUNC
                    | F64X2_NEAREST
                    | F32X4_ABS
                    | F32X4_CONVERT_I32X4_S
                    | F32X4_CONVERT_I32X4_U
                    | F64X2_PROMOTE_LOW_F32X4
                    | F64X2_CONVERT_LOW_I32X4_S
                    | F64X2_CONVERT_LOW_I32X4_U
                    | I32X4_TRUNC_SAT_F32X4_S
                    | I32X4_TRUNC_SAT_F32X4_U
                    | I32X4_TRUNC_SAT_F64X2_S_ZERO
                    | I32X4_TRUNC_SAT_F64X2_U_ZERO
                    | I32X4_RELAXED_TRUNC_F32X4_S
                    | I32X4_RELAXED_TRUNC_F32X4_U
                    | I32X4_RELAXED_TRUNC_F64X2_S_ZERO
                    | I32X4_RELAXED_TRUNC_F64X2_U_ZERO
            ) =>
            {
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }

            op if matches!(
                op,
                V128_AND
                    | V128_ANDNOT
                    | V128_OR
                    | V128_XOR
                    | I8X16_SWIZZLE
                    | I8X16_RELAXED_SWIZZLE
                    | I8X16_NARROW_I16X8_S
                    | I8X16_NARROW_I16X8_U
                    | I8X16_MIN_S
                    | I8X16_MIN_U
                    | I8X16_MAX_S
                    | I8X16_MAX_U
                    | I8X16_AVGR_U
                    | I8X16_ADD
                    | I8X16_SUB
                    | I16X8_MIN_S
                    | I16X8_MIN_U
                    | I16X8_MAX_S
                    | I16X8_MAX_U
                    | I16X8_AVGR_U
                    | I16X8_Q15MULR_SAT_S
                    | I16X8_RELAXED_Q15MULR_S
                    | I16X8_NARROW_I32X4_S
                    | I16X8_NARROW_I32X4_U
                    | I16X8_ADD
                    | I16X8_SUB
                    | I16X8_MUL
                    | I16X8_RELAXED_DOT_I8X16_I7X16_S
                    | I16X8_EXTMUL_LOW_I8X16_S
                    | I16X8_EXTMUL_HIGH_I8X16_S
                    | I16X8_EXTMUL_LOW_I8X16_U
                    | I16X8_EXTMUL_HIGH_I8X16_U
                    | I32X4_MIN_S
                    | I32X4_MIN_U
                    | I32X4_MAX_S
                    | I32X4_MAX_U
                    | I32X4_ADD
                    | I32X4_SUB
                    | I32X4_MUL
                    | I32X4_DOT_I16X8_S
                    | I32X4_EXTMUL_LOW_I16X8_S
                    | I32X4_EXTMUL_HIGH_I16X8_S
                    | I32X4_EXTMUL_LOW_I16X8_U
                    | I32X4_EXTMUL_HIGH_I16X8_U
                    | I64X2_ADD
                    | I64X2_SUB
                    | I64X2_MUL
                    | I64X2_EXTMUL_LOW_I32X4_S
                    | I64X2_EXTMUL_HIGH_I32X4_S
                    | I64X2_EXTMUL_LOW_I32X4_U
                    | I64X2_EXTMUL_HIGH_I32X4_U
                    | F64X2_ADD
                    | F64X2_SUB
                    | F64X2_MUL
                    | I8X16_ADD_SAT_S
                    | I8X16_ADD_SAT_U
                    | I16X8_ADD_SAT_S
                    | I16X8_ADD_SAT_U
                    | I8X16_SUB_SAT_S
                    | I8X16_SUB_SAT_U
                    | I16X8_SUB_SAT_S
                    | I16X8_SUB_SAT_U
                    | I8X16_EQ
                    | I8X16_NE
                    | I8X16_LT_S
                    | I8X16_LT_U
                    | I8X16_GT_S
                    | I8X16_GT_U
                    | I8X16_LE_S
                    | I8X16_LE_U
                    | I8X16_GE_S
                    | I8X16_GE_U
                    | I16X8_EQ
                    | I16X8_NE
                    | I16X8_LT_S
                    | I16X8_LT_U
                    | I16X8_GT_S
                    | I16X8_GT_U
                    | I16X8_LE_S
                    | I16X8_LE_U
                    | I16X8_GE_S
                    | I16X8_GE_U
                    | I32X4_EQ
                    | I32X4_NE
                    | I32X4_LT_S
                    | I32X4_LT_U
                    | I32X4_GT_S
                    | I32X4_GT_U
                    | I32X4_LE_S
                    | I32X4_LE_U
                    | I32X4_GE_S
                    | I32X4_GE_U
                    | F32X4_EQ
                    | F32X4_NE
                    | F32X4_LT
                    | F32X4_GT
                    | F32X4_LE
                    | F32X4_GE
                    | F32X4_ADD
                    | F32X4_SUB
                    | F32X4_MUL
                    | F32X4_RELAXED_MIN
                    | F32X4_RELAXED_MAX
                    | F32X4_MAX
                    | F32X4_PMIN
                    | F32X4_PMAX
                    | I64X2_EQ
                    | I64X2_NE
                    | I64X2_LT_S
                    | I64X2_GT_S
                    | I64X2_LE_S
                    | I64X2_GE_S
                    | F64X2_EQ
                    | F64X2_NE
                    | F64X2_LT
                    | F64X2_GT
                    | F64X2_LE
                    | F64X2_GE
                    | F32X4_MIN
                    | F64X2_PMIN
                    | F64X2_PMAX
                    | F64X2_RELAXED_MIN
                    | F64X2_RELAXED_MAX
                    | F64X2_MIN
                    | F64X2_MAX
                    | F32X4_DIV
                    | F64X2_DIV
            ) =>
            {
                self.context.pop_val(Some(V128))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }

            op if matches!(
                op,
                V128_BITSELECT
                    | I8X16_RELAXED_LANESELECT
                    | I16X8_RELAXED_LANESELECT
                    | I32X4_RELAXED_LANESELECT
                    | I64X2_RELAXED_LANESELECT
                    | F32X4_RELAXED_MADD
                    | F32X4_RELAXED_NMADD
                    | F64X2_RELAXED_MADD
                    | F64X2_RELAXED_NMADD
                    | I32X4_RELAXED_DOT_I8X16_I7X16_ADD_S
            ) =>
            {
                self.context.pop_val(Some(V128))?;
                self.context.pop_val(Some(V128))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }

            op if matches!(
                op,
                I8X16_SHL
                    | I8X16_SHR_S
                    | I8X16_SHR_U
                    | I16X8_SHL
                    | I16X8_SHR_S
                    | I16X8_SHR_U
                    | I32X4_SHL
                    | I32X4_SHR_S
                    | I32X4_SHR_U
                    | I64X2_SHL
                    | I64X2_SHR_S
                    | I64X2_SHR_U
            ) =>
            {
                self.context.pop_val(Some(I32))?;
                self.context.pop_val(Some(V128))?;
                self.context.push_val(V128)
            }
            _ => Err(WasmError::invalid("SIMD support is not yet implemented")),
        }
    }
}

// ============================================================================
// Control Flow Context (no jump table, no max_stack_height)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameType {
    Function,
    Block,
    Loop,
    If,
    Else,
    TryTable,
}

struct ControlFrame {
    frame_type: FrameType,
    function_type: Rc<FunctionType>,
    height: usize,
    unreachable: bool,
    inits_height: usize,
}

impl ControlFrame {
    fn new(
        frame_type: FrameType,
        function_type: Rc<FunctionType>,
        height: usize,
        unreachable: bool,
        inits_height: usize,
    ) -> Self {
        ControlFrame {
            frame_type,
            function_type,
            height,
            unreachable,
            inits_height,
        }
    }

    fn frame_type(&self) -> FrameType {
        self.frame_type
    }

    fn function_type(&self) -> Rc<FunctionType> {
        self.function_type.clone()
    }

    fn is_unreachable(&self) -> bool {
        self.unreachable
    }

    fn height(&self) -> usize {
        self.height
    }

    fn label_types(&self) -> collections::Vec<ValueType> {
        if self.frame_type == FrameType::Loop {
            self.function_type.params().iter().copied().collect()
        } else {
            self.function_type.results().iter().copied().collect()
        }
    }
}

struct Context {
    types: TypeContext,
    control_frames: collections::Vec<ControlFrame>,
    all_locals: collections::Vec<ValueType>,
    val_stack: collections::Vec<ValueType>,
    locals_init: collections::Vec<bool>,
    inits: collections::Vec<u32>,
}

impl Context {
    fn new(types: TypeContext, params: &[ValueType], locals: &[ValueType]) -> Self {
        let mut all_locals = collections::Vec::new();
        all_locals.extend_from_slice(params);
        all_locals.extend_from_slice(locals);

        let num_locals = all_locals.len();
        let num_params = params.len();

        let mut locals_init = collections::vec![false; num_locals];
        for slot in locals_init.iter_mut().take(num_params) {
            *slot = true;
        }
        for (slot, local) in locals_init
            .iter_mut()
            .zip(all_locals.iter())
            .skip(num_params)
        {
            *slot = local.is_defaultable();
        }

        Context {
            types,
            control_frames: collections::Vec::new(),
            all_locals,
            val_stack: collections::Vec::new(),
            locals_init,
            inits: collections::Vec::new(),
        }
    }

    fn push_vals(&mut self, vals: &[ValueType]) -> Result<(), WasmError> {
        vals.iter().try_for_each(|v| self.push_val(*v))
    }

    fn push_val(&mut self, val: ValueType) -> Result<(), WasmError> {
        self.val_stack.push(val);
        Ok(())
    }

    fn pop_val(&mut self, expected: Option<ValueType>) -> Result<ValueType, WasmError> {
        use ValueType::*;
        if self.control_frames.is_empty() {
            return Err(WasmError::invalid(
                "popping value while control frame stack is empty".into(),
            ));
        }
        let current_frame = self.control_frames.last().unwrap();
        if current_frame.is_unreachable() && current_frame.height() == self.val_stack.len() {
            return Ok(Unknown);
        }
        if current_frame.height() >= self.val_stack.len() {
            return Err(WasmError::invalid("stack underflow"));
        }
        if self.val_stack.is_empty() {
            return Err(WasmError::invalid("cannot pop from an empty stack"));
        }
        let actual = self.val_stack.pop().unwrap();

        if let Some(expected_type) = expected {
            if !actual.is_subtype_of(&expected_type, &self.types) {
                return Err(WasmError::invalid("type mismatch: expected , got"));
            }
        }

        Ok(actual)
    }

    fn pop_vals(
        &mut self,
        expected_vals: &[ValueType],
    ) -> Result<collections::Vec<ValueType>, WasmError> {
        let mut popped_vals = collections::Vec::new();
        for &expected in expected_vals.iter().rev() {
            let popped = self.pop_val(Some(expected))?;
            popped_vals.push(popped);
        }
        popped_vals.reverse();
        Ok(popped_vals)
    }

    fn pop_ref_type(&mut self) -> Result<ValueType, WasmError> {
        let val = self.pop_val(None)?;
        if !val.is_ref() && val != ValueType::Unknown {
            return Err(WasmError::invalid("expected reference type, got"));
        }
        Ok(val)
    }

    fn push_ctrl(
        &mut self,
        frame_type: FrameType,
        function_type: Rc<FunctionType>,
    ) -> Result<(), WasmError> {
        let inits_height = self.inits.len();
        self.control_frames.push(ControlFrame::new(
            frame_type,
            function_type.clone(),
            self.val_stack.len(),
            false,
            inits_height,
        ));
        if !matches!(frame_type, FrameType::Function) {
            self.push_vals(function_type.params())?;
        }
        Ok(())
    }

    fn pop_ctrl(&mut self) -> Result<ControlFrame, WasmError> {
        if self.control_frames.is_empty() {
            return Err(WasmError::invalid(
                "cannot pop from an empty control frame stack".into(),
            ));
        }
        let function_type = self.control_frames.last().unwrap().function_type();
        self.pop_vals(function_type.results())?;
        let frame = self.control_frames.pop().unwrap();
        if frame.height() != self.val_stack.len() {
            return Err(WasmError::invalid("invalid stack height"));
        }
        self.reset_locals(frame.inits_height);
        Ok(frame)
    }

    fn mark_unreachable(&mut self) -> Result<(), WasmError> {
        if self.control_frames.is_empty() {
            return Err(WasmError::invalid("control frame stack is empty"));
        }
        let current_frame = self.control_frames.last_mut().unwrap();
        if self.val_stack.len() < current_frame.height() {
            return Err(WasmError::invalid("invalid stack height"));
        }
        self.val_stack.truncate(current_frame.height());
        current_frame.unreachable = true;
        Ok(())
    }

    fn frame_at(&self, label_index: u32) -> Result<&ControlFrame, WasmError> {
        let labels = self.control_frames.len();
        if label_index as usize >= labels {
            return Err(WasmError::invalid("invalid frame index"));
        }
        Ok(&self.control_frames[labels - label_index as usize - 1])
    }

    fn frame_last(&self) -> Result<&ControlFrame, WasmError> {
        if self.control_frames.is_empty() {
            return Err(WasmError::invalid("control frame stack is empty"));
        }
        Ok(&self.control_frames[0])
    }

    fn set_local_initialized(&mut self, local_index: usize) {
        if local_index < self.locals_init.len() && !self.locals_init[local_index] {
            self.locals_init[local_index] = true;
            self.inits.push(local_index as u32);
        }
    }

    fn reset_locals(&mut self, height: usize) {
        while self.inits.len() > height {
            if let Some(local_idx) = self.inits.pop() {
                if (local_idx as usize) < self.locals_init.len() {
                    self.locals_init[local_idx as usize] = false;
                }
            }
        }
    }
}
