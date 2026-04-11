use crate::collections;

use crate::{
    error::WasmError,
    extract_imm,
    module::{
        entities::{Element, ElementInit, FunctionSpec, FunctionType},
        Module,
    },
    op_decoder::{BlockType, Immediate, OpStream, OpcodeHandler},
    opcodes::{Opcode, OpcodeFC, WasmOpcode},
    utils::limits::Limitable,
    value_type::{HeapType, RefType, ValueType},
};
use tracked_alloc::rc::Rc;

pub(super) struct FunctionValidator<'a> {
    module: &'a Module,
    function: &'a FunctionSpec,
    context: Context,
}

impl<'a> OpcodeHandler for FunctionValidator<'a> {
    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            match decoded.wasm_op {
                WasmOpcode::OP(op) => self.on_op(
                    op,
                    decoded.op_offset,
                    decoded.next_op_offset,
                    decoded.imm.clone(),
                )?,
                WasmOpcode::FC(op) => self.on_op_fc(
                    op,
                    decoded.op_offset,
                    decoded.next_op_offset,
                    decoded.imm.clone(),
                )?,
                WasmOpcode::FD(_op) => {
                    // SIMD validation: accept all FD-prefixed ops
                    // (detailed SIMD validation is out of scope for nano)
                }
            }
        }
        Ok(())
    }

    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
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
            if !actual.is_compatible_with(expected) {
                return Err(WasmError::invalid(
                    "function return type mismatch, expected: , actual",
                ));
            }
        }

        Ok(())
    }
}

impl<'a> FunctionValidator<'a> {
    pub(super) fn new(module: &'a Module, function: &'a FunctionSpec) -> Result<Self, WasmError> {
        let mut context = Context::new(function.func_type().params(), function.locals());

        context.push_ctrl(FrameType::Function, function.func_type_rc())?;

        Ok(FunctionValidator {
            module,
            function,
            context,
        })
    }

    fn get_block_type(&self, block_type: BlockType) -> Result<Rc<FunctionType>, WasmError> {
        match block_type {
            BlockType::Empty => Ok(Rc::new(FunctionType::new(
                collections::Vec::new(),
                collections::Vec::new(),
            ))),
            BlockType::ValueType(value_type) => Ok(Rc::new(FunctionType::new(
                collections::Vec::new(),
                collections::vec![value_type],
            ))),
            BlockType::TypeIndex(type_index) => self
                .module
                .types()
                .get(type_index as u32)
                .cloned()
                .ok_or_else(|| WasmError::malformed("block type index out of range")),
        }
    }

    fn get_local_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let local_index = *extract_imm!(imm, Immediate::LocalIndex) as u32;
        let local_type = self
            .context
            .all_locals
            .get(local_index as usize)
            .ok_or_else(|| WasmError::invalid("local index out of range"))?;
        Ok(*local_type)
    }

    fn get_global_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let global_index = *extract_imm!(imm, Immediate::GlobalIndex) as usize;
        let global = self
            .module
            .globals()
            .get(global_index)
            .ok_or_else(|| WasmError::invalid("global index out of range"))?;
        Ok(global.value_type())
    }

    fn get_table_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let table_index = *extract_imm!(imm, Immediate::TableIndex) as usize;
        let table = self
            .module
            .tables()
            .get(table_index)
            .ok_or_else(|| WasmError::invalid("table index out of range"))?;
        Ok(table.value_type())
    }

    fn get_table_index_type(&self, imm: &Immediate) -> Result<ValueType, WasmError> {
        let table_index = *extract_imm!(imm, Immediate::TableIndex) as usize;
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

    fn handle_load<T: Sized>(
        &mut self,
        imm: Immediate,
        val_type: ValueType,
    ) -> Result<(), WasmError> {
        use ValueType::*;
        let Immediate::MemArg {
            align,
            memidx,
            offset,
        } = imm
        else {
            unreachable!()
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
        imm: Immediate,
        val_type: ValueType,
    ) -> Result<(), WasmError> {
        use ValueType::*;
        let Immediate::MemArg {
            align,
            memidx,
            offset,
        } = imm
        else {
            unreachable!()
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

    fn on_op(
        &mut self,
        op: Opcode,
        _op_offset: usize,
        _next_op_offset: usize,
        imm: Immediate,
    ) -> Result<(), WasmError> {
        use Opcode::*;
        use ValueType::*;
        match op {
            NOP | PREFIX_FC | PREFIX_FD => Ok(()),
            UNREACHABLE => self.context.mark_unreachable(),
            BLOCK => {
                let block_type = extract_imm!(imm, Immediate::Block);
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::Block, function_type)?;
                Ok(())
            }
            LOOP => {
                let block_type = extract_imm!(imm, Immediate::Block);
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::Loop, function_type)?;
                Ok(())
            }
            IF => {
                let block_type = extract_imm!(imm, Immediate::Block);
                let function_type = self.get_block_type(block_type)?;
                self.context.pop_val(Some(I32))?;
                self.context.pop_vals(function_type.params())?;
                self.context.push_ctrl(FrameType::If, function_type)?;
                Ok(())
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
                let label_index = extract_imm!(imm, Immediate::LabelIndex);
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
                let (mut labels, default) = extract_imm!(imm, Immediate::BrLabels, tuple);
                self.context.pop_val(Some(I32))?;
                let default_label_types = self.context.frame_at(default)?.label_types();
                let default_arity = default_label_types.len();
                labels.push(default);
                labels.iter().try_for_each(|&label| {
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
                let function_index = extract_imm!(imm, Immediate::FunctionIndex);
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
                let Immediate::CallIndirectArgs { typeidx, tableidx } = imm else {
                    unreachable!()
                };
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
                    .get(typeidx)
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
            RETURN_CALL | RETURN_CALL_INDIRECT => Err(WasmError::invalid("Opcode not implemented")),
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
                let select_types = extract_imm!(imm, Immediate::SelectTypes);
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
                let local_index = extract_imm!(imm, Immediate::LocalIndex);
                let local_type = self.get_local_type(&imm)?;
                if local_index as usize >= self.context.locals_init.len() {
                    return Err(WasmError::invalid("local index out of range"));
                }
                if !self.context.locals_init[local_index as usize] {
                    return Err(WasmError::invalid("uninitialized local"));
                }
                self.context.push_val(local_type)
            }
            LOCAL_SET => {
                let local_index = extract_imm!(imm, Immediate::LocalIndex);
                let local_type = self.get_local_type(&imm)?;
                self.context.pop_val(Some(local_type))?;
                self.context.set_local_initialized(local_index as usize);
                Ok(())
            }
            LOCAL_TEE => {
                let local_index = extract_imm!(imm, Immediate::LocalIndex);
                let local_type = self.get_local_type(&imm)?;
                self.context.pop_val(Some(local_type))?;
                self.context.set_local_initialized(local_index as usize);
                self.context.push_val(local_type)
            }
            GLOBAL_GET => {
                let global_type = self.get_global_type(&imm)?;
                self.context.push_val(global_type)
            }
            GLOBAL_SET => {
                let global_index = extract_imm!(imm, Immediate::GlobalIndex) as usize;
                let global = self
                    .module
                    .globals()
                    .get(global_index)
                    .ok_or_else(|| WasmError::invalid("global not found"))?;
                if !global.mutable() {
                    return Err(WasmError::invalid("Global is immutable"));
                }
                let global_type = self.get_global_type(&imm)?;
                self.context.pop_val(Some(global_type))?;
                Ok(())
            }
            TABLE_GET => {
                let table_type = self.get_table_type(&imm)?;
                let index_type = self.get_table_index_type(&imm)?;
                self.context.pop_val(Some(index_type))?;
                self.context.push_val(table_type)
            }
            TABLE_SET => {
                let table_type = self.get_table_type(&imm)?;
                let index_type = self.get_table_index_type(&imm)?;
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
                let memidx = extract_imm!(imm, Immediate::MemoryIndex) as usize;
                if memidx >= self.module.memories().len() {
                    return Err(WasmError::invalid("unknown memory"));
                }
                let mem = &self.module.memories()[memidx];
                let is_mem64 = mem.spec().limits().is64;
                let size_type = if is_mem64 { I64 } else { I32 };
                self.context.push_val(size_type)
            }
            MEMORY_GROW => {
                let memidx = extract_imm!(imm, Immediate::MemoryIndex) as usize;
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
                let ref_type = extract_imm!(imm, Immediate::RefType);
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
            REF_FUNC => {
                let function_index = extract_imm!(imm, Immediate::FunctionIndex);
                if function_index as usize >= self.module.functions().len() {
                    return Err(WasmError::invalid("function index out of range"));
                }

                // Check if the function is declared in any element section
                let mut is_declared = false;
                for element in self.module.elements() {
                    match element {
                        Element::Active { init, .. }
                        | Element::Passive { init, .. }
                        | Element::Declarative { init, .. } => match init {
                            ElementInit::FunctionIndexes(indices) => {
                                if indices.contains(&(function_index as usize)) {
                                    is_declared = true;
                                    break;
                                }
                            }
                            ElementInit::InitExprs { .. } => {
                                is_declared = true;
                                break;
                            }
                        },
                    }
                }

                if !is_declared {
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
        imm: Immediate,
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
                    Immediate::MemoryInitArgs { dataidx, memidx } => (dataidx, memidx),
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
                    Immediate::MemoryCopyArgs { dstidx, srcidx } => (dstidx, srcidx),
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
                    Immediate::MemoryIndex(memidx) => memidx,
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
                    Immediate::DataIndex(dataidx) => dataidx,
                    _ => return Err(WasmError::invalid("invalid data index")),
                };
                if dataidx as usize >= self.module.data().len() {
                    return Err(WasmError::invalid("invalid data index"));
                }
                Ok(())
            }
            TABLE_INIT => {
                let (elemidx, tableidx) = match imm {
                    Immediate::TableInitArgs { elemidx, tableidx } => (elemidx, tableidx),
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
                if !elem_type.is_compatible_with(&table_type) {
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
                    Immediate::ElementIndex(elemidx) => elemidx,
                    _ => return Err(WasmError::invalid("invalid element index")),
                };
                if elemidx as usize >= self.module.elements().len() {
                    return Err(WasmError::invalid("invalid element index"));
                }
                Ok(())
            }
            TABLE_COPY => {
                let (dstidx, srcidx) = match imm {
                    Immediate::TableCopyArgs { dstidx, srcidx } => (dstidx, srcidx),
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
                if src_type != dst_type && !src_type.is_compatible_with(&dst_type) {
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
                    Immediate::TableIndex(tableidx) => tableidx,
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
                    Immediate::TableIndex(tableidx) => tableidx,
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
                    Immediate::TableIndex(tableidx) => tableidx,
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
                if !ref_val.is_compatible_with(&table_type) {
                    return Err(WasmError::invalid(
                        "table fill type mismatch: expected , got",
                    ));
                }
                self.context.pop_val(Some(idx_type))?; // dest
                Ok(())
            }
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
    control_frames: collections::Vec<ControlFrame>,
    all_locals: collections::Vec<ValueType>,
    val_stack: collections::Vec<ValueType>,
    locals_init: collections::Vec<bool>,
    inits: collections::Vec<u32>,
}

impl Context {
    fn new(params: &[ValueType], locals: &[ValueType]) -> Self {
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
            if !actual.is_compatible_with(&expected_type) {
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
