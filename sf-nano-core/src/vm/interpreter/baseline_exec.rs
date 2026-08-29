//! Test-only raw-Wasm baseline executor prototype.
//!
//! Production still executes the folded interpreter exclusively. This module
//! answers a narrower architecture question: can the raw cursor and the eager
//! control artifact drive execution without allocating or rebuilding a native
//! instruction stream?

use super::baseline_artifact::{BaselineArtifact, BaselineFunction, BrTableRange, ControlTarget};
use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::Module;
use crate::op_decoder::raw_cursor::{RawDecodeError, RawImmediate, RawOpCursor};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::utils::limits::Limitable;
use crate::value_type::ValueType;
use crate::Value;

use super::InterpInstanceAccess;

const MAX_BASELINE_CALL_DEPTH: usize = 4096;
const MAX_BASELINE_ACTIVATIONS: usize = MAX_BASELINE_CALL_DEPTH + 1;
/// Match the hosted interpreter's default two-MiB Wasm stack budget.
const MAX_BASELINE_VALUE_SLOTS: usize = (2 * 1024 * 1024) / core::mem::size_of::<u64>();

#[derive(Debug)]
pub(super) enum BaselineExecError {
    Wasm(WasmError),
    Unsupported {
        opcode: Option<WasmOpcode>,
        pc: usize,
        feature: &'static str,
    },
}

impl core::fmt::Display for BaselineExecError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Wasm(error) => write!(formatter, "{error}"),
            Self::Unsupported {
                opcode,
                pc,
                feature,
            } => write!(
                formatter,
                "baseline MVP unsupported {feature} at byte {pc}: {opcode:?}"
            ),
        }
    }
}

impl From<WasmError> for BaselineExecError {
    fn from(error: WasmError) -> Self {
        Self::Wasm(error)
    }
}

impl From<RawDecodeError> for BaselineExecError {
    fn from(error: RawDecodeError) -> Self {
        match error {
            RawDecodeError::Decode(error) => Self::Wasm(error),
            RawDecodeError::Unsupported { opcode, offset } => Self::Unsupported {
                opcode: Some(opcode),
                pc: offset,
                feature: "raw decoder opcode",
            },
            RawDecodeError::InvalidPc { .. } => {
                Self::Wasm(WasmError::invalid("baseline MVP raw pc is out of bounds"))
            }
        }
    }
}

pub(super) struct BaselineDriver<'artifact, 'instance> {
    access: InterpInstanceAccess<'instance>,
    artifact: &'artifact BaselineArtifact,
}

impl<'artifact, 'instance> BaselineDriver<'artifact, 'instance> {
    pub(super) const fn new(
        access: InterpInstanceAccess<'instance>,
        artifact: &'artifact BaselineArtifact,
    ) -> Self {
        Self { access, artifact }
    }

    pub(super) fn invoke_export(
        self,
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        let function =
            self.access
                .with_instance(|instance| {
                    instance.module().functions().iter().position(|function| {
                        function.export_names().iter().any(|name| name == export)
                    })
                })?
                .ok_or_else(|| WasmError::invalid("baseline MVP export was not found"))?;
        let mut frame = BaselineFrame::new(self.access, self.artifact, function, args)?;
        frame.run()?;
        frame.results()
    }
}

#[derive(Clone, Copy, Debug)]
struct BaselineActivation {
    function_index: usize,
    pc: usize,
    stp: usize,
    /// Parameters followed by zero-initialized declared locals.
    locals_base: usize,
    /// First operand slot; everything below belongs to locals or a caller.
    operand_base: usize,
    /// Caller's staged-argument base, overwritten in place by results.
    return_base: usize,
}

#[derive(Clone, Copy)]
enum BaselineImmediate {
    None,
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    LocalIndex(u32),
    FunctionIndex(u32),
    GlobalIndex(u32),
    MemoryIndex(u32),
    MemArg { offset: u64, memidx: u32 },
    Other,
}

#[derive(Clone, Copy)]
struct BaselineDecoded {
    wasm_op: WasmOpcode,
    start: usize,
    end: usize,
    imm: BaselineImmediate,
}

pub(super) struct BaselineFrame<'artifact, 'instance> {
    access: InterpInstanceAccess<'instance>,
    artifact: &'artifact BaselineArtifact,
    root_function: usize,
    activations: Vec<BaselineActivation>,
    values: Vec<u64>,
    max_activations: usize,
    max_value_slots: usize,
}

impl<'artifact, 'instance> BaselineFrame<'artifact, 'instance> {
    pub(super) fn new(
        access: InterpInstanceAccess<'instance>,
        artifact: &'artifact BaselineArtifact,
        function_index: usize,
        args: &[Value],
    ) -> Result<Self, BaselineExecError> {
        let mut frame = Self {
            access,
            artifact,
            root_function: function_index,
            activations: Vec::with_capacity(1),
            values: Vec::new(),
            max_activations: MAX_BASELINE_ACTIVATIONS,
            max_value_slots: MAX_BASELINE_VALUE_SLOTS,
        };
        frame.restart(args)?;
        Ok(frame)
    }

    pub(super) fn run(&mut self) -> Result<(), BaselineExecError> {
        while !self.activations.is_empty() {
            self.step()?;
        }
        Ok(())
    }

    fn restart(&mut self, args: &[Value]) -> Result<(), BaselineExecError> {
        self.activations.clear();
        self.values.clear();
        self.enter_root(args)
    }

    fn invoke_again(
        &mut self,
        args: &[Value],
        results: &mut [Value],
    ) -> Result<(), BaselineExecError> {
        self.restart(args)?;
        self.run()?;
        self.copy_results_into(results)
    }

    fn results(&self) -> Result<Vec<Value>, BaselineExecError> {
        if !self.activations.is_empty() {
            return Err(WasmError::invalid("baseline MVP execution is not finished").into());
        }
        let result_count = self.root_result_count()?;
        if self.values.len() != result_count {
            return Err(WasmError::invalid("baseline MVP result stack shape mismatch").into());
        }
        let mut results = Vec::with_capacity(result_count);
        for (index, &raw) in self.values.iter().enumerate() {
            let value_type = self.root_result_type(index)?;
            results.push(Value::from_raw(raw, value_type));
        }
        Ok(results)
    }

    fn copy_results_into(&self, results: &mut [Value]) -> Result<(), BaselineExecError> {
        if !self.activations.is_empty() {
            return Err(WasmError::invalid("baseline MVP execution is not finished").into());
        }
        let result_count = self.root_result_count()?;
        if self.values.len() != result_count || results.len() != result_count {
            return Err(WasmError::invalid("baseline MVP result stack shape mismatch").into());
        }
        for (index, (output, &raw)) in results.iter_mut().zip(&self.values).enumerate() {
            let value_type = self.root_result_type(index)?;
            *output = Value::from_raw(raw, value_type);
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), BaselineExecError> {
        let (function_index, pc) = {
            let activation = self.current_activation()?;
            (activation.function_index, activation.pc)
        };
        let (raw, code_len) = self.access.with_instance(|instance| {
            let function = instance
                .module()
                .functions()
                .get(function_index)
                .ok_or_else(|| {
                    WasmError::invalid("baseline MVP function index is out of bounds")
                })?;
            let spec = function
                .spec()
                .ok_or_else(|| WasmError::invalid("baseline MVP activation targets an import"))?;
            let mut cursor = RawOpCursor::at(spec.code(), pc);
            let raw = cursor
                .next()?
                .ok_or_else(|| WasmError::invalid("baseline MVP reached code end without end"))?;
            Ok::<_, BaselineExecError>((
                BaselineDecoded {
                    wasm_op: raw.wasm_op,
                    start: raw.start,
                    end: raw.end,
                    imm: copy_immediate(raw.imm),
                },
                spec.code().len(),
            ))
        })??;
        self.current_activation_mut()?.pc = raw.end;
        if let WasmOpcode::FC(opcode) = raw.wasm_op {
            if self.exec_saturating_conversion(opcode)? {
                return Ok(());
            }
            return self.unsupported(Some(raw.wasm_op), raw.start, "prefixed opcode");
        }
        let WasmOpcode::OP(opcode) = raw.wasm_op else {
            return self.unsupported(Some(raw.wasm_op), raw.start, "prefixed opcode");
        };
        match opcode {
            Opcode::NOP | Opcode::BLOCK | Opcode::LOOP => {}
            Opcode::I32_CONST => {
                let BaselineImmediate::I32(value) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP i32.const mismatch").into());
                };
                self.push(value as u32 as u64)?;
            }
            Opcode::I64_CONST => {
                let BaselineImmediate::I64(value) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP i64.const mismatch").into());
                };
                self.push(value as u64)?;
            }
            Opcode::F32_CONST => {
                let BaselineImmediate::F32(bits) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP f32.const mismatch").into());
                };
                self.push(bits as u64)?;
            }
            Opcode::F64_CONST => {
                let BaselineImmediate::F64(bits) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP f64.const mismatch").into());
                };
                self.push(bits)?;
            }
            Opcode::LOCAL_GET => {
                let local = raw_local(raw.imm)?;
                let slot = self.local_slot(local)?;
                let value = *self
                    .values
                    .get(slot)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))?;
                self.push(value)?;
            }
            Opcode::LOCAL_SET => {
                let local = raw_local(raw.imm)?;
                let value = self.pop()?;
                let slot = self.local_slot(local)?;
                *self
                    .values
                    .get_mut(slot)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))? =
                    value;
            }
            Opcode::LOCAL_TEE => {
                let local = raw_local(raw.imm)?;
                let value = self.peek()?;
                let slot = self.local_slot(local)?;
                *self
                    .values
                    .get_mut(slot)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))? =
                    value;
            }
            Opcode::GLOBAL_GET => {
                let global = raw_global(raw.imm)?;
                self.require_i32_global(global, raw.wasm_op, raw.start)?;
                let value = self
                    .access
                    .with_instance(|instance| instance.global_get_for_frame(global))?;
                self.push(value)?;
            }
            Opcode::GLOBAL_SET => {
                let global = raw_global(raw.imm)?;
                self.require_i32_global(global, raw.wasm_op, raw.start)?;
                let value = self.pop()?;
                self.access
                    .with_instance_mut(|instance| instance.global_set_from_frame(global, value))?;
            }
            opcode @ (Opcode::I32_LOAD
            | Opcode::I64_LOAD
            | Opcode::F32_LOAD
            | Opcode::F64_LOAD
            | Opcode::I32_LOAD8_S
            | Opcode::I32_LOAD8_U
            | Opcode::I32_LOAD16_S
            | Opcode::I32_LOAD16_U
            | Opcode::I64_LOAD8_S
            | Opcode::I64_LOAD8_U
            | Opcode::I64_LOAD16_S
            | Opcode::I64_LOAD16_U
            | Opcode::I64_LOAD32_S
            | Opcode::I64_LOAD32_U) => self.exec_memory_load(opcode, raw.imm)?,
            opcode @ (Opcode::I32_STORE
            | Opcode::I64_STORE
            | Opcode::F32_STORE
            | Opcode::F64_STORE
            | Opcode::I32_STORE8
            | Opcode::I32_STORE16
            | Opcode::I64_STORE8
            | Opcode::I64_STORE16
            | Opcode::I64_STORE32) => self.exec_memory_store(opcode, raw.imm)?,
            Opcode::MEMORY_SIZE => {
                let memory = raw_memory(raw.imm)?;
                self.memory_is_64(memory)?;
                let pages = self
                    .access
                    .with_instance(|instance| instance.memory_size(memory))?;
                self.push(pages)?;
            }
            Opcode::MEMORY_GROW => {
                let memory = raw_memory(raw.imm)?;
                self.memory_is_64(memory)?;
                let delta = self.pop()?;
                let previous = self
                    .access
                    .with_instance_mut(|instance| instance.memory_grow(memory, delta))??;
                self.push(previous)?;
            }
            Opcode::DROP => {
                self.pop()?;
            }
            Opcode::SELECT => {
                let condition = self.pop()?;
                let otherwise = self.pop()?;
                let selected = self.pop()?;
                self.push(if condition != 0 { selected } else { otherwise })?;
            }
            Opcode::IF => {
                let condition = self.pop()?;
                let target = self.current_target()?;
                if condition == 0 {
                    self.apply_target(target)?;
                } else {
                    self.advance_stp()?;
                }
            }
            Opcode::ELSE | Opcode::BR | Opcode::RETURN => {
                let target = self.current_target()?;
                self.apply_target(target)?;
            }
            Opcode::BR_IF => {
                let condition = self.pop()?;
                let target = self.current_target()?;
                if condition != 0 {
                    self.apply_target(target)?;
                } else {
                    self.advance_stp()?;
                }
            }
            Opcode::BR_TABLE => {
                let selector = self.pop()? as u32 as usize;
                let table = self.current_br_table(raw.start)?;
                let target_count = table.targets_len as usize;
                if target_count == 0 {
                    return Err(WasmError::invalid("baseline MVP empty br_table metadata").into());
                }
                let target_offset = selector.min(target_count - 1);
                let target_index = table.targets_start as usize + target_offset;
                let stp = self.current_activation()?.stp;
                let relative_stp = u32::try_from(stp)
                    .map_err(|_| WasmError::invalid("baseline MVP side-table pointer overflow"))?;
                let function = self.current_baseline_function()?;
                if function.absolute_stp(relative_stp) != Some(table.targets_start as usize) {
                    return Err(WasmError::invalid("baseline MVP br_table pointer mismatch").into());
                }
                let target = self
                    .artifact
                    .control_targets
                    .get(target_index)
                    .copied()
                    .ok_or_else(|| WasmError::invalid("baseline MVP br_table target missing"))?;
                self.apply_target(target)?;
            }
            Opcode::CALL => {
                let callee = raw_function(raw.imm)?;
                self.enter_call(callee, raw.start)?;
            }
            Opcode::END => {
                if raw.end == code_len {
                    self.finish_activation()?;
                }
            }
            Opcode::UNREACHABLE => return Err(WasmError::trap("unreachable").into()),
            _ if self.exec_numeric(opcode)? => {}
            _ => return self.unsupported(Some(raw.wasm_op), raw.start, "MVP opcode"),
        }
        Ok(())
    }

    fn exec_memory_load(
        &mut self,
        opcode: Opcode,
        immediate: BaselineImmediate,
    ) -> Result<(), BaselineExecError> {
        let (memory, offset) = raw_memarg(immediate)?;
        let address_is_64 = self.memory_is_64(memory)?;
        let address = self.pop()?;
        let address = if address_is_64 {
            address
        } else {
            address as u32 as u64
        };
        let size = match opcode {
            Opcode::I32_LOAD | Opcode::F32_LOAD | Opcode::I64_LOAD32_S | Opcode::I64_LOAD32_U => 4,
            Opcode::I64_LOAD | Opcode::F64_LOAD => 8,
            Opcode::I32_LOAD8_S
            | Opcode::I32_LOAD8_U
            | Opcode::I64_LOAD8_S
            | Opcode::I64_LOAD8_U => 1,
            Opcode::I32_LOAD16_S
            | Opcode::I32_LOAD16_U
            | Opcode::I64_LOAD16_S
            | Opcode::I64_LOAD16_U => 2,
            _ => return Err(WasmError::internal("baseline MVP load opcode mismatch").into()),
        };
        let loaded = self
            .access
            .with_instance(|instance| instance.mem_load(address, memory, offset, size))??;
        let value = match opcode {
            Opcode::I32_LOAD | Opcode::F32_LOAD | Opcode::I32_LOAD8_U | Opcode::I32_LOAD16_U => {
                loaded
            }
            Opcode::I64_LOAD
            | Opcode::F64_LOAD
            | Opcode::I64_LOAD8_U
            | Opcode::I64_LOAD16_U
            | Opcode::I64_LOAD32_U => loaded,
            Opcode::I32_LOAD8_S => loaded as i8 as i32 as u32 as u64,
            Opcode::I32_LOAD16_S => loaded as i16 as i32 as u32 as u64,
            Opcode::I64_LOAD8_S => loaded as i8 as i64 as u64,
            Opcode::I64_LOAD16_S => loaded as i16 as i64 as u64,
            Opcode::I64_LOAD32_S => loaded as i32 as i64 as u64,
            _ => return Err(WasmError::internal("baseline MVP load opcode mismatch").into()),
        };
        self.push(value)
    }

    fn exec_memory_store(
        &mut self,
        opcode: Opcode,
        immediate: BaselineImmediate,
    ) -> Result<(), BaselineExecError> {
        let (memory, offset) = raw_memarg(immediate)?;
        let address_is_64 = self.memory_is_64(memory)?;
        let value = self.pop()?;
        let address = self.pop()?;
        let address = if address_is_64 {
            address
        } else {
            address as u32 as u64
        };
        let size = match opcode {
            Opcode::I32_STORE | Opcode::F32_STORE | Opcode::I64_STORE32 => 4,
            Opcode::I64_STORE | Opcode::F64_STORE => 8,
            Opcode::I32_STORE8 | Opcode::I64_STORE8 => 1,
            Opcode::I32_STORE16 | Opcode::I64_STORE16 => 2,
            _ => return Err(WasmError::internal("baseline MVP store opcode mismatch").into()),
        };
        self.access.with_instance_mut(|instance| {
            instance.mem_store(address, memory, offset, size, value)
        })??;
        Ok(())
    }

    fn memory_is_64(&self, memory: usize) -> Result<bool, BaselineExecError> {
        self.access
            .with_instance(|instance| {
                instance
                    .module()
                    .memories()
                    .get(memory)
                    .map(|memory| memory.limits().is64)
                    .ok_or_else(|| WasmError::invalid("baseline MVP memory index overflow"))
            })?
            .map_err(Into::into)
    }

    fn require_i32_global(
        &self,
        global: usize,
        opcode: WasmOpcode,
        pc: usize,
    ) -> Result<(), BaselineExecError> {
        let value_type = self.access.with_instance(|instance| {
            instance
                .module()
                .globals()
                .get(global)
                .map(|global| global.value_type())
        })?;
        match value_type {
            Some(ValueType::I32) => Ok(()),
            Some(_) => Err(BaselineExecError::Unsupported {
                opcode: Some(opcode),
                pc,
                feature: "non-i32 global",
            }),
            None => Err(WasmError::invalid("baseline MVP global index overflow").into()),
        }
    }

    fn exec_numeric(&mut self, opcode: Opcode) -> Result<bool, BaselineExecError> {
        macro_rules! unary {
            ($operation:expr) => {{
                let value = self.pop()?;
                self.push(($operation)(value))?;
            }};
        }
        macro_rules! binary {
            ($operation:expr) => {{
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                self.push(($operation)(lhs, rhs))?;
            }};
        }

        match opcode {
            Opcode::I32_ADD => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).wrapping_add(rhs as u32) as u64 })
            }
            Opcode::I32_SUB => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).wrapping_sub(rhs as u32) as u64 })
            }
            Opcode::I32_MUL => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).wrapping_mul(rhs as u32) as u64 })
            }
            Opcode::I32_DIV_S => {
                let rhs = self.pop()? as u32 as i32;
                let lhs = self.pop()? as u32 as i32;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                let (value, overflow) = lhs.overflowing_div(rhs);
                if overflow {
                    return Err(WasmError::trap("integer overflow").into());
                }
                self.push(value as u32 as u64)?;
            }
            Opcode::I32_DIV_U => {
                let rhs = self.pop()? as u32;
                let lhs = self.pop()? as u32;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push((lhs / rhs) as u64)?;
            }
            Opcode::I32_REM_S => {
                let rhs = self.pop()? as u32 as i32;
                let lhs = self.pop()? as u32 as i32;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push(lhs.wrapping_rem(rhs) as u32 as u64)?;
            }
            Opcode::I32_REM_U => {
                let rhs = self.pop()? as u32;
                let lhs = self.pop()? as u32;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push((lhs % rhs) as u64)?;
            }
            Opcode::I32_AND => binary!(|lhs: u64, rhs: u64| (lhs as u32 & rhs as u32) as u64),
            Opcode::I32_OR => binary!(|lhs: u64, rhs: u64| (lhs as u32 | rhs as u32) as u64),
            Opcode::I32_XOR => binary!(|lhs: u64, rhs: u64| (lhs as u32 ^ rhs as u32) as u64),
            Opcode::I32_SHL => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).wrapping_shl(rhs as u32) as u64 })
            }
            Opcode::I32_SHR_S => binary!(|lhs: u64, rhs: u64| {
                (lhs as u32 as i32).wrapping_shr(rhs as u32) as u32 as u64
            }),
            Opcode::I32_SHR_U => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).wrapping_shr(rhs as u32) as u64 })
            }
            Opcode::I32_ROTL => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).rotate_left(rhs as u32 & 31) as u64 })
            }
            Opcode::I32_ROTR => {
                binary!(|lhs: u64, rhs: u64| { (lhs as u32).rotate_right(rhs as u32 & 31) as u64 })
            }
            Opcode::I32_CLZ => unary!(|value: u64| (value as u32).leading_zeros() as u64),
            Opcode::I32_CTZ => unary!(|value: u64| (value as u32).trailing_zeros() as u64),
            Opcode::I32_POPCNT => unary!(|value: u64| (value as u32).count_ones() as u64),
            Opcode::I32_EXTEND8_S => {
                unary!(|value: u64| value as u32 as i8 as i32 as u32 as u64)
            }
            Opcode::I32_EXTEND16_S => {
                unary!(|value: u64| value as u32 as i16 as i32 as u32 as u64)
            }
            Opcode::I32_EQZ => unary!(|value: u64| u64::from(value as u32 == 0)),
            Opcode::I32_EQ => {
                binary!(|lhs: u64, rhs: u64| u64::from(lhs as u32 == rhs as u32))
            }
            Opcode::I32_NE => {
                binary!(|lhs: u64, rhs: u64| u64::from(lhs as u32 != rhs as u32))
            }
            Opcode::I32_LT_S => binary!(|lhs: u64, rhs: u64| {
                u64::from((lhs as u32 as i32) < (rhs as u32 as i32))
            }),
            Opcode::I32_LT_U => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as u32) < (rhs as u32)))
            }
            Opcode::I32_GT_S => binary!(|lhs: u64, rhs: u64| {
                u64::from((lhs as u32 as i32) > (rhs as u32 as i32))
            }),
            Opcode::I32_GT_U => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as u32) > (rhs as u32)))
            }
            Opcode::I32_LE_S => binary!(|lhs: u64, rhs: u64| {
                u64::from((lhs as u32 as i32) <= (rhs as u32 as i32))
            }),
            Opcode::I32_LE_U => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as u32) <= (rhs as u32)))
            }
            Opcode::I32_GE_S => binary!(|lhs: u64, rhs: u64| {
                u64::from((lhs as u32 as i32) >= (rhs as u32 as i32))
            }),
            Opcode::I32_GE_U => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as u32) >= (rhs as u32)))
            }

            Opcode::I64_ADD => binary!(u64::wrapping_add),
            Opcode::I64_SUB => binary!(u64::wrapping_sub),
            Opcode::I64_MUL => binary!(u64::wrapping_mul),
            Opcode::I64_DIV_S => {
                let rhs = self.pop()? as i64;
                let lhs = self.pop()? as i64;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                let (value, overflow) = lhs.overflowing_div(rhs);
                if overflow {
                    return Err(WasmError::trap("integer overflow").into());
                }
                self.push(value as u64)?;
            }
            Opcode::I64_DIV_U => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push(lhs / rhs)?;
            }
            Opcode::I64_REM_S => {
                let rhs = self.pop()? as i64;
                let lhs = self.pop()? as i64;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push(lhs.wrapping_rem(rhs) as u64)?;
            }
            Opcode::I64_REM_U => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                if rhs == 0 {
                    return Err(WasmError::trap("integer divide by zero").into());
                }
                self.push(lhs % rhs)?;
            }
            Opcode::I64_AND => binary!(|lhs: u64, rhs: u64| lhs & rhs),
            Opcode::I64_OR => binary!(|lhs: u64, rhs: u64| lhs | rhs),
            Opcode::I64_XOR => binary!(|lhs: u64, rhs: u64| lhs ^ rhs),
            Opcode::I64_SHL => binary!(|lhs: u64, rhs: u64| lhs.wrapping_shl(rhs as u32)),
            Opcode::I64_SHR_S => {
                binary!(|lhs: u64, rhs: u64| { (lhs as i64).wrapping_shr(rhs as u32) as u64 })
            }
            Opcode::I64_SHR_U => binary!(|lhs: u64, rhs: u64| lhs.wrapping_shr(rhs as u32)),
            Opcode::I64_ROTL => {
                binary!(|lhs: u64, rhs: u64| lhs.rotate_left((rhs & 63) as u32))
            }
            Opcode::I64_ROTR => {
                binary!(|lhs: u64, rhs: u64| lhs.rotate_right((rhs & 63) as u32))
            }
            Opcode::I64_CLZ => unary!(|value: u64| value.leading_zeros() as u64),
            Opcode::I64_CTZ => unary!(|value: u64| value.trailing_zeros() as u64),
            Opcode::I64_POPCNT => unary!(|value: u64| value.count_ones() as u64),
            Opcode::I64_EXTEND8_S => unary!(|value: u64| value as i8 as i64 as u64),
            Opcode::I64_EXTEND16_S => unary!(|value: u64| value as i16 as i64 as u64),
            Opcode::I64_EXTEND32_S => unary!(|value: u64| value as i32 as i64 as u64),
            Opcode::I64_EQZ => unary!(|value: u64| u64::from(value == 0)),
            Opcode::I64_EQ => binary!(|lhs: u64, rhs: u64| u64::from(lhs == rhs)),
            Opcode::I64_NE => binary!(|lhs: u64, rhs: u64| u64::from(lhs != rhs)),
            Opcode::I64_LT_S => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as i64) < (rhs as i64)))
            }
            Opcode::I64_LT_U => binary!(|lhs: u64, rhs: u64| u64::from(lhs < rhs)),
            Opcode::I64_GT_S => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as i64) > (rhs as i64)))
            }
            Opcode::I64_GT_U => binary!(|lhs: u64, rhs: u64| u64::from(lhs > rhs)),
            Opcode::I64_LE_S => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as i64) <= (rhs as i64)))
            }
            Opcode::I64_LE_U => binary!(|lhs: u64, rhs: u64| u64::from(lhs <= rhs)),
            Opcode::I64_GE_S => {
                binary!(|lhs: u64, rhs: u64| u64::from((lhs as i64) >= (rhs as i64)))
            }
            Opcode::I64_GE_U => binary!(|lhs: u64, rhs: u64| u64::from(lhs >= rhs)),

            Opcode::F32_ABS => {
                unary!(|value: u64| { f32::from_bits(value as u32).abs().to_bits() as u64 })
            }
            Opcode::F32_NEG => {
                unary!(|value: u64| { (-f32::from_bits(value as u32)).to_bits() as u64 })
            }
            Opcode::F32_CEIL => unary!(|value: u64| {
                super::fmath::ceil32(f32::from_bits(value as u32)).to_bits() as u64
            }),
            Opcode::F32_FLOOR => unary!(|value: u64| {
                super::fmath::floor32(f32::from_bits(value as u32)).to_bits() as u64
            }),
            Opcode::F32_TRUNC => unary!(|value: u64| {
                super::fmath::trunc32(f32::from_bits(value as u32)).to_bits() as u64
            }),
            Opcode::F32_NEAREST => unary!(|value: u64| {
                super::fmath::nearest32(f32::from_bits(value as u32)).to_bits() as u64
            }),
            Opcode::F32_SQRT => unary!(|value: u64| {
                super::fmath::sqrt32(f32::from_bits(value as u32)).to_bits() as u64
            }),
            Opcode::F32_ADD => binary!(|lhs: u64, rhs: u64| {
                (f32::from_bits(lhs as u32) + f32::from_bits(rhs as u32)).to_bits() as u64
            }),
            Opcode::F32_SUB => binary!(|lhs: u64, rhs: u64| {
                (f32::from_bits(lhs as u32) - f32::from_bits(rhs as u32)).to_bits() as u64
            }),
            Opcode::F32_MUL => binary!(|lhs: u64, rhs: u64| {
                (f32::from_bits(lhs as u32) * f32::from_bits(rhs as u32)).to_bits() as u64
            }),
            Opcode::F32_DIV => binary!(|lhs: u64, rhs: u64| {
                (f32::from_bits(lhs as u32) / f32::from_bits(rhs as u32)).to_bits() as u64
            }),
            Opcode::F32_MIN => binary!(|lhs: u64, rhs: u64| {
                super::exec::wasm_min_f32(f32::from_bits(lhs as u32), f32::from_bits(rhs as u32))
                    .to_bits() as u64
            }),
            Opcode::F32_MAX => binary!(|lhs: u64, rhs: u64| {
                super::exec::wasm_max_f32(f32::from_bits(lhs as u32), f32::from_bits(rhs as u32))
                    .to_bits() as u64
            }),
            Opcode::F32_COPYSIGN => binary!(|lhs: u64, rhs: u64| {
                f32::from_bits(lhs as u32)
                    .copysign(f32::from_bits(rhs as u32))
                    .to_bits() as u64
            }),
            Opcode::F32_EQ => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) == f32::from_bits(rhs as u32))
            }),
            Opcode::F32_NE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) != f32::from_bits(rhs as u32))
            }),
            Opcode::F32_LT => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) < f32::from_bits(rhs as u32))
            }),
            Opcode::F32_GT => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) > f32::from_bits(rhs as u32))
            }),
            Opcode::F32_LE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) <= f32::from_bits(rhs as u32))
            }),
            Opcode::F32_GE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f32::from_bits(lhs as u32) >= f32::from_bits(rhs as u32))
            }),

            Opcode::F64_ABS => {
                unary!(|value: u64| f64::from_bits(value).abs().to_bits())
            }
            Opcode::F64_NEG => unary!(|value: u64| (-f64::from_bits(value)).to_bits()),
            Opcode::F64_CEIL => {
                unary!(|value: u64| super::fmath::ceil64(f64::from_bits(value)).to_bits())
            }
            Opcode::F64_FLOOR => {
                unary!(|value: u64| super::fmath::floor64(f64::from_bits(value)).to_bits())
            }
            Opcode::F64_TRUNC => {
                unary!(|value: u64| super::fmath::trunc64(f64::from_bits(value)).to_bits())
            }
            Opcode::F64_NEAREST => {
                unary!(|value: u64| super::fmath::nearest64(f64::from_bits(value)).to_bits())
            }
            Opcode::F64_SQRT => {
                unary!(|value: u64| super::fmath::sqrt64(f64::from_bits(value)).to_bits())
            }
            Opcode::F64_ADD => binary!(|lhs: u64, rhs: u64| {
                (f64::from_bits(lhs) + f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_SUB => binary!(|lhs: u64, rhs: u64| {
                (f64::from_bits(lhs) - f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_MUL => binary!(|lhs: u64, rhs: u64| {
                (f64::from_bits(lhs) * f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_DIV => binary!(|lhs: u64, rhs: u64| {
                (f64::from_bits(lhs) / f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_MIN => binary!(|lhs: u64, rhs: u64| {
                super::exec::wasm_min_f64(f64::from_bits(lhs), f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_MAX => binary!(|lhs: u64, rhs: u64| {
                super::exec::wasm_max_f64(f64::from_bits(lhs), f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_COPYSIGN => binary!(|lhs: u64, rhs: u64| {
                f64::from_bits(lhs).copysign(f64::from_bits(rhs)).to_bits()
            }),
            Opcode::F64_EQ => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) == f64::from_bits(rhs))
            }),
            Opcode::F64_NE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) != f64::from_bits(rhs))
            }),
            Opcode::F64_LT => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) < f64::from_bits(rhs))
            }),
            Opcode::F64_GT => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) > f64::from_bits(rhs))
            }),
            Opcode::F64_LE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) <= f64::from_bits(rhs))
            }),
            Opcode::F64_GE => binary!(|lhs: u64, rhs: u64| {
                u64::from(f64::from_bits(lhs) >= f64::from_bits(rhs))
            }),

            Opcode::I32_WRAP_I64 => unary!(|value: u64| value as u32 as u64),
            Opcode::I64_EXTEND_I32_S => {
                unary!(|value: u64| value as u32 as i32 as i64 as u64)
            }
            Opcode::I64_EXTEND_I32_U => unary!(|value: u64| value as u32 as u64),
            Opcode::I32_TRUNC_F32_S => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(
                    super::exec::trunc_checked(value as f64, -2147483648.0, 2147483648.0)? as i64
                        as u32 as u64,
                )?;
            }
            Opcode::I32_TRUNC_F32_U => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(
                    super::exec::trunc_checked(value as f64, 0.0, 4294967296.0)? as u64 as u32
                        as u64,
                )?;
            }
            Opcode::I32_TRUNC_F64_S => {
                let value = f64::from_bits(self.pop()?);
                self.push(
                    super::exec::trunc_checked(value, -2147483648.0, 2147483648.0)? as i64 as u32
                        as u64,
                )?;
            }
            Opcode::I32_TRUNC_F64_U => {
                let value = f64::from_bits(self.pop()?);
                self.push(
                    super::exec::trunc_checked(value, 0.0, 4294967296.0)? as u64 as u32 as u64,
                )?;
            }
            Opcode::I64_TRUNC_F32_S => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(super::exec::trunc_checked(
                    value as f64,
                    -9223372036854775808.0,
                    9223372036854775808.0,
                )? as i64 as u64)?;
            }
            Opcode::I64_TRUNC_F32_U => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(
                    super::exec::trunc_checked(value as f64, 0.0, 18446744073709551616.0)? as u64,
                )?;
            }
            Opcode::I64_TRUNC_F64_S => {
                let value = f64::from_bits(self.pop()?);
                self.push(super::exec::trunc_checked(
                    value,
                    -9223372036854775808.0,
                    9223372036854775808.0,
                )? as i64 as u64)?;
            }
            Opcode::I64_TRUNC_F64_U => {
                let value = f64::from_bits(self.pop()?);
                self.push(super::exec::trunc_checked(value, 0.0, 18446744073709551616.0)? as u64)?;
            }
            Opcode::F32_CONVERT_I32_S => {
                unary!(|value: u64| { ((value as u32 as i32) as f32).to_bits() as u64 })
            }
            Opcode::F32_CONVERT_I32_U => {
                unary!(|value: u64| (value as u32 as f32).to_bits() as u64)
            }
            Opcode::F32_CONVERT_I64_S => {
                unary!(|value: u64| (value as i64 as f32).to_bits() as u64)
            }
            Opcode::F32_CONVERT_I64_U => {
                unary!(|value: u64| (value as f32).to_bits() as u64)
            }
            Opcode::F32_DEMOTE_F64 => {
                unary!(|value: u64| (f64::from_bits(value) as f32).to_bits() as u64)
            }
            Opcode::F64_CONVERT_I32_S => {
                unary!(|value: u64| ((value as u32 as i32) as f64).to_bits())
            }
            Opcode::F64_CONVERT_I32_U => {
                unary!(|value: u64| (value as u32 as f64).to_bits())
            }
            Opcode::F64_CONVERT_I64_S => {
                unary!(|value: u64| (value as i64 as f64).to_bits())
            }
            Opcode::F64_CONVERT_I64_U => unary!(|value: u64| (value as f64).to_bits()),
            Opcode::F64_PROMOTE_F32 => {
                unary!(|value: u64| (f32::from_bits(value as u32) as f64).to_bits())
            }
            Opcode::I32_REINTERPRET_F32 | Opcode::F32_REINTERPRET_I32 => {
                unary!(|value: u64| value as u32 as u64)
            }
            Opcode::I64_REINTERPRET_F64 | Opcode::F64_REINTERPRET_I64 => {
                unary!(|value: u64| value)
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn exec_saturating_conversion(&mut self, opcode: OpcodeFC) -> Result<bool, BaselineExecError> {
        match opcode {
            OpcodeFC::I32_TRUNC_SAT_F32_S => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(value as i32 as u32 as u64)?;
            }
            OpcodeFC::I32_TRUNC_SAT_F32_U => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(value as u32 as u64)?;
            }
            OpcodeFC::I32_TRUNC_SAT_F64_S => {
                let value = f64::from_bits(self.pop()?);
                self.push(value as i32 as u32 as u64)?;
            }
            OpcodeFC::I32_TRUNC_SAT_F64_U => {
                let value = f64::from_bits(self.pop()?);
                self.push(value as u32 as u64)?;
            }
            OpcodeFC::I64_TRUNC_SAT_F32_S => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(value as i64 as u64)?;
            }
            OpcodeFC::I64_TRUNC_SAT_F32_U => {
                let value = f32::from_bits(self.pop()? as u32);
                self.push(value as u64)?;
            }
            OpcodeFC::I64_TRUNC_SAT_F64_S => {
                let value = f64::from_bits(self.pop()?);
                self.push(value as i64 as u64)?;
            }
            OpcodeFC::I64_TRUNC_SAT_F64_U => {
                let value = f64::from_bits(self.pop()?);
                self.push(value as u64)?;
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn current_target(&self) -> Result<ControlTarget, BaselineExecError> {
        let activation = self.current_activation()?;
        let function = self.current_baseline_function()?;
        let relative = u32::try_from(activation.stp)
            .map_err(|_| WasmError::invalid("baseline MVP side-table pointer overflow"))?;
        let index = function
            .absolute_stp(relative)
            .ok_or_else(|| WasmError::invalid("baseline MVP side-table index overflow"))?;
        if index >= function.control_targets.end {
            return Err(WasmError::invalid("baseline MVP side-table pointer overflow").into());
        }
        self.artifact
            .control_targets
            .get(index)
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline MVP control target missing").into())
    }

    fn apply_target(&mut self, target: ControlTarget) -> Result<(), BaselineExecError> {
        let (operand_base, function_index) = {
            let activation = self.current_activation()?;
            (activation.operand_base, activation.function_index)
        };
        let max_operand_height =
            self.baseline_function(function_index)?.max_operand_height as usize;
        let keep = target.keep_arity as usize;
        let base = operand_base
            .checked_add(target.target_stack_height as usize)
            .ok_or_else(|| WasmError::invalid("baseline MVP branch stack overflow"))?;
        let source = self
            .values
            .len()
            .checked_sub(keep)
            .ok_or_else(|| WasmError::invalid("baseline MVP branch value underflow"))?;
        let new_len = base
            .checked_add(keep)
            .ok_or_else(|| WasmError::invalid("baseline MVP branch stack overflow"))?;
        let frame_limit = operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline MVP frame limit overflow"))?;
        if base > source || source < operand_base || new_len > frame_limit {
            return Err(WasmError::invalid("baseline MVP branch stack shape mismatch").into());
        }
        self.values.copy_within(source..source + keep, base);
        self.values.truncate(new_len);
        self.baseline_function(function_index)?
            .absolute_stp(target.target_stp)
            .ok_or_else(|| WasmError::invalid("baseline MVP target side-table overflow"))?;
        let activation = self.current_activation_mut()?;
        activation.pc = target.target_pc as usize;
        activation.stp = target.target_stp as usize;
        Ok(())
    }

    fn push(&mut self, value: u64) -> Result<(), BaselineExecError> {
        let activation = *self.current_activation()?;
        let max_operand_height = self
            .baseline_function(activation.function_index)?
            .max_operand_height as usize;
        let frame_limit = activation
            .operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline MVP frame limit overflow"))?;
        if self.values.len() >= frame_limit || self.values.len() == self.values.capacity() {
            return Err(WasmError::invalid("baseline MVP operand capacity exhausted").into());
        }
        self.values.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<u64, BaselineExecError> {
        let operand_base = self.current_activation()?.operand_base;
        if self.values.len() == operand_base {
            return Err(WasmError::invalid("baseline MVP operand stack underflow").into());
        }
        self.values
            .pop()
            .ok_or_else(|| WasmError::internal("baseline MVP value stack disappeared").into())
    }

    fn peek(&self) -> Result<u64, BaselineExecError> {
        let operand_base = self.current_activation()?.operand_base;
        if self.values.len() == operand_base {
            return Err(WasmError::invalid("baseline MVP operand stack underflow").into());
        }
        self.values
            .last()
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline MVP operand stack underflow").into())
    }

    fn enter_root(&mut self, args: &[Value]) -> Result<(), BaselineExecError> {
        let (parameter_count, local_count) = self.access.with_instance(|instance| {
            let function = instance
                .module()
                .functions()
                .get(self.root_function)
                .ok_or_else(|| {
                    WasmError::invalid("baseline MVP function index is out of bounds")
                })?;
            let spec = function.spec().ok_or(BaselineExecError::Unsupported {
                opcode: None,
                pc: 0,
                feature: "imported function",
            })?;
            let function_type = function.func_type();
            if args.len() != function_type.params().len() {
                return Err(WasmError::invalid("baseline MVP argument count mismatch").into());
            }
            validate_scalar_function(
                function_type.params(),
                function_type.results(),
                spec.locals(),
            )?;
            Ok::<_, BaselineExecError>((function_type.params().len(), spec.locals().len()))
        })??;
        let max_operand_height = self
            .baseline_function(self.root_function)?
            .max_operand_height as usize;
        let operand_base = parameter_count
            .checked_add(local_count)
            .ok_or_else(|| WasmError::invalid("baseline MVP local count overflow"))?;
        let required = operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline MVP frame size overflow"))?;
        self.reserve_value_slots(required)?;
        self.reserve_activation()?;
        for (index, value) in args.iter().enumerate() {
            let expected = self.access.with_instance(|instance| {
                instance.module().functions()[self.root_function]
                    .func_type()
                    .params()[index]
            })?;
            self.values.push(scalar_to_raw(*value, expected)?);
        }
        self.values.resize(operand_base, 0);
        self.activations.push(BaselineActivation {
            function_index: self.root_function,
            pc: 0,
            stp: 0,
            locals_base: 0,
            operand_base,
            return_base: 0,
        });
        Ok(())
    }

    fn enter_call(&mut self, callee: usize, source_pc: usize) -> Result<(), BaselineExecError> {
        let (parameter_count, local_count) = self.access.with_instance(|instance| {
            let function =
                instance.module().functions().get(callee).ok_or_else(|| {
                    WasmError::invalid("baseline MVP callee index is out of bounds")
                })?;
            let spec = function.spec().ok_or(BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::CALL)),
                pc: source_pc,
                feature: "imported call",
            })?;
            let function_type = function.func_type();
            validate_scalar_function(
                function_type.params(),
                function_type.results(),
                spec.locals(),
            )?;
            Ok::<_, BaselineExecError>((function_type.params().len(), spec.locals().len()))
        })??;
        let max_operand_height = self.baseline_function(callee)?.max_operand_height as usize;
        let caller_operand_base = self.current_activation()?.operand_base;
        // The caller's arguments already occupy the value-stack tail. Reuse
        // them as the callee's parameter locals, append only declared locals,
        // then overwrite this same range with results on return.
        let return_base = self
            .values
            .len()
            .checked_sub(parameter_count)
            .filter(|&base| base >= caller_operand_base)
            .ok_or_else(|| WasmError::invalid("baseline MVP call argument underflow"))?;
        let locals_base = return_base;
        let operand_base = return_base
            .checked_add(parameter_count)
            .and_then(|base| base.checked_add(local_count))
            .ok_or_else(|| WasmError::invalid("baseline MVP callee locals overflow"))?;
        let required = operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline MVP callee frame overflow"))?;
        self.reserve_value_slots(required)?;
        self.reserve_activation()?;
        self.values.resize(operand_base, 0);
        self.activations.push(BaselineActivation {
            function_index: callee,
            pc: 0,
            stp: 0,
            locals_base,
            operand_base,
            return_base,
        });
        Ok(())
    }

    fn finish_activation(&mut self) -> Result<(), BaselineExecError> {
        let activation = *self.current_activation()?;
        let result_count = self.access.with_instance(|instance| {
            instance
                .module()
                .functions()
                .get(activation.function_index)
                .map(|function| function.func_type().results().len())
                .ok_or_else(|| WasmError::invalid("baseline MVP return function is missing"))
        })??;
        let expected_len = activation
            .operand_base
            .checked_add(result_count)
            .ok_or_else(|| WasmError::invalid("baseline MVP result stack overflow"))?;
        if self.values.len() != expected_len {
            return Err(WasmError::invalid("baseline MVP result stack shape mismatch").into());
        }
        let result_end = activation
            .return_base
            .checked_add(result_count)
            .ok_or_else(|| WasmError::invalid("baseline MVP result destination overflow"))?;
        self.values.copy_within(
            activation.operand_base..expected_len,
            activation.return_base,
        );
        self.values.truncate(result_end);
        self.activations.pop();
        Ok(())
    }

    fn reserve_activation(&mut self) -> Result<(), BaselineExecError> {
        if self.activations.len() >= self.max_activations {
            return Err(WasmError::trap("call stack exhausted").into());
        }
        if self.activations.len() == self.activations.capacity() {
            self.activations
                .try_reserve(1)
                .map_err(|_| WasmError::trap("call stack exhausted"))?;
        }
        Ok(())
    }

    fn reserve_value_slots(&mut self, required: usize) -> Result<(), BaselineExecError> {
        if required > self.max_value_slots {
            return Err(WasmError::trap("call stack exhausted").into());
        }
        if required > self.values.capacity() {
            self.values
                .try_reserve(required.saturating_sub(self.values.len()))
                .map_err(|_| WasmError::trap("call stack exhausted"))?;
        }
        Ok(())
    }

    fn current_activation(&self) -> Result<&BaselineActivation, BaselineExecError> {
        self.activations
            .last()
            .ok_or_else(|| WasmError::invalid("baseline MVP has no active function").into())
    }

    fn current_activation_mut(&mut self) -> Result<&mut BaselineActivation, BaselineExecError> {
        self.activations
            .last_mut()
            .ok_or_else(|| WasmError::invalid("baseline MVP has no active function").into())
    }

    fn baseline_function(&self, index: usize) -> Result<&BaselineFunction, BaselineExecError> {
        self.artifact
            .functions
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| WasmError::invalid("baseline MVP artifact function is missing").into())
    }

    fn current_baseline_function(&self) -> Result<&BaselineFunction, BaselineExecError> {
        self.baseline_function(self.current_activation()?.function_index)
    }

    fn root_result_count(&self) -> Result<usize, BaselineExecError> {
        self.access
            .with_instance(|instance| {
                instance
                    .module()
                    .functions()
                    .get(self.root_function)
                    .map(|function| function.func_type().results().len())
                    .ok_or_else(|| WasmError::invalid("baseline MVP result function is missing"))
            })?
            .map_err(Into::into)
    }

    fn root_result_type(&self, index: usize) -> Result<ValueType, BaselineExecError> {
        self.access
            .with_instance(|instance| {
                instance
                    .module()
                    .functions()
                    .get(self.root_function)
                    .and_then(|function| function.func_type().results().get(index))
                    .copied()
                    .ok_or_else(|| WasmError::invalid("baseline MVP result type is missing"))
            })?
            .map_err(Into::into)
    }

    fn current_br_table(&self, source_pc: usize) -> Result<BrTableRange, BaselineExecError> {
        let function = self.current_baseline_function()?;
        self.artifact.br_tables[function.br_tables.clone()]
            .iter()
            .find(|table| table.source_pc as usize == source_pc)
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline MVP br_table metadata missing").into())
    }

    fn local_slot(&self, local: usize) -> Result<usize, BaselineExecError> {
        let activation = self.current_activation()?;
        let slot = activation
            .locals_base
            .checked_add(local)
            .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))?;
        (slot < activation.operand_base)
            .then_some(slot)
            .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow").into())
    }

    fn advance_stp(&mut self) -> Result<(), BaselineExecError> {
        let activation = self.current_activation_mut()?;
        activation.stp = activation
            .stp
            .checked_add(1)
            .ok_or_else(|| WasmError::invalid("baseline MVP side-table pointer overflow"))?;
        Ok(())
    }

    fn unsupported<T>(
        &self,
        opcode: Option<WasmOpcode>,
        pc: usize,
        feature: &'static str,
    ) -> Result<T, BaselineExecError> {
        Err(BaselineExecError::Unsupported {
            opcode,
            pc,
            feature,
        })
    }
}

fn copy_immediate(immediate: RawImmediate<'_>) -> BaselineImmediate {
    match immediate {
        RawImmediate::None => BaselineImmediate::None,
        RawImmediate::I32(value) => BaselineImmediate::I32(value),
        RawImmediate::I64(value) => BaselineImmediate::I64(value),
        RawImmediate::F32(value) => BaselineImmediate::F32(value),
        RawImmediate::F64(value) => BaselineImmediate::F64(value),
        RawImmediate::LocalIndex(value) => BaselineImmediate::LocalIndex(value),
        RawImmediate::FunctionIndex(value) => BaselineImmediate::FunctionIndex(value),
        RawImmediate::GlobalIndex(value) => BaselineImmediate::GlobalIndex(value),
        RawImmediate::MemoryIndex(value) => BaselineImmediate::MemoryIndex(value),
        RawImmediate::MemArg { offset, memidx, .. } => BaselineImmediate::MemArg { offset, memidx },
        _ => BaselineImmediate::Other,
    }
}

fn raw_local(immediate: BaselineImmediate) -> Result<usize, BaselineExecError> {
    let BaselineImmediate::LocalIndex(local) = immediate else {
        return Err(WasmError::internal("baseline MVP local immediate mismatch").into());
    };
    Ok(local as usize)
}

fn raw_function(immediate: BaselineImmediate) -> Result<usize, BaselineExecError> {
    let BaselineImmediate::FunctionIndex(function) = immediate else {
        return Err(WasmError::internal("baseline MVP function immediate mismatch").into());
    };
    Ok(function as usize)
}

fn raw_global(immediate: BaselineImmediate) -> Result<usize, BaselineExecError> {
    let BaselineImmediate::GlobalIndex(global) = immediate else {
        return Err(WasmError::internal("baseline MVP global immediate mismatch").into());
    };
    Ok(global as usize)
}

fn raw_memory(immediate: BaselineImmediate) -> Result<usize, BaselineExecError> {
    let BaselineImmediate::MemoryIndex(memory) = immediate else {
        return Err(WasmError::internal("baseline MVP memory immediate mismatch").into());
    };
    Ok(memory as usize)
}

fn raw_memarg(immediate: BaselineImmediate) -> Result<(usize, u64), BaselineExecError> {
    let BaselineImmediate::MemArg { offset, memidx } = immediate else {
        return Err(WasmError::internal("baseline MVP memarg immediate mismatch").into());
    };
    Ok((memidx as usize, offset))
}

fn validate_scalar_function(
    params: &[ValueType],
    results: &[ValueType],
    locals: &[ValueType],
) -> Result<(), BaselineExecError> {
    if params
        .iter()
        .chain(results)
        .chain(locals)
        .any(|&value_type| !is_mvp_scalar(value_type))
    {
        return Err(BaselineExecError::Unsupported {
            opcode: None,
            pc: 0,
            feature: "non-scalar function frame",
        });
    }
    Ok(())
}

fn is_mvp_scalar(value_type: ValueType) -> bool {
    matches!(
        value_type,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
    )
}

fn scalar_to_raw(value: Value, expected: ValueType) -> Result<u64, BaselineExecError> {
    if value.value_type() != expected || !is_mvp_scalar(expected) {
        return Err(WasmError::invalid("baseline MVP argument type mismatch").into());
    }
    Ok(value.to_raw())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::vm::engine::{Engine, Tier};
    use crate::vm::interpreter::baseline_artifact::artifact_test_guard;
    use crate::vm::interpreter::predecode::build_baseline_artifact;
    use crate::vm::interpreter::InterpInstance;
    use crate::vm::link::LinkRegistry;
    use crate::Instance;
    use std::string::ToString;
    use std::vec;
    use std::vec::Vec as StdVec;

    fn baseline(
        wasm: &[u8],
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        baseline_with_imports(wasm, export, args, &[])
    }

    fn baseline_with_imports(
        wasm: &[u8],
        export: &str,
        args: &[Value],
        imports: &[crate::Import],
    ) -> Result<Vec<Value>, BaselineExecError> {
        let _guard = artifact_test_guard();
        let module = Module::new("baseline-exec", wasm).expect("module");
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut instance = Instance::from_module(&engine, module, imports).expect("instance");
        instance
            .with_interp_mut(|instance| {
                BaselineDriver::new(InterpInstanceAccess::borrowed(instance), &artifact)
                    .invoke_export(export, args)
            })
            .expect("interpreter instance")
    }

    fn native(wasm: &[u8], export: &str, args: &[Value]) -> Result<StdVec<Value>, WasmError> {
        native_with_imports(wasm, export, args, &[])
    }

    fn native_with_imports(
        wasm: &[u8],
        export: &str,
        args: &[Value],
        imports: &[crate::Import],
    ) -> Result<StdVec<Value>, WasmError> {
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut instance = Instance::new(&engine, wasm, imports).expect("instance");
        instance.invoke(export, args).map(StdVec::from)
    }

    fn built_interp(module: Module, imports: &[crate::Import]) -> InterpInstance {
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let registry = LinkRegistry::new();
        let (_, instance_backref) = registry.reserve_instance();
        InterpInstance::build(
            &engine,
            module,
            None,
            imports,
            None,
            registry.arenas(),
            instance_backref,
        )
        .expect("build interpreter instance")
    }

    fn initialized_interp(module: Module, imports: &[crate::Import]) -> InterpInstance {
        InterpInstance::initialize(built_interp(module, imports))
            .map_err(|(_, error)| error)
            .expect("initialize interpreter instance")
    }

    fn baseline_on(
        instance: &mut InterpInstance,
        artifact: &BaselineArtifact,
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        BaselineDriver::new(InterpInstanceAccess::borrowed(instance), artifact)
            .invoke_export(export, args)
    }

    fn assert_values_equal(baseline: &[Value], native: &[Value]) {
        assert_eq!(baseline.len(), native.len());
        for (baseline, native) in baseline.iter().zip(native) {
            match (*baseline, *native) {
                (Value::F32(lhs), Value::F32(rhs)) if lhs.is_nan() && rhs.is_nan() => {}
                (Value::F64(lhs), Value::F64(rhs)) if lhs.is_nan() && rhs.is_nan() => {}
                (Value::F32(lhs), Value::F32(rhs)) => {
                    assert_eq!(lhs.to_bits(), rhs.to_bits())
                }
                (Value::F64(lhs), Value::F64(rhs)) => {
                    assert_eq!(lhs.to_bits(), rhs.to_bits())
                }
                (lhs, rhs) => assert_eq!(lhs, rhs),
            }
        }
    }

    fn assert_trap_equal(wasm: &[u8], export: &str) {
        let baseline = baseline(wasm, export, &[]).expect_err("baseline must trap");
        let BaselineExecError::Wasm(baseline) = baseline else {
            panic!("baseline returned unsupported instead of a trap");
        };
        let native = native(wasm, export, &[]).expect_err("native must trap");
        assert_eq!(baseline.to_string(), native.to_string());
    }

    #[test]
    fn raw_loop_driver_matches_folded_interpreter() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "sum") (param i32) (result i32) (local i32)
                    block $exit
                        loop $again
                            local.get 0
                            i32.eqz
                            br_if $exit
                            local.get 1
                            local.get 0
                            i32.add
                            local.set 1
                            local.get 0
                            i32.const 1
                            i32.sub
                            local.set 0
                            br $again
                        end
                    end
                    local.get 1))"#,
        )
        .expect("wat");
        for input in [0, 1, 10, 100] {
            let args = [Value::I32(input)];
            let baseline = baseline(&wasm, "sum", &args).expect("baseline");
            let native = native(&wasm, "sum", &args).expect("native");
            assert_eq!(baseline.as_slice(), native.as_slice());
        }
    }

    #[test]
    fn iterative_fibonacci_matches_folded_interpreter() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "fib") (param $n i32) (result i32)
                    (local $a i32) (local $b i32) (local $next i32)
                    i32.const 0 local.set $a
                    i32.const 1 local.set $b
                    block $exit
                        loop $again
                            local.get $n
                            i32.eqz
                            br_if $exit
                            local.get $a
                            local.get $b
                            i32.add
                            local.set $next
                            local.get $b
                            local.set $a
                            local.get $next
                            local.set $b
                            local.get $n
                            i32.const 1
                            i32.sub
                            local.set $n
                            br $again
                        end
                    end
                    local.get $a))"#,
        )
        .expect("wat");
        for input in [0, 1, 2, 10, 20] {
            let args = [Value::I32(input)];
            let baseline = baseline(&wasm, "fib", &args).expect("baseline");
            let native = native(&wasm, "fib", &args).expect("native");
            assert_values_equal(&baseline, &native);
        }
    }

    #[test]
    fn multivalue_branch_and_br_table_match_folded_interpreter() {
        let wasm = wat::parse_str(
            r#"(module
                (type $pair (func (param i32 i32) (result i32 i32)))
                (func (export "multi") (param i32 i32) (result i32)
                    block (result i32)
                        local.get 0
                        local.get 1
                        block (type $pair)
                            br 0
                            unreachable
                        end
                        i32.add
                    end)
                (func (export "table") (param i32) (result i32)
                    block $outer (result i32)
                        block $inner (result i32)
                            i32.const 11
                            local.get 0
                            br_table $inner $outer
                        end
                        i32.const 1
                        i32.add
                    end)
                (func (export "early") (param i32) (result i32)
                    local.get 0
                    return
                    unreachable))"#,
        )
        .expect("wat");
        for args in [
            [Value::I32(7), Value::I32(9)],
            [Value::I32(-1), Value::I32(3)],
        ] {
            let baseline = baseline(&wasm, "multi", &args).expect("baseline multi");
            let native = native(&wasm, "multi", &args).expect("native multi");
            assert_values_equal(&baseline, &native);
        }
        for selector in [0, 1, 99] {
            let args = [Value::I32(selector)];
            let baseline = baseline(&wasm, "table", &args).expect("baseline table");
            let native = native(&wasm, "table", &args).expect("native table");
            assert_values_equal(&baseline, &native);
        }
        let args = [Value::I32(73)];
        let baseline = baseline(&wasm, "early", &args).expect("baseline return");
        let native = native(&wasm, "early", &args).expect("native return");
        assert_values_equal(&baseline, &native);
    }

    #[test]
    fn nested_local_calls_preserve_arguments_locals_and_multiple_results() {
        let wasm = wat::parse_str(
            r#"(module
                (func $pair (param $value i32) (result i32 i64) (local $zero i32)
                    local.get $value
                    local.get $zero
                    i32.add
                    local.get $value
                    i64.extend_i32_s)
                (func $early (param $value i32) (result i32)
                    local.get $value
                    i32.const 2
                    i32.mul
                    return
                    unreachable)
                (func $middle (param $value i32) (result i32 i64 i32)
                    local.get $value
                    call $pair
                    local.get $value
                    call $early)
                (func (export "nested") (param $value i32) (result i32 i64 i32)
                    local.get $value
                    call $middle))"#,
        )
        .expect("wat");
        for input in [-17, 0, 31] {
            let args = [Value::I32(input)];
            let baseline = baseline(&wasm, "nested", &args).expect("baseline nested calls");
            let native = native(&wasm, "nested", &args).expect("native nested calls");
            assert_values_equal(&baseline, &native);
        }
    }

    #[test]
    fn recursive_local_call_matches_folded_interpreter() {
        let wasm = wat::parse_str(
            r#"(module
                (func $sum (export "sum") (param $value i32) (result i32)
                    local.get $value
                    i32.eqz
                    if (result i32)
                        i32.const 0
                    else
                        local.get $value
                        local.get $value
                        i32.const 1
                        i32.sub
                        call $sum
                        i32.add
                    end))"#,
        )
        .expect("wat");
        for input in [0, 1, 32, 200] {
            let args = [Value::I32(input)];
            let baseline = baseline(&wasm, "sum", &args).expect("baseline recursion");
            let native = native(&wasm, "sum", &args).expect("native recursion");
            assert_values_equal(&baseline, &native);
        }
    }

    #[test]
    fn recursive_local_call_exhausts_explicit_activation_stack() {
        let wasm = wat::parse_str(
            r#"(module
                (func $recurse (export "recurse")
                    call $recurse))"#,
        )
        .expect("wat");
        assert_trap_equal(&wasm, "recurse");
    }

    #[test]
    fn recursive_call_depth_matches_folded_4095_4096_boundary() {
        let wasm = wat::parse_str(
            r#"(module
                (func $depth (export "depth") (param $remaining i32) (result i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        i32.const 73
                    else
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        call $depth
                        i32.const 0
                        i32.add
                    end))"#,
        )
        .expect("wat");
        for depth in [4095, 4096] {
            let args = [Value::I32(depth)];
            let baseline = baseline(&wasm, "depth", &args).expect("baseline depth boundary");
            let native = native(&wasm, "depth", &args).expect("native depth boundary");
            assert_values_equal(&baseline, &native);
        }
    }

    #[test]
    fn scalar_numeric_compare_and_conversion_results_match() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "scalar")
                      (param $a i32) (param $b i64) (param $c f32) (param $d f64)
                      (result i32 i64 f32 f64 i32 i64 f32 f64)
                    local.get $a i32.const 3 i32.mul i32.const 1 i32.add
                    local.get $b i64.const 5 i64.xor i64.extend16_s
                    local.get $c f32.abs f32.sqrt f32.const 2 f32.add
                    local.get $d f64.nearest f64.const -0 f64.max
                    local.get $a i32.const 0 i32.lt_s
                    local.get $a i64.extend_i32_s
                    local.get $b f32.convert_i64_s
                    local.get $c f64.promote_f32)
                (func (export "sat") (param f64) (result i32 i64)
                    local.get 0 i32.trunc_sat_f64_s
                    local.get 0 i64.trunc_sat_f64_u)
                (func (export "nan") (result f32 f64)
                    f32.const nan:0x200000 f32.sqrt
                    f64.const nan:0x4000000000000 f64.nearest))"#,
        )
        .expect("wat");
        let args = [
            Value::I32(-7),
            Value::I64(-123),
            Value::F32(-9.0),
            Value::F64(3.5),
        ];
        let baseline_values = baseline(&wasm, "scalar", &args).expect("baseline scalar");
        let native_values = native(&wasm, "scalar", &args).expect("native scalar");
        assert_values_equal(&baseline_values, &native_values);
        for input in [f64::NEG_INFINITY, -7.9, 5.2, f64::INFINITY, f64::NAN] {
            let args = [Value::F64(input)];
            let baseline_values = baseline(&wasm, "sat", &args).expect("baseline sat");
            let native_values = native(&wasm, "sat", &args).expect("native sat");
            assert_values_equal(&baseline_values, &native_values);
        }
        let baseline_values = baseline(&wasm, "nan", &[]).expect("baseline nan");
        let native_values = native(&wasm, "nan", &[]).expect("native nan");
        assert_values_equal(&baseline_values, &native_values);
    }

    #[test]
    fn scalar_traps_match_folded_interpreter() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "zero") (result i32)
                    i32.const 1 i32.const 0 i32.div_s)
                (func (export "overflow") (result i32)
                    i32.const -2147483648 i32.const -1 i32.div_s)
                (func (export "nan") (result i32)
                    f64.const nan i32.trunc_f64_s)
                (func (export "dead") unreachable))"#,
        )
        .expect("wat");
        for export in ["zero", "overflow", "nan", "dead"] {
            assert_trap_equal(&wasm, export);
        }
    }

    #[test]
    fn mvp_memory_load_store_widths_and_active_data_match_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1 1)
                (data (i32.const 48) "\01\02\03\04")
                (func (export "ops")
                      (result i32 i32 i32 i32 i32
                              i64 i64 i64 i64 i64 i64 i64
                              f32 f64 i32)
                    i32.const 0 i32.const 0xffff8080 i32.store
                    i32.const 0 i32.load
                    i32.const 0 i32.load8_s
                    i32.const 0 i32.load8_u
                    i32.const 0 i32.load16_s
                    i32.const 0 i32.load16_u

                    i32.const 8 i64.const 0xffffffff80008080 i64.store
                    i32.const 8 i64.load
                    i32.const 8 i64.load8_s
                    i32.const 8 i64.load8_u
                    i32.const 8 i64.load16_s
                    i32.const 8 i64.load16_u
                    i32.const 8 i64.load32_s
                    i32.const 8 i64.load32_u

                    i32.const 24 f32.const 3.25 f32.store
                    i32.const 24 f32.load
                    i32.const 32 f64.const -9.5 f64.store
                    i32.const 32 f64.load
                    i32.const 48 i32.load)
                (func (export "size") (result i32) memory.size)
                (func (export "grow-fail") (result i32)
                    i32.const 1 memory.grow))"#,
        )
        .expect("wat");
        for export in ["ops", "size", "grow-fail"] {
            let baseline = baseline(&wasm, export, &[]).expect("baseline memory32");
            let folded = native(&wasm, export, &[]).expect("folded memory32");
            assert_values_equal(&baseline, &folded);
        }
    }

    #[test]
    fn memory32_and_memory64_oob_and_effective_address_overflow_match_folded() {
        let memory32 = wat::parse_str(
            r#"(module
                (memory 1)
                (func (export "oob") (result i32)
                    i32.const 65534 i32.load)
                (func (export "offset") (result i32)
                    i32.const -1 i32.load offset=1))"#,
        )
        .expect("memory32 wat");
        for export in ["oob", "offset"] {
            let baseline = baseline(&memory32, export, &[]).expect_err("baseline must trap");
            let BaselineExecError::Wasm(baseline) = baseline else {
                panic!("baseline returned unsupported for {export}");
            };
            let folded = native(&memory32, export, &[]).expect_err("folded must trap");
            assert_eq!(baseline, folded);
        }

        let memory64 = wat::parse_str(
            r#"(module
                (memory i64 1 2)
                (func (export "oob") (result i64)
                    i64.const 65530 i64.load)
                (func (export "overflow") (result i32)
                    i64.const -1 i32.load offset=1))"#,
        )
        .expect("memory64 wat");
        for export in ["oob", "overflow"] {
            let baseline = baseline(&memory64, export, &[]).expect_err("baseline must trap");
            let BaselineExecError::Wasm(baseline) = baseline else {
                panic!("baseline returned unsupported for {export}");
            };
            let folded = native(&memory64, export, &[]).expect_err("folded must trap");
            assert_eq!(baseline, folded);
        }
    }

    #[test]
    fn memory64_size_grow_and_scalar_access_match_folded_statefully() {
        let wasm = wat::parse_str(
            r#"(module
                (memory i64 1 2)
                (func (export "roundtrip") (param i64) (result i64)
                    i64.const 8 local.get 0 i64.store
                    i64.const 8 i64.load)
                (func (export "float") (param f64) (result f64)
                    i64.const 24 local.get 0 f64.store
                    i64.const 24 f64.load)
                (func (export "size") (result i64) memory.size)
                (func (export "grow") (param i64) (result i64)
                    local.get 0 memory.grow))"#,
        )
        .expect("wat");
        let baseline_module = Module::new("baseline-memory64", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&baseline_module).expect("artifact");
        let mut baseline_instance = initialized_interp(baseline_module, &[]);
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut folded = Instance::new(&engine, &wasm, &[]).expect("folded instance");

        for (export, args) in [
            ("roundtrip", vec![Value::I64(0x0123_4567_89ab_cdef)]),
            ("float", vec![Value::F64(-17.25)]),
            ("size", vec![]),
            ("grow", vec![Value::I64(1)]),
            ("size", vec![]),
            ("grow", vec![Value::I64(1)]),
        ] {
            let baseline = baseline_on(&mut baseline_instance, &artifact, export, &args)
                .expect("baseline memory64");
            let native = folded.invoke(export, &args).expect("folded memory64");
            assert_values_equal(&baseline, &native);
        }
    }

    #[test]
    fn active_data_and_shared_memory_import_use_real_instance_state() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "memory" (memory 1 2))
                (data (i32.const 12) "\2a\00\00\00")
                (func (export "run") (param i32) (result i32 i32)
                    i32.const 4 local.get 0 i32.store
                    i32.const 4 i32.load
                    i32.const 12 i32.load))"#,
        )
        .expect("wat");
        let limits = crate::utils::limits::Limits::new(1, Some(2)).expect("limits");
        let config = Config::new();
        let baseline_memory =
            crate::vm::entities::MemInst::new(&config, limits.clone()).expect("baseline memory");
        let folded_memory =
            crate::vm::entities::MemInst::new(&config, limits.clone()).expect("folded memory");
        let baseline_import = crate::Import::memory_with_state(
            "host",
            "memory",
            limits.clone(),
            Some(baseline_memory),
        );
        let folded_import =
            crate::Import::memory_with_state("host", "memory", limits, Some(folded_memory));
        let args = [Value::I32(0x1234_5678)];
        let baseline = baseline_with_imports(&wasm, "run", &args, &[baseline_import])
            .expect("baseline shared memory");
        let folded = native_with_imports(&wasm, "run", &args, &[folded_import])
            .expect("folded shared memory");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(0x1234_5678), Value::I32(42)]);
    }

    #[test]
    fn private_and_shared_i32_globals_match_folded_statefully() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "shared" (global $shared (mut i32)))
                (global $private (mut i32) (i32.const 3))
                (func (export "step") (param i32) (result i32 i32)
                    local.get 0 global.set $private
                    global.get $private
                    global.get $shared i32.const 1 i32.add global.set $shared
                    global.get $shared))"#,
        )
        .expect("wat");
        let baseline_state = crate::vm::entities::GlobalInst::new_raw(10, true, ValueType::I32);
        let folded_state = crate::vm::entities::GlobalInst::new_raw(10, true, ValueType::I32);
        let baseline_import = crate::Import::global_with_state(
            "host",
            "shared",
            crate::vm::imports::ImportedGlobalState {
                global: baseline_state.clone(),
                type_ctx: None,
            },
        );
        let folded_import = crate::Import::global_with_state(
            "host",
            "shared",
            crate::vm::imports::ImportedGlobalState {
                global: folded_state.clone(),
                type_ctx: None,
            },
        );

        let baseline_module = Module::new("baseline-globals", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&baseline_module).expect("artifact");
        let mut baseline_instance = initialized_interp(baseline_module, &[baseline_import]);
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut folded = Instance::new(&engine, &wasm, &[folded_import]).expect("folded instance");
        for input in [7, -9] {
            let args = [Value::I32(input)];
            let baseline = baseline_on(&mut baseline_instance, &artifact, "step", &args)
                .expect("baseline globals");
            let native = folded.invoke("step", &args).expect("folded globals");
            assert_values_equal(&baseline, &native);
        }
        assert_eq!(baseline_state.raw(), folded_state.raw());
        assert_eq!(baseline_state.raw(), 12);
    }

    #[test]
    fn stateful_ops_work_and_imported_call_ops_remain_explicit() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (global $g (mut i32) (i32.const 0))
                (func (export "memory") (result i32) i32.const 0 i32.load)
                (func (export "global") (result i32) global.get $g))"#,
        )
        .expect("wat");
        for export in ["memory", "global"] {
            let baseline = baseline(&wasm, export, &[]).expect("baseline stateful op");
            let folded = native(&wasm, export, &[]).expect("folded stateful op");
            assert_values_equal(&baseline, &folded);
        }

        let imported = wat::parse_str(
            r#"(module
                (import "host" "f" (func $host))
                (func $local (result i32) i32.const 37)
                (func (export "idle"))
                (func (export "local") (result i32) call $local)
                (func (export "run") call $host))"#,
        )
        .expect("wat");
        let host = || crate::Import::func("host", "f", |_caller, _args, _results| Ok(()));
        baseline_with_imports(&imported, "idle", &[], &[host()])
            .expect("an unused import does not block local execution");
        assert_eq!(
            baseline_with_imports(&imported, "local", &[], &[host()])
                .expect("local call after import"),
            [Value::I32(37)]
        );
        let error = baseline_with_imports(&imported, "run", &[], &[host()])
            .expect_err("imported call unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::CALL)),
                feature: "imported call",
                ..
            }
        ));

        let exceptions = wat::parse_str(
            r#"(module
                (tag $e (param i32))
                (func (export "catch") (result i32)
                    (block $handler (result i32)
                        (try_table (result i32) (catch $e $handler)
                            (throw $e (i32.const 7))
                            (i32.const 2))
                        return)
                    return))"#,
        )
        .expect("wat");
        let error = baseline(&exceptions, "catch", &[]).expect_err("EH unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::TRY_TABLE)),
                feature: "MVP opcode",
                ..
            }
        ));
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn warm_recursive_invocation_has_zero_allocations_and_reallocations() {
        let wasm = wat::parse_str(
            r#"(module
                (func $sum (export "sum") (param $value i32) (result i32)
                    local.get $value
                    i32.eqz
                    if (result i32)
                        i32.const 0
                    else
                        local.get $value
                        local.get $value
                        i32.const 1
                        i32.sub
                        call $sum
                        i32.add
                    end))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-allocation", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = initialized_interp(module, &[]);
        let args = [Value::I32(128)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            0,
            &args,
        )
        .expect("frame");
        frame.run().expect("warm-up invocation");
        assert_eq!(frame.values.as_slice(), &[8256]);
        let values_pointer = frame.values.as_ptr();
        let activations_pointer = frame.activations.as_ptr();
        let values_capacity = frame.values.capacity();
        let activations_capacity = frame.activations.capacity();
        let mut output = [Value::I32(0)];
        let (result, census) =
            crate::test_alloc::measure(|| frame.invoke_again(&args, &mut output));
        result.expect("run");
        assert_eq!(census, crate::test_alloc::Census::default());
        assert_eq!(output, [Value::I32(8256)]);
        assert_eq!(frame.values.as_slice(), &[8256]);
        assert_eq!(frame.values.as_ptr(), values_pointer);
        assert_eq!(frame.activations.as_ptr(), activations_pointer);
        assert_eq!(frame.values.capacity(), values_capacity);
        assert_eq!(frame.activations.capacity(), activations_capacity);
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn warm_instance_memory_and_global_invocation_has_zero_allocations() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (global $state (mut i32) (i32.const 0))
                (func (export "run") (param $value i32) (result i32)
                    i32.const 0 local.get $value i32.store
                    local.get $value global.set $state
                    i32.const 0 i32.load
                    global.get $state
                    i32.add))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-instance-allocation", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = initialized_interp(module, &[]);
        let args = [Value::I32(21)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            0,
            &args,
        )
        .expect("frame");
        frame.run().expect("warm-up invocation");
        assert_eq!(frame.values.as_slice(), &[42]);
        let values_pointer = frame.values.as_ptr();
        let activations_pointer = frame.activations.as_ptr();
        let values_capacity = frame.values.capacity();
        let activations_capacity = frame.activations.capacity();
        let mut output = [Value::I32(0)];
        let (result, census) =
            crate::test_alloc::measure(|| frame.invoke_again(&args, &mut output));
        result.expect("warm instance invocation");
        assert_eq!(census, crate::test_alloc::Census::default());
        assert_eq!(output, [Value::I32(42)]);
        assert_eq!(frame.values.as_ptr(), values_pointer);
        assert_eq!(frame.activations.as_ptr(), activations_pointer);
        assert_eq!(frame.values.capacity(), values_capacity);
        assert_eq!(frame.activations.capacity(), activations_capacity);
    }

    #[test]
    fn instance_access_memory_and_global_keep_borrows_scoped() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (global $state (mut i32) (i32.const 5))
                (func (export "run") (param $value i32) (result i32)
                    i32.const 0 local.get $value i32.store
                    i32.const 0 i32.load
                    global.get $state
                    i32.add))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-instance-borrows", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        // Build the real runtime entities but deliberately stop before native
        // linking, so this ownership oracle can execute under Miri.
        let mut instance = built_interp(module, &[]);
        let args = [Value::I32(37)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            0,
            &args,
        )
        .expect("frame");
        frame.run().expect("baseline run");
        assert_eq!(frame.results().expect("results"), [Value::I32(42)]);
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn trapped_recursion_can_restart_without_allocating() {
        let wasm = wat::parse_str(
            r#"(module
                (func $run (export "run")
                      (param $remaining i32) (param $must_trap i32) (result i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        local.get $must_trap
                        if (result i32)
                            unreachable
                        else
                            i32.const 42
                        end
                    else
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        local.get $must_trap
                        call $run
                    end))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-trap-restart", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = initialized_interp(module, &[]);
        let trap_args = [Value::I32(64), Value::I32(1)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            0,
            &trap_args,
        )
        .expect("frame");
        let error = frame.run().expect_err("deep invocation must trap");
        assert!(matches!(
            error,
            BaselineExecError::Wasm(WasmError::Trap("unreachable"))
        ));

        let values_pointer = frame.values.as_ptr();
        let activations_pointer = frame.activations.as_ptr();
        let values_capacity = frame.values.capacity();
        let activations_capacity = frame.activations.capacity();
        let ok_args = [Value::I32(64), Value::I32(0)];
        let mut output = [Value::I32(0)];
        let (result, census) =
            crate::test_alloc::measure(|| frame.invoke_again(&ok_args, &mut output));
        result.expect("restart after trap");
        assert_eq!(census, crate::test_alloc::Census::default());
        assert_eq!(output, [Value::I32(42)]);
        assert_eq!(frame.values.as_ptr(), values_pointer);
        assert_eq!(frame.activations.as_ptr(), activations_pointer);
        assert_eq!(frame.values.capacity(), values_capacity);
        assert_eq!(frame.activations.capacity(), activations_capacity);
    }
}
