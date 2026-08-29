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
use crate::value_type::ValueType;
use crate::Value;

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

pub(super) struct BaselineDriver<'a> {
    module: &'a Module,
    artifact: &'a BaselineArtifact,
}

impl<'a> BaselineDriver<'a> {
    pub(super) const fn new(module: &'a Module, artifact: &'a BaselineArtifact) -> Self {
        Self { module, artifact }
    }

    pub(super) fn invoke_export(
        &self,
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        if self
            .module
            .functions()
            .iter()
            .any(|function| function.spec().is_none())
        {
            return Err(BaselineExecError::Unsupported {
                opcode: None,
                pc: 0,
                feature: "imported functions",
            });
        }
        let function = self
            .module
            .functions()
            .iter()
            .position(|function| function.export_names().iter().any(|name| name == export))
            .ok_or_else(|| WasmError::invalid("baseline MVP export was not found"))?;
        let mut frame = BaselineFrame::new(self.module, self.artifact, function, args)?;
        frame.run()?;
        frame.results()
    }
}

pub(super) struct BaselineFrame<'a> {
    code: &'a [u8],
    function: &'a BaselineFunction,
    all_targets: &'a [ControlTarget],
    br_tables: &'a [BrTableRange],
    result_types: &'a [ValueType],
    pc: usize,
    stp: usize,
    locals: Vec<u64>,
    stack: Vec<u64>,
    finished: bool,
}

impl<'a> BaselineFrame<'a> {
    pub(super) fn new(
        module: &'a Module,
        artifact: &'a BaselineArtifact,
        function_index: usize,
        args: &[Value],
    ) -> Result<Self, BaselineExecError> {
        let function = module
            .functions()
            .get(function_index)
            .ok_or_else(|| WasmError::invalid("baseline MVP function index is out of bounds"))?;
        let spec = function.spec().ok_or(BaselineExecError::Unsupported {
            opcode: None,
            pc: 0,
            feature: "imported function",
        })?;
        let function_type = function.func_type();
        if args.len() != function_type.params().len() {
            return Err(WasmError::invalid("baseline MVP argument count mismatch").into());
        }
        let artifact_function = artifact
            .functions
            .get(function_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| WasmError::invalid("baseline MVP artifact function is missing"))?;

        let local_count = args
            .len()
            .checked_add(spec.locals().len())
            .ok_or_else(|| WasmError::invalid("baseline MVP local count overflow"))?;
        let mut locals = Vec::with_capacity(local_count);
        for (value, &expected) in args.iter().zip(function_type.params()) {
            locals.push(scalar_to_raw(*value, expected)?);
        }
        for &local_type in spec.locals() {
            if !is_mvp_scalar(local_type) {
                return Err(BaselineExecError::Unsupported {
                    opcode: None,
                    pc: 0,
                    feature: "non-scalar local",
                });
            }
            locals.push(0);
        }
        if function_type
            .results()
            .iter()
            .any(|&value_type| !is_mvp_scalar(value_type))
        {
            return Err(BaselineExecError::Unsupported {
                opcode: None,
                pc: 0,
                feature: "non-scalar result",
            });
        }
        let stack_capacity = artifact_function.max_operand_height as usize;
        Ok(Self {
            code: spec.code(),
            function: artifact_function,
            all_targets: &artifact.control_targets,
            br_tables: &artifact.br_tables[artifact_function.br_tables.clone()],
            result_types: function_type.results(),
            pc: 0,
            stp: 0,
            locals,
            stack: Vec::with_capacity(stack_capacity),
            finished: false,
        })
    }

    pub(super) fn run(&mut self) -> Result<(), BaselineExecError> {
        while !self.finished {
            self.step()?;
        }
        Ok(())
    }

    fn results(&self) -> Result<Vec<Value>, BaselineExecError> {
        if self.stack.len() != self.result_types.len() {
            return Err(WasmError::invalid("baseline MVP result stack shape mismatch").into());
        }
        let mut results = Vec::with_capacity(self.result_types.len());
        for (&raw, &value_type) in self.stack.iter().zip(self.result_types) {
            results.push(Value::from_raw(raw, value_type));
        }
        Ok(results)
    }

    fn step(&mut self) -> Result<(), BaselineExecError> {
        let mut cursor = RawOpCursor::at(self.code, self.pc);
        let raw = cursor
            .next()?
            .ok_or_else(|| WasmError::invalid("baseline MVP reached code end without end"))?;
        self.pc = raw.end;
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
                let RawImmediate::I32(value) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP i32.const mismatch").into());
                };
                self.push(value as u32 as u64)?;
            }
            Opcode::I64_CONST => {
                let RawImmediate::I64(value) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP i64.const mismatch").into());
                };
                self.push(value as u64)?;
            }
            Opcode::F32_CONST => {
                let RawImmediate::F32(bits) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP f32.const mismatch").into());
                };
                self.push(bits as u64)?;
            }
            Opcode::F64_CONST => {
                let RawImmediate::F64(bits) = raw.imm else {
                    return Err(WasmError::internal("baseline MVP f64.const mismatch").into());
                };
                self.push(bits)?;
            }
            Opcode::LOCAL_GET => {
                let local = raw_local(raw.imm)?;
                let value = *self
                    .locals
                    .get(local)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))?;
                self.push(value)?;
            }
            Opcode::LOCAL_SET => {
                let local = raw_local(raw.imm)?;
                let value = self.pop()?;
                *self
                    .locals
                    .get_mut(local)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))? =
                    value;
            }
            Opcode::LOCAL_TEE => {
                let local = raw_local(raw.imm)?;
                let value = self.peek()?;
                *self
                    .locals
                    .get_mut(local)
                    .ok_or_else(|| WasmError::invalid("baseline MVP local index overflow"))? =
                    value;
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
                    self.stp += 1;
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
                    self.stp += 1;
                }
            }
            Opcode::BR_TABLE => {
                let selector = self.pop()? as u32 as usize;
                let table = self
                    .br_tables
                    .iter()
                    .find(|table| table.source_pc as usize == raw.start)
                    .copied()
                    .ok_or_else(|| WasmError::invalid("baseline MVP br_table metadata missing"))?;
                let target_count = table.targets_len as usize;
                if target_count == 0 {
                    return Err(WasmError::invalid("baseline MVP empty br_table metadata").into());
                }
                let target_offset = selector.min(target_count - 1);
                let target_index = table.targets_start as usize + target_offset;
                let expected_stp = table
                    .targets_start
                    .checked_sub(self.function.control_targets.start as u32)
                    .ok_or_else(|| WasmError::invalid("baseline MVP br_table base mismatch"))?
                    as usize;
                if self.stp != expected_stp {
                    return Err(WasmError::invalid("baseline MVP br_table pointer mismatch").into());
                }
                let target =
                    self.all_targets.get(target_index).copied().ok_or_else(|| {
                        WasmError::invalid("baseline MVP br_table target missing")
                    })?;
                self.apply_target(target)?;
            }
            Opcode::END => {
                if raw.end == self.code.len() {
                    self.finished = true;
                }
            }
            Opcode::UNREACHABLE => return Err(WasmError::trap("unreachable").into()),
            _ if self.exec_numeric(opcode)? => {}
            _ => return self.unsupported(Some(raw.wasm_op), raw.start, "MVP opcode"),
        }
        Ok(())
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
        let index = self
            .function
            .control_targets
            .start
            .checked_add(self.stp)
            .ok_or_else(|| WasmError::invalid("baseline MVP side-table index overflow"))?;
        if index >= self.function.control_targets.end {
            return Err(WasmError::invalid("baseline MVP side-table pointer overflow").into());
        }
        self.all_targets
            .get(index)
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline MVP control target missing").into())
    }

    fn apply_target(&mut self, target: ControlTarget) -> Result<(), BaselineExecError> {
        let keep = target.keep_arity as usize;
        let base = target.target_stack_height as usize;
        let source = self
            .stack
            .len()
            .checked_sub(keep)
            .ok_or_else(|| WasmError::invalid("baseline MVP branch value underflow"))?;
        let new_len = base
            .checked_add(keep)
            .ok_or_else(|| WasmError::invalid("baseline MVP branch stack overflow"))?;
        if base > source || new_len > self.stack.capacity() {
            return Err(WasmError::invalid("baseline MVP branch stack shape mismatch").into());
        }
        self.stack.copy_within(source..source + keep, base);
        self.stack.truncate(new_len);
        self.pc = target.target_pc as usize;
        self.stp = target.target_stp as usize;
        Ok(())
    }

    fn push(&mut self, value: u64) -> Result<(), BaselineExecError> {
        if self.stack.len() == self.stack.capacity() {
            return Err(WasmError::invalid("baseline MVP operand capacity exhausted").into());
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<u64, BaselineExecError> {
        self.stack
            .pop()
            .ok_or_else(|| WasmError::invalid("baseline MVP operand stack underflow").into())
    }

    fn peek(&self) -> Result<u64, BaselineExecError> {
        self.stack
            .last()
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline MVP operand stack underflow").into())
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

fn raw_local(immediate: RawImmediate<'_>) -> Result<usize, BaselineExecError> {
    let RawImmediate::LocalIndex(local) = immediate else {
        return Err(WasmError::internal("baseline MVP local immediate mismatch").into());
    };
    Ok(local as usize)
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
    use crate::vm::interpreter::predecode::build_baseline_artifact;
    use crate::Instance;
    use std::string::ToString;
    use std::vec::Vec as StdVec;

    fn baseline(
        wasm: &[u8],
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        let module = Module::new("baseline-exec", wasm).expect("module");
        let artifact = build_baseline_artifact(&module).expect("artifact");
        BaselineDriver::new(&module, &artifact).invoke_export(export, args)
    }

    fn native(wasm: &[u8], export: &str, args: &[Value]) -> Result<StdVec<Value>, WasmError> {
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut instance = Instance::new(&engine, wasm, &[]).expect("instance");
        instance.invoke(export, args).map(StdVec::from)
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
    fn unsupported_stateful_and_call_ops_are_explicit() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (global $g (mut i32) (i32.const 0))
                (func $callee)
                (func (export "memory") (result i32) i32.const 0 i32.load)
                (func (export "global") (result i32) global.get $g)
                (func (export "call") call $callee))"#,
        )
        .expect("wat");
        for export in ["memory", "global", "call"] {
            let error = baseline(&wasm, export, &[]).expect_err("must be unsupported");
            assert!(matches!(
                error,
                BaselineExecError::Unsupported {
                    opcode: Some(_),
                    feature: "MVP opcode",
                    ..
                }
            ));
        }

        let imported = wat::parse_str(
            r#"(module
                (import "host" "f" (func))
                (func (export "run")))"#,
        )
        .expect("wat");
        let error = baseline(&imported, "run", &[]).expect_err("imports unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: None,
                feature: "imported functions",
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
                feature: "raw decoder opcode",
                ..
            }
        ));
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn execution_loop_has_zero_allocations_and_reallocations() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "count") (param i32) (result i32) (local i32)
                    block $exit
                        loop $again
                            local.get 0 i32.eqz br_if $exit
                            local.get 1 i32.const 1 i32.add local.set 1
                            local.get 0 i32.const 1 i32.sub local.set 0
                            br $again
                        end
                    end
                    local.get 1))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-allocation", &wasm).expect("module");
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut frame =
            BaselineFrame::new(&module, &artifact, 0, &[Value::I32(10_000)]).expect("frame");
        let locals_pointer = frame.locals.as_ptr();
        let stack_pointer = frame.stack.as_ptr();
        let locals_capacity = frame.locals.capacity();
        let stack_capacity = frame.stack.capacity();
        let (result, census) = crate::test_alloc::measure(|| frame.run());
        result.expect("run");
        assert_eq!(census, crate::test_alloc::Census::default());
        assert_eq!(frame.locals.as_ptr(), locals_pointer);
        assert_eq!(frame.stack.as_ptr(), stack_pointer);
        assert_eq!(frame.locals.capacity(), locals_capacity);
        assert_eq!(frame.stack.capacity(), stack_capacity);
    }
}
