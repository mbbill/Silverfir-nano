//! Raw-Wasm whole-function executor and its standalone test oracle.
//!
//! Validator-enabled interpreter instances may execute preflighted `Raw`
//! functions through [`RawStepper`]. Tests retain a standalone Owned-slot
//! scheduler as an independent differential oracle.

use super::baseline_artifact::{BaselineArtifact, BaselineFunction, BrTableRange, ControlTarget};
#[cfg(test)]
use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::Module;
#[cfg(sf_module_validator)]
use crate::op_decoder::raw_cursor::RawOp;
use crate::op_decoder::raw_cursor::{RawDecodeError, RawImmediate, RawOpCursor};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::utils::limits::Limitable;
use crate::value_type::ValueType;
#[cfg(test)]
use crate::Value;

#[cfg(all(test, sf_module_validator))]
use super::baseline_function_plan::{select_function_plans, FunctionPlanKind};
#[cfg(test)]
use super::exec::{PreparedCall, ResolvedIndirectCall};
use super::InterpInstance;
#[cfg(test)]
use super::InterpInstanceAccess;

#[cfg(test)]
const MAX_BASELINE_CALL_DEPTH: usize = 4096;
#[cfg(test)]
const MAX_BASELINE_ACTIVATIONS: usize = MAX_BASELINE_CALL_DEPTH + 1;
/// Match the hosted interpreter's default two-MiB Wasm stack budget.
#[cfg(test)]
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

#[cfg(test)]
pub(super) struct BaselineDriver<'artifact, 'instance> {
    access: InterpInstanceAccess<'instance>,
    artifact: &'artifact BaselineArtifact,
}

#[cfg(test)]
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

#[cfg(test)]
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
    #[cfg(test)]
    CallIndirect {
        typeidx: u32,
        tableidx: u32,
    },
    GlobalIndex(u32),
    MemoryIndex(u32),
    MemArg {
        offset: u64,
        memidx: u32,
    },
    Other,
}

#[derive(Clone, Copy)]
struct BaselineDecoded {
    wasm_op: WasmOpcode,
    start: usize,
    end: usize,
    imm: BaselineImmediate,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RawFrameState {
    pub(super) pc: usize,
    pub(super) stp: usize,
    pub(super) top: usize,
    pub(super) operand_base: usize,
    pub(super) return_base: usize,
}

pub(super) enum RawStepExit {
    Continue,
    Call { callee: usize, arg_base: usize },
    Return,
}

#[derive(Clone, Copy)]
enum RawOpcodeKind {
    Structural,
    I32Const,
    I64Const,
    F32Const,
    F64Const,
    LocalGet,
    LocalSet,
    LocalTee,
    GlobalGet,
    GlobalSet,
    Drop,
    Select,
    If,
    Branch,
    BrIf,
    BrTable,
    Numeric(Opcode),
    Saturating(OpcodeFC),
    MemoryLoad(Opcode),
    MemoryStore(Opcode),
    MemorySize,
    MemoryGrow,
    Call,
    End,
    Unreachable,
}

fn raw_stepper_opcode_kind(opcode: WasmOpcode) -> Option<RawOpcodeKind> {
    match opcode {
        WasmOpcode::OP(Opcode::NOP | Opcode::BLOCK | Opcode::LOOP) => {
            Some(RawOpcodeKind::Structural)
        }
        WasmOpcode::OP(Opcode::I32_CONST) => Some(RawOpcodeKind::I32Const),
        WasmOpcode::OP(Opcode::I64_CONST) => Some(RawOpcodeKind::I64Const),
        WasmOpcode::OP(Opcode::F32_CONST) => Some(RawOpcodeKind::F32Const),
        WasmOpcode::OP(Opcode::F64_CONST) => Some(RawOpcodeKind::F64Const),
        WasmOpcode::OP(Opcode::LOCAL_GET) => Some(RawOpcodeKind::LocalGet),
        WasmOpcode::OP(Opcode::LOCAL_SET) => Some(RawOpcodeKind::LocalSet),
        WasmOpcode::OP(Opcode::LOCAL_TEE) => Some(RawOpcodeKind::LocalTee),
        WasmOpcode::OP(Opcode::GLOBAL_GET) => Some(RawOpcodeKind::GlobalGet),
        WasmOpcode::OP(Opcode::GLOBAL_SET) => Some(RawOpcodeKind::GlobalSet),
        WasmOpcode::OP(Opcode::DROP) => Some(RawOpcodeKind::Drop),
        WasmOpcode::OP(Opcode::SELECT) => Some(RawOpcodeKind::Select),
        WasmOpcode::OP(Opcode::IF) => Some(RawOpcodeKind::If),
        WasmOpcode::OP(Opcode::ELSE | Opcode::BR | Opcode::RETURN) => Some(RawOpcodeKind::Branch),
        WasmOpcode::OP(Opcode::BR_IF) => Some(RawOpcodeKind::BrIf),
        WasmOpcode::OP(Opcode::BR_TABLE) => Some(RawOpcodeKind::BrTable),
        WasmOpcode::OP(opcode)
            if (Opcode::I32_EQZ as u8..=Opcode::I64_EXTEND32_S as u8).contains(&(opcode as u8)) =>
        {
            Some(RawOpcodeKind::Numeric(opcode))
        }
        WasmOpcode::OP(
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
            | Opcode::I64_LOAD32_U),
        ) => Some(RawOpcodeKind::MemoryLoad(opcode)),
        WasmOpcode::OP(
            opcode @ (Opcode::I32_STORE
            | Opcode::I64_STORE
            | Opcode::F32_STORE
            | Opcode::F64_STORE
            | Opcode::I32_STORE8
            | Opcode::I32_STORE16
            | Opcode::I64_STORE8
            | Opcode::I64_STORE16
            | Opcode::I64_STORE32),
        ) => Some(RawOpcodeKind::MemoryStore(opcode)),
        WasmOpcode::OP(Opcode::MEMORY_SIZE) => Some(RawOpcodeKind::MemorySize),
        WasmOpcode::OP(Opcode::MEMORY_GROW) => Some(RawOpcodeKind::MemoryGrow),
        WasmOpcode::OP(Opcode::CALL) => Some(RawOpcodeKind::Call),
        WasmOpcode::OP(Opcode::END) => Some(RawOpcodeKind::End),
        WasmOpcode::OP(Opcode::UNREACHABLE) => Some(RawOpcodeKind::Unreachable),
        WasmOpcode::FC(opcode) => match opcode {
            OpcodeFC::I32_TRUNC_SAT_F32_S
            | OpcodeFC::I32_TRUNC_SAT_F32_U
            | OpcodeFC::I32_TRUNC_SAT_F64_S
            | OpcodeFC::I32_TRUNC_SAT_F64_U
            | OpcodeFC::I64_TRUNC_SAT_F32_S
            | OpcodeFC::I64_TRUNC_SAT_F32_U
            | OpcodeFC::I64_TRUNC_SAT_F64_S
            | OpcodeFC::I64_TRUNC_SAT_F64_U => Some(RawOpcodeKind::Saturating(opcode)),
            _ => None,
        },
        WasmOpcode::OP(_) | WasmOpcode::FB(_) | WasmOpcode::FD(_) => None,
    }
}

pub(super) fn raw_stepper_supports_opcode(opcode: WasmOpcode) -> bool {
    raw_stepper_opcode_kind(opcode).is_some()
}

fn raw_stepper_i32_global(module: &Module, global: usize) -> Option<bool> {
    module
        .globals()
        .get(global)
        .map(|global| global.value_type() == ValueType::I32)
}

#[cfg(sf_module_validator)]
fn raw_stepper_supports_raw_op(module: &Module, raw: &RawOp<'_>) -> bool {
    if !raw_stepper_supports_opcode(raw.wasm_op) {
        return false;
    }
    match raw.wasm_op {
        WasmOpcode::OP(Opcode::GLOBAL_GET | Opcode::GLOBAL_SET) => {
            let RawImmediate::GlobalIndex(global) = raw.imm else {
                return false;
            };
            raw_stepper_i32_global(module, global as usize) == Some(true)
        }
        _ => true,
    }
}

#[cfg(sf_module_validator)]
pub(super) fn raw_stepper_supports_function(
    module: &Module,
    artifact: &BaselineArtifact,
    function_index: usize,
) -> bool {
    let Some(function) = module.functions().get(function_index) else {
        return false;
    };
    let Some(spec) = function.spec() else {
        return false;
    };
    if artifact
        .functions
        .get(function_index)
        .and_then(Option::as_ref)
        .is_none()
        || function
            .func_type()
            .params()
            .iter()
            .chain(function.func_type().results())
            .chain(spec.locals())
            .any(|&value_type| !is_mvp_scalar(value_type))
    {
        return false;
    }
    let mut cursor = RawOpCursor::new(spec.code());
    loop {
        match cursor.next() {
            Ok(Some(raw)) if raw_stepper_supports_raw_op(module, &raw) => {}
            Ok(Some(_)) | Err(_) => return false,
            Ok(None) => return true,
        }
    }
}

pub(super) enum RawSlots<'a> {
    #[cfg(test)]
    Owned(&'a mut Vec<u64>),
    Fixed {
        slots: &'a mut [u64],
        top: &'a mut usize,
    },
}

impl RawSlots<'_> {
    fn len(&self) -> usize {
        match self {
            #[cfg(test)]
            Self::Owned(values) => values.len(),
            Self::Fixed { top, .. } => **top,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        match self {
            #[cfg(test)]
            Self::Owned(values) => values.get(index).copied(),
            Self::Fixed { slots, top } => (index < **top).then(|| slots[index]),
        }
    }

    fn set(&mut self, index: usize, value: u64) -> Result<(), BaselineExecError> {
        match self {
            #[cfg(test)]
            Self::Owned(values) => values
                .get_mut(index)
                .map(|slot| *slot = value)
                .ok_or_else(|| WasmError::invalid("baseline raw slot overflow").into()),
            Self::Fixed { slots, top } => {
                if index >= **top {
                    return Err(WasmError::invalid("baseline raw slot overflow").into());
                }
                slots[index] = value;
                Ok(())
            }
        }
    }

    fn push(&mut self, value: u64) -> Result<(), BaselineExecError> {
        match self {
            #[cfg(test)]
            Self::Owned(values) => {
                if values.len() == values.capacity() {
                    return Err(WasmError::invalid("baseline raw slot capacity exhausted").into());
                }
                values.push(value);
                Ok(())
            }
            Self::Fixed { slots, top } => {
                let slot = slots
                    .get_mut(**top)
                    .ok_or_else(|| WasmError::trap("call stack exhausted"))?;
                *slot = value;
                **top += 1;
                Ok(())
            }
        }
    }

    fn pop(&mut self, operand_base: usize) -> Result<u64, BaselineExecError> {
        if self.len() == operand_base {
            return Err(WasmError::invalid("baseline raw operand stack underflow").into());
        }
        match self {
            #[cfg(test)]
            Self::Owned(values) => values
                .pop()
                .ok_or_else(|| WasmError::invalid("baseline raw operand stack underflow").into()),
            Self::Fixed { slots, top } => {
                **top -= 1;
                Ok(slots[**top])
            }
        }
    }

    fn copy_within(&mut self, source: core::ops::Range<usize>, destination: usize) {
        match self {
            #[cfg(test)]
            Self::Owned(values) => values.copy_within(source, destination),
            Self::Fixed { slots, top } => slots[..**top].copy_within(source, destination),
        }
    }

    fn truncate(&mut self, len: usize) {
        match self {
            #[cfg(test)]
            Self::Owned(values) => values.truncate(len),
            Self::Fixed { top, .. } => **top = (**top).min(len),
        }
    }
}

trait ScalarExecutor {
    fn push(&mut self, value: u64) -> Result<(), BaselineExecError>;
    fn pop(&mut self) -> Result<u64, BaselineExecError>;

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
}

pub(super) struct RawStepper<'a, 'slots> {
    pub(super) instance: &'a mut InterpInstance,
    pub(super) artifact: &'a BaselineArtifact,
    pub(super) function_index: usize,
    pub(super) frame_base: usize,
    pub(super) state: &'a mut RawFrameState,
    pub(super) slots: RawSlots<'slots>,
    pub(super) acc: &'a mut u64,
}

impl ScalarExecutor for RawStepper<'_, '_> {
    fn push(&mut self, value: u64) -> Result<(), BaselineExecError> {
        let frame_limit = self
            .state
            .operand_base
            .checked_add(self.baseline_function()?.max_operand_height as usize)
            .ok_or_else(|| WasmError::invalid("baseline raw frame limit overflow"))?;
        if self.slots.len() >= frame_limit {
            return Err(WasmError::invalid("baseline raw operand capacity exhausted").into());
        }
        self.slots.push(value)
    }

    fn pop(&mut self) -> Result<u64, BaselineExecError> {
        self.slots.pop(self.state.operand_base)
    }
}

impl RawStepper<'_, '_> {
    fn baseline_function(&self) -> Result<&BaselineFunction, BaselineExecError> {
        self.artifact
            .functions
            .get(self.function_index)
            .and_then(Option::as_ref)
            .ok_or_else(|| WasmError::invalid("baseline raw artifact function is missing").into())
    }

    fn current_target(&self) -> Result<ControlTarget, BaselineExecError> {
        let function = self.baseline_function()?;
        let relative = u32::try_from(self.state.stp)
            .map_err(|_| WasmError::invalid("baseline raw side-table pointer overflow"))?;
        let index = function
            .absolute_stp(relative)
            .ok_or_else(|| WasmError::invalid("baseline raw side-table index overflow"))?;
        if index >= function.control_targets.end {
            return Err(WasmError::invalid("baseline raw side-table pointer overflow").into());
        }
        self.artifact
            .control_targets
            .get(index)
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline raw control target missing").into())
    }

    fn apply_target(&mut self, target: ControlTarget) -> Result<(), BaselineExecError> {
        let max_operand_height = self.baseline_function()?.max_operand_height as usize;
        let keep = target.keep_arity as usize;
        let base = self
            .state
            .operand_base
            .checked_add(target.target_stack_height as usize)
            .ok_or_else(|| WasmError::invalid("baseline raw branch stack overflow"))?;
        let source = self
            .slots
            .len()
            .checked_sub(keep)
            .ok_or_else(|| WasmError::invalid("baseline raw branch value underflow"))?;
        let new_len = base
            .checked_add(keep)
            .ok_or_else(|| WasmError::invalid("baseline raw branch stack overflow"))?;
        let frame_limit = self
            .state
            .operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline raw frame limit overflow"))?;
        if base > source || source < self.state.operand_base || new_len > frame_limit {
            return Err(WasmError::invalid("baseline raw branch stack shape mismatch").into());
        }
        self.slots.copy_within(source..source + keep, base);
        self.slots.truncate(new_len);
        self.baseline_function()?
            .absolute_stp(target.target_stp)
            .ok_or_else(|| WasmError::invalid("baseline raw target side-table overflow"))?;
        self.state.pc = target.target_pc as usize;
        self.state.stp = target.target_stp as usize;
        self.state.top = new_len;
        Ok(())
    }

    fn current_br_table(&self, source_pc: usize) -> Result<BrTableRange, BaselineExecError> {
        let function = self.baseline_function()?;
        self.artifact.br_tables[function.br_tables.clone()]
            .iter()
            .find(|table| table.source_pc as usize == source_pc)
            .copied()
            .ok_or_else(|| WasmError::invalid("baseline raw br_table metadata missing").into())
    }

    fn advance_stp(&mut self) -> Result<(), BaselineExecError> {
        self.state.stp = self
            .state
            .stp
            .checked_add(1)
            .ok_or_else(|| WasmError::invalid("baseline raw side-table pointer overflow"))?;
        Ok(())
    }

    fn peek(&self) -> Result<u64, BaselineExecError> {
        if self.slots.len() == self.state.operand_base {
            return Err(WasmError::invalid("baseline raw operand stack underflow").into());
        }
        self.slots
            .get(self.slots.len() - 1)
            .ok_or_else(|| WasmError::invalid("baseline raw operand stack underflow").into())
    }

    fn memory_is_64(&self, memory: usize) -> Result<bool, BaselineExecError> {
        self.instance
            .module()
            .memories()
            .get(memory)
            .map(|memory| memory.limits().is64)
            .ok_or_else(|| WasmError::invalid("baseline raw memory index overflow").into())
    }

    fn require_i32_global(
        &self,
        global: usize,
        opcode: WasmOpcode,
        pc: usize,
    ) -> Result<(), BaselineExecError> {
        match raw_stepper_i32_global(self.instance.module(), global) {
            Some(true) => Ok(()),
            Some(false) => Err(BaselineExecError::Unsupported {
                opcode: Some(opcode),
                pc,
                feature: "non-i32 global",
            }),
            None => Err(WasmError::invalid("baseline raw global index overflow").into()),
        }
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
            _ => return Err(WasmError::internal("baseline raw load opcode mismatch").into()),
        };
        let loaded = self.instance.mem_load(address, memory, offset, size)?;
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
            _ => return Err(WasmError::internal("baseline raw load opcode mismatch").into()),
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
            _ => return Err(WasmError::internal("baseline raw store opcode mismatch").into()),
        };
        self.instance
            .mem_store(address, memory, offset, size, value)?;
        Ok(())
    }

    pub(super) fn step(&mut self) -> Result<RawStepExit, BaselineExecError> {
        let (decoded, code_len, result_count) = {
            let function = self
                .instance
                .module()
                .functions()
                .get(self.function_index)
                .ok_or_else(|| WasmError::invalid("mixed raw function index overflow"))?;
            let spec = function
                .spec()
                .ok_or_else(|| WasmError::invalid("mixed raw activation targets an import"))?;
            let mut cursor = RawOpCursor::at(spec.code(), self.state.pc);
            let raw = cursor
                .next()?
                .ok_or_else(|| WasmError::invalid("mixed raw reached code end"))?;
            (
                BaselineDecoded {
                    wasm_op: raw.wasm_op,
                    start: raw.start,
                    end: raw.end,
                    imm: copy_immediate(raw.imm),
                },
                spec.code().len(),
                function.func_type().results().len(),
            )
        };
        self.step_decoded(decoded, code_len, result_count)
    }

    fn step_decoded(
        &mut self,
        decoded: BaselineDecoded,
        code_len: usize,
        result_count: usize,
    ) -> Result<RawStepExit, BaselineExecError> {
        self.state.pc = decoded.end;
        if !raw_stepper_supports_opcode(decoded.wasm_op) {
            return Err(BaselineExecError::Unsupported {
                opcode: Some(decoded.wasm_op),
                pc: decoded.start,
                feature: "mixed raw MVP opcode",
            });
        }
        let kind = raw_stepper_opcode_kind(decoded.wasm_op)
            .expect("RawStepper capability and opcode kind must agree");
        match kind {
            RawOpcodeKind::Structural => {}
            RawOpcodeKind::I32Const => {
                let BaselineImmediate::I32(value) = decoded.imm else {
                    return Err(WasmError::internal("mixed raw i32.const mismatch").into());
                };
                self.slots.push(value as u32 as u64)?;
            }
            RawOpcodeKind::I64Const => {
                let BaselineImmediate::I64(value) = decoded.imm else {
                    return Err(WasmError::internal("mixed raw i64.const mismatch").into());
                };
                self.slots.push(value as u64)?;
            }
            RawOpcodeKind::F32Const => {
                let BaselineImmediate::F32(value) = decoded.imm else {
                    return Err(WasmError::internal("mixed raw f32.const mismatch").into());
                };
                self.slots.push(value as u64)?;
            }
            RawOpcodeKind::F64Const => {
                let BaselineImmediate::F64(value) = decoded.imm else {
                    return Err(WasmError::internal("mixed raw f64.const mismatch").into());
                };
                self.slots.push(value)?;
            }
            RawOpcodeKind::LocalGet => {
                let local = raw_local(decoded.imm)?;
                let slot = self
                    .frame_base
                    .checked_add(local)
                    .filter(|&slot| slot < self.state.operand_base)
                    .ok_or_else(|| WasmError::invalid("mixed raw local index overflow"))?;
                let value = self
                    .slots
                    .get(slot)
                    .ok_or_else(|| WasmError::invalid("mixed raw local index overflow"))?;
                self.slots.push(value)?;
            }
            RawOpcodeKind::LocalSet => {
                let local = raw_local(decoded.imm)?;
                let value = self.slots.pop(self.state.operand_base)?;
                let slot = self
                    .frame_base
                    .checked_add(local)
                    .filter(|&slot| slot < self.state.operand_base)
                    .ok_or_else(|| WasmError::invalid("mixed raw local index overflow"))?;
                self.slots.set(slot, value)?;
            }
            RawOpcodeKind::LocalTee => {
                let local = raw_local(decoded.imm)?;
                let value = self.peek()?;
                let slot = self
                    .frame_base
                    .checked_add(local)
                    .filter(|&slot| slot < self.state.operand_base)
                    .ok_or_else(|| WasmError::invalid("mixed raw local index overflow"))?;
                self.slots.set(slot, value)?;
            }
            RawOpcodeKind::GlobalGet => {
                let global = raw_global(decoded.imm)?;
                self.require_i32_global(global, decoded.wasm_op, decoded.start)?;
                let value = self.instance.global_get_for_frame(global);
                self.push(value)?;
            }
            RawOpcodeKind::GlobalSet => {
                let global = raw_global(decoded.imm)?;
                self.require_i32_global(global, decoded.wasm_op, decoded.start)?;
                let value = self.pop()?;
                self.instance.global_set_from_frame(global, value);
            }
            RawOpcodeKind::Drop => {
                self.slots.pop(self.state.operand_base)?;
            }
            RawOpcodeKind::Select => {
                let condition = self.pop()?;
                let otherwise = self.pop()?;
                let selected = self.pop()?;
                self.push(if condition != 0 { selected } else { otherwise })?;
            }
            RawOpcodeKind::If => {
                let condition = self.pop()?;
                let target = self.current_target()?;
                if condition == 0 {
                    self.apply_target(target)?;
                } else {
                    self.advance_stp()?;
                }
            }
            RawOpcodeKind::Branch => {
                let target = self.current_target()?;
                self.apply_target(target)?;
            }
            RawOpcodeKind::BrIf => {
                let condition = self.pop()?;
                let target = self.current_target()?;
                if condition != 0 {
                    self.apply_target(target)?;
                } else {
                    self.advance_stp()?;
                }
            }
            RawOpcodeKind::BrTable => {
                let selector = self.pop()? as u32 as usize;
                let table = self.current_br_table(decoded.start)?;
                let target_count = table.targets_len as usize;
                if target_count == 0 {
                    return Err(WasmError::invalid("baseline raw empty br_table metadata").into());
                }
                let target_offset = selector.min(target_count - 1);
                let target_index = table.targets_start as usize + target_offset;
                let relative_stp = u32::try_from(self.state.stp)
                    .map_err(|_| WasmError::invalid("baseline raw side-table overflow"))?;
                let function = self.baseline_function()?;
                if function.absolute_stp(relative_stp) != Some(table.targets_start as usize) {
                    return Err(WasmError::invalid("baseline raw br_table pointer mismatch").into());
                }
                let target = self
                    .artifact
                    .control_targets
                    .get(target_index)
                    .copied()
                    .ok_or_else(|| WasmError::invalid("baseline raw br_table target missing"))?;
                self.apply_target(target)?;
            }
            RawOpcodeKind::Numeric(opcode) => {
                if !self.exec_numeric(opcode)? {
                    return Err(
                        WasmError::internal("RawStepper numeric capability mismatch").into(),
                    );
                }
            }
            RawOpcodeKind::Saturating(opcode) => {
                if !self.exec_saturating_conversion(opcode)? {
                    return Err(
                        WasmError::internal("RawStepper saturating capability mismatch").into(),
                    );
                }
            }
            RawOpcodeKind::MemoryLoad(opcode) => {
                self.exec_memory_load(opcode, decoded.imm)?;
            }
            RawOpcodeKind::MemoryStore(opcode) => {
                self.exec_memory_store(opcode, decoded.imm)?;
            }
            RawOpcodeKind::MemorySize => {
                let memory = raw_memory(decoded.imm)?;
                self.memory_is_64(memory)?;
                let pages = self.instance.memory_size(memory);
                self.push(pages)?;
            }
            RawOpcodeKind::MemoryGrow => {
                let memory = raw_memory(decoded.imm)?;
                self.memory_is_64(memory)?;
                let delta = self.pop()?;
                let previous = self.instance.memory_grow(memory, delta)?;
                self.push(previous)?;
            }
            RawOpcodeKind::Call => {
                let callee = raw_function(decoded.imm)?;
                let parameter_count = self
                    .instance
                    .module()
                    .functions()
                    .get(callee)
                    .ok_or_else(|| WasmError::invalid("mixed raw callee overflow"))?
                    .func_type()
                    .params()
                    .len();
                let argument = self
                    .slots
                    .len()
                    .checked_sub(parameter_count)
                    .filter(|&base| base >= self.state.operand_base)
                    .ok_or_else(|| WasmError::invalid("mixed raw call argument underflow"))?;
                return Ok(RawStepExit::Call {
                    callee,
                    arg_base: argument - self.frame_base,
                });
            }
            RawOpcodeKind::End if decoded.end == code_len => {
                let expected_top = self
                    .state
                    .operand_base
                    .checked_add(result_count)
                    .ok_or_else(|| WasmError::invalid("mixed raw result range overflow"))?;
                if self.slots.len() != expected_top {
                    return Err(WasmError::invalid("mixed raw result stack shape mismatch").into());
                }
                self.slots.copy_within(
                    self.state.operand_base..expected_top,
                    self.state.return_base,
                );
                let result_end = self.state.return_base + result_count;
                self.slots.truncate(result_end);
                self.state.top = result_end;
                *self.acc = self.slots.get(self.state.return_base).unwrap_or(0);
                return Ok(RawStepExit::Return);
            }
            RawOpcodeKind::End => {}
            RawOpcodeKind::Unreachable => return Err(WasmError::trap("unreachable").into()),
        }
        self.state.top = self.slots.len();
        Ok(RawStepExit::Continue)
    }
}

#[cfg(test)]
pub(super) struct BaselineFrame<'artifact, 'instance> {
    access: InterpInstanceAccess<'instance>,
    artifact: &'artifact BaselineArtifact,
    root_function: usize,
    activations: Vec<BaselineActivation>,
    values: Vec<u64>,
    max_activations: usize,
    max_value_slots: usize,
}

#[cfg(test)]
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
            results.push(self.frame_raw_to_external_value(raw, value_type)?);
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
            *output = self.frame_raw_to_external_value(raw, value_type)?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), BaselineExecError> {
        let (function_index, pc) = {
            let activation = self.current_activation()?;
            (activation.function_index, activation.pc)
        };
        let (raw, code_len, result_count) = self.access.with_instance(|instance| {
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
                function.func_type().results().len(),
            ))
        })??;
        if raw_stepper_supports_opcode(raw.wasm_op) {
            let activation = *self.current_activation()?;
            let mut state = RawFrameState {
                pc: activation.pc,
                stp: activation.stp,
                top: self.values.len(),
                operand_base: activation.operand_base,
                return_base: activation.return_base,
            };
            let mut acc = 0u64;
            let artifact = self.artifact;
            let function_index = activation.function_index;
            let frame_base = activation.locals_base;
            let values = &mut self.values;
            let exit = self.access.with_instance_mut(|instance| {
                RawStepper {
                    instance,
                    artifact,
                    function_index,
                    frame_base,
                    state: &mut state,
                    slots: RawSlots::Owned(values),
                    acc: &mut acc,
                }
                .step_decoded(raw, code_len, result_count)
            })??;
            if let Some(current) = self.activations.last_mut() {
                current.pc = state.pc;
                current.stp = state.stp;
            }
            return match exit {
                RawStepExit::Continue => Ok(()),
                RawStepExit::Call { callee, .. } => self.exec_direct_call(callee, raw.start),
                RawStepExit::Return => {
                    self.activations.pop();
                    Ok(())
                }
            };
        }
        self.current_activation_mut()?.pc = raw.end;
        if matches!(raw.wasm_op, WasmOpcode::FC(_)) {
            return self.unsupported(Some(raw.wasm_op), raw.start, "prefixed opcode");
        }
        let WasmOpcode::OP(opcode) = raw.wasm_op else {
            return self.unsupported(Some(raw.wasm_op), raw.start, "prefixed opcode");
        };
        match opcode {
            Opcode::RETURN_CALL => {
                let callee = raw_function(raw.imm)?;
                self.enter_tail_call(callee, raw.start)?;
            }
            Opcode::CALL_INDIRECT => {
                let (expected_type, table_index) = raw_call_indirect(raw.imm)?;
                self.exec_indirect_call(expected_type, table_index, raw.start)?;
            }
            Opcode::RETURN_CALL_INDIRECT => {
                return self.unsupported(Some(raw.wasm_op), raw.start, "return_call_indirect");
            }
            Opcode::CALL_REF | Opcode::RETURN_CALL_REF => {
                return self.unsupported(Some(raw.wasm_op), raw.start, "call_ref");
            }
            _ => return self.unsupported(Some(raw.wasm_op), raw.start, "MVP opcode"),
        }
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
            validate_baseline_function(
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
            let raw = self.external_value_to_frame_raw(*value, expected)?;
            self.values.push(raw);
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

    fn exec_direct_call(
        &mut self,
        callee: usize,
        source_pc: usize,
    ) -> Result<(), BaselineExecError> {
        let (is_import, parameter_count, result_count) =
            self.access.with_instance(|instance| {
                let function = instance.module().functions().get(callee).ok_or_else(|| {
                    WasmError::invalid("baseline MVP callee index is out of bounds")
                })?;
                let function_type = function.func_type();
                if function_type
                    .params()
                    .iter()
                    .chain(function_type.results())
                    .any(|&value_type| !is_baseline_slot_type(value_type))
                {
                    return Err(BaselineExecError::Unsupported {
                        opcode: Some(WasmOpcode::OP(Opcode::CALL)),
                        pc: source_pc,
                        feature: "unsupported imported call type",
                    });
                }
                Ok::<_, BaselineExecError>((
                    function.spec().is_none(),
                    function_type.params().len(),
                    function_type.results().len(),
                ))
            })??;
        if !is_import {
            return self.enter_call(callee, source_pc);
        }

        let caller_operand_base = self.current_activation()?.operand_base;
        let argument_base = self
            .values
            .len()
            .checked_sub(parameter_count)
            .filter(|&base| base >= caller_operand_base)
            .ok_or_else(|| WasmError::invalid("baseline MVP call argument underflow"))?;
        let call = self.access.with_instance(|instance| {
            instance.prepare_import_call(callee, &self.values, argument_base)
        })??;
        self.run_prepared_call(&call, argument_base, result_count)
    }

    fn exec_indirect_call(
        &mut self,
        expected_type: u32,
        table_index: u32,
        source_pc: usize,
    ) -> Result<(), BaselineExecError> {
        let table_index = table_index as usize;
        let closed = self
            .access
            .with_instance(|instance| instance.call_indirect_table_is_closed(table_index))?;
        if !closed {
            return Err(BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::CALL_INDIRECT)),
                pc: source_pc,
                feature: "open call_indirect table",
            });
        }
        let (parameter_count, result_count) = self.access.with_instance(|instance| {
            let function_type = instance
                .module()
                .types()
                .get_function_type(expected_type)
                .ok_or_else(|| WasmError::trap("indirect call type mismatch"))?;
            if function_type
                .params()
                .iter()
                .chain(function_type.results())
                .any(|&value_type| !is_baseline_slot_type(value_type))
            {
                return Err(BaselineExecError::Unsupported {
                    opcode: Some(WasmOpcode::OP(Opcode::CALL_INDIRECT)),
                    pc: source_pc,
                    feature: "unsupported call_indirect type",
                });
            }
            Ok::<_, BaselineExecError>((
                function_type.params().len(),
                function_type.results().len(),
            ))
        })??;
        let table_element = self.pop()?;
        let caller_operand_base = self.current_activation()?.operand_base;
        let argument_base = self
            .values
            .len()
            .checked_sub(parameter_count)
            .filter(|&base| base >= caller_operand_base)
            .ok_or_else(|| WasmError::invalid("baseline MVP call argument underflow"))?;
        let resolved = self.access.with_instance(|instance| {
            instance.resolve_call_indirect(
                &self.values,
                table_element,
                argument_base,
                table_index,
                expected_type,
            )
        })??;
        match resolved {
            ResolvedIndirectCall::Local(callee) => self.exec_direct_call(callee, source_pc),
            ResolvedIndirectCall::External(call) => {
                self.run_prepared_call(&call, argument_base, result_count)
            }
        }
    }

    fn run_prepared_call(
        &mut self,
        call: &PreparedCall,
        argument_base: usize,
        result_count: usize,
    ) -> Result<(), BaselineExecError> {
        // `call` owns arguments, result types, names, and shared entity
        // handles. No materialized instance reference survives this point, so
        // host callbacks and linked foreign instances may safely re-enter the
        // runtime world before results are localized in a fresh short scope.
        let returned = InterpInstance::run_external_call(&self.access, call)?;
        if returned.len() != result_count {
            return Err(WasmError::invalid("baseline MVP external result count mismatch").into());
        }
        self.values.truncate(argument_base);
        for value in returned {
            self.push(value)?;
        }
        Ok(())
    }

    fn enter_tail_call(
        &mut self,
        callee: usize,
        source_pc: usize,
    ) -> Result<(), BaselineExecError> {
        let (parameter_count, local_count) = self.access.with_instance(|instance| {
            let function =
                instance.module().functions().get(callee).ok_or_else(|| {
                    WasmError::invalid("baseline MVP callee index is out of bounds")
                })?;
            let spec = function.spec().ok_or(BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::RETURN_CALL)),
                pc: source_pc,
                feature: "imported return_call",
            })?;
            let function_type = function.func_type();
            validate_baseline_function(
                function_type.params(),
                function_type.results(),
                spec.locals(),
            )?;
            Ok::<_, BaselineExecError>((function_type.params().len(), spec.locals().len()))
        })??;
        let max_operand_height = self.baseline_function(callee)?.max_operand_height as usize;
        let activation = *self.current_activation()?;
        let argument_base = self
            .values
            .len()
            .checked_sub(parameter_count)
            .filter(|&base| base >= activation.operand_base)
            .ok_or_else(|| WasmError::invalid("baseline MVP call argument underflow"))?;
        let locals_base = activation.return_base;
        let parameters_end = locals_base
            .checked_add(parameter_count)
            .ok_or_else(|| WasmError::invalid("baseline MVP tail parameters overflow"))?;
        let operand_base = parameters_end
            .checked_add(local_count)
            .ok_or_else(|| WasmError::invalid("baseline MVP tail locals overflow"))?;
        let required = operand_base
            .checked_add(max_operand_height)
            .ok_or_else(|| WasmError::invalid("baseline MVP tail frame overflow"))?;
        self.reserve_value_slots(required)?;

        // Tail calls preserve the current activation's caller destination.
        // Move staged arguments over the old frame, discard its locals and
        // operands, then zero only the replacement callee's declared locals.
        self.values
            .copy_within(argument_base..argument_base + parameter_count, locals_base);
        self.values.truncate(parameters_end);
        self.values.resize(operand_base, 0);
        *self.current_activation_mut()? = BaselineActivation {
            function_index: callee,
            pc: 0,
            stp: 0,
            locals_base,
            operand_base,
            return_base: activation.return_base,
        };
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
            validate_baseline_function(
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

    fn external_value_to_frame_raw(
        &self,
        value: Value,
        expected: ValueType,
    ) -> Result<u64, BaselineExecError> {
        let type_matches = match (value, expected) {
            (Value::I32(_), ValueType::I32)
            | (Value::I64(_), ValueType::I64)
            | (Value::F32(_), ValueType::F32)
            | (Value::F64(_), ValueType::F64)
            | (Value::Ref(_, _), ValueType::Ref(_)) => true,
            _ => false,
        };
        if !type_matches || !is_baseline_slot_type(expected) {
            return Err(WasmError::invalid("baseline MVP argument type mismatch").into());
        }
        let raw = self.access.with_instance(|instance| {
            let value = instance.localize_value_for_type(value, expected);
            super::value_to_raw_for_interp(&value)
        })??;
        Ok(raw)
    }

    fn frame_raw_to_external_value(
        &self,
        raw: u64,
        value_type: ValueType,
    ) -> Result<Value, BaselineExecError> {
        let value = super::raw_to_value_for_interp(raw, value_type)?;
        self.access
            .with_instance(|instance| instance.absolutize_value_for_type(value, value_type))
            .map_err(Into::into)
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
        #[cfg(test)]
        RawImmediate::CallIndirect { typeidx, tableidx } => {
            BaselineImmediate::CallIndirect { typeidx, tableidx }
        }
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

#[cfg(test)]
fn raw_call_indirect(immediate: BaselineImmediate) -> Result<(u32, u32), BaselineExecError> {
    let BaselineImmediate::CallIndirect { typeidx, tableidx } = immediate else {
        return Err(WasmError::internal("baseline MVP call_indirect immediate mismatch").into());
    };
    Ok((typeidx, tableidx))
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

#[cfg(test)]
fn validate_baseline_function(
    params: &[ValueType],
    results: &[ValueType],
    locals: &[ValueType],
) -> Result<(), BaselineExecError> {
    if params
        .iter()
        .chain(results)
        .any(|&value_type| !is_baseline_slot_type(value_type))
    {
        return Err(BaselineExecError::Unsupported {
            opcode: None,
            pc: 0,
            feature: "unsupported function boundary type",
        });
    }
    if locals.iter().any(|&value_type| !is_mvp_scalar(value_type)) {
        return Err(BaselineExecError::Unsupported {
            opcode: None,
            pc: 0,
            feature: "reference locals",
        });
    }
    Ok(())
}

#[cfg(test)]
fn is_baseline_slot_type(value_type: ValueType) -> bool {
    is_mvp_scalar(value_type) || matches!(value_type, ValueType::Ref(_))
}

fn is_mvp_scalar(value_type: ValueType) -> bool {
    matches!(
        value_type,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::value_type::RefType;
    use crate::vm::engine::{Engine, Tier};
    use crate::vm::interpreter::baseline_artifact::artifact_test_guard;
    use crate::vm::interpreter::predecode::build_baseline_artifact;
    use crate::vm::link::LinkRegistry;
    use crate::vm::value::RefValue;
    use crate::Instance;
    use core::cell::Cell;
    use std::rc::Rc;
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
        let instance = Instance::from_module(&engine, module, imports).expect("instance");
        baseline_on_instance(&instance, &artifact, export, args)
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
        #[cfg(sf_module_validator)]
        let baseline =
            super::super::ValidatedBaselinePlan::validate(&module).expect("validated baseline");
        InterpInstance::build(
            &engine,
            module,
            #[cfg(sf_module_validator)]
            baseline,
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

    fn baseline_on_instance(
        instance: &Instance,
        artifact: &BaselineArtifact,
        export: &str,
        args: &[Value],
    ) -> Result<Vec<Value>, BaselineExecError> {
        let token = instance.checkout_interp_for_test()?;
        BaselineDriver::new(InterpInstanceAccess::checked_out(token), artifact)
            .invoke_export(export, args)
    }

    fn baseline_with_peak_activations(
        wasm: &[u8],
        export: &str,
        args: &[Value],
    ) -> Result<(Vec<Value>, usize), BaselineExecError> {
        let _guard = artifact_test_guard();
        let module = Module::new("baseline-tail-depth", wasm).expect("module");
        let function = module
            .functions()
            .iter()
            .position(|function| function.export_names().iter().any(|name| name == export))
            .expect("export");
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = built_interp(module, &[]);
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            function,
            args,
        )?;
        let mut peak = frame.activations.len();
        while !frame.activations.is_empty() {
            frame.step()?;
            peak = peak.max(frame.activations.len());
        }
        Ok((frame.results()?, peak))
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
    fn raw_stepper_owned_and_fixed_slots_share_one_semantics() {
        let wasm = wat::parse_str(
            r#"(module
                (func (export "add") (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.add))"#,
        )
        .expect("wat");
        let module = Module::new("raw-slots-oracle", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        assert!(matches!(
            artifact.function_eligibility(&module, 0),
            Some(
                crate::vm::interpreter::baseline_artifact::BaselineFunctionEligibility::Baseline(_,)
            )
        ));
        let mut instance = built_interp(module, &[]);

        let mut owned = Vec::with_capacity(8);
        owned.push(40);
        let mut owned_state = RawFrameState {
            pc: 0,
            stp: 0,
            top: 1,
            operand_base: 1,
            return_base: 0,
        };
        let mut owned_acc = 0;
        loop {
            let exit = RawStepper {
                instance: &mut instance,
                artifact: &artifact,
                function_index: 0,
                frame_base: 0,
                state: &mut owned_state,
                slots: RawSlots::Owned(&mut owned),
                acc: &mut owned_acc,
            }
            .step()
            .expect("owned step");
            match exit {
                RawStepExit::Continue => {}
                RawStepExit::Return => break,
                RawStepExit::Call { callee, arg_base } => {
                    panic!("unexpected call {callee} at {arg_base}")
                }
            }
        }

        let mut fixed = [0u64; 8];
        fixed[0] = 40;
        let mut fixed_top = 1usize;
        let mut fixed_state = RawFrameState {
            pc: 0,
            stp: 0,
            top: 1,
            operand_base: 1,
            return_base: 0,
        };
        let mut fixed_acc = 0;
        loop {
            let exit = RawStepper {
                instance: &mut instance,
                artifact: &artifact,
                function_index: 0,
                frame_base: 0,
                state: &mut fixed_state,
                slots: RawSlots::Fixed {
                    slots: &mut fixed,
                    top: &mut fixed_top,
                },
                acc: &mut fixed_acc,
            }
            .step()
            .expect("fixed step");
            match exit {
                RawStepExit::Continue => {}
                RawStepExit::Return => break,
                RawStepExit::Call { callee, arg_base } => {
                    panic!("unexpected call {callee} at {arg_base}")
                }
            }
        }

        assert_eq!(owned.as_slice(), &[42]);
        assert_eq!(&fixed[..fixed_top], &[42]);
        assert_eq!((owned_acc, fixed_acc), (42, 42));
        assert_eq!((owned_state.stp, fixed_state.stp), (0, 0));
    }

    #[cfg(sf_module_validator)]
    #[test]
    fn whole_function_mixed_direct_calls_cross_both_modes() {
        let wasm = wat::parse_str(
            r#"(module
                (func $raw_add (param i32) (result i32)
                    local.get 0 i32.const 2 i32.add)
                (func $fold_mul (param i32) (result i32)
                    local.get 0 i32.const 3 i32.mul)
                (func (export "fold_to_raw") (param i32) (result i32)
                    local.get 0 call $raw_add i32.const 5 i32.add)
                (func (export "raw_to_fold") (param i32) (result i32)
                    local.get 0 call $fold_mul i32.const 7 i32.add)
                (func $raw_pair (param i32) (result i32 i32)
                    local.get 0
                    local.get 0 i32.const 1 i32.add)
                (func $fold_pair (param i32) (result i32 i32)
                    local.get 0
                    local.get 0 i32.const 4 i32.add)
                (func (export "fold_to_raw_pair") (param i32) (result i32)
                    local.get 0 call $raw_pair i32.add)
                (func (export "raw_to_fold_pair") (param i32) (result i32)
                    local.get 0 call $fold_pair i32.add))"#,
        )
        .expect("wat");
        let module = Module::new("whole-mixed-direct", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut mixed = built_interp(module, &[]);
        mixed
            .set_whole_function_plan_for_test(
                artifact,
                vec![
                    FunctionPlanKind::Raw,
                    FunctionPlanKind::FullFold,
                    FunctionPlanKind::FullFold,
                    FunctionPlanKind::Raw,
                    FunctionPlanKind::Raw,
                    FunctionPlanKind::FullFold,
                    FunctionPlanKind::FullFold,
                    FunctionPlanKind::Raw,
                ],
            )
            .expect("mixed plan");
        let mut mixed = InterpInstance::initialize(mixed)
            .map_err(|(_, error)| error)
            .expect("mixed initialize");
        assert!(mixed.whole_direct_call_is_slow_for_test(2, 0));

        for (export, input) in [
            ("fold_to_raw", 11u64),
            ("raw_to_fold", 13u64),
            ("fold_to_raw_pair", 17u64),
            ("raw_to_fold_pair", 19u64),
        ] {
            let function = mixed.find_export(export).expect("mixed export");
            let mut result = [0u64; 1];
            mixed
                .invoke(function, &[input], &mut result)
                .expect("mixed invoke");
            let folded =
                native(&wasm, export, &[Value::I32(input as i32)]).expect("folded differential");
            assert_eq!(result[0], folded[0].to_raw());
        }

        #[cfg(not(feature = "memprof"))]
        {
            let fold_to_raw = mixed.find_export("fold_to_raw").expect("mixed export");
            let raw_to_fold = mixed.find_export("raw_to_fold").expect("mixed export");
            let mut left = [0u64; 1];
            let mut right = [0u64; 1];
            let (outcome, census) = crate::test_alloc::measure(|| {
                mixed.invoke(fold_to_raw, &[23], &mut left)?;
                mixed.invoke(raw_to_fold, &[29], &mut right)
            });
            outcome.expect("warm mixed invoke");
            assert_eq!(census, crate::test_alloc::Census::default());
            assert_eq!((left[0], right[0]), (30, 94));
        }
    }

    #[cfg(sf_module_validator)]
    #[test]
    fn whole_plan_preflights_selector_and_forced_raw_masks() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (data (i32.const 0) "\07\00\00\00")
                (global $state (mut i32) (i32.const 0))
                (func $pure (export "pure") (param i32) (result i32)
                    local.get 0 i32.const 2 i32.add)
                (func (export "with_if") (param i32) (result i32)
                    local.get 0 i32.eqz
                    if (result i32)
                        i32.const 7
                    else
                        local.get 0
                    end)
                (func (export "explicit") (result i32)
                    i32.const 9 return)
                (func (export "cross") (param i32) (result i32)
                    local.get 0 call $pure
                    i32.eqz
                    if (result i32)
                        i32.const 1
                    else
                        i32.const 2
                    end)
                (func (export "memory") (result i32)
                    i32.const 0 i32.load)
                (func (export "global") (param i32) (result i32)
                    local.get 0 global.set $state
                    global.get $state))"#,
        )
        .expect("wat");
        let module = Module::new("whole-plan-selector", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let selected = select_function_plans(&module, &artifact).expect("selector plan");
        assert_eq!(selected, [FunctionPlanKind::Raw; 6]);

        let mut selected_instance = built_interp(module, &[]);
        selected_instance
            .set_whole_function_plan_for_test(artifact, selected)
            .expect("install selector plan");
        assert_eq!(
            (0..6)
                .map(|index| selected_instance.whole_function_mode(index))
                .collect::<StdVec<_>>(),
            [FunctionPlanKind::Raw; 6]
        );
        let mut selected_instance = InterpInstance::initialize(selected_instance)
            .map_err(|(_, error)| error)
            .expect("initialize selected mixed instance");
        for (export, args, expected) in [
            ("pure", &[Value::I32(40)][..], Value::I32(42)),
            ("with_if", &[Value::I32(0)][..], Value::I32(7)),
            ("explicit", &[][..], Value::I32(9)),
            ("cross", &[Value::I32(5)][..], Value::I32(2)),
            ("memory", &[][..], Value::I32(7)),
            ("global", &[Value::I32(11)][..], Value::I32(11)),
        ] {
            let function = selected_instance.find_export(export).expect("export");
            let raw_args = args.iter().map(Value::to_raw).collect::<StdVec<_>>();
            let mut result = [0u64; 1];
            selected_instance
                .invoke(function, &raw_args, &mut result)
                .expect("preflighted mixed invoke");
            assert_eq!(result[0], expected.to_raw(), "{export}");
        }

        #[cfg(not(feature = "memprof"))]
        {
            let memory = selected_instance.find_export("memory").expect("export");
            let global = selected_instance.find_export("global").expect("export");
            let mut memory_result = [0u64; 1];
            let mut global_result = [0u64; 1];
            let (outcome, census) = crate::test_alloc::measure(|| {
                selected_instance.invoke(memory, &[], &mut memory_result)?;
                selected_instance.invoke(global, &[13], &mut global_result)
            });
            outcome.expect("warm stateful mixed invoke");
            assert_eq!(census, crate::test_alloc::Census::default());
            assert_eq!((memory_result[0], global_result[0]), (7, 13));
        }

        let forced_module = Module::new("whole-plan-forced", &wasm).expect("module");
        let forced_artifact = build_baseline_artifact(&forced_module).expect("artifact");
        let mut forced = built_interp(forced_module, &[]);
        forced
            .set_whole_function_plan_for_test(forced_artifact, vec![FunctionPlanKind::Raw; 6])
            .expect("forced plan must be safely promoted");
        for index in 0..6 {
            assert_eq!(forced.whole_function_mode(index), FunctionPlanKind::Raw);
        }
    }

    #[cfg(sf_module_validator)]
    #[test]
    fn forced_raw_promotes_non_i32_globals_but_keeps_i32_globals() {
        let wasm = wat::parse_str(
            r#"(module
                (global $wide (mut i64) (i64.const 5))
                (global $narrow (mut i32) (i32.const 7))
                (func (export "wide") (param i64) (result i64)
                    local.get 0 global.set $wide
                    global.get $wide)
                (func (export "narrow") (param i32) (result i32)
                    local.get 0 global.set $narrow
                    global.get $narrow))"#,
        )
        .expect("wat");
        let module = Module::new("whole-plan-global-types", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = built_interp(module, &[]);
        instance
            .set_whole_function_plan_for_test(artifact, vec![FunctionPlanKind::Raw; 2])
            .expect("forced plan must be safely promoted");
        assert_eq!(instance.whole_function_mode(0), FunctionPlanKind::FullFold);
        assert_eq!(instance.whole_function_mode(1), FunctionPlanKind::Raw);

        let mut instance = InterpInstance::initialize(instance)
            .map_err(|(_, error)| error)
            .expect("initialize mixed instance");
        for (export, input) in [
            ("wide", 0x0123_4567_89ab_cdef),
            ("narrow", 0x0000_0000_89ab_cdef),
        ] {
            let function = instance.find_export(export).expect("export");
            let mut result = [0u64; 1];
            instance
                .invoke(function, &[input], &mut result)
                .expect("global roundtrip");
            assert_eq!(result[0], input, "{export}");
        }
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
    fn direct_tail_wrapper_and_multivalue_results_match_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $target (func (param i32 i64) (result i32 i64)))
                (type $wrapper (func (param i32) (result i32 i64)))
                (func $pair (type $target) (param $value i32) (param $wide i64)
                    (result i32 i64) (local $zero i32)
                    local.get $value
                    local.get $zero
                    i32.add
                    local.get $wide)
                (func $wrapper (export "wrapper") (type $wrapper) (param $value i32)
                    (result i32 i64) (local i64)
                    local.get $value
                    local.get $value
                    i64.extend_i32_s
                    return_call $pair)
                (func (export "nested") (type $wrapper) (param $value i32)
                    (result i32 i64) (local $wide i64)
                    i32.const 777
                    local.get $value
                    call $wrapper
                    local.set $wide
                    i32.add
                    local.get $wide))"#,
        )
        .expect("wat");
        for input in [-17, 0, 31] {
            let args = [Value::I32(input)];
            let (baseline, peak) =
                baseline_with_peak_activations(&wasm, "wrapper", &args).expect("baseline tail");
            let folded = native(&wasm, "wrapper", &args).expect("folded tail");
            assert_values_equal(&baseline, &folded);
            assert_eq!(peak, 1, "acyclic tail wrapper pushed an activation");

            let (baseline, peak) =
                baseline_with_peak_activations(&wasm, "nested", &args).expect("nested tail");
            let folded = native(&wasm, "nested", &args).expect("folded nested tail");
            assert_values_equal(&baseline, &folded);
            assert_eq!(peak, 2, "nested tail call failed to reuse return_base");
        }
    }

    #[test]
    fn self_and_mutual_tail_calls_keep_one_activation_beyond_depth_limit() {
        let self_tail = wat::parse_str(
            r#"(module
                (func $count (export "count") (param $remaining i32) (result i32)
                    (local $must_reset i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        local.get $must_reset
                    else
                        i32.const 99
                        local.set $must_reset
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        return_call $count
                    end))"#,
        )
        .expect("self-tail wat");
        let args = [Value::I32(5000)];
        let (baseline, peak) =
            baseline_with_peak_activations(&self_tail, "count", &args).expect("baseline self tail");
        let folded = native(&self_tail, "count", &args).expect("folded self tail");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(0)], "tail call did not zero locals");
        assert_eq!(peak, 1, "self tail recursion grew activation depth");

        let mutual_tail = wat::parse_str(
            r#"(module
                (type $count (func (param i32) (result i32)))
                (func $even (export "count") (type $count) (param $remaining i32) (result i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        i32.const 42
                    else
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        return_call $odd
                    end)
                (func $odd (type $count) (param $remaining i32) (result i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        i32.const 41
                    else
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        return_call $even
                    end))"#,
        )
        .expect("mutual-tail wat");
        let args = [Value::I32(5001)];
        let (baseline, peak) = baseline_with_peak_activations(&mutual_tail, "count", &args)
            .expect("baseline mutual tail");
        let folded = native(&mutual_tail, "count", &args).expect("folded mutual tail");
        assert_values_equal(&baseline, &folded);
        assert_eq!(peak, 1, "mutual tail recursion grew activation depth");
    }

    #[test]
    fn tail_replacement_runs_without_native_linking() {
        let wasm = wat::parse_str(
            r#"(module
                (func $count (export "count") (param $remaining i32) (result i32)
                    (local $must_reset i32)
                    local.get $remaining
                    i32.eqz
                    if (result i32)
                        local.get $must_reset
                    else
                        i32.const 99
                        local.set $must_reset
                        local.get $remaining
                        i32.const 1
                        i32.sub
                        return_call $count
                    end))"#,
        )
        .expect("wat");
        let (result, peak) = baseline_with_peak_activations(&wasm, "count", &[Value::I32(32)])
            .expect("baseline tail without native linking");
        assert_eq!(result, [Value::I32(0)]);
        assert_eq!(peak, 1);
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
    fn stateful_local_and_imported_calls_work_while_eh_remains_explicit() {
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
        baseline_with_imports(&imported, "run", &[], &[host()]).expect("direct imported call");

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

    #[test]
    fn inert_typed_import_and_local_index_after_import_do_not_dispatch_host() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "host" "typed" (func $typed (type $unary)))
                (func $local (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 1 i32.add)
                (func (export "idle") (result i32) i32.const 7)
                (func (export "local") (param i32) (result i32)
                    local.get 0 call $local))"#,
        )
        .expect("wat");
        let calls = Rc::new(Cell::new(0usize));
        let callback_calls = Rc::clone(&calls);
        let import = crate::Import::func_typed(
            "host",
            "typed",
            move |_caller, args, results| {
                callback_calls.set(callback_calls.get() + 1);
                results[0] = args[0];
                Ok(())
            },
            crate::FunctionType::new(
                crate::collections::vec![ValueType::I32],
                crate::collections::vec![ValueType::I32],
            ),
        );
        assert_eq!(
            baseline_with_imports(&wasm, "idle", &[], &[import]).expect("idle baseline"),
            [Value::I32(7)]
        );
        assert_eq!(calls.get(), 0);

        let calls = Rc::new(Cell::new(0usize));
        let callback_calls = Rc::clone(&calls);
        let import = crate::Import::func_typed(
            "host",
            "typed",
            move |_caller, args, results| {
                callback_calls.set(callback_calls.get() + 1);
                results[0] = args[0];
                Ok(())
            },
            crate::FunctionType::new(
                crate::collections::vec![ValueType::I32],
                crate::collections::vec![ValueType::I32],
            ),
        );
        assert_eq!(
            baseline_with_imports(&wasm, "local", &[Value::I32(41)], &[import])
                .expect("local after import"),
            [Value::I32(42)]
        );
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn imported_host_multivalue_results_match_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $host (func (param i32 i64) (result i64 i32)))
                (import "host" "multi" (func $multi (type $host)))
                (func (export "run") (param i32 i64) (result i64 i32)
                    local.get 0 local.get 1 call $multi))"#,
        )
        .expect("wat");
        let function_type = crate::FunctionType::new(
            crate::collections::vec![ValueType::I32, ValueType::I64],
            crate::collections::vec![ValueType::I64, ValueType::I32],
        );
        let host = || {
            crate::Import::func_typed(
                "host",
                "multi",
                |_caller, args, results| {
                    let [Value::I32(lhs), Value::I64(rhs)] = args else {
                        return Err(WasmError::internal("host multivalue arguments"));
                    };
                    results[0] = Value::I64(rhs.wrapping_add(*lhs as i64));
                    results[1] = Value::I32(lhs.wrapping_sub(*rhs as i32));
                    Ok(())
                },
                function_type.clone(),
            )
        };
        let args = [Value::I32(17), Value::I64(1_000_000_003)];
        let baseline = baseline_with_imports(&wasm, "run", &args, &[host()])
            .expect("baseline imported multivalue");
        let folded = native_with_imports(&wasm, "run", &args, &[host()])
            .expect("folded imported multivalue");
        assert_values_equal(&baseline, &folded);
    }

    #[test]
    fn imported_host_trap_matches_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "trap" (func $trap))
                (func (export "run") call $trap))"#,
        )
        .expect("wat");
        let host = || {
            crate::Import::func("host", "trap", |_caller, _args, _results| {
                Err(WasmError::trap("host trap sentinel"))
            })
        };
        let baseline =
            baseline_with_imports(&wasm, "run", &[], &[host()]).expect_err("baseline host trap");
        let BaselineExecError::Wasm(baseline) = baseline else {
            panic!("baseline host trap became unsupported");
        };
        let folded =
            native_with_imports(&wasm, "run", &[], &[host()]).expect_err("folded host trap");
        assert_eq!(baseline, folded);
    }

    #[test]
    fn imported_return_call_is_unsupported_before_callback() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "tail" (func $tail))
                (func (export "run")
                    return_call $tail))"#,
        )
        .expect("wat");
        let calls = Rc::new(Cell::new(0usize));
        let callback_calls = Rc::clone(&calls);
        let import = crate::Import::func("host", "tail", move |_caller, _args, _results| {
            callback_calls.set(callback_calls.get() + 1);
            Ok(())
        });
        let error = baseline_with_imports(&wasm, "run", &[], &[import])
            .expect_err("imported tail call must be rejected before dispatch");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::RETURN_CALL)),
                feature: "imported return_call",
                ..
            }
        ));
        assert_eq!(
            calls.get(),
            0,
            "imported tail callback ran before rejection"
        );
    }

    #[test]
    fn host_throw_after_side_effect_is_terminal_and_never_fallback_eligible() {
        let wasm = wat::parse_str(
            r#"(module
                (type $exception (func (param i32)))
                (import "host" "exception" (tag $e (type $exception)))
                (import "host" "throw" (func $throw))
                (func (export "run") call $throw))"#,
        )
        .expect("wat");
        let tag_type = crate::FunctionType::new(
            crate::collections::vec![ValueType::I32],
            crate::collections::vec![],
        );
        let (tag_import, tag) = crate::Import::tag_typed_with_handle("host", "exception", tag_type);
        let effects = Rc::new(Cell::new(0usize));
        let callback_effects = Rc::clone(&effects);
        let function_import =
            crate::Import::func("host", "throw", move |_caller, _args, _results| {
                callback_effects.set(callback_effects.get() + 1);
                Err(WasmError::HostThrow {
                    tag,
                    args: vec![Value::I32(7)],
                })
            });
        let error = baseline_with_imports(&wasm, "run", &[], &[tag_import, function_import])
            .expect_err("host throw must terminate baseline execution");
        let BaselineExecError::Wasm(WasmError::HostThrow {
            tag: actual_tag,
            args,
        }) = error
        else {
            panic!("post-effect host throw became fallback-eligible: {error}");
        };
        assert_eq!(actual_tag, tag);
        assert_eq!(args, [Value::I32(7)]);
        assert_eq!(effects.get(), 1, "host side effect ran more than once");
    }

    #[test]
    fn external_exception_after_side_effect_is_terminal_and_preserved() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "throw" (func $throw))
                (func (export "run") call $throw))"#,
        )
        .expect("wat");
        let effects = Rc::new(Cell::new(0usize));
        let callback_effects = Rc::clone(&effects);
        let exn = RefValue::new(123);
        let tag = crate::vm::tag::TagIdentity::mint_fresh();
        let import = crate::Import::func("host", "throw", move |_caller, _args, _results| {
            callback_effects.set(callback_effects.get() + 1);
            Err(WasmError::Exception {
                exn,
                tag,
                module_tag_name: Some("sentinel".to_string()),
            })
        });
        let error = baseline_with_imports(&wasm, "run", &[], &[import])
            .expect_err("external exception must terminate baseline execution");
        assert_eq!(
            match error {
                BaselineExecError::Wasm(error) => error,
                error => panic!("external exception became fallback-eligible: {error}"),
            },
            WasmError::Exception {
                exn,
                tag,
                module_tag_name: Some("sentinel".to_string()),
            }
        );
        assert_eq!(effects.get(), 1, "external side effect ran more than once");
    }

    #[test]
    fn imported_host_caller_memory_access_matches_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $touch (func (param i32) (result i32)))
                (import "host" "touch" (func $touch (type $touch)))
                (memory 1)
                (data (i32.const 0) "\07")
                (func (export "run") (param i32) (result i32)
                    local.get 0 call $touch
                    i32.const 0 i32.load8_u
                    i32.add))"#,
        )
        .expect("wat");
        let function_type = crate::FunctionType::new(
            crate::collections::vec![ValueType::I32],
            crate::collections::vec![ValueType::I32],
        );
        let host = || {
            crate::Import::func_typed(
                "host",
                "touch",
                |caller, args, results| {
                    let Value::I32(replacement) = args[0] else {
                        return Err(WasmError::internal("host memory argument"));
                    };
                    let memory = caller
                        .memory_mut()
                        .ok_or_else(|| WasmError::trap("host memory missing"))?;
                    let previous = memory[0];
                    memory[0] = replacement as u8;
                    results[0] = Value::I32(previous as i32);
                    Ok(())
                },
                function_type.clone(),
            )
        };
        let args = [Value::I32(12)];
        let baseline =
            baseline_with_imports(&wasm, "run", &args, &[host()]).expect("baseline caller memory");
        let folded =
            native_with_imports(&wasm, "run", &args, &[host()]).expect("folded caller memory");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(19)]);
    }

    #[test]
    fn imported_host_funcref_argument_and_result_cross_absolute_boundary() {
        let wasm = wat::parse_str(
            r#"(module
                (type $echo (func (param funcref) (result funcref)))
                (import "host" "echo" (func $echo (type $echo)))
                (func $target (export "target"))
                (func (export "run") (param funcref) (result funcref)
                    local.get 0 call $echo))"#,
        )
        .expect("wat");
        let expected = Rc::new(Cell::new(None));
        let callback_expected = Rc::clone(&expected);
        let import = crate::Import::func_typed(
            "host",
            "echo",
            move |_caller, args, results| {
                let Value::Ref(handle, _) = args[0] else {
                    return Err(WasmError::internal("host funcref argument"));
                };
                if Some(handle) != callback_expected.get() {
                    return Err(WasmError::trap("host funcref was not absolute"));
                }
                results[0] = args[0];
                Ok(())
            },
            crate::FunctionType::new(
                crate::collections::vec![ValueType::Ref(RefType::funcref())],
                crate::collections::vec![ValueType::Ref(RefType::funcref())],
            ),
        );
        let module = Module::new("baseline-funcref-import", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut instance = Instance::from_module(&engine, module, &[import]).expect("instance");
        let handle = instance.function_handle_at(1).expect("target identity");
        expected.set(Some(handle));
        let args = [Value::Ref(handle, RefType::funcref())];
        let baseline = baseline_on_instance(&instance, &artifact, "run", &args)
            .expect("baseline funcref import");
        let folded = instance
            .invoke("run", &args)
            .expect("folded funcref import");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, args);
    }

    #[test]
    fn linked_foreign_interpreter_function_matches_folded() {
        let provider_wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (func (export "plus") (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 40 i32.add))"#,
        )
        .expect("provider wat");
        let importer_wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "provider" "plus" (func $plus (type $unary)))
                (func (export "run") (param i32) (result i32)
                    local.get 0 call $plus
                    i32.const 2 i32.add))"#,
        )
        .expect("importer wat");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut world = crate::RuntimeWorld::new();
        let provider_module = Module::new("linked-provider", &provider_wasm).expect("module");
        let provider = world
            .instantiate(&engine, provider_module, &[])
            .expect("provider instance");
        let provider_instance = world.instance(provider).expect("provider facade");
        let handle = provider_instance
            .function_handle_at(0)
            .expect("provider function identity");
        let function_type = provider_instance
            .function_type_at(0)
            .expect("provider function type");
        let import = crate::Import::linked_func_typed("provider", "plus", handle, function_type);

        let importer_module = Module::new("linked-importer", &importer_wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&importer_module).expect("artifact");
        let importer = world
            .instantiate(&engine, importer_module, &[import])
            .expect("importer instance");
        let args = [Value::I32(0)];
        let baseline = baseline_on_instance(
            world.instance(importer).expect("importer facade"),
            &artifact,
            "run",
            &args,
        )
        .expect("baseline linked call");
        let folded = world
            .invoke(importer, "run", &args)
            .expect("folded linked call");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(42)]);
    }

    #[test]
    fn closed_table_local_call_indirect_and_equivalent_types_match_folded() {
        let wasm = wat::parse_str(
            r#"(module
                (type $actual (func (param i32) (result i32)))
                (type $equivalent (func (param i32) (result i32)))
                (table 1 1 funcref)
                (func $add (type $actual) (param i32) (result i32)
                    local.get 0 i32.const 5 i32.add)
                (elem (i32.const 0) $add)
                (func (export "run") (param i32) (result i32)
                    local.get 0 i32.const 0
                    call_indirect (type $equivalent)))"#,
        )
        .expect("wat");
        for input in [-9, 0, 37] {
            let args = [Value::I32(input)];
            let baseline = baseline(&wasm, "run", &args).expect("baseline call_indirect");
            let folded = native(&wasm, "run", &args).expect("folded call_indirect");
            assert_values_equal(&baseline, &folded);
        }
    }

    #[test]
    fn closed_table_call_indirect_traps_match_folded_exactly() {
        let out_of_bounds = wat::parse_str(
            r#"(module
                (type $result (func (result i32)))
                (table 1 1 funcref)
                (func $value (type $result) (result i32) i32.const 1)
                (elem (i32.const 0) $value)
                (func (export "run") (result i32)
                    i32.const 1 call_indirect (type $result)))"#,
        )
        .expect("out-of-bounds wat");
        let null = wat::parse_str(
            r#"(module
                (type $result (func (result i32)))
                (table 1 1 funcref)
                (func (export "run") (result i32)
                    i32.const 0 call_indirect (type $result)))"#,
        )
        .expect("null wat");
        let wrong_type = wat::parse_str(
            r#"(module
                (type $expected (func (param i32) (result i32)))
                (type $actual (func (param i64) (result i64)))
                (table 1 1 funcref)
                (func $value (type $actual) (param i64) (result i64) local.get 0)
                (elem (i32.const 0) $value)
                (func (export "run") (result i32)
                    i32.const 7 i32.const 0
                    call_indirect (type $expected)))"#,
        )
        .expect("wrong-type wat");
        for (wasm, expected) in [
            (out_of_bounds, "Trap: undefined element"),
            (null, "Trap: uninitialized element"),
            (wrong_type, "Trap: indirect call type mismatch"),
        ] {
            let baseline = baseline(&wasm, "run", &[]).expect_err("baseline trap");
            let BaselineExecError::Wasm(baseline) = baseline else {
                panic!("call_indirect trap became unsupported");
            };
            let folded = native(&wasm, "run", &[]).expect_err("folded trap");
            assert_eq!(baseline, folded);
            assert_eq!(baseline.to_string(), expected);
        }
    }

    #[test]
    fn closed_table_imported_host_call_indirect_is_not_misclassified_local() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "host" "add" (func $add (type $unary)))
                (table 1 1 funcref)
                (elem (i32.const 0) $add)
                (func (export "run") (param i32) (result i32)
                    local.get 0 i32.const 0 call_indirect (type $unary)))"#,
        )
        .expect("wat");
        let function_type = crate::FunctionType::new(
            crate::collections::vec![ValueType::I32],
            crate::collections::vec![ValueType::I32],
        );
        let host = || {
            crate::Import::func_typed(
                "host",
                "add",
                |_caller, args, results| {
                    let Value::I32(value) = args[0] else {
                        return Err(WasmError::internal("host indirect argument"));
                    };
                    results[0] = Value::I32(value.wrapping_add(40));
                    Ok(())
                },
                function_type.clone(),
            )
        };
        let args = [Value::I32(2)];
        let baseline = baseline_with_imports(&wasm, "run", &args, &[host()])
            .expect("baseline host call_indirect");
        let folded =
            native_with_imports(&wasm, "run", &args, &[host()]).expect("folded host call_indirect");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(42)]);
    }

    #[test]
    fn closed_table_foreign_linked_funcref_uses_external_barrier() {
        let provider_wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (func (export "plus") (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 40 i32.add))"#,
        )
        .expect("provider wat");
        let importer_wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (import "provider" "plus" (func $plus (type $unary)))
                (table 1 1 funcref)
                (elem (i32.const 0) $plus)
                (func (export "run") (param i32) (result i32)
                    local.get 0 i32.const 0 call_indirect (type $unary)
                    i32.const 2 i32.add))"#,
        )
        .expect("importer wat");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let mut world = crate::RuntimeWorld::new();
        let provider = world
            .instantiate(
                &engine,
                Module::new("indirect-provider", &provider_wasm).expect("module"),
                &[],
            )
            .expect("provider instance");
        let provider_instance = world.instance(provider).expect("provider facade");
        let handle = provider_instance
            .function_handle_at(0)
            .expect("provider function identity");
        let function_type = provider_instance
            .function_type_at(0)
            .expect("provider function type");
        let import = crate::Import::linked_func_typed("provider", "plus", handle, function_type);
        let importer_module = Module::new("indirect-importer", &importer_wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&importer_module).expect("artifact");
        let importer = world
            .instantiate(&engine, importer_module, &[import])
            .expect("importer instance");
        let args = [Value::I32(0)];
        let baseline = baseline_on_instance(
            world.instance(importer).expect("importer facade"),
            &artifact,
            "run",
            &args,
        )
        .expect("baseline foreign call_indirect");
        let folded = world
            .invoke(importer, "run", &args)
            .expect("folded foreign call_indirect");
        assert_values_equal(&baseline, &folded);
        assert_eq!(baseline, [Value::I32(42)]);
    }

    #[test]
    fn open_table_and_deferred_indirect_forms_are_explicitly_unsupported() {
        let open_table = wat::parse_str(
            r#"(module
                (type $result (func (result i32)))
                (table (export "table") 1 1 funcref)
                (func $value (type $result) (result i32) i32.const 1)
                (elem (i32.const 0) $value)
                (func (export "run") (result i32)
                    i32.const 0 call_indirect (type $result)))"#,
        )
        .expect("open table wat");
        let error = baseline(&open_table, "run", &[]).expect_err("open table unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::CALL_INDIRECT)),
                feature: "open call_indirect table",
                ..
            }
        ));

        let tail = wat::parse_str(
            r#"(module
                (type $result (func (result i32)))
                (table 1 1 funcref)
                (func $value (type $result) (result i32) i32.const 1)
                (elem (i32.const 0) $value)
                (func (export "run") (result i32)
                    i32.const 0 return_call_indirect (type $result)))"#,
        )
        .expect("tail wat");
        let error = baseline(&tail, "run", &[]).expect_err("tail indirect unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::RETURN_CALL_INDIRECT)),
                feature: "return_call_indirect",
                ..
            }
        ));

        let call_ref = wat::parse_str(
            r#"(module
                (type $result (func (result i32)))
                (func (export "run") (param (ref null $result)) (result i32)
                    local.get 0 call_ref $result))"#,
        )
        .expect("call_ref wat");
        let args = [Value::Ref(RefValue::null(), RefType::nullable_concrete(0))];
        let error = baseline(&call_ref, "run", &args).expect_err("call_ref unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::CALL_REF)),
                feature: "call_ref",
                ..
            }
        ));

        let mutation = wat::parse_str(
            r#"(module
                (table 1 1 funcref)
                (func (export "run") (param funcref)
                    i32.const 0 local.get 0 table.set))"#,
        )
        .expect("table mutation wat");
        let args = [Value::Ref(RefValue::null(), RefType::funcref())];
        let error = baseline(&mutation, "run", &args).expect_err("table.set unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::TABLE_SET)),
                feature: "MVP opcode",
                ..
            }
        ));
    }

    #[test]
    fn active_try_with_imported_call_is_explicitly_unsupported() {
        let wasm = wat::parse_str(
            r#"(module
                (import "host" "call" (func $call))
                (func (export "run")
                    block $done
                        try_table (catch_all $done)
                            call $call
                        end
                    end))"#,
        )
        .expect("wat");
        let calls = Rc::new(Cell::new(0usize));
        let callback_calls = Rc::clone(&calls);
        let import = crate::Import::func("host", "call", move |_caller, _args, _results| {
            callback_calls.set(callback_calls.get() + 1);
            Ok(())
        });
        let error = baseline_with_imports(&wasm, "run", &[], &[import])
            .expect_err("active EH is unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::TRY_TABLE)),
                feature: "MVP opcode",
                ..
            }
        ));
        assert_eq!(calls.get(), 0);

        let indirect = wat::parse_str(
            r#"(module
                (type $call (func))
                (import "host" "call" (func $call (type $call)))
                (table 1 1 funcref)
                (elem (i32.const 0) $call)
                (func (export "run")
                    block $done
                        try_table (catch_all $done)
                            i32.const 0 call_indirect (type $call)
                        end
                    end))"#,
        )
        .expect("indirect wat");
        let calls = Rc::new(Cell::new(0usize));
        let callback_calls = Rc::clone(&calls);
        let import = crate::Import::func_typed(
            "host",
            "call",
            move |_caller, _args, _results| {
                callback_calls.set(callback_calls.get() + 1);
                Ok(())
            },
            crate::FunctionType::new(crate::collections::vec![], crate::collections::vec![]),
        );
        let error = baseline_with_imports(&indirect, "run", &[], &[import])
            .expect_err("active indirect EH is unsupported");
        assert!(matches!(
            error,
            BaselineExecError::Unsupported {
                opcode: Some(WasmOpcode::OP(Opcode::TRY_TABLE)),
                feature: "MVP opcode",
                ..
            }
        ));
        assert_eq!(calls.get(), 0);
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

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn warm_closed_table_local_call_indirect_has_zero_allocations() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (table 1 1 funcref)
                (func $add (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 5 i32.add)
                (elem (i32.const 0) $add)
                (func (export "run") (param i32) (result i32)
                    local.get 0 i32.const 0 call_indirect (type $unary)))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-indirect-allocation", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let mut instance = initialized_interp(module, &[]);
        let args = [Value::I32(37)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            1,
            &args,
        )
        .expect("frame");
        frame.run().expect("warm-up indirect invocation");
        assert_eq!(frame.values.as_slice(), &[42]);
        let values_pointer = frame.values.as_ptr();
        let activations_pointer = frame.activations.as_ptr();
        let values_capacity = frame.values.capacity();
        let activations_capacity = frame.activations.capacity();
        let mut output = [Value::I32(0)];
        let (result, census) =
            crate::test_alloc::measure(|| frame.invoke_again(&args, &mut output));
        result.expect("warm indirect invocation");
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

    #[test]
    fn imported_call_releases_instance_materialization_before_host_memory_borrow() {
        let wasm = wat::parse_str(
            r#"(module
                (type $host (func (param i32) (result i32)))
                (import "host" "touch" (func $touch (type $host)))
                (memory 1)
                (func (export "run") (param i32) (result i32)
                    local.get 0 call $touch
                    i32.const 0 i32.load8_u
                    i32.add))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-import-borrows", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("engine");
        let registry = LinkRegistry::new();
        let (_, instance_backref) = registry.reserve_instance();
        #[cfg(sf_module_validator)]
        let validated_baseline =
            super::super::ValidatedBaselinePlan::validate(&module).expect("validated baseline");
        let host = InterpInstance::boxed_caller_host(|_module, _name, caller, args, results| {
            let memory = caller
                .memory_mut()
                .ok_or_else(|| WasmError::trap("host memory missing"))?;
            memory[0] = args[0] as u8;
            results[0] = (args[0] as u32).wrapping_add(1) as u64;
            Ok(())
        });
        // Stop before native linking so both Miri aliasing models can execute
        // the alternate driver and its owned external-call barrier.
        let mut instance = InterpInstance::build(
            &engine,
            module,
            #[cfg(sf_module_validator)]
            validated_baseline,
            Some(host),
            &[],
            None,
            registry.arenas(),
            instance_backref,
        )
        .expect("build interpreter instance");
        let args = [Value::I32(10)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            1,
            &args,
        )
        .expect("frame");
        frame.run().expect("baseline imported call");
        assert_eq!(frame.results().expect("results"), [Value::I32(21)]);
    }

    #[test]
    fn call_indirect_resolver_keeps_instance_borrows_scoped() {
        let wasm = wat::parse_str(
            r#"(module
                (type $unary (func (param i32) (result i32)))
                (table 1 1 funcref)
                (func $add (type $unary) (param i32) (result i32)
                    local.get 0 i32.const 5 i32.add)
                (elem (i32.const 0) $add)
                (func (export "run") (param i32) (result i32)
                    local.get 0 i32.const 0 call_indirect (type $unary)))"#,
        )
        .expect("wat");
        let module = Module::new("baseline-indirect-borrows", &wasm).expect("module");
        let _guard = artifact_test_guard();
        let artifact = build_baseline_artifact(&module).expect("artifact");
        // Build the real table entity and apply its active segment, but stop
        // before native linking so Miri can execute the common resolver.
        let mut instance = built_interp(module, &[]);
        instance
            .apply_element_segments()
            .expect("active element segment");
        let args = [Value::I32(37)];
        let mut frame = BaselineFrame::new(
            InterpInstanceAccess::borrowed(&mut instance),
            &artifact,
            1,
            &args,
        )
        .expect("frame");
        frame.run().expect("baseline call_indirect");
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
