//! Test-only raw-Wasm baseline executor prototype.
//!
//! Production still executes the folded interpreter exclusively. This module
//! answers a narrower architecture question: can the raw cursor and the eager
//! control artifact drive execution without allocating or rebuilding a native
//! instruction stream?

use super::baseline_artifact::{BaselineArtifact, BaselineFunction, ControlTarget};
use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::Module;
use crate::op_decoder::raw_cursor::{RawDecodeError, RawImmediate, RawOpCursor};
use crate::opcodes::{Opcode, WasmOpcode};
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
            Opcode::I32_EQZ => {
                let value = self.pop()? as u32;
                self.push(u64::from(value == 0))?;
            }
            Opcode::I32_ADD => self.binary_i32(u32::wrapping_add)?,
            Opcode::I32_SUB => self.binary_i32(u32::wrapping_sub)?,
            Opcode::I32_MUL => self.binary_i32(u32::wrapping_mul)?,
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
            Opcode::END => {
                if raw.end == self.code.len() {
                    self.finished = true;
                }
            }
            Opcode::UNREACHABLE => return Err(WasmError::trap("unreachable").into()),
            _ => return self.unsupported(Some(raw.wasm_op), raw.start, "MVP opcode"),
        }
        Ok(())
    }

    fn binary_i32(&mut self, operation: fn(u32, u32) -> u32) -> Result<(), BaselineExecError> {
        let rhs = self.pop()? as u32;
        let lhs = self.pop()? as u32;
        self.push(operation(lhs, rhs) as u64)
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
}
