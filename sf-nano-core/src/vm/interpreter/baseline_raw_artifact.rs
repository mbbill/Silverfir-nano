//! Direct raw-Wasm construction of the diagnostic baseline artifact.
//!
//! Unlike the original oracle in `predecode`, this scanner never constructs
//! a folded instruction cell. It owns only a height/reachability control
//! model and feeds raw structural events into the shared artifact assembler.
//! It is deliberately available only with the standalone module validator:
//! raw height accounting is not a replacement for full Wasm type validation.

use super::baseline_artifact::{BaselineArtifact, BaselineFunctionBuilder};
use crate::error::WasmError;
use crate::module::validator::Validator;
use crate::module::Module;
use crate::op_decoder::raw_cursor::{
    RawBlockType, RawDecodeError, RawImmediate, RawOp, RawOpCursor,
};
use crate::op_decoder::{BlockType, DecodedOp, Immediate};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};

#[derive(Debug)]
pub(super) enum RawArtifactError {
    Wasm(WasmError),
    Unsupported { opcode: WasmOpcode, offset: usize },
}

impl core::fmt::Display for RawArtifactError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Wasm(error) => write!(formatter, "{error}"),
            Self::Unsupported { opcode, offset } => {
                write!(
                    formatter,
                    "raw artifact unsupported {opcode:?} at byte {offset}"
                )
            }
        }
    }
}

impl From<WasmError> for RawArtifactError {
    fn from(error: WasmError) -> Self {
        Self::Wasm(error)
    }
}

impl From<RawDecodeError> for RawArtifactError {
    fn from(error: RawDecodeError) -> Self {
        match error {
            RawDecodeError::Decode(error) => Self::Wasm(error),
            RawDecodeError::Unsupported { opcode, offset } => Self::Unsupported { opcode, offset },
            RawDecodeError::InvalidPc { .. } => {
                Self::Wasm(WasmError::invalid("raw artifact pc out of range"))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct ScanFrame {
    base: usize,
    params: usize,
    results: usize,
    is_loop: bool,
    is_if: bool,
    dead_entry: bool,
    end_targeted: bool,
    saw_else: bool,
    then_fell_live: bool,
}

pub(super) struct RawFunctionScanner {
    height: usize,
    result_count: usize,
    frames: crate::collections::Vec<ScanFrame>,
    dead: bool,
    finished: bool,
}

pub(super) enum DecodedLayoutError {
    Ineligible,
    Wasm(WasmError),
}

impl From<WasmError> for DecodedLayoutError {
    fn from(error: WasmError) -> Self {
        Self::Wasm(error)
    }
}

impl RawFunctionScanner {
    pub(super) fn new(result_count: usize) -> Self {
        Self {
            height: 0,
            result_count,
            frames: crate::collections::Vec::new(),
            dead: false,
            finished: false,
        }
    }

    pub(super) fn height(&self) -> usize {
        self.height
    }

    pub(super) fn dead(&self) -> bool {
        self.dead
    }

    pub(super) fn ensure_finished(&self) -> Result<(), WasmError> {
        if self.finished && self.frames.is_empty() {
            Ok(())
        } else {
            Err(WasmError::invalid("raw artifact function did not close"))
        }
    }

    pub(super) fn apply_decoded(
        &mut self,
        module: &Module,
        decoded: &DecodedOp,
    ) -> Result<(), DecodedLayoutError> {
        if matches!(decoded.wasm_op, WasmOpcode::FB(_) | WasmOpcode::FD(_))
            || matches!(
                decoded.wasm_op,
                WasmOpcode::OP(Opcode::THROW | Opcode::THROW_REF)
            )
        {
            return Err(DecodedLayoutError::Ineligible);
        }
        if self.dead {
            self.apply_dead_decoded(module, decoded)?;
            return Ok(());
        }
        match decoded.wasm_op {
            WasmOpcode::OP(opcode) => self.apply_decoded_op(module, opcode, &decoded.imm),
            WasmOpcode::FC(opcode) => self.apply_fc(opcode).map_err(Into::into),
            WasmOpcode::FB(_) | WasmOpcode::FD(_) => unreachable!(),
        }
    }

    fn apply_dead_decoded(
        &mut self,
        module: &Module,
        decoded: &DecodedOp,
    ) -> Result<(), WasmError> {
        let WasmOpcode::OP(opcode) = decoded.wasm_op else {
            return Ok(());
        };
        match opcode {
            Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                let Immediate::Block(block) = &decoded.imm else {
                    return Err(
                        WasmError::internal("decoded artifact block immediate mismatch").into(),
                    );
                };
                let (params, results) = decoded_block_arity(module, block)?;
                let base = self.height.saturating_sub(params);
                self.frames.push(ScanFrame {
                    base,
                    params,
                    results,
                    is_loop: opcode == Opcode::LOOP,
                    is_if: opcode == Opcode::IF,
                    dead_entry: true,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::TRY_TABLE => {
                let Immediate::TryTable { block_type, .. } = &decoded.imm else {
                    return Err(WasmError::internal(
                        "decoded artifact try_table immediate mismatch",
                    )
                    .into());
                };
                let (params, results) = decoded_block_arity(module, block_type)?;
                let base = self.height.saturating_sub(params);
                self.frames.push(ScanFrame {
                    base,
                    params,
                    results,
                    is_loop: false,
                    is_if: false,
                    dead_entry: true,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::ELSE => {
                let frame = self
                    .frames
                    .last_mut()
                    .ok_or_else(|| WasmError::invalid("decoded artifact else without frame"))?;
                frame.saw_else = true;
                self.height = frame.base + frame.params;
                self.dead = frame.dead_entry;
            }
            Opcode::END => {
                if let Some(frame) = self.frames.pop() {
                    let live_after = !frame.dead_entry
                        && (frame.end_targeted
                            || frame.then_fell_live
                            || (frame.is_if && !frame.saw_else));
                    self.height = frame.base + frame.results;
                    self.dead = !live_after;
                } else {
                    self.finished = true;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn apply_decoded_op(
        &mut self,
        module: &Module,
        opcode: Opcode,
        immediate: &Immediate,
    ) -> Result<(), DecodedLayoutError> {
        match opcode {
            Opcode::NOP => {}
            Opcode::UNREACHABLE => self.dead = true,
            Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                let Immediate::Block(block) = immediate else {
                    return Err(
                        WasmError::internal("decoded artifact block immediate mismatch").into(),
                    );
                };
                let (params, results) = decoded_block_arity(module, block)?;
                if opcode == Opcode::IF {
                    self.pop(1)?;
                }
                self.require(params)?;
                self.frames.push(ScanFrame {
                    base: self.height - params,
                    params,
                    results,
                    is_loop: opcode == Opcode::LOOP,
                    is_if: opcode == Opcode::IF,
                    dead_entry: false,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::TRY_TABLE => {
                let Immediate::TryTable {
                    block_type,
                    catches,
                } = immediate
                else {
                    return Err(WasmError::internal(
                        "decoded artifact try_table immediate mismatch",
                    )
                    .into());
                };
                for catch in catches {
                    self.mark_branch_target(catch.label_idx);
                }
                let (params, results) = decoded_block_arity(module, block_type)?;
                self.require(params)?;
                self.frames.push(ScanFrame {
                    base: self.height - params,
                    params,
                    results,
                    is_loop: false,
                    is_if: false,
                    dead_entry: false,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::ELSE => {
                let frame = self
                    .frames
                    .last_mut()
                    .ok_or_else(|| WasmError::invalid("decoded artifact else without frame"))?;
                if !frame.is_if || self.height != frame.base + frame.results {
                    return Err(
                        WasmError::invalid("decoded artifact else stack shape mismatch").into(),
                    );
                }
                frame.saw_else = true;
                frame.then_fell_live = true;
                self.height = frame.base + frame.params;
            }
            Opcode::END => {
                if let Some(frame) = self.frames.pop() {
                    if self.height != frame.base + frame.results {
                        return Err(WasmError::invalid(
                            "decoded artifact end stack shape mismatch",
                        )
                        .into());
                    }
                } else {
                    if self.height != self.result_count {
                        return Err(WasmError::invalid(
                            "decoded artifact function result height mismatch",
                        )
                        .into());
                    }
                    self.finished = true;
                }
            }
            Opcode::BR => {
                let depth = decoded_label(immediate)?;
                self.mark_branch_target(depth);
                self.require(self.branch_arity(depth))?;
                self.dead = true;
            }
            Opcode::BR_IF => {
                let depth = decoded_label(immediate)?;
                self.mark_branch_target(depth);
                self.pop(1)?;
                self.require(self.branch_arity(depth))?;
            }
            Opcode::BR_TABLE => {
                let Immediate::BrLabels(labels, default) = immediate else {
                    return Err(WasmError::internal(
                        "decoded artifact br_table immediate mismatch",
                    )
                    .into());
                };
                self.pop(1)?;
                for &depth in labels {
                    self.mark_branch_target(depth);
                    self.require(self.branch_arity(depth))?;
                }
                self.mark_branch_target(*default);
                self.require(self.branch_arity(*default))?;
                self.dead = true;
            }
            Opcode::RETURN => {
                self.require(self.result_count)?;
                self.dead = true;
            }
            Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL => {
                let depth = decoded_label(immediate)?;
                self.mark_branch_target(depth);
                self.require(1)?;
                self.require(self.branch_arity(depth))?;
                if opcode == Opcode::BR_ON_NON_NULL {
                    self.pop(1)?;
                }
            }
            Opcode::CALL | Opcode::RETURN_CALL => {
                let Immediate::FunctionIndex(index) = immediate else {
                    return Err(
                        WasmError::internal("decoded artifact call immediate mismatch").into(),
                    );
                };
                let function = module
                    .functions()
                    .get(*index as usize)
                    .ok_or_else(|| WasmError::invalid("decoded artifact call target overflow"))?;
                self.pop(function.func_type().params().len())?;
                if opcode == Opcode::RETURN_CALL {
                    self.dead = true;
                } else {
                    self.push(function.func_type().results().len())?;
                }
            }
            Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT => {
                let Immediate::CallIndirectArgs { typeidx, .. } = immediate else {
                    return Err(WasmError::internal(
                        "decoded artifact call_indirect immediate mismatch",
                    )
                    .into());
                };
                let function = module
                    .types()
                    .get_function_type(*typeidx)
                    .ok_or_else(|| WasmError::invalid("decoded artifact call type overflow"))?;
                self.pop(function.params().len() + 1)?;
                if opcode == Opcode::RETURN_CALL_INDIRECT {
                    self.dead = true;
                } else {
                    self.push(function.results().len())?;
                }
            }
            Opcode::CALL_REF | Opcode::RETURN_CALL_REF => {
                let Immediate::TypeIndex(typeidx) = immediate else {
                    return Err(WasmError::internal(
                        "decoded artifact call_ref immediate mismatch",
                    )
                    .into());
                };
                let function = module
                    .types()
                    .get_function_type(*typeidx)
                    .ok_or_else(|| WasmError::invalid("decoded artifact call type overflow"))?;
                self.pop(function.params().len() + 1)?;
                if opcode == Opcode::RETURN_CALL_REF {
                    self.dead = true;
                } else {
                    self.push(function.results().len())?;
                }
            }
            _ => {
                let (pops, pushes) = simple_effect(opcode).ok_or(DecodedLayoutError::Ineligible)?;
                self.pop(pops)?;
                self.push(pushes)?;
            }
        }
        Ok(())
    }

    fn apply(&mut self, module: &Module, raw: &RawOp<'_>) -> Result<(), WasmError> {
        if self.dead {
            return self.apply_dead(module, raw);
        }
        match raw.wasm_op {
            WasmOpcode::OP(opcode) => self.apply_op(module, opcode, raw.imm),
            WasmOpcode::FC(opcode) => self.apply_fc(opcode),
            opcode => Err(WasmError::invalid(match opcode {
                WasmOpcode::FB(_) => "raw artifact does not support GC opcodes",
                WasmOpcode::FD(_) => "raw artifact does not support SIMD opcodes",
                _ => "raw artifact opcode family is unsupported",
            })),
        }
    }

    fn apply_dead(&mut self, module: &Module, raw: &RawOp<'_>) -> Result<(), WasmError> {
        let WasmOpcode::OP(opcode) = raw.wasm_op else {
            return Ok(());
        };
        match opcode {
            Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                let RawImmediate::Block(block) = raw.imm else {
                    return Err(WasmError::internal("raw artifact block immediate mismatch"));
                };
                let (params, results) = block_arity(module, block)?;
                let base = self.height.saturating_sub(params);
                self.frames.push(ScanFrame {
                    base,
                    params,
                    results,
                    is_loop: opcode == Opcode::LOOP,
                    is_if: opcode == Opcode::IF,
                    dead_entry: true,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::TRY_TABLE => {
                let RawImmediate::TryTable { block, .. } = raw.imm else {
                    return Err(WasmError::internal(
                        "raw artifact try_table immediate mismatch",
                    ));
                };
                let (params, results) = block_arity(module, block)?;
                let base = self.height.saturating_sub(params);
                self.frames.push(ScanFrame {
                    base,
                    params,
                    results,
                    is_loop: false,
                    is_if: false,
                    dead_entry: true,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::ELSE => {
                let frame = self
                    .frames
                    .last_mut()
                    .ok_or_else(|| WasmError::invalid("raw artifact else without frame"))?;
                frame.saw_else = true;
                self.height = frame.base + frame.params;
                self.dead = frame.dead_entry;
            }
            Opcode::END => {
                if let Some(frame) = self.frames.pop() {
                    let live_after = !frame.dead_entry
                        && (frame.end_targeted
                            || frame.then_fell_live
                            || (frame.is_if && !frame.saw_else));
                    self.height = frame.base + frame.results;
                    self.dead = !live_after;
                } else {
                    self.finished = true;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn apply_op(
        &mut self,
        module: &Module,
        opcode: Opcode,
        immediate: RawImmediate<'_>,
    ) -> Result<(), WasmError> {
        match opcode {
            Opcode::NOP => {}
            Opcode::UNREACHABLE => self.dead = true,
            Opcode::BLOCK | Opcode::LOOP | Opcode::IF => {
                let RawImmediate::Block(block) = immediate else {
                    return Err(WasmError::internal("raw artifact block immediate mismatch"));
                };
                let (params, results) = block_arity(module, block)?;
                if opcode == Opcode::IF {
                    self.pop(1)?;
                }
                self.require(params)?;
                self.frames.push(ScanFrame {
                    base: self.height - params,
                    params,
                    results,
                    is_loop: opcode == Opcode::LOOP,
                    is_if: opcode == Opcode::IF,
                    dead_entry: false,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::TRY_TABLE => {
                let RawImmediate::TryTable { block, catches } = immediate else {
                    return Err(WasmError::internal(
                        "raw artifact try_table immediate mismatch",
                    ));
                };
                for catch in catches.iter() {
                    self.mark_branch_target(catch?.label_depth);
                }
                let (params, results) = block_arity(module, block)?;
                self.require(params)?;
                self.frames.push(ScanFrame {
                    base: self.height - params,
                    params,
                    results,
                    is_loop: false,
                    is_if: false,
                    dead_entry: false,
                    end_targeted: false,
                    saw_else: false,
                    then_fell_live: false,
                });
            }
            Opcode::ELSE => {
                let frame = self
                    .frames
                    .last_mut()
                    .ok_or_else(|| WasmError::invalid("raw artifact else without frame"))?;
                if !frame.is_if || self.height != frame.base + frame.results {
                    return Err(WasmError::invalid("raw artifact else stack shape mismatch"));
                }
                frame.saw_else = true;
                frame.then_fell_live = true;
                self.height = frame.base + frame.params;
            }
            Opcode::END => {
                if let Some(frame) = self.frames.pop() {
                    if self.height != frame.base + frame.results {
                        return Err(WasmError::invalid("raw artifact end stack shape mismatch"));
                    }
                } else {
                    if self.height != self.result_count {
                        return Err(WasmError::invalid(
                            "raw artifact function result height mismatch",
                        ));
                    }
                    self.finished = true;
                }
            }
            Opcode::BR => {
                let depth = label(immediate)?;
                self.mark_branch_target(depth);
                self.require(self.branch_arity(depth))?;
                self.dead = true;
            }
            Opcode::BR_IF => {
                let depth = label(immediate)?;
                self.mark_branch_target(depth);
                self.pop(1)?;
                self.require(self.branch_arity(depth))?;
            }
            Opcode::BR_TABLE => {
                let RawImmediate::BrTable { labels, default } = immediate else {
                    return Err(WasmError::internal(
                        "raw artifact br_table immediate mismatch",
                    ));
                };
                self.pop(1)?;
                for depth in labels.iter() {
                    let depth = depth?;
                    self.mark_branch_target(depth);
                    self.require(self.branch_arity(depth))?;
                }
                self.mark_branch_target(default);
                self.require(self.branch_arity(default))?;
                self.dead = true;
            }
            Opcode::RETURN => {
                self.require(self.result_count)?;
                self.dead = true;
            }
            Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL => {
                let depth = label(immediate)?;
                self.mark_branch_target(depth);
                self.require(1)?;
                if opcode == Opcode::BR_ON_NULL {
                    self.require(self.branch_arity(depth))?;
                } else {
                    self.require(self.branch_arity(depth))?;
                    self.pop(1)?;
                }
            }
            Opcode::CALL | Opcode::RETURN_CALL => {
                let RawImmediate::FunctionIndex(index) = immediate else {
                    return Err(WasmError::internal("raw artifact call immediate mismatch"));
                };
                let function = module
                    .functions()
                    .get(index as usize)
                    .ok_or_else(|| WasmError::invalid("raw artifact call target overflow"))?;
                let params = function.func_type().params().len();
                let results = function.func_type().results().len();
                self.pop(params)?;
                if opcode == Opcode::RETURN_CALL {
                    self.dead = true;
                } else {
                    self.push(results)?;
                }
            }
            Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT => {
                let RawImmediate::CallIndirect { typeidx, .. } = immediate else {
                    return Err(WasmError::internal(
                        "raw artifact call_indirect immediate mismatch",
                    ));
                };
                let function = module
                    .types()
                    .get_function_type(typeidx)
                    .ok_or_else(|| WasmError::invalid("raw artifact call type overflow"))?;
                self.pop(function.params().len() + 1)?;
                if opcode == Opcode::RETURN_CALL_INDIRECT {
                    self.dead = true;
                } else {
                    self.push(function.results().len())?;
                }
            }
            Opcode::CALL_REF | Opcode::RETURN_CALL_REF => {
                let RawImmediate::TypeIndex(typeidx) = immediate else {
                    return Err(WasmError::internal(
                        "raw artifact call_ref immediate mismatch",
                    ));
                };
                let function = module
                    .types()
                    .get_function_type(typeidx)
                    .ok_or_else(|| WasmError::invalid("raw artifact call_ref type overflow"))?;
                self.pop(function.params().len() + 1)?;
                if opcode == Opcode::RETURN_CALL_REF {
                    self.dead = true;
                } else {
                    self.push(function.results().len())?;
                }
            }
            _ => {
                let (pops, pushes) = simple_effect(opcode).ok_or_else(|| {
                    WasmError::invalid("raw artifact primary opcode effect is unsupported")
                })?;
                self.pop(pops)?;
                self.push(pushes)?;
            }
        }
        Ok(())
    }

    fn apply_fc(&mut self, opcode: OpcodeFC) -> Result<(), WasmError> {
        use OpcodeFC::*;
        let (pops, pushes) = match opcode {
            I32_TRUNC_SAT_F32_S | I32_TRUNC_SAT_F32_U | I32_TRUNC_SAT_F64_S
            | I32_TRUNC_SAT_F64_U | I64_TRUNC_SAT_F32_S | I64_TRUNC_SAT_F32_U
            | I64_TRUNC_SAT_F64_S | I64_TRUNC_SAT_F64_U => (1, 1),
            MEMORY_INIT | MEMORY_COPY | MEMORY_FILL | TABLE_INIT | TABLE_COPY | TABLE_FILL => {
                (3, 0)
            }
            DATA_DROP | ELEM_DROP => (0, 0),
            TABLE_GROW => (2, 1),
            TABLE_SIZE => (0, 1),
        };
        self.pop(pops)?;
        self.push(pushes)
    }

    fn mark_branch_target(&mut self, depth: u32) {
        let depth = depth as usize;
        if depth < self.frames.len() {
            let index = self.frames.len() - 1 - depth;
            if !self.frames[index].is_loop {
                self.frames[index].end_targeted = true;
            }
        }
    }

    fn branch_arity(&self, depth: u32) -> usize {
        let depth = depth as usize;
        if depth >= self.frames.len() {
            return self.result_count;
        }
        let frame = self.frames[self.frames.len() - 1 - depth];
        if frame.is_loop {
            frame.params
        } else {
            frame.results
        }
    }

    fn require(&self, count: usize) -> Result<(), WasmError> {
        if self.height < count {
            Err(WasmError::invalid("raw artifact operand stack underflow"))
        } else {
            Ok(())
        }
    }

    fn pop(&mut self, count: usize) -> Result<(), WasmError> {
        self.require(count)?;
        self.height -= count;
        Ok(())
    }

    fn push(&mut self, count: usize) -> Result<(), WasmError> {
        self.height = self
            .height
            .checked_add(count)
            .ok_or_else(|| WasmError::invalid("raw artifact operand stack overflow"))?;
        Ok(())
    }
}

pub(super) fn build_baseline_artifact_raw(
    module: &Module,
) -> Result<BaselineArtifact, RawArtifactError> {
    // Publication is conditional on a complete validation pass. The scanner
    // checks a few bounds as it consumes metadata, but intentionally does not
    // grow a second, incomplete type validator inside the interpreter.
    Validator::new(module).validate()?;

    let mut artifact = BaselineArtifact::new(module.functions().len());
    for (function_index, function) in module.functions().iter().enumerate() {
        let Some(spec) = function.spec() else {
            continue;
        };
        let raw_end = spec
            .code_offset()
            .checked_add(spec.code().len())
            .ok_or_else(|| WasmError::invalid("raw artifact code range overflow"))?;
        let mut builder = BaselineFunctionBuilder::new(
            function_index,
            spec.code_offset()..raw_end,
            function.func_type().results().len(),
        )?;
        let mut scanner = RawFunctionScanner::new(function.func_type().results().len());
        let mut cursor = RawOpCursor::new(spec.code());
        while let Some(raw) = cursor.next()? {
            let event = builder.plan_raw(module, &raw, scanner.height, scanner.dead)?;
            scanner.apply(module, &raw)?;
            builder.commit(event, scanner.height)?;
        }
        if !scanner.finished || !scanner.frames.is_empty() {
            return Err(WasmError::invalid("raw artifact function did not close").into());
        }
        artifact.publish_function(function_index, builder.finish()?)?;
    }
    Ok(artifact)
}

fn label(immediate: RawImmediate<'_>) -> Result<u32, WasmError> {
    let RawImmediate::LabelIndex(depth) = immediate else {
        return Err(WasmError::internal(
            "raw artifact branch immediate mismatch",
        ));
    };
    Ok(depth)
}

fn decoded_label(immediate: &Immediate) -> Result<u32, WasmError> {
    let Immediate::LabelIndex(depth) = immediate else {
        return Err(WasmError::internal(
            "decoded artifact branch immediate mismatch",
        ));
    };
    Ok(*depth)
}

fn block_arity(module: &Module, block: RawBlockType) -> Result<(usize, usize), WasmError> {
    match block {
        RawBlockType::Empty => Ok((0, 0)),
        RawBlockType::Value(_) => Ok((0, 1)),
        RawBlockType::TypeIndex(index) => module
            .types()
            .get_function_type(index as u32)
            .map(|function| (function.params().len(), function.results().len()))
            .ok_or_else(|| WasmError::invalid("raw artifact block type overflow")),
    }
}

fn decoded_block_arity(module: &Module, block: &BlockType) -> Result<(usize, usize), WasmError> {
    match *block {
        BlockType::Empty => Ok((0, 0)),
        BlockType::ValueType(_) => Ok((0, 1)),
        BlockType::TypeIndex(index) => module
            .types()
            .get_function_type(index as u32)
            .map(|function| (function.params().len(), function.results().len()))
            .ok_or_else(|| WasmError::invalid("decoded artifact block type overflow")),
    }
}

fn simple_effect(opcode: Opcode) -> Option<(usize, usize)> {
    use Opcode::*;
    let effect = match opcode {
        LOCAL_GET | GLOBAL_GET | MEMORY_SIZE | I32_CONST | I64_CONST | F32_CONST | F64_CONST
        | REF_NULL | REF_FUNC => (0, 1),
        LOCAL_SET | GLOBAL_SET | DROP => (1, 0),
        LOCAL_TEE | MEMORY_GROW | REF_IS_NULL | REF_AS_NON_NULL | I32_EQZ | I64_EQZ | I32_CLZ
        | I32_CTZ | I32_POPCNT | I64_CLZ | I64_CTZ | I64_POPCNT | I32_WRAP_I64
        | I32_TRUNC_F32_S | I32_TRUNC_F32_U | I32_TRUNC_F64_S | I32_TRUNC_F64_U
        | I64_EXTEND_I32_S | I64_EXTEND_I32_U | I64_TRUNC_F32_S | I64_TRUNC_F32_U
        | I64_TRUNC_F64_S | I64_TRUNC_F64_U | F32_CONVERT_I32_S | F32_CONVERT_I32_U
        | F32_CONVERT_I64_S | F32_CONVERT_I64_U | F32_DEMOTE_F64 | F64_CONVERT_I32_S
        | F64_CONVERT_I32_U | F64_CONVERT_I64_S | F64_CONVERT_I64_U | F64_PROMOTE_F32
        | I32_REINTERPRET_F32 | I64_REINTERPRET_F64 | F32_REINTERPRET_I32 | F64_REINTERPRET_I64
        | I32_EXTEND8_S | I32_EXTEND16_S | I64_EXTEND8_S | I64_EXTEND16_S | I64_EXTEND32_S
        | F32_ABS | F32_NEG | F32_CEIL | F32_FLOOR | F32_TRUNC | F32_NEAREST | F32_SQRT
        | F64_ABS | F64_NEG | F64_CEIL | F64_FLOOR | F64_TRUNC | F64_NEAREST | F64_SQRT => (1, 1),
        TABLE_GET | I32_LOAD | I32_LOAD8_S | I32_LOAD8_U | I32_LOAD16_S | I32_LOAD16_U
        | I64_LOAD | I64_LOAD8_S | I64_LOAD8_U | I64_LOAD16_S | I64_LOAD16_U | I64_LOAD32_S
        | I64_LOAD32_U | F32_LOAD | F64_LOAD => (1, 1),
        TABLE_SET | I32_STORE | I32_STORE8 | I32_STORE16 | I64_STORE | I64_STORE8 | I64_STORE16
        | I64_STORE32 | F32_STORE | F64_STORE => (2, 0),
        SELECT | SELECT_T => (3, 1),
        REF_EQ | I32_EQ | I32_NE | I32_LT_S | I32_LT_U | I32_GT_S | I32_GT_U | I32_LE_S
        | I32_LE_U | I32_GE_S | I32_GE_U | I64_EQ | I64_NE | I64_LT_S | I64_LT_U | I64_GT_S
        | I64_GT_U | I64_LE_S | I64_LE_U | I64_GE_S | I64_GE_U | F32_EQ | F32_NE | F32_LT
        | F32_GT | F32_LE | F32_GE | F64_EQ | F64_NE | F64_LT | F64_GT | F64_LE | F64_GE => (2, 1),
        I32_ADD | I32_SUB | I32_MUL | I32_DIV_S | I32_DIV_U | I32_REM_S | I32_REM_U | I32_AND
        | I32_OR | I32_XOR | I32_SHL | I32_SHR_S | I32_SHR_U | I32_ROTL | I32_ROTR | I64_ADD
        | I64_SUB | I64_MUL | I64_DIV_S | I64_DIV_U | I64_REM_S | I64_REM_U | I64_AND | I64_OR
        | I64_XOR | I64_SHL | I64_SHR_S | I64_SHR_U | I64_ROTL | I64_ROTR | F32_ADD | F32_SUB
        | F32_MUL | F32_DIV | F32_MIN | F32_MAX | F32_COPYSIGN | F64_ADD | F64_SUB | F64_MUL
        | F64_DIV | F64_MIN | F64_MAX | F64_COPYSIGN => (2, 1),
        _ => return None,
    };
    Some(effect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::baseline_artifact::{
        artifact_build_count, artifact_test_guard, BaselineArtifact,
    };
    use crate::vm::interpreter::predecode::build_baseline_artifact;

    fn artifacts(wat: &str) -> (BaselineArtifact, BaselineArtifact) {
        let _guard = artifact_test_guard();
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new("raw-artifact", &wasm).expect("module");
        let oracle = build_baseline_artifact(&module).expect("predecode oracle");
        let raw = build_baseline_artifact_raw(&module).expect("raw artifact");
        (oracle, raw)
    }

    fn assert_artifacts_equal(wat: &str) {
        let (oracle, raw) = artifacts(wat);
        assert_eq!(raw, oracle);
    }

    #[test]
    fn raw_structure_matches_if_loop_multivalue_and_br_table_oracles() {
        for wat in [
            r#"(module
                (func (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 7
                    else
                        i32.const 9
                    end))"#,
            r#"(module
                (func (local i32)
                    (loop $again
                        local.get 0
                        i32.const 1
                        i32.add
                        local.tee 0
                        i32.const 10
                        i32.lt_u
                        br_if $again)))"#,
            r#"(module
                (type $pair (func (param i32 i32) (result i32 i32)))
                (func (param i32 i32 i32) (result i32)
                    block $exit (result i32)
                        local.get 0
                        local.get 1
                        local.get 2
                        br_table $exit $exit $exit
                    end)
                (func (result i32)
                    i32.const 1
                    i32.const 2
                    block (type $pair)
                        br 0
                        unreachable
                    end
                    i32.add))"#,
        ] {
            assert_artifacts_equal(wat);
        }
    }

    #[test]
    fn raw_dead_if_preserves_the_oracle_residual_height() {
        assert_artifacts_equal(
            r#"(module
                (func
                    i32.const 99
                    unreachable
                    if
                        nop
                    else
                        nop
                    end))"#,
        );
    }

    #[test]
    fn full_validation_rejects_bad_indices_before_artifact_publication() {
        let _guard = artifact_test_guard();
        for wat in [
            "(module (func ref.func 1 drop))",
            "(module (func local.get 0 drop))",
            "(module (func global.get 0 drop))",
            "(module (func i32.const 0 table.get 0 drop))",
            "(module (func memory.size 0 drop))",
            "(module (func i32.const 0 i32.load 0 drop))",
            "(module (func ref.null func call_ref 0))",
            "(module (type (func)) (func i32.const 0 call_indirect (type 0)))",
        ] {
            let wasm = wat::parse_str(wat).expect("encode invalid-index fixture");
            let module = Module::new("raw-artifact-invalid-index", &wasm).expect("parse module");
            let before = artifact_build_count();
            let error = build_baseline_artifact_raw(&module).expect_err("validation must reject");
            assert!(
                matches!(error, RawArtifactError::Wasm(WasmError::Invalid(_))),
                "unexpected error for {wat}: {error}"
            );
            assert_eq!(
                artifact_build_count(),
                before,
                "invalid module published an artifact for {wat}"
            );
        }
    }

    #[test]
    fn raw_try_table_matches_nested_and_typed_catch_metadata() {
        for wat in [
            r#"(module
                (func
                    block $outer
                        try_table (catch_all $outer)
                            block $inner
                                try_table (catch_all $inner)
                                    unreachable
                                end
                            end
                        end
                    end))"#,
            r#"(module
                (tag $e (param i32))
                (func (result i32)
                    block $handler (result i32)
                        try_table (result i32) (catch $e $handler)
                            i32.const 2
                        end
                    end))"#,
        ] {
            assert_artifacts_equal(wat);
        }
    }

    #[test]
    fn raw_call_edges_match_direct_tail_table_and_ref_oracles() {
        assert_artifacts_equal(
            r#"(module
                (type $u (func (param i32) (result i32)))
                (table 1 funcref)
                (func $id (type $u) (param i32) (result i32) local.get 0)
                (elem (i32.const 0) $id)
                (func $tail (type $u) (param i32) (result i32)
                    local.get 0
                    return_call $id)
                (func (type $u) (param i32) (result i32)
                    block $done
                        loop $hot
                            local.get 0
                            call $tail
                            drop
                            br $done
                        end
                    end
                    local.get 0)
                (func (type $u) (param i32) (result i32)
                    local.get 0
                    i32.const 0
                    call_indirect (type $u))
                (func (type $u) (param i32) (result i32)
                    local.get 0
                    ref.func $id
                    call_ref $u))"#,
        );
    }

    #[test]
    fn raw_one_pass_matches_the_predecoder_retry_result() {
        assert_artifacts_equal(
            r#"(module
                (func (param $x i32) (param $n i32) (result i32)
                    local.get $x
                    loop $unsafe (param i32) (result i32)
                        drop
                        i32.const 42
                        local.get $n
                        i32.const 1
                        i32.sub
                        local.tee $n
                        br_if $unsafe
                    end
                    drop
                    i32.const 3
                    local.set $n
                    local.get $n
                    loop $safe (param i32) (result i32)
                        i32.const 1
                        i32.sub
                        local.tee $n
                        local.get $n
                        br_if $safe
                    end
                    drop
                    local.get $x))"#,
        );
    }

    #[test]
    fn try_table_ranges_survive_canonical_loop_safety_retry() {
        let (oracle, raw) = artifacts(
            r#"(module
                (tag $e (param i32))
                (func (param $x i32) (param $n i32) (result i32)
                    local.get $x
                    loop $unsafe (param i32) (result i32)
                        drop
                        i32.const 42
                        local.get $n
                        i32.const 1
                        i32.sub
                        local.tee $n
                        br_if $unsafe
                    end
                    drop
                    block $handler (result i32)
                        try_table (result i32) (catch $e $handler)
                            local.get $x
                        end
                    end))"#,
        );
        assert_eq!(raw, oracle);
        assert_eq!(oracle.try_tables.len(), 1, "retry duplicated try metadata");
        assert_eq!(oracle.catches.len(), 1, "retry duplicated catch metadata");
        let function = oracle.functions[0].as_ref().expect("local function");
        assert_eq!(function.try_tables, 0..1);
        assert_eq!(function.catches, 0..1);
    }

    #[test]
    fn unsupported_throw_is_transactional_and_does_not_poison_next_build() {
        let _guard = artifact_test_guard();
        let unsupported = wat::parse_str(
            r#"(module
                (tag $e)
                (func i32.const 1 drop)
                (func (throw $e)))"#,
        )
        .expect("wat");
        let module = Module::new("raw-artifact-unsupported", &unsupported).expect("module");
        let error = build_baseline_artifact_raw(&module).expect_err("throw unsupported");
        assert!(matches!(
            error,
            RawArtifactError::Unsupported {
                opcode: WasmOpcode::OP(Opcode::THROW),
                ..
            }
        ));

        let supported = wat::parse_str("(module (func (block $out br $out)))").expect("wat");
        let module = Module::new("raw-artifact-after-error", &supported).expect("module");
        let oracle = build_baseline_artifact(&module).expect("oracle after error");
        let raw = build_baseline_artifact_raw(&module).expect("raw after error");
        assert_eq!(raw, oracle);
    }

    #[test]
    fn checked_in_scalar_wasm_corpus_matches_the_oracle() {
        let _guard = artifact_test_guard();
        for (name, wasm) in [
            (
                "fib-min",
                &include_bytes!("../../../../benchmarks/wasi/fib/fib_min.wasm")[..],
            ),
            (
                "sha256",
                &include_bytes!("../../../../benchmarks/wasi/sha256/sha256.wasm")[..],
            ),
            (
                "coremark",
                &include_bytes!("../../../../benchmarks/wasi/coremark/coremark.wasm")[..],
            ),
        ] {
            let module = Module::new(name, wasm).expect("module");
            let oracle = build_baseline_artifact(&module).expect("predecode oracle");
            let raw = build_baseline_artifact_raw(&module).expect("raw artifact");
            assert_eq!(raw, oracle, "{name}");
        }
    }
}
