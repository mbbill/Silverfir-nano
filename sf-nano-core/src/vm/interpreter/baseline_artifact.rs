//! Diagnostic-only eager baseline representation.
//!
//! This records the information a future raw-Wasm execution tier would need
//! without changing which code representation production builds or executes.
//! The collector is driven by the interpreter predecoder in tests so its
//! accepted input language stays identical to today's interpreter. A future
//! production baseline should move the structural collection into a cheaper
//! raw scan (or the predecoder's validation work) and must not materialize
//! [`super::instr::Instr`] cells for baseline-only functions.

use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::Module;
use crate::op_decoder::{
    raw_cursor::{RawBlockType, RawImmediate, RawOp},
    CatchClauseKind, DecodedOp, Immediate,
};
use crate::opcodes::{Opcode, OpcodeFB, WasmOpcode};
use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};

const UNRESOLVED: u32 = u32::MAX;

static ARTIFACT_BUILD_COUNT: AtomicUsize = AtomicUsize::new(0);
static ARTIFACT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(super) fn artifact_test_guard() -> std::sync::MutexGuard<'static, ()> {
    ARTIFACT_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BaselineArtifact {
    pub(crate) functions: Vec<Option<BaselineFunction>>,
    pub(crate) control_targets: Vec<ControlTarget>,
    pub(crate) br_tables: Vec<BrTableRange>,
    pub(crate) try_tables: Vec<TryTableMeta>,
    pub(crate) catches: Vec<CatchMeta>,
    pub(crate) direct_calls: Vec<DirectCallEdge>,
    pub(crate) indirect_calls: Vec<IndirectCallSite>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BaselineFunction {
    pub(crate) raw_code: Range<usize>,
    pub(crate) max_operand_height: u32,
    pub(crate) max_control_height: u32,
    pub(crate) control_targets: Range<usize>,
    pub(crate) br_tables: Range<usize>,
    pub(crate) try_tables: Range<usize>,
    pub(crate) catches: Range<usize>,
    pub(crate) direct_calls: Range<usize>,
    pub(crate) indirect_calls: Range<usize>,
}

impl BaselineFunction {
    /// Translate a function-relative side-table cursor into the module-wide
    /// flat arena. The cursor may equal the range end when control transfers
    /// past this function's final side entry.
    pub(crate) fn absolute_stp(&self, relative: u32) -> Option<usize> {
        let absolute = self.control_targets.start.checked_add(relative as usize)?;
        (absolute <= self.control_targets.end).then_some(absolute)
    }
}

/// One sequential side-table entry for a raw control transfer.
///
/// `target_stp` is relative to the function's first target. `source_pc` and
/// `target_pc` are byte offsets within the function expression, not absolute
/// module offsets. `target_stack_height` excludes the values preserved by
/// `keep_arity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ControlTarget {
    pub(crate) source_pc: u32,
    pub(crate) target_pc: u32,
    pub(crate) target_stp: u32,
    pub(crate) target_stack_height: u32,
    pub(crate) keep_arity: u32,
    pub(crate) eh_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BrTableRange {
    pub(crate) source_pc: u32,
    pub(crate) targets_start: u32,
    pub(crate) targets_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TryTableMeta {
    pub(crate) source_pc: u32,
    pub(crate) catches_start: u32,
    pub(crate) catches_len: u32,
    /// Number of active try frames after entering this try_table.
    pub(crate) active_eh_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CatchMeta {
    pub(crate) source_pc: u32,
    pub(crate) kind: CatchClauseKind,
    pub(crate) tag_index: Option<u32>,
    pub(crate) payload_arity: u32,
    pub(crate) forwards_exn: bool,
    pub(crate) target_pc: u32,
    pub(crate) target_stp: u32,
    pub(crate) target_stack_height: u32,
    pub(crate) keep_arity: u32,
    pub(crate) eh_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DirectCallEdge {
    pub(crate) caller: u32,
    pub(crate) callee: u32,
    pub(crate) source_pc: u32,
    pub(crate) tail: bool,
    pub(crate) loop_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IndirectCallKind {
    Table { table_index: u32 },
    Ref,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndirectCallSite {
    pub(crate) function: u32,
    pub(crate) source_pc: u32,
    pub(crate) expected_type: u32,
    pub(crate) kind: IndirectCallKind,
    pub(crate) tail: bool,
    pub(crate) loop_depth: u32,
}

impl BaselineArtifact {
    pub(super) fn new(function_count: usize) -> Self {
        ARTIFACT_BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut functions = Vec::with_capacity(function_count);
        functions.resize_with(function_count, || None);
        Self {
            functions,
            control_targets: Vec::new(),
            br_tables: Vec::new(),
            try_tables: Vec::new(),
            catches: Vec::new(),
            direct_calls: Vec::new(),
            indirect_calls: Vec::new(),
        }
    }

    pub(super) fn publish_function(
        &mut self,
        index: usize,
        mut parts: BaselineFunctionParts,
    ) -> Result<(), WasmError> {
        let control_start = self.control_targets.len();
        let control_base = to_u32(control_start, "baseline control target base overflow")?;
        for table in &mut parts.br_tables {
            table.targets_start = table
                .targets_start
                .checked_add(control_base)
                .ok_or_else(|| WasmError::invalid("baseline br_table range overflow"))?;
        }

        let catch_start = self.catches.len();
        let catch_base = to_u32(catch_start, "baseline catch base overflow")?;
        for table in &mut parts.try_tables {
            table.catches_start = table
                .catches_start
                .checked_add(catch_base)
                .ok_or_else(|| WasmError::invalid("baseline try_table range overflow"))?;
        }

        let br_start = self.br_tables.len();
        let try_start = self.try_tables.len();
        let direct_start = self.direct_calls.len();
        let indirect_start = self.indirect_calls.len();

        self.control_targets.extend(parts.control_targets);
        self.br_tables.extend(parts.br_tables);
        self.try_tables.extend(parts.try_tables);
        self.catches.extend(parts.catches);
        self.direct_calls.extend(parts.direct_calls);
        self.indirect_calls.extend(parts.indirect_calls);

        let function = self
            .functions
            .get_mut(index)
            .ok_or_else(|| WasmError::invalid("baseline function index overflow"))?;
        *function = Some(BaselineFunction {
            raw_code: parts.raw_code,
            max_operand_height: parts.max_operand_height,
            max_control_height: parts.max_control_height,
            control_targets: control_start..self.control_targets.len(),
            br_tables: br_start..self.br_tables.len(),
            try_tables: try_start..self.try_tables.len(),
            catches: catch_start..self.catches.len(),
            direct_calls: direct_start..self.direct_calls.len(),
            indirect_calls: indirect_start..self.indirect_calls.len(),
        });
        Ok(())
    }
}

pub(crate) fn artifact_build_count() -> usize {
    ARTIFACT_BUILD_COUNT.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ArtifactFrameKind {
    Function,
    Block,
    Loop,
    If,
    Else,
    TryTable,
}

struct ArtifactFrame {
    kind: ArtifactFrameKind,
    base: u32,
    params: u32,
    results: u32,
    label_pc: u32,
    label_stp: u32,
    if_false: Option<usize>,
    fixups: Vec<TargetPatch>,
}

#[derive(Clone, Copy)]
enum TargetPatch {
    Control(usize),
    Catch(usize),
}

#[derive(Clone, Copy)]
pub(super) struct TargetSpec {
    frame: usize,
    stack_height: u32,
    keep_arity: u32,
    eh_depth: u32,
}

pub(super) struct PendingCatch {
    kind: CatchClauseKind,
    tag_index: Option<u32>,
    payload_arity: u32,
    forwards_exn: bool,
    target: TargetSpec,
}

pub(super) enum ArtifactEvent {
    None,
    Push {
        kind: ArtifactFrameKind,
        source_pc: u32,
        next_pc: u32,
        base: u32,
        params: u32,
        results: u32,
        catches: Vec<PendingCatch>,
    },
    Else {
        source_pc: u32,
        next_pc: u32,
        target: TargetSpec,
    },
    End {
        source_pc: u32,
        next_pc: u32,
    },
    Branch {
        source_pc: u32,
        target: TargetSpec,
    },
    BrTable {
        source_pc: u32,
        targets: Vec<TargetSpec>,
    },
    DirectCall(DirectCallEdge),
    IndirectCall(IndirectCallSite),
}

pub(super) struct BaselineFunctionBuilder {
    function: u32,
    raw_code: Range<usize>,
    n_results: u32,
    frames: Vec<ArtifactFrame>,
    max_operand_height: u32,
    max_control_height: u32,
    control_targets: Vec<ControlTarget>,
    br_tables: Vec<BrTableRange>,
    try_tables: Vec<TryTableMeta>,
    catches: Vec<CatchMeta>,
    direct_calls: Vec<DirectCallEdge>,
    indirect_calls: Vec<IndirectCallSite>,
}

pub(super) struct BaselineFunctionParts {
    raw_code: Range<usize>,
    max_operand_height: u32,
    max_control_height: u32,
    control_targets: Vec<ControlTarget>,
    br_tables: Vec<BrTableRange>,
    try_tables: Vec<TryTableMeta>,
    catches: Vec<CatchMeta>,
    direct_calls: Vec<DirectCallEdge>,
    indirect_calls: Vec<IndirectCallSite>,
}

impl BaselineFunctionBuilder {
    pub(super) fn new(
        function: usize,
        raw_code: Range<usize>,
        n_results: usize,
    ) -> Result<Self, WasmError> {
        let function = to_u32(function, "baseline function index overflow")?;
        let n_results = to_u32(n_results, "baseline function result arity overflow")?;
        let mut builder = Self {
            function,
            raw_code,
            n_results,
            frames: Vec::new(),
            max_operand_height: 0,
            max_control_height: 0,
            control_targets: Vec::new(),
            br_tables: Vec::new(),
            try_tables: Vec::new(),
            catches: Vec::new(),
            direct_calls: Vec::new(),
            indirect_calls: Vec::new(),
        };
        builder.reset();
        Ok(builder)
    }

    /// Reset all contents before a safety re-decode of the same function.
    pub(super) fn reset(&mut self) {
        self.frames.clear();
        self.frames.push(ArtifactFrame {
            kind: ArtifactFrameKind::Function,
            base: 0,
            params: 0,
            results: self.n_results,
            label_pc: UNRESOLVED,
            label_stp: 0,
            if_false: None,
            fixups: Vec::new(),
        });
        self.max_operand_height = 0;
        self.max_control_height = 1;
        self.control_targets.clear();
        self.br_tables.clear();
        self.try_tables.clear();
        self.catches.clear();
        self.direct_calls.clear();
        self.indirect_calls.clear();
    }

    pub(super) fn plan(
        &self,
        module: &Module,
        decoded: &DecodedOp,
        stack_height: usize,
        dead: bool,
    ) -> Result<ArtifactEvent, WasmError> {
        let source_pc = to_u32(decoded.op_offset, "baseline source pc overflow")?;
        let next_pc = to_u32(decoded.next_op_offset, "baseline next pc overflow")?;
        let loop_depth = self.loop_depth()?;

        let base_op = match decoded.wasm_op {
            WasmOpcode::OP(op) => Some(op),
            _ => None,
        };
        if dead
            && !matches!(
                base_op,
                Some(
                    Opcode::BLOCK
                        | Opcode::LOOP
                        | Opcode::IF
                        | Opcode::TRY_TABLE
                        | Opcode::ELSE
                        | Opcode::END
                )
            )
        {
            return Ok(ArtifactEvent::None);
        }

        match decoded.wasm_op {
            WasmOpcode::OP(Opcode::BLOCK | Opcode::LOOP | Opcode::IF) => {
                let Immediate::Block(block_type) = &decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected block immediate",
                    ));
                };
                let (params, results) = super::predecode::block_arity(module.types(), block_type)?;
                // The production predecoder's dead structural path does not
                // pop an if condition: it only preserves nesting until a
                // merge can revive control. In particular, residual symbolic
                // values below `unreachable` stay in its stack. Mirror that
                // accepted representation exactly; a future full-validator
                // artifact may deliberately choose the validator's
                // polymorphic-stack height instead.
                let condition = usize::from(!dead && base_op == Some(Opcode::IF));
                let required = params as usize + condition;
                let base = if dead {
                    stack_height.saturating_sub(required)
                } else {
                    stack_height.checked_sub(required).ok_or_else(|| {
                        WasmError::invalid("baseline block stack height underflow")
                    })?
                };
                let kind = match base_op {
                    Some(Opcode::BLOCK) => ArtifactFrameKind::Block,
                    Some(Opcode::LOOP) => ArtifactFrameKind::Loop,
                    Some(Opcode::IF) => ArtifactFrameKind::If,
                    _ => unreachable!(),
                };
                Ok(ArtifactEvent::Push {
                    kind,
                    source_pc,
                    next_pc,
                    base: to_u32(base, "baseline block height overflow")?,
                    params,
                    results,
                    catches: Vec::new(),
                })
            }
            WasmOpcode::OP(Opcode::TRY_TABLE) => {
                let Immediate::TryTable {
                    block_type,
                    catches,
                } = &decoded.imm
                else {
                    return Err(WasmError::internal(
                        "baseline collector expected try_table immediate",
                    ));
                };
                let (params, results) = super::predecode::block_arity(module.types(), block_type)?;
                let base = if dead {
                    stack_height.saturating_sub(params as usize)
                } else {
                    stack_height.checked_sub(params as usize).ok_or_else(|| {
                        WasmError::invalid("baseline try_table stack height underflow")
                    })?
                };
                let mut pending = Vec::with_capacity(catches.len());
                for clause in catches {
                    let target = self.target_spec(clause.label_idx)?;
                    let (payload_arity, forwards_exn) = match clause.kind {
                        CatchClauseKind::Catch | CatchClauseKind::CatchRef => {
                            let tag = clause.tag_idx.ok_or_else(|| {
                                WasmError::invalid("baseline typed catch has no tag")
                            })?;
                            let arity = module
                                .tags()
                                .get(tag as usize)
                                .map(|tag| tag.func_type().params().len())
                                .ok_or_else(|| WasmError::invalid("baseline catch tag overflow"))?;
                            (
                                to_u32(arity, "baseline catch payload arity overflow")?,
                                clause.kind == CatchClauseKind::CatchRef,
                            )
                        }
                        CatchClauseKind::CatchAll => (0, false),
                        CatchClauseKind::CatchAllRef => (0, true),
                    };
                    pending.push(PendingCatch {
                        kind: clause.kind,
                        tag_index: clause.tag_idx,
                        payload_arity,
                        forwards_exn,
                        target,
                    });
                }
                Ok(ArtifactEvent::Push {
                    kind: ArtifactFrameKind::TryTable,
                    source_pc,
                    next_pc,
                    base: to_u32(base, "baseline try_table height overflow")?,
                    params,
                    results,
                    catches: pending,
                })
            }
            WasmOpcode::OP(Opcode::ELSE) => Ok(ArtifactEvent::Else {
                source_pc,
                next_pc,
                target: self.target_spec(0)?,
            }),
            WasmOpcode::OP(Opcode::END) => Ok(ArtifactEvent::End { source_pc, next_pc }),
            WasmOpcode::OP(
                Opcode::BR | Opcode::BR_IF | Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL,
            ) => {
                let Immediate::LabelIndex(depth) = decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected branch label",
                    ));
                };
                Ok(ArtifactEvent::Branch {
                    source_pc,
                    target: self.target_spec(depth)?,
                })
            }
            WasmOpcode::FB(OpcodeFB::BR_ON_CAST | OpcodeFB::BR_ON_CAST_FAIL) => {
                let Immediate::BrOnCast { label_idx, .. } = decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected br_on_cast label",
                    ));
                };
                Ok(ArtifactEvent::Branch {
                    source_pc,
                    target: self.target_spec(label_idx)?,
                })
            }
            WasmOpcode::OP(Opcode::BR_TABLE) => {
                let Immediate::BrLabels(labels, default) = &decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected br_table labels",
                    ));
                };
                let mut targets = Vec::with_capacity(labels.len() + 1);
                for &label in labels {
                    targets.push(self.target_spec(label)?);
                }
                targets.push(self.target_spec(*default)?);
                Ok(ArtifactEvent::BrTable { source_pc, targets })
            }
            WasmOpcode::OP(Opcode::RETURN) => Ok(ArtifactEvent::Branch {
                source_pc,
                target: self.function_target()?,
            }),
            WasmOpcode::OP(Opcode::CALL | Opcode::RETURN_CALL) => {
                let Immediate::FunctionIndex(callee) = decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected direct call target",
                    ));
                };
                Ok(ArtifactEvent::DirectCall(DirectCallEdge {
                    caller: self.function,
                    callee,
                    source_pc,
                    tail: base_op == Some(Opcode::RETURN_CALL),
                    loop_depth,
                }))
            }
            WasmOpcode::OP(Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT) => {
                let Immediate::CallIndirectArgs { typeidx, tableidx } = decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected indirect call immediate",
                    ));
                };
                Ok(ArtifactEvent::IndirectCall(IndirectCallSite {
                    function: self.function,
                    source_pc,
                    expected_type: typeidx,
                    kind: IndirectCallKind::Table {
                        table_index: tableidx,
                    },
                    tail: base_op == Some(Opcode::RETURN_CALL_INDIRECT),
                    loop_depth,
                }))
            }
            WasmOpcode::OP(Opcode::CALL_REF | Opcode::RETURN_CALL_REF) => {
                let Immediate::TypeIndex(expected_type) = decoded.imm else {
                    return Err(WasmError::internal(
                        "baseline collector expected call_ref type",
                    ));
                };
                Ok(ArtifactEvent::IndirectCall(IndirectCallSite {
                    function: self.function,
                    source_pc,
                    expected_type,
                    kind: IndirectCallKind::Ref,
                    tail: base_op == Some(Opcode::RETURN_CALL_REF),
                    loop_depth,
                }))
            }
            _ => Ok(ArtifactEvent::None),
        }
    }

    /// Plan the same side-table event directly from the allocation-free raw
    /// cursor. This is intentionally independent of `DecodedOp` and the
    /// folded predecoder: the only shared state is this artifact assembler.
    pub(super) fn plan_raw(
        &self,
        module: &Module,
        decoded: &RawOp<'_>,
        stack_height: usize,
        dead: bool,
    ) -> Result<ArtifactEvent, WasmError> {
        let source_pc = to_u32(decoded.start, "baseline source pc overflow")?;
        let next_pc = to_u32(decoded.end, "baseline next pc overflow")?;
        let loop_depth = self.loop_depth()?;
        let base_op = match decoded.wasm_op {
            WasmOpcode::OP(op) => Some(op),
            _ => None,
        };
        if dead
            && !matches!(
                base_op,
                Some(
                    Opcode::BLOCK
                        | Opcode::LOOP
                        | Opcode::IF
                        | Opcode::TRY_TABLE
                        | Opcode::ELSE
                        | Opcode::END
                )
            )
        {
            return Ok(ArtifactEvent::None);
        }

        match decoded.wasm_op {
            WasmOpcode::OP(Opcode::BLOCK | Opcode::LOOP | Opcode::IF) => {
                let RawImmediate::Block(block) = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected block immediate",
                    ));
                };
                let (params, results) = raw_block_arity(module, block)?;
                let condition = usize::from(!dead && base_op == Some(Opcode::IF));
                let required = params as usize + condition;
                let base = if dead {
                    stack_height.saturating_sub(required)
                } else {
                    stack_height.checked_sub(required).ok_or_else(|| {
                        WasmError::invalid("raw baseline block stack height underflow")
                    })?
                };
                let kind = match base_op {
                    Some(Opcode::BLOCK) => ArtifactFrameKind::Block,
                    Some(Opcode::LOOP) => ArtifactFrameKind::Loop,
                    Some(Opcode::IF) => ArtifactFrameKind::If,
                    _ => unreachable!(),
                };
                Ok(ArtifactEvent::Push {
                    kind,
                    source_pc,
                    next_pc,
                    base: to_u32(base, "raw baseline block height overflow")?,
                    params,
                    results,
                    catches: Vec::new(),
                })
            }
            WasmOpcode::OP(Opcode::TRY_TABLE) => {
                let RawImmediate::TryTable { block, catches } = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected try_table immediate",
                    ));
                };
                let (params, results) = raw_block_arity(module, block)?;
                let base = if dead {
                    stack_height.saturating_sub(params as usize)
                } else {
                    stack_height.checked_sub(params as usize).ok_or_else(|| {
                        WasmError::invalid("raw baseline try_table stack height underflow")
                    })?
                };
                let mut pending = Vec::with_capacity(catches.len() as usize);
                for catch in catches.iter() {
                    let catch = catch?;
                    let target = self.target_spec(catch.label_depth)?;
                    let (payload_arity, forwards_exn) = match catch.kind {
                        CatchClauseKind::Catch | CatchClauseKind::CatchRef => {
                            let tag = catch.tag_index.ok_or_else(|| {
                                WasmError::invalid("raw baseline typed catch has no tag")
                            })?;
                            let arity = module
                                .tags()
                                .get(tag as usize)
                                .map(|tag| tag.func_type().params().len())
                                .ok_or_else(|| {
                                    WasmError::invalid("raw baseline catch tag overflow")
                                })?;
                            (
                                to_u32(arity, "raw baseline catch payload arity overflow")?,
                                catch.kind == CatchClauseKind::CatchRef,
                            )
                        }
                        CatchClauseKind::CatchAll => (0, false),
                        CatchClauseKind::CatchAllRef => (0, true),
                    };
                    pending.push(PendingCatch {
                        kind: catch.kind,
                        tag_index: catch.tag_index,
                        payload_arity,
                        forwards_exn,
                        target,
                    });
                }
                Ok(ArtifactEvent::Push {
                    kind: ArtifactFrameKind::TryTable,
                    source_pc,
                    next_pc,
                    base: to_u32(base, "raw baseline try_table height overflow")?,
                    params,
                    results,
                    catches: pending,
                })
            }
            WasmOpcode::OP(Opcode::ELSE) => Ok(ArtifactEvent::Else {
                source_pc,
                next_pc,
                target: self.target_spec(0)?,
            }),
            WasmOpcode::OP(Opcode::END) => Ok(ArtifactEvent::End { source_pc, next_pc }),
            WasmOpcode::OP(
                Opcode::BR | Opcode::BR_IF | Opcode::BR_ON_NULL | Opcode::BR_ON_NON_NULL,
            ) => {
                let RawImmediate::LabelIndex(depth) = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected branch label",
                    ));
                };
                Ok(ArtifactEvent::Branch {
                    source_pc,
                    target: self.target_spec(depth)?,
                })
            }
            WasmOpcode::OP(Opcode::BR_TABLE) => {
                let RawImmediate::BrTable { labels, default } = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected br_table labels",
                    ));
                };
                let mut targets = Vec::with_capacity(labels.len() as usize + 1);
                for label in labels.iter() {
                    targets.push(self.target_spec(label?)?);
                }
                targets.push(self.target_spec(default)?);
                Ok(ArtifactEvent::BrTable { source_pc, targets })
            }
            WasmOpcode::OP(Opcode::RETURN) => Ok(ArtifactEvent::Branch {
                source_pc,
                target: self.function_target()?,
            }),
            WasmOpcode::OP(Opcode::CALL | Opcode::RETURN_CALL) => {
                let RawImmediate::FunctionIndex(callee) = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected direct call target",
                    ));
                };
                Ok(ArtifactEvent::DirectCall(DirectCallEdge {
                    caller: self.function,
                    callee,
                    source_pc,
                    tail: base_op == Some(Opcode::RETURN_CALL),
                    loop_depth,
                }))
            }
            WasmOpcode::OP(Opcode::CALL_INDIRECT | Opcode::RETURN_CALL_INDIRECT) => {
                let RawImmediate::CallIndirect { typeidx, tableidx } = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected indirect call immediate",
                    ));
                };
                Ok(ArtifactEvent::IndirectCall(IndirectCallSite {
                    function: self.function,
                    source_pc,
                    expected_type: typeidx,
                    kind: IndirectCallKind::Table {
                        table_index: tableidx,
                    },
                    tail: base_op == Some(Opcode::RETURN_CALL_INDIRECT),
                    loop_depth,
                }))
            }
            WasmOpcode::OP(Opcode::CALL_REF | Opcode::RETURN_CALL_REF) => {
                let RawImmediate::TypeIndex(expected_type) = decoded.imm else {
                    return Err(WasmError::internal(
                        "raw baseline collector expected call_ref type",
                    ));
                };
                Ok(ArtifactEvent::IndirectCall(IndirectCallSite {
                    function: self.function,
                    source_pc,
                    expected_type,
                    kind: IndirectCallKind::Ref,
                    tail: base_op == Some(Opcode::RETURN_CALL_REF),
                    loop_depth,
                }))
            }
            _ => Ok(ArtifactEvent::None),
        }
    }

    pub(super) fn commit(
        &mut self,
        event: ArtifactEvent,
        stack_height: usize,
    ) -> Result<(), WasmError> {
        match event {
            ArtifactEvent::None => {}
            ArtifactEvent::Push {
                kind,
                source_pc,
                next_pc,
                base,
                params,
                results,
                catches,
            } => {
                if kind == ArtifactFrameKind::TryTable {
                    let catches_start = to_u32(
                        self.catches.len(),
                        "baseline try_table catch range overflow",
                    )?;
                    for catch in catches {
                        self.add_catch(source_pc, catch)?;
                    }
                    self.try_tables.push(TryTableMeta {
                        source_pc,
                        catches_start,
                        catches_len: to_u32(
                            self.catches.len() - catches_start as usize,
                            "baseline try_table catch length overflow",
                        )?,
                        active_eh_depth: self
                            .eh_depth()?
                            .checked_add(1)
                            .ok_or_else(|| WasmError::invalid("baseline EH depth overflow"))?,
                    });
                }

                let label_stp = self.current_stp()?;
                let label_pc = if kind == ArtifactFrameKind::Loop {
                    next_pc
                } else {
                    UNRESOLVED
                };
                self.frames.push(ArtifactFrame {
                    kind,
                    base,
                    params,
                    results,
                    label_pc,
                    label_stp,
                    if_false: None,
                    fixups: Vec::new(),
                });
                if kind == ArtifactFrameKind::If {
                    // Unlike a branch to the if label, a false condition
                    // enters the else arm with the block parameters. Keep it
                    // separate from the frame's end fixups: an `else` patches
                    // it immediately, while an if-without-else patches it at
                    // `end`.
                    let target = self.control_targets.len();
                    self.control_targets.push(ControlTarget {
                        source_pc,
                        target_pc: UNRESOLVED,
                        target_stp: UNRESOLVED,
                        target_stack_height: base,
                        keep_arity: params,
                        eh_depth: self.eh_depth()?,
                    });
                    self.frames.last_mut().expect("if frame").if_false = Some(target);
                }
            }
            ArtifactEvent::Else {
                source_pc,
                next_pc,
                target,
            } => {
                self.add_control_target(source_pc, target)?;
                let target_stp = self.current_stp()?;
                let frame = self
                    .frames
                    .last_mut()
                    .ok_or_else(|| WasmError::invalid("baseline else without frame"))?;
                if frame.kind != ArtifactFrameKind::If {
                    return Err(WasmError::invalid("baseline else without if"));
                }
                let false_target = frame
                    .if_false
                    .take()
                    .ok_or_else(|| WasmError::invalid("baseline if target already patched"))?;
                frame.kind = ArtifactFrameKind::Else;
                self.patch(TargetPatch::Control(false_target), next_pc, target_stp)?;
            }
            ArtifactEvent::End { source_pc, next_pc } => {
                let frame = self
                    .frames
                    .pop()
                    .ok_or_else(|| WasmError::invalid("baseline end without frame"))?;
                let target_pc = if frame.kind == ArtifactFrameKind::Function {
                    source_pc
                } else {
                    next_pc
                };
                let target_stp = self.current_stp()?;
                if let Some(false_target) = frame.if_false {
                    self.patch(TargetPatch::Control(false_target), target_pc, target_stp)?;
                }
                for fixup in frame.fixups {
                    self.patch(fixup, target_pc, target_stp)?;
                }
            }
            ArtifactEvent::Branch { source_pc, target } => {
                self.add_control_target(source_pc, target)?;
            }
            ArtifactEvent::BrTable { source_pc, targets } => {
                let targets_start = to_u32(
                    self.control_targets.len(),
                    "baseline br_table target range overflow",
                )?;
                let targets_len = to_u32(targets.len(), "baseline br_table arity overflow")?;
                for target in targets {
                    self.add_control_target(source_pc, target)?;
                }
                self.br_tables.push(BrTableRange {
                    source_pc,
                    targets_start,
                    targets_len,
                });
            }
            ArtifactEvent::DirectCall(edge) => self.direct_calls.push(edge),
            ArtifactEvent::IndirectCall(site) => self.indirect_calls.push(site),
        }
        self.max_operand_height = self
            .max_operand_height
            .max(to_u32(stack_height, "baseline operand height overflow")?);
        self.max_control_height = self.max_control_height.max(to_u32(
            self.frames.len(),
            "baseline control height overflow",
        )?);
        Ok(())
    }

    pub(super) fn observe_height(&mut self, stack_height: usize) -> Result<(), WasmError> {
        self.commit(ArtifactEvent::None, stack_height)
    }

    pub(super) fn finish(self) -> Result<BaselineFunctionParts, WasmError> {
        if !self.frames.is_empty() {
            return Err(WasmError::invalid(
                "baseline artifact ended with open control frames",
            ));
        }
        if self
            .control_targets
            .iter()
            .any(|target| target.target_pc == UNRESOLVED || target.target_stp == UNRESOLVED)
            || self
                .catches
                .iter()
                .any(|catch| catch.target_pc == UNRESOLVED || catch.target_stp == UNRESOLVED)
        {
            return Err(WasmError::invalid(
                "baseline artifact has unresolved target",
            ));
        }
        Ok(BaselineFunctionParts {
            raw_code: self.raw_code,
            max_operand_height: self.max_operand_height,
            max_control_height: self.max_control_height,
            control_targets: self.control_targets,
            br_tables: self.br_tables,
            try_tables: self.try_tables,
            catches: self.catches,
            direct_calls: self.direct_calls,
            indirect_calls: self.indirect_calls,
        })
    }

    fn function_target(&self) -> Result<TargetSpec, WasmError> {
        let depth = self
            .frames
            .len()
            .checked_sub(1)
            .ok_or_else(|| WasmError::invalid("baseline function frame missing"))?;
        self.target_spec(to_u32(depth, "baseline function label depth overflow")?)
    }

    fn target_spec(&self, depth: u32) -> Result<TargetSpec, WasmError> {
        let frame = self
            .frames
            .len()
            .checked_sub(depth as usize + 1)
            .ok_or_else(|| WasmError::invalid("baseline branch label overflow"))?;
        let target = &self.frames[frame];
        let keep_arity = if target.kind == ArtifactFrameKind::Loop {
            target.params
        } else {
            target.results
        };
        let mut eh_depth = 0u32;
        for active in &self.frames[..=frame] {
            if active.kind == ArtifactFrameKind::TryTable {
                eh_depth = eh_depth
                    .checked_add(1)
                    .ok_or_else(|| WasmError::invalid("baseline EH depth overflow"))?;
            }
        }
        if target.kind == ArtifactFrameKind::TryTable {
            eh_depth -= 1;
        }
        Ok(TargetSpec {
            frame,
            stack_height: target.base,
            keep_arity,
            eh_depth,
        })
    }

    fn add_control_target(
        &mut self,
        source_pc: u32,
        target: TargetSpec,
    ) -> Result<usize, WasmError> {
        let (target_pc, target_stp) = self.target_location(target.frame)?;
        let index = self.control_targets.len();
        self.control_targets.push(ControlTarget {
            source_pc,
            target_pc,
            target_stp,
            target_stack_height: target.stack_height,
            keep_arity: target.keep_arity,
            eh_depth: target.eh_depth,
        });
        if target_pc == UNRESOLVED {
            self.frames[target.frame]
                .fixups
                .push(TargetPatch::Control(index));
        }
        Ok(index)
    }

    fn add_catch(&mut self, source_pc: u32, pending: PendingCatch) -> Result<(), WasmError> {
        let (target_pc, target_stp) = self.target_location(pending.target.frame)?;
        let index = self.catches.len();
        self.catches.push(CatchMeta {
            source_pc,
            kind: pending.kind,
            tag_index: pending.tag_index,
            payload_arity: pending.payload_arity,
            forwards_exn: pending.forwards_exn,
            target_pc,
            target_stp,
            target_stack_height: pending.target.stack_height,
            keep_arity: pending.target.keep_arity,
            eh_depth: pending.target.eh_depth,
        });
        if target_pc == UNRESOLVED {
            self.frames[pending.target.frame]
                .fixups
                .push(TargetPatch::Catch(index));
        }
        Ok(())
    }

    fn target_location(&self, frame: usize) -> Result<(u32, u32), WasmError> {
        let frame = self
            .frames
            .get(frame)
            .ok_or_else(|| WasmError::invalid("baseline target frame overflow"))?;
        if frame.kind == ArtifactFrameKind::Loop {
            Ok((frame.label_pc, frame.label_stp))
        } else {
            Ok((UNRESOLVED, UNRESOLVED))
        }
    }

    fn patch(
        &mut self,
        patch: TargetPatch,
        target_pc: u32,
        target_stp: u32,
    ) -> Result<(), WasmError> {
        match patch {
            TargetPatch::Control(index) => {
                let target = self
                    .control_targets
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("baseline control target fixup overflow"))?;
                target.target_pc = target_pc;
                target.target_stp = target_stp;
            }
            TargetPatch::Catch(index) => {
                let target = self
                    .catches
                    .get_mut(index)
                    .ok_or_else(|| WasmError::invalid("baseline catch fixup overflow"))?;
                target.target_pc = target_pc;
                target.target_stp = target_stp;
            }
        }
        Ok(())
    }

    fn current_stp(&self) -> Result<u32, WasmError> {
        to_u32(
            self.control_targets.len(),
            "baseline side-table pointer overflow",
        )
    }

    fn loop_depth(&self) -> Result<u32, WasmError> {
        to_u32(
            self.frames
                .iter()
                .filter(|frame| frame.kind == ArtifactFrameKind::Loop)
                .count(),
            "baseline loop depth overflow",
        )
    }

    fn eh_depth(&self) -> Result<u32, WasmError> {
        to_u32(
            self.frames
                .iter()
                .filter(|frame| frame.kind == ArtifactFrameKind::TryTable)
                .count(),
            "baseline EH depth overflow",
        )
    }
}

fn to_u32(value: usize, message: &'static str) -> Result<u32, WasmError> {
    u32::try_from(value).map_err(|_| WasmError::invalid(message))
}

fn raw_block_arity(module: &Module, block: RawBlockType) -> Result<(u32, u32), WasmError> {
    match block {
        RawBlockType::Empty => Ok((0, 0)),
        RawBlockType::Value(_) => Ok((0, 1)),
        RawBlockType::TypeIndex(index) => module
            .types()
            .get_function_type(index as u32)
            .map(|function| {
                (
                    function.params().len() as u32,
                    function.results().len() as u32,
                )
            })
            .ok_or_else(|| WasmError::invalid("raw baseline block type index out of range")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::module::Module;
    use crate::vm::engine::{Engine, Tier};
    use crate::vm::interpreter::instr::Op;
    use crate::vm::interpreter::predecode::{
        build_baseline_artifact, predecode_function, PredecodedFunction,
    };
    use crate::vm::tag::TagIdentity;
    use crate::vm::value::RefValue;
    use crate::{Instance, Value};
    use std::vec::Vec as StdVec;

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        artifact_test_guard()
    }

    fn parse_artifact(wat: &str) -> (StdVec<u8>, Module, BaselineArtifact) {
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new("baseline-artifact", &wasm).expect("module");
        let artifact = build_baseline_artifact(&module).expect("baseline artifact");
        (wasm, module, artifact)
    }

    fn predecoded(module: &Module, function: usize) -> PredecodedFunction {
        let tags: StdVec<TagIdentity> = module
            .tags()
            .iter()
            .map(|_| TagIdentity::mint_fresh())
            .collect();
        let handles: StdVec<RefValue> = (0..module.functions().len()).map(RefValue::new).collect();
        predecode_function(module, &tags, &handles, function).expect("predecoded function")
    }

    fn invoke(wasm: &[u8], name: &str, args: &[Value]) -> Result<StdVec<Value>, WasmError> {
        let engine = Engine::new(Config::new().tier(Tier::Interp)).expect("interp engine");
        let mut instance = Instance::new(&engine, wasm, &[]).expect("instance");
        instance.invoke(name, args).map(StdVec::from)
    }

    #[test]
    fn if_else_raw_targets_are_a_stable_golden() {
        let _guard = test_guard();
        let before = artifact_build_count();
        let (_wasm, module, artifact) = parse_artifact(
            r#"(module
                (func (export "run") (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 7
                    else
                        i32.const 9
                    end))"#,
        );
        assert_eq!(artifact_build_count(), before + 1);
        assert_eq!(artifact.functions.len(), 1);
        let function = artifact.functions[0].as_ref().expect("local function");
        let spec = module.functions()[0].spec().expect("function spec");
        assert_eq!(
            function.raw_code,
            spec.code_offset()..spec.code_offset() + spec.code().len()
        );
        assert_eq!(function.max_operand_height, 1);
        assert_eq!(function.max_control_height, 2);
        assert_eq!(function.control_targets, 0..2);
        assert_eq!(function.br_tables, 0..0);
        assert_eq!(function.try_tables, 0..0);
        assert_eq!(function.catches, 0..0);
        assert_eq!(function.direct_calls, 0..0);
        assert_eq!(function.indirect_calls, 0..0);
        assert_eq!(
            artifact.control_targets.as_slice(),
            &[
                ControlTarget {
                    source_pc: 2,
                    target_pc: 7,
                    target_stp: 2,
                    target_stack_height: 0,
                    keep_arity: 0,
                    eh_depth: 0,
                },
                ControlTarget {
                    source_pc: 6,
                    target_pc: 10,
                    target_stp: 2,
                    target_stack_height: 0,
                    keep_arity: 1,
                    eh_depth: 0,
                },
            ]
        );
        assert!(artifact.br_tables.is_empty());
        assert!(artifact.try_tables.is_empty());
        assert!(artifact.catches.is_empty());
        assert!(artifact.direct_calls.is_empty());
        assert!(artifact.indirect_calls.is_empty());

        let native = predecoded(&module, 0);
        let ops: StdVec<Op> = native.code.iter().map(|ins| ins.op).collect();
        assert_eq!(
            ops,
            [Op::BrIfNot, Op::MovConst, Op::Br, Op::MovConst, Op::Return]
        );
        assert_eq!(native.code[0].c, 3);
        assert_eq!(native.code[2].c, 4);
    }

    #[test]
    fn dead_if_keeps_the_predecoder_residual_stack_height() {
        let _guard = test_guard();
        let (wasm, module, artifact) = parse_artifact(
            r#"(module
                (func (export "run")
                    i32.const 99
                    unreachable
                    if
                        nop
                    else
                        nop
                    end))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        assert_eq!(function.max_operand_height, 1);
        assert_eq!(function.max_control_height, 2);
        let targets = &artifact.control_targets[function.control_targets.clone()];
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(|target| {
            target.target_stack_height == 1 && target.keep_arity == 0 && target.target_stp == 2
        }));

        let native = predecoded(&module, 0);
        assert!(native.code.iter().any(|ins| ins.op == Op::Unreachable));
        assert!(matches!(invoke(&wasm, "run", &[]), Err(WasmError::Trap(_))));
    }

    #[test]
    fn loop_backedge_agrees_with_the_native_target_oracle() {
        let _guard = test_guard();
        let (wasm, module, artifact) = parse_artifact(
            r#"(module
                (func (export "run") (local i32)
                    (loop $l
                        local.get 0
                        i32.const 1
                        i32.add
                        local.set 0
                        local.get 0
                        i32.const 10
                        i32.lt_u
                        br_if $l)))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        assert_eq!(function.max_operand_height, 2);
        assert_eq!(function.max_control_height, 2);
        let targets = &artifact.control_targets[function.control_targets.clone()];
        assert_eq!(targets.len(), 1);
        assert_eq!(
            targets[0],
            ControlTarget {
                source_pc: 14,
                target_pc: 2,
                target_stp: 0,
                target_stack_height: 0,
                keep_arity: 0,
                eh_depth: 0,
            }
        );

        let native = predecoded(&module, 0);
        let branch = native
            .code
            .iter()
            .find(|ins| ins.op == Op::I32_BrLtU)
            .expect("fused native backedge");
        assert_eq!(branch.c, 0);
        assert!(invoke(&wasm, "run", &[]).expect("run").is_empty());
    }

    #[test]
    fn br_table_and_multivalue_branch_ranges_are_flat_goldens() {
        let _guard = test_guard();
        let (wasm, module, artifact) = parse_artifact(
            r#"(module
                (func (export "run")
                      (param $discarded i32) (param $value i32)
                      (param $selector i32) (result i32)
                    (block $exit (result i32)
                        local.get $discarded
                        local.get $value
                        local.get $selector
                        br_table $exit $exit $exit)))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        assert_eq!(function.br_tables, 0..1);
        let table = artifact.br_tables[0];
        assert_eq!(table.targets_len, 3);
        let start = table.targets_start as usize;
        let targets = &artifact.control_targets[start..start + table.targets_len as usize];
        assert_eq!(targets.len(), 3);
        assert!(targets.iter().all(|target| {
            target.source_pc == table.source_pc
                && target.target_pc == targets[0].target_pc
                && target.target_stp == 3
                && target.target_stack_height == 0
                && target.keep_arity == 1
                && target.eh_depth == 0
        }));

        let native = predecoded(&module, 0);
        let native_targets = native.br_table(0).expect("native br_table");
        assert_eq!(native_targets.len(), targets.len());
        assert!(native_targets
            .iter()
            .all(|target| *target == native_targets[0]));
        assert_eq!(
            invoke(
                &wasm,
                "run",
                &[Value::I32(11), Value::I32(22), Value::I32(1)],
            )
            .expect("run"),
            [Value::I32(22)]
        );

        let (_wasm, _module, multivalue) = parse_artifact(
            r#"(module
                (type $pair (func (param i32 i32) (result i32 i32)))
                (func (export "run") (result i32)
                    i32.const 1
                    i32.const 2
                    (block (type $pair)
                        br 0
                        unreachable)
                    i32.add))"#,
        );
        let function = multivalue.functions[0].as_ref().expect("function");
        let branch = &multivalue.control_targets[function.control_targets.clone()][0];
        assert_eq!(branch.target_stack_height, 0);
        assert_eq!(branch.keep_arity, 2);
    }

    #[test]
    fn direct_tail_and_dynamic_call_facts_keep_loop_depth_and_type() {
        let _guard = test_guard();
        let (wasm, _module, artifact) = parse_artifact(
            r#"(module
                (type $u (func (param i32) (result i32)))
                (table 1 funcref)
                (func $id (type $u) (param i32) (result i32) local.get 0)
                (elem (i32.const 0) $id)
                (func $tail (type $u) (param i32) (result i32)
                    local.get 0
                    return_call $id)
                (func (export "direct") (type $u) (param i32) (result i32)
                    (block $done
                        (loop $hot
                            local.get 0
                            call $tail
                            drop
                            br $done))
                    local.get 0)
                (func (export "indirect") (type $u) (param i32) (result i32)
                    local.get 0
                    i32.const 0
                    call_indirect (type $u))
                (func (export "by_ref") (type $u) (param i32) (result i32)
                    local.get 0
                    ref.func $id
                    call_ref $u)
                (func (export "tail_ref") (type $u) (param i32) (result i32)
                    local.get 0
                    ref.func $id
                    return_call_ref $u))"#,
        );
        assert_eq!(artifact.direct_calls.len(), 2);
        assert_eq!(
            artifact.direct_calls.as_slice(),
            &[
                DirectCallEdge {
                    caller: 1,
                    callee: 0,
                    source_pc: 2,
                    tail: true,
                    loop_depth: 0,
                },
                DirectCallEdge {
                    caller: 2,
                    callee: 1,
                    source_pc: 6,
                    tail: false,
                    loop_depth: 1,
                },
            ]
        );
        assert_eq!(artifact.indirect_calls.len(), 3);
        assert_eq!(
            artifact
                .indirect_calls
                .iter()
                .map(|site| (
                    site.function,
                    site.expected_type,
                    site.kind,
                    site.tail,
                    site.loop_depth
                ))
                .collect::<StdVec<_>>(),
            [
                (3, 0, IndirectCallKind::Table { table_index: 0 }, false, 0),
                (4, 0, IndirectCallKind::Ref, false, 0),
                (5, 0, IndirectCallKind::Ref, true, 0),
            ]
        );
        for export in ["direct", "indirect", "by_ref", "tail_ref"] {
            assert_eq!(
                invoke(&wasm, export, &[Value::I32(37)]).expect(export),
                [Value::I32(37)]
            );
        }
    }

    #[test]
    fn explicit_return_and_typed_catch_preserve_stack_and_eh_targets() {
        let _guard = test_guard();
        let (return_wasm, return_module, returns) = parse_artifact(
            r#"(module
                (func (export "run") (param i32) (result i32)
                    local.get 0
                    return
                    unreachable))"#,
        );
        let function = returns.functions[0].as_ref().expect("function");
        let targets = &returns.control_targets[function.control_targets.clone()];
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].source_pc, 2);
        assert_eq!(targets[0].target_pc, 4);
        assert_eq!(targets[0].target_stp, 1);
        assert_eq!(targets[0].target_stack_height, 0);
        assert_eq!(targets[0].keep_arity, 1);
        let native = predecoded(&return_module, 0);
        let native_return = native
            .code
            .iter()
            .find(|ins| ins.op == Op::Return)
            .expect("native return");
        assert_eq!((native_return.a, native_return.b), (1, 1));
        assert_eq!(
            invoke(&return_wasm, "run", &[Value::I32(91)]).expect("run"),
            [Value::I32(91)]
        );

        let (catch_wasm, catch_module, catches) = parse_artifact(
            r#"(module
                (tag $e (param i32))
                (func (export "catch") (result i32)
                    (block $h (result i32)
                        (try_table (result i32) (catch $e $h)
                            (throw $e (i32.const 7))
                            (i32.const 2))
                        (return))
                    (return)))"#,
        );
        let catch = catches.catches[0];
        assert_eq!(catch.kind, CatchClauseKind::Catch);
        assert_eq!(catch.tag_index, Some(0));
        assert_eq!(catch.payload_arity, 1);
        assert!(!catch.forwards_exn);
        assert_eq!(catch.target_stack_height, 0);
        assert_eq!(catch.keep_arity, 1);
        assert_eq!(catch.eh_depth, 0);
        let native = predecoded(&catch_module, 0);
        let throw_pc = native
            .code
            .iter()
            .position(|ins| ins.op == Op::Throw)
            .expect("throw cell") as u32;
        let handler = native.exception_handlers_at(throw_pc)[0];
        assert_eq!(handler.payload_arity, catch.payload_arity);
        assert_eq!(handler.forwards_exn, catch.forwards_exn);
        assert_eq!(handler.target_base, catch.target_stack_height);
        assert_eq!(
            invoke(&catch_wasm, "catch", &[]).expect("catch"),
            [Value::I32(7)]
        );
    }

    #[test]
    fn nested_try_metadata_records_target_eh_depth() {
        let _guard = test_guard();
        let (_wasm, _module, artifact) = parse_artifact(
            r#"(module
                (func
                    (block $outer
                        (try_table (catch_all $outer)
                            (block $inner
                                (try_table (catch_all $inner)
                                    unreachable))))))"#,
        );
        assert_eq!(artifact.try_tables.len(), 2);
        assert_eq!(artifact.try_tables[0].active_eh_depth, 1);
        assert_eq!(artifact.try_tables[1].active_eh_depth, 2);
        assert_eq!(artifact.catches.len(), 2);
        assert_eq!(artifact.catches[0].kind, CatchClauseKind::CatchAll);
        assert_eq!(artifact.catches[0].eh_depth, 0);
        assert_eq!(artifact.catches[1].kind, CatchClauseKind::CatchAll);
        assert_eq!(artifact.catches[1].eh_depth, 1);
    }

    #[test]
    fn try_table_catch_ref_metadata_matches_native_exception_oracle() {
        let _guard = test_guard();
        let (wasm, module, artifact) = parse_artifact(
            r#"(module
                (tag $e)
                (func (export "go")
                    (block $h (result exnref)
                        (try_table (catch_ref $e $h)
                            (throw $e))
                        unreachable)
                    throw_ref))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        assert_eq!(function.try_tables, 0..1);
        assert_eq!(function.catches, 0..1);
        assert_eq!(artifact.try_tables[0].catches_start, 0);
        assert_eq!(artifact.try_tables[0].catches_len, 1);
        assert_eq!(artifact.try_tables[0].active_eh_depth, 1);
        let catch = artifact.catches[0];
        assert_eq!(catch.source_pc, artifact.try_tables[0].source_pc);
        assert_eq!(catch.kind, CatchClauseKind::CatchRef);
        assert_eq!(catch.tag_index, Some(0));
        assert_eq!(catch.payload_arity, 0);
        assert!(catch.forwards_exn);
        assert_eq!(catch.target_stack_height, 0);
        assert_eq!(catch.keep_arity, 1);
        assert_eq!(catch.eh_depth, 0);
        assert_ne!(catch.target_pc, UNRESOLVED);
        assert_ne!(catch.target_stp, UNRESOLVED);

        let native = predecoded(&module, 0);
        let throw_pc = native
            .code
            .iter()
            .position(|ins| ins.op == Op::Throw)
            .expect("throw cell") as u32;
        let handlers = native.exception_handlers_at(throw_pc);
        assert_eq!(handlers.len(), 1);
        assert_eq!(handlers[0].payload_arity, catch.payload_arity);
        assert_eq!(handlers[0].forwards_exn, catch.forwards_exn);
        assert_eq!(handlers[0].target_base, catch.target_stack_height);
        assert_eq!(native.code[handlers[0].target as usize].op, Op::ThrowRef);
        assert!(matches!(
            invoke(&wasm, "go", &[]),
            Err(WasmError::Exception { .. })
        ));
    }

    #[test]
    fn runtime_neither_rebuilds_nor_mutates_the_finished_artifact() {
        let _guard = test_guard();
        let (_wasm, _module, imports) = parse_artifact(
            r#"(module
                (import "host" "unused" (func))
                (func (export "run") (result i32) i32.const 42))"#,
        );
        assert!(imports.functions[0].is_none());
        assert!(imports.functions[1].is_some());

        let (wasm, _module, artifact) =
            parse_artifact(r#"(module (func (export "run") (result i32) i32.const 42))"#);
        let snapshot = artifact.clone();
        let builds = artifact_build_count();
        assert_eq!(invoke(&wasm, "run", &[]).expect("run"), [Value::I32(42)]);
        assert_eq!(artifact_build_count(), builds);
        assert_eq!(artifact, snapshot);
    }
}
