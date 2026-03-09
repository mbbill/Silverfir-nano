//! Lockstep differential debugging contract.
//!
//! This module is intentionally `std`-only and debug-only. It defines the
//! shared checkpoint model that both the base and native backends will use for
//! resumable differential execution.
//!
//! Normal release builds must not depend on these types.

use alloc::{string::String, vec, vec::Vec};

use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::vm::lowered::{IrOp, IrOpKind, OpIndex};
use crate::vm::{
    backend::BackendKind,
    compile::{CompileContext, StackTracker},
    entities::{FunctionInst, ModuleInst},
    planner::CompilePlan,
    store::Store,
};

/// Coarse-to-fine checkpoint modes for differential execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointMode {
    /// Compare only at function entry and function exit.
    Function,
    /// Compare at function boundaries plus control/call boundaries.
    Control,
    /// Compare only within an already narrowed function/region at additional
    /// semantic commit points selected by the debug plan.
    Scoped,
}

/// Stable kind of checkpoint site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointKind {
    FunctionEntry,
    BeforeOp,
    FunctionExit,
}

/// Stable checkpoint identifier used by both base and native backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId {
    pub func_idx: u32,
    pub ordinal: u32,
    pub kind: CheckpointKind,
    pub lowered_op: Option<OpIndex>,
}

/// Concrete checkpoint site in one lowered function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointSite {
    pub id: CheckpointId,
}

/// Per-function checkpoint plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPlan {
    func_idx: u32,
    mode: CheckpointMode,
    sites: Vec<CheckpointSite>,
}

impl CheckpointPlan {
    /// Build a checkpoint plan for one lowered function.
    pub fn for_function(func_idx: u32, ir: &[IrOp], mode: CheckpointMode) -> Self {
        Self::for_function_with_ordinal_base(func_idx, ir, mode, 0)
    }

    fn for_function_with_ordinal_base(
        func_idx: u32,
        ir: &[IrOp],
        mode: CheckpointMode,
        ordinal_base: u32,
    ) -> Self {
        let mut sites = Vec::new();
        let mut ordinal = ordinal_base;

        sites.push(CheckpointSite {
            id: CheckpointId {
                func_idx,
                ordinal,
                kind: CheckpointKind::FunctionEntry,
                lowered_op: None,
            },
        });
        ordinal += 1;

        for (idx, op) in ir.iter().enumerate() {
            let op_idx = OpIndex::new(idx);
            if idx != 0 && should_checkpoint_before(mode, &op.kind) {
                sites.push(CheckpointSite {
                    id: CheckpointId {
                        func_idx,
                        ordinal,
                        kind: CheckpointKind::BeforeOp,
                        lowered_op: Some(op_idx),
                    },
                });
                ordinal += 1;
            }
        }

        Self {
            func_idx,
            mode,
            sites,
        }
    }

    /// Translate checkpoint sites into native group barriers.
    ///
    /// `break_before[i]` means no native group may cross from `i - 1` into
    /// lowered op `i`. `break_after[i]` means no native group may cross from
    /// lowered op `i` into `i + 1`.
    pub fn native_barriers(&self, ir_len: usize) -> CheckpointBarriers {
        let mut break_before = vec![false; ir_len];
        let mut break_after = vec![false; ir_len];

        for site in &self.sites {
            match (site.id.kind, site.id.lowered_op) {
                (CheckpointKind::BeforeOp, Some(op_idx)) => {
                    let idx = op_idx.as_usize();
                    if idx < ir_len {
                        break_before[idx] = true;
                    }
                }
                _ => {}
            }
        }

        CheckpointBarriers {
            break_before,
            break_after,
        }
    }

    #[inline]
    pub fn func_idx(&self) -> u32 {
        self.func_idx
    }

    #[inline]
    pub fn mode(&self) -> CheckpointMode {
        self.mode
    }

    #[inline]
    pub fn sites(&self) -> &[CheckpointSite] {
        &self.sites
    }

    #[inline]
    pub fn entry_site(&self) -> Option<CheckpointSite> {
        self.sites
            .iter()
            .find(|site| site.id.kind == CheckpointKind::FunctionEntry)
            .copied()
    }
}

/// Program-wide checkpoint plan with globally unique ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramCheckpointPlan {
    mode: CheckpointMode,
    sites: Vec<CheckpointSite>,
    function_plans: Vec<Option<CheckpointPlan>>,
}

impl ProgramCheckpointPlan {
    pub fn for_module(
        module: &ModuleInst,
        store: &Store,
        mode: CheckpointMode,
    ) -> Result<Self, WasmError> {
        let mut sites = Vec::new();
        let mut function_plans = Vec::with_capacity(module.functions.len());
        let mut ordinal_base = 0u32;

        for (func_idx, func_inst) in module.functions.iter().enumerate() {
            let FunctionInst::Local { spec, .. } = func_inst else {
                function_plans.push(None);
                continue;
            };
            let plan = build_function_plan(
                spec,
                module,
                store,
                func_idx as u32,
                mode,
                ordinal_base,
            )?;
            ordinal_base += plan.sites().len() as u32;
            sites.extend_from_slice(plan.sites());
            function_plans.push(Some(plan));
        }

        Ok(Self {
            mode,
            sites,
            function_plans,
        })
    }

    pub fn for_function(
        module: &ModuleInst,
        store: &Store,
        func_idx: u32,
        mode: CheckpointMode,
    ) -> Result<Self, WasmError> {
        let mut sites = Vec::new();
        let mut function_plans = vec![None; module.functions.len()];

        let Some(func_inst) = module.functions.get(func_idx as usize) else {
            return Err(WasmError::invalid(alloc::format!(
                "checkpoint function index out of range: {}",
                func_idx
            )));
        };
        let FunctionInst::Local { spec, .. } = func_inst else {
            return Err(WasmError::invalid(alloc::format!(
                "checkpoint target function {} is not local",
                func_idx
            )));
        };

        let plan = build_function_plan(spec, module, store, func_idx, mode, 0)?;
        sites.extend_from_slice(plan.sites());
        function_plans[func_idx as usize] = Some(plan);

        Ok(Self {
            mode,
            sites,
            function_plans,
        })
    }

    #[inline]
    pub fn mode(&self) -> CheckpointMode {
        self.mode
    }

    #[inline]
    pub fn sites(&self) -> &[CheckpointSite] {
        &self.sites
    }

    pub fn site_by_ordinal(&self, ordinal: u32) -> Option<CheckpointSite> {
        self.sites
            .iter()
            .find(|site| site.id.ordinal == ordinal)
            .copied()
    }

    pub fn function_plan(&self, func_idx: u32) -> Option<&CheckpointPlan> {
        self.function_plans
            .get(func_idx as usize)
            .and_then(|plan| plan.as_ref())
    }
}

/// Snapshot payload compared at one checkpoint.
///
/// This is intentionally canonical state only: frame words, globals, and page
/// digests. Later backends may narrow or enrich how each field is collected,
/// but the comparison contract should stay at this level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSnapshot {
    pub call_depth: u32,
    pub result_values: Vec<u64>,
    pub globals: Vec<u64>,
    pub memory_page_hashes: Vec<u64>,
    pub trap: Option<String>,
}

/// Result of stepping one backend forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOutcome {
    Checkpoint {
        site: CheckpointSite,
        snapshot: CheckpointSnapshot,
    },
    Finished {
        snapshot: CheckpointSnapshot,
    },
}

/// Resumable execution API used by the lockstep differential harness.
///
/// Backends must advance execution until the next planned checkpoint, then
/// return canonical state without recording a full trace log.
pub trait Stepper {
    fn checkpoint_plan(&self) -> &ProgramCheckpointPlan;
    fn step_until_checkpoint(&mut self) -> StepOutcome;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockstepMismatch {
    pub left: StepOutcome,
    pub right: StepOutcome,
}

pub fn compare_steppers(
    left: &mut impl Stepper,
    right: &mut impl Stepper,
) -> Result<(), LockstepMismatch> {
    loop {
        let left_outcome = left.step_until_checkpoint();
        let right_outcome = right.step_until_checkpoint();
        if left_outcome != right_outcome {
            return Err(LockstepMismatch {
                left: left_outcome,
                right: right_outcome,
            });
        }
        if matches!(left_outcome, StepOutcome::Finished { .. }) {
            return Ok(());
        }
    }
}

/// Native compilation barriers derived from a checkpoint plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointBarriers {
    pub break_before: Vec<bool>,
    pub break_after: Vec<bool>,
}

fn should_checkpoint_before(mode: CheckpointMode, kind: &IrOpKind) -> bool {
    match mode {
        CheckpointMode::Function => false,
        CheckpointMode::Control => is_control_or_call_boundary(kind),
        CheckpointMode::Scoped => is_control_or_call_boundary(kind),
    }
}

fn is_control_or_call_boundary(kind: &IrOpKind) -> bool {
    matches!(
        kind,
        IrOpKind::Br { .. }
            | IrOpKind::BrIfSimple
            | IrOpKind::BrIf { .. }
            | IrOpKind::BrTable { .. }
            | IrOpKind::If
            | IrOpKind::Else
            | IrOpKind::ReturnVoid { .. }
            | IrOpKind::ReturnOne { .. }
            | IrOpKind::Return { .. }
            | IrOpKind::Unreachable
            | IrOpKind::CallExternal { .. }
            | IrOpKind::CallInternal { .. }
            | IrOpKind::CallIndirect { .. }
    )
}

fn build_function_plan(
    spec: &FunctionSpec,
    module: &ModuleInst,
    store: &Store,
    func_idx: u32,
    mode: CheckpointMode,
    ordinal_base: u32,
) -> Result<CheckpointPlan, WasmError> {
    let compile_plan = CompilePlan::for_config(
        spec.code(),
        spec.func_type().params().len() + spec.locals().len(),
        BackendKind::Base.compile_config(),
    );
    let ctx = CompileContext::new(
        Some(&module.types),
        store,
        module,
        spec.func_type().results().len(),
    );
    let mut stack = StackTracker::new(
        compile_plan.config(),
        spec.func_type().params().len(),
        spec.locals().len(),
        spec.func_type().results().len(),
    );
    let ir = crate::vm::lowered::lower_to_ir(spec.code(), &ctx, &mut stack, compile_plan.hot_locals())?;
    Ok(CheckpointPlan::for_function_with_ordinal_base(
        func_idx,
        &ir,
        mode,
        ordinal_base,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::rc::Rc;
    use crate::module::entities::FunctionSpec;
    use crate::module::type_context::TypeContext;
    use crate::module::type_defs::FunctionType;
    use crate::vm::lowered::SlotRef;
    use crate::value_type::ValueType;
    use crate::vm::entities::{FunctionInst, ModuleInst};
    use crate::vm::store::Store;

    fn op(kind: IrOpKind, pre_height: u16) -> IrOp {
        IrOp {
            kind,
            variant: 0,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    #[test]
    fn function_mode_only_has_entry_and_exit() {
        let ir = vec![
            op(IrOpKind::I32Const { value: 1 }, 0),
            op(IrOpKind::I32Add, 2),
            op(
                IrOpKind::ReturnOne {
                    frame_size: 0,
                    operand_base_offset: 0,
                    height: 1,
                },
                1,
            ),
        ];

        let plan = CheckpointPlan::for_function(7, &ir, CheckpointMode::Function);
        assert_eq!(plan.sites().len(), 1);
        assert_eq!(plan.sites()[0].id.kind, CheckpointKind::FunctionEntry);
        assert_eq!(plan.sites()[0].id.lowered_op, None);
    }

    #[test]
    fn control_mode_marks_control_and_call_boundaries() {
        let ir = vec![
            op(IrOpKind::I32Const { value: 1 }, 0),
            op(
                IrOpKind::CallIndirect {
                    type_idx: 0,
                    table_idx: 0,
                    delta: SlotRef::operand_relative(0),
                    operand_base_offset: 8,
                    height: 1,
                },
                1,
            ),
            op(
                IrOpKind::BrIf {
                    stack_drop: 1,
                    arity: 1,
                    height: 2,
                    operand_base_offset: 8,
                },
                2,
            ),
            op(IrOpKind::End, 1),
        ];

        let plan = CheckpointPlan::for_function(3, &ir, CheckpointMode::Control);
        let ids: Vec<(CheckpointKind, Option<OpIndex>)> = plan
            .sites()
            .iter()
            .map(|site| (site.id.kind, site.id.lowered_op))
            .collect();

        assert_eq!(
            ids,
            vec![
                (CheckpointKind::FunctionEntry, None),
                (CheckpointKind::BeforeOp, Some(OpIndex::new(1))),
                (CheckpointKind::BeforeOp, Some(OpIndex::new(2))),
            ]
        );
    }

    #[test]
    fn scoped_mode_defaults_to_control_boundaries() {
        let ir = vec![
            op(IrOpKind::I32Const { value: 1 }, 0),
            op(IrOpKind::I32Const { value: 2 }, 1),
            op(
                IrOpKind::CallIndirect {
                    type_idx: 0,
                    table_idx: 0,
                    delta: SlotRef::operand_relative(0),
                    operand_base_offset: 8,
                    height: 1,
                },
                1,
            ),
        ];

        let plan = CheckpointPlan::for_function(11, &ir, CheckpointMode::Scoped);
        assert_eq!(plan.sites().len(), 2);
        assert_eq!(plan.sites()[1].id.lowered_op, Some(OpIndex::new(2)));
    }

    #[test]
    fn scoped_mode_native_barriers_split_selected_boundaries() {
        let ir = vec![
            op(IrOpKind::I32Const { value: 1 }, 0),
            op(IrOpKind::I32Const { value: 2 }, 1),
            op(
                IrOpKind::CallIndirect {
                    type_idx: 0,
                    table_idx: 0,
                    delta: SlotRef::operand_relative(0),
                    operand_base_offset: 8,
                    height: 1,
                },
                1,
            ),
        ];

        let plan = CheckpointPlan::for_function(11, &ir, CheckpointMode::Scoped);
        let barriers = plan.native_barriers(ir.len());

        assert_eq!(barriers.break_before, vec![false, false, true]);
        assert_eq!(barriers.break_after, vec![false, false, false]);
    }

    #[test]
    fn program_plan_assigns_global_ordinals_across_functions() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = ModuleInst::new("test".into(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });
        let FunctionInst::Local { spec, .. } = &mut module.functions[0] else {
            unreachable!();
        };
        spec.set_code(crate::module::entities::Bytecode::from(&[0x41, 0x01, 0x0f, 0x0b][..]));
        let FunctionInst::Local { spec, .. } = &mut module.functions[1] else {
            unreachable!();
        };
        spec.set_code(crate::module::entities::Bytecode::from(&[0x41, 0x02, 0x0f, 0x0b][..]));

        let store = Store::new(module);
        let plan = ProgramCheckpointPlan::for_module(store.module(), &store, CheckpointMode::Function)
            .expect("program checkpoint plan");

        assert_eq!(plan.sites().len(), 2);
        for (expected, site) in plan.sites().iter().enumerate() {
            assert_eq!(site.id.ordinal as usize, expected);
        }
        assert_eq!(plan.sites()[0].id.func_idx, 0);
        assert_eq!(plan.sites().last().map(|site| site.id.func_idx), Some(1));
        assert!(plan.function_plan(0).is_some());
        assert!(plan.function_plan(1).is_some());
    }

    #[cfg(all(feature = "micro-jit", target_arch = "aarch64"))]
    #[test]
    fn fast_and_native_steppers_agree_on_simple_function_run() {
        fn make_store() -> Store {
            let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
            let types = TypeContext::new(vec![sig.clone()]);
            let mut module = ModuleInst::new("test".into(), types);
            module.functions.push(FunctionInst::Local {
                spec: FunctionSpec::new(sig, 0),
                type_index: 0,
            });
            let FunctionInst::Local { spec, .. } = &mut module.functions[0] else {
                unreachable!();
            };
            spec.set_code(crate::module::entities::Bytecode::from(&[
                0x41, 0x01,
                0x41, 0x02,
                0x6a,
                0x0b,
            ][..]));
            Store::new(module)
        }

        let mut fast_store = make_store();
        let mut native_store = make_store();
        let plan = ProgramCheckpointPlan::for_module(
            fast_store.module(),
            &fast_store,
            CheckpointMode::Function,
        )
        .expect("program checkpoint plan");

        let fast_func = fast_store.function(0) as *const FunctionInst;
        let native_func = native_store.function(0) as *const FunctionInst;
        let mut fast = crate::vm::interp::fast::lockstep::FastStepper::eval(
            unsafe { &*fast_func },
            &mut fast_store,
            &[],
            plan.clone(),
        )
        .expect("fast stepper");
        let mut native = crate::vm::native::lockstep::NativeStepper::eval(
            unsafe { &*native_func },
            &mut native_store,
            &[],
            plan,
        )
        .expect("native stepper");
        compare_steppers(&mut fast, &mut native).expect("lockstep compare");
    }
}
