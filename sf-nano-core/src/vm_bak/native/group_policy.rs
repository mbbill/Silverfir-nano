use crate::vm::{
    lir::{self, IrOpKind},
    plan::{CompileConfig, GroupTerminator, HotLocalPlan, SemanticGroupPolicy},
    wasm::{
        semantic_ir::{SemanticOp, SemanticOpKind},
        CoreOpKind,
    },
};

use super::{arm64::op_meta, bridge};

pub struct NativeGroupPolicy {
    hot_locals: HotLocalPlan,
    config: CompileConfig,
    frame_size: usize,
}

impl NativeGroupPolicy {
    pub fn new(hot_locals: HotLocalPlan, config: CompileConfig, frame_size: usize) -> Self {
        Self {
            hot_locals,
            config,
            frame_size,
        }
    }

    fn hot_mask(&self) -> [bool; 3] {
        self.hot_locals.hot_mask()
    }

    fn lowered_kind(&self, op: &SemanticOp) -> IrOpKind {
        lir::lower::preview_lowered_kind(
            op.kind.clone(),
            self.hot_locals,
            self.config,
            self.frame_size,
            op.pre_height,
        )
    }

    fn supports_lowered(&self, kind: &IrOpKind) -> bool {
        if !op_meta::supports_kind(kind) || !op_meta::memidx_is_supported(kind) {
            return false;
        }

        if let IrOpKind::LocalTeeHot { reg } = kind {
            if (*reg as usize) >= 3 || !self.hot_mask()[*reg as usize] {
                return false;
            }
        }

        true
    }
}

impl SemanticGroupPolicy for NativeGroupPolicy {
    fn can_group(&self, op: &SemanticOp) -> bool {
        let kind = self.lowered_kind(op);

        if matches!(kind, IrOpKind::BrTable { .. }) {
            return false;
        }

        if bridge::cold_helper_kind(&kind).is_some() {
            return false;
        }

        if self.supports_lowered(&kind) {
            return true;
        }

        match kind {
            IrOpKind::Br {
                stack_drop, arity, ..
            } => (stack_drop == 0 || arity == 0) && op.alt_target.is_some(),
            IrOpKind::BrIfSimple => op.alt_target.is_some(),
            IrOpKind::BrIf { stack_drop, .. } => stack_drop == 0 && op.alt_target.is_some(),
            IrOpKind::If => op.alt_target.is_some(),
            IrOpKind::ReturnVoid { .. }
            | IrOpKind::ReturnOne { .. }
            | IrOpKind::Return { .. }
            | IrOpKind::Unreachable => true,
            _ => false,
        }
    }

    fn is_redirect_marker(&self, op: &SemanticOp) -> bool {
        matches!(
            op.kind,
            SemanticOpKind::Block { .. } | SemanticOpKind::Loop { .. } | SemanticOpKind::End
        )
    }

    fn terminator(&self, op: &SemanticOp) -> Option<GroupTerminator> {
        match &op.kind {
            SemanticOpKind::Br { stack_drop, arity }
                if (*stack_drop == 0 || *arity == 0) && op.alt_target.is_some() =>
            {
                Some(GroupTerminator::Br)
            }
            SemanticOpKind::BrIf { stack_drop, arity }
                if *stack_drop == 0 && op.alt_target.is_some() =>
            {
                Some(GroupTerminator::BrIf)
            }
            SemanticOpKind::If { .. } if op.alt_target.is_some() => Some(GroupTerminator::If),
            SemanticOpKind::ReturnVoid => Some(GroupTerminator::ReturnVoid),
            SemanticOpKind::ReturnOne => Some(GroupTerminator::ReturnOne),
            SemanticOpKind::Return { .. } => Some(GroupTerminator::Return),
            SemanticOpKind::Core(CoreOpKind::Unreachable) => Some(GroupTerminator::Unreachable),
            _ => None,
        }
    }
}
