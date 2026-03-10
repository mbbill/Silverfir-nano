//! ARM64/native op metadata.

use crate::vm::{lir::legacy::ir::LirOpKind, native::helper_meta::ColdHelperKind};

/// Coarse native compilation support class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeOpClass {
    NativeGroupable,
    NativeSingleton,
    ColdHelper(ColdHelperKind),
    MarkerOnly,
    Unsupported,
}

pub fn classify(kind: &LirOpKind) -> NativeOpClass {
    match kind {
        LirOpKind::Core(_) => NativeOpClass::NativeGroupable,
        LirOpKind::InitLocals { .. }
        | LirOpKind::CacheTransfer(_)
        | LirOpKind::LocalGetHot { .. }
        | LirOpKind::LocalSetHot { .. }
        | LirOpKind::LocalTeeHot { .. }
        | LirOpKind::LocalGetFrame { .. }
        | LirOpKind::LocalSetFrame { .. }
        | LirOpKind::LocalTeeFrame { .. }
        | LirOpKind::Branch { .. }
        | LirOpKind::BrTable { .. }
        | LirOpKind::Return { .. } => NativeOpClass::NativeSingleton,
        LirOpKind::Marker(_) => NativeOpClass::MarkerOnly,
        LirOpKind::CallExternal { .. } => NativeOpClass::ColdHelper(ColdHelperKind::CallExternal),
        LirOpKind::CallIndirect { .. } => NativeOpClass::ColdHelper(ColdHelperKind::CallIndirect),
        LirOpKind::CallInternal { .. } => NativeOpClass::NativeSingleton,
    }
}
