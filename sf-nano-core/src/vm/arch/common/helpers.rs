use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineConvertOp, MachineFloatWidth, MachineTrapKind,
};

// ── Trap code mapping ────────────────────────────────────────────────────────

pub(crate) fn trap_code(kind: MachineTrapKind) -> u64 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::InvalidConversion => 7,
        MachineTrapKind::StackOverflow => 8,
        MachineTrapKind::HelperFailure => 9,
    }
}

pub(crate) const MACHINE_TRAP_KIND_COUNT: usize = 10;

pub(crate) fn trap_kind_index(kind: MachineTrapKind) -> usize {
    trap_code(kind) as usize
}

pub(crate) fn trap_error(kind: MachineTrapKind) -> WasmError {
    match kind {
        MachineTrapKind::Unreachable => WasmError::trap("unreachable executed".into()),
        MachineTrapKind::MemoryOutOfBounds => WasmError::trap("out of bounds memory access".into()),
        MachineTrapKind::TableOutOfBounds => WasmError::trap("out of bounds table access".into()),
        MachineTrapKind::InvalidFunctionReference => {
            WasmError::trap("invalid function reference".into())
        }
        MachineTrapKind::IndirectCallTypeMismatch => {
            WasmError::trap("indirect call type mismatch".into())
        }
        MachineTrapKind::IntegerDivideByZero => WasmError::trap("integer divide by zero".into()),
        MachineTrapKind::IntegerOverflow => WasmError::trap("integer overflow".into()),
        MachineTrapKind::InvalidConversion => {
            WasmError::trap("invalid conversion to integer".into())
        }
        MachineTrapKind::StackOverflow => WasmError::exhaustion("stack overflow".into()),
        MachineTrapKind::HelperFailure => WasmError::trap("native call failed".into()),
    }
}

pub(crate) fn convert_result_float_width(op: MachineConvertOp) -> Option<MachineFloatWidth> {
    Some(match op {
        MachineConvertOp::F32ConvertI32S
        | MachineConvertOp::F32ConvertI32U
        | MachineConvertOp::F32ConvertI64S
        | MachineConvertOp::F32ConvertI64U
        | MachineConvertOp::F32DemoteF64
        | MachineConvertOp::F32ReinterpretI32 => MachineFloatWidth::F32,
        MachineConvertOp::F64ConvertI32S
        | MachineConvertOp::F64ConvertI32U
        | MachineConvertOp::F64ConvertI64S
        | MachineConvertOp::F64ConvertI64U
        | MachineConvertOp::F64PromoteF32
        | MachineConvertOp::F64ReinterpretI64 => MachineFloatWidth::F64,
        _ => return None,
    })
}

// ── Page alignment ───────────────────────────────────────────────────────────

/// Align a function start to reduce instruction-cache and iTLB pressure.
///
/// 1. Cache-line alignment: round up to 64 bytes.
/// 2. Page-boundary avoidance: if the function fits within one 16 KB page
///    but would straddle a boundary, bump to the next page (up to 1 KB pad).
#[inline]
pub(crate) fn page_align_function(offset: usize, func_size: usize) -> usize {
    let aligned = (offset + 63) & !63;

    if func_size == 0 {
        return aligned;
    }

    const PAGE_SIZE: usize = 16384;
    const MAX_PADDING: usize = 1024;

    if func_size <= PAGE_SIZE {
        let start_page = aligned / PAGE_SIZE;
        let end_page = (aligned + func_size - 1) / PAGE_SIZE;
        if start_page != end_page {
            let next_page = (start_page + 1) * PAGE_SIZE;
            let padding = next_page - aligned;
            if padding <= MAX_PADDING {
                return next_page;
            }
        }
    }
    aligned
}

// ── Fallthrough check ────────────────────────────────────────────────────────

use crate::vm::machine::machine_ir::{MachineBlockId, MachineValue};

/// Returns true if jumping to `target` with `args` can be elided because the
/// target is the physical fallthrough and the args are an identity mapping.
pub(crate) fn is_fallthrough_edge(
    target: MachineBlockId,
    args: &[MachineValue],
    fallthrough: Option<MachineBlockId>,
    blocks: &[crate::vm::machine::machine_ir::MachineBlock],
) -> bool {
    if fallthrough != Some(target) {
        return false;
    }
    let Some(block) = blocks.get(target.as_usize()) else {
        return false;
    };
    if block.params.len() != args.len() {
        return false;
    }
    block
        .params
        .iter()
        .zip(args.iter())
        .all(|(param, arg)| match arg {
            MachineValue::Reg(r) | MachineValue::ReservedReg(r) => *r == param.reg,
            MachineValue::Imm64(_) => false,
        })
}
