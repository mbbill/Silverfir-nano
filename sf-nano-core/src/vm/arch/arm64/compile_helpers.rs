//! Free helper functions: constant materialization, condition mapping,
//! trap codes, float-width queries, and trapping/saturating truncation helpers.

use crate::error::WasmError;
use crate::vm::machine::machine_ir::{
    MachineCompareKind, MachineConvertOp, MachineFloatWidth, MachineMemWidth, MachineSign,
    MachineTrapKind,
};

use super::emit::Arm64TextEmitter;
use super::enc::{self, Cond};
use super::reg::Arm64Reg;

use crate::vm::raw_value::{as_f32, as_f64, from_i32, from_i64};

pub(super) fn materialize_u64_into(text: &mut Arm64TextEmitter, dst: Arm64Reg, value: u64) {
    if value == 0 {
        text.emit_u32(enc::mov_reg_64(dst, Arm64Reg::Xzr));
        return;
    }

    let chunks = [
        (value & 0xffff) as u16,
        ((value >> 16) & 0xffff) as u16,
        ((value >> 32) & 0xffff) as u16,
        ((value >> 48) & 0xffff) as u16,
    ];

    // Find first non-zero chunk for movz, then movk for remaining non-zero chunks
    let mut first = true;
    for (i, &chunk) in chunks.iter().enumerate() {
        if chunk != 0 || first && i == 3 {
            // Must emit at least one instruction
            if first {
                text.emit_u32(enc::movz_64(dst, chunk, (i as u32) * 16));
                first = false;
            } else {
                text.emit_u32(enc::movk_64(dst, chunk, (i as u32) * 16));
            }
        }
    }
    if first {
        // All chunks are zero but value != 0 (shouldn't happen, covered above)
        text.emit_u32(enc::movz_64(dst, 0, 0));
    }
}

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cond {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cond::Eq,
        (MachineCompareKind::Ne, _) => Cond::Ne,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cond::Lt,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cond::Lo,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cond::Gt,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cond::Hi,
        (MachineCompareKind::Le, MachineSign::Signed) => Cond::Le,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cond::Ls,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cond::Ge,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cond::Hs,
    }
}

pub(super) fn trap_code(kind: MachineTrapKind) -> u64 {
    match kind {
        MachineTrapKind::Unreachable => 0,
        MachineTrapKind::MemoryOutOfBounds => 1,
        MachineTrapKind::TableOutOfBounds => 2,
        MachineTrapKind::InvalidFunctionReference => 3,
        MachineTrapKind::IndirectCallTypeMismatch => 4,
        MachineTrapKind::IntegerDivideByZero => 5,
        MachineTrapKind::IntegerOverflow => 6,
        MachineTrapKind::StackOverflow => 7,
        MachineTrapKind::HelperFailure => 8,
    }
}

pub(super) const MACHINE_TRAP_KIND_COUNT: usize = 9;

pub(super) fn trap_kind_index(kind: MachineTrapKind) -> usize {
    trap_code(kind) as usize
}

/// Map a Wasm float comparison kind to an ARM64 condition code.
///
/// Wasm float comparisons treat unordered (NaN) as false for all relations
/// except Ne (Ne returns true when either is NaN).
pub(super) fn map_float_cond(kind: MachineCompareKind) -> Cond {
    match kind {
        MachineCompareKind::Eq => Cond::Eq,
        MachineCompareKind::Ne => Cond::Ne,
        MachineCompareKind::Lt => Cond::Mi,
        MachineCompareKind::Gt => Cond::Gt,
        MachineCompareKind::Le => Cond::Ls,
        MachineCompareKind::Ge => Cond::Ge,
    }
}

pub(super) fn mem_width_bytes(width: MachineMemWidth) -> i64 {
    match width {
        MachineMemWidth::U8 => 1,
        MachineMemWidth::U16 => 2,
        MachineMemWidth::U32 => 4,
        MachineMemWidth::U64 => 8,
    }
}

pub(super) fn convert_result_float_width(op: MachineConvertOp) -> Option<MachineFloatWidth> {
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

pub(super) fn convert_op_code(op: MachineConvertOp) -> u64 {
    match op {
        MachineConvertOp::I32TruncF32S => 0,
        MachineConvertOp::I32TruncF32U => 1,
        MachineConvertOp::I32TruncF64S => 2,
        MachineConvertOp::I32TruncF64U => 3,
        MachineConvertOp::I64TruncF32S => 4,
        MachineConvertOp::I64TruncF32U => 5,
        MachineConvertOp::I64TruncF64S => 6,
        MachineConvertOp::I64TruncF64U => 7,
        MachineConvertOp::I32TruncSatF32S => 8,
        MachineConvertOp::I32TruncSatF32U => 9,
        MachineConvertOp::I32TruncSatF64S => 10,
        MachineConvertOp::I32TruncSatF64U => 11,
        MachineConvertOp::I64TruncSatF32S => 12,
        MachineConvertOp::I64TruncSatF32U => 13,
        MachineConvertOp::I64TruncSatF64S => 14,
        MachineConvertOp::I64TruncSatF64U => 15,
        _ => u64::MAX,
    }
}

/// Return type for trapping truncation helpers.
/// On ARM64 C ABI, a 2-field repr(C) struct of u64s is returned in X0 and X1.
#[repr(C)]
pub(crate) struct TruncResult {
    pub status: u64,
    pub value: u64,
}

/// Trapping truncation helper called from generated code.
/// Returns status in X0 (0 = ok) and result in X1 via struct return.
pub(crate) unsafe extern "C" fn arm64_trapping_trunc(
    ctx: *mut crate::vm::runtime::context::NativeContext,
    src_bits: u64,
    op_code: u64,
) -> TruncResult {
    let result = match op_code {
        0 => trunc_f32_to_i32_s(src_bits as u32),
        1 => trunc_f32_to_i32_u(src_bits as u32),
        2 => trunc_f64_to_i32_s(src_bits),
        3 => trunc_f64_to_i32_u(src_bits),
        4 => trunc_f32_to_i64_s(src_bits as u32),
        5 => trunc_f32_to_i64_u(src_bits as u32),
        6 => trunc_f64_to_i64_s(src_bits),
        7 => trunc_f64_to_i64_u(src_bits),
        _ => Err(WasmError::trap("invalid trunc op".into())),
    };
    match result {
        Ok(value) => TruncResult { status: 0, value },
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            TruncResult {
                status: 1,
                value: 0,
            }
        }
    }
}

// Trapping truncation implementations (matching Wasm spec)

fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0_f32 || value < -2147483648.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0 || value <= -2147483649.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0_f32 || value < -9223372036854775808.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0 || value < -9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

