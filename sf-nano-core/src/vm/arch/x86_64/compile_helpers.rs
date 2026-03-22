//! x86_64 backend: free helper functions, mapping tables, and truncation helpers.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineBlockId, MachineCompareKind, MachineConvertOp, MachineFloatWidth, MachineProgram,
        MachineReg, MachineSign, MachineTrapKind, MachineValue,
    },
};

use super::compile::FunctionCompiler;
use super::enc::Cc;

pub(super) fn defaulted_fp_transient_count(program: &MachineProgram) -> usize {
    if program.fp_transient_count != 0 {
        return program.fp_transient_count as usize;
    }
    let fp_bank_count = program.reg_count.saturating_sub(program.first_fp_reg) as usize;
    fp_bank_count.min(2)
}

pub(super) fn is_fallthrough_edge(
    compiler: &FunctionCompiler<'_>,
    target: MachineBlockId,
    args: &[MachineValue],
    fallthrough: Option<MachineBlockId>,
) -> bool {
    fallthrough == Some(target) && compiler.is_identity_edge(target, args)
}

pub(super) fn map_int_cond(kind: MachineCompareKind, sign: MachineSign) -> Cc {
    match (kind, sign) {
        (MachineCompareKind::Eq, _) => Cc::E,
        (MachineCompareKind::Ne, _) => Cc::NE,
        (MachineCompareKind::Lt, MachineSign::Signed) => Cc::L,
        (MachineCompareKind::Lt, MachineSign::Unsigned) => Cc::B,
        (MachineCompareKind::Gt, MachineSign::Signed) => Cc::G,
        (MachineCompareKind::Gt, MachineSign::Unsigned) => Cc::A,
        (MachineCompareKind::Le, MachineSign::Signed) => Cc::LE,
        (MachineCompareKind::Le, MachineSign::Unsigned) => Cc::BE,
        (MachineCompareKind::Ge, MachineSign::Signed) => Cc::GE,
        (MachineCompareKind::Ge, MachineSign::Unsigned) => Cc::AE,
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

/// Map Wasm float comparison kind to x86_64 condition code.
///
/// x86_64 UCOMISD/UCOMISS set flags:
///   ordered & equal: ZF=1, PF=0, CF=0 → use E (but need to check PF for NaN)
///   unordered (NaN): ZF=1, PF=1, CF=1
///
/// Wasm semantics: NaN is false for all relations except Ne (Ne is true for NaN).
pub(super) fn map_float_cond(kind: MachineCompareKind) -> Cc {
    match kind {
        // Eq: ordered & equal. Use E with NaN handled by caller if needed.
        // After UCOMISD: E is true when ZF=1 (both equal and unordered).
        // For proper NaN handling, caller must also check PF. But for now,
        // use Cc::E — the lowering ensures NaN is handled separately if needed.
        // Actually, for Wasm: use AE-based pairs. Simplest:
        //   Eq → JE (then JNP to skip NaN case — but we'll handle this in float_branch)
        MachineCompareKind::Eq => Cc::E,
        // Ne → JNE
        MachineCompareKind::Ne => Cc::NE,
        // Lt: ordered & less-than → JB (CF=1, for unsigned/unordered comparison)
        MachineCompareKind::Lt => Cc::B,
        // Gt: ordered & greater-than → JA
        MachineCompareKind::Gt => Cc::A,
        // Le: ordered & less-or-equal → JBE
        MachineCompareKind::Le => Cc::BE,
        // Ge: ordered & greater-or-equal → JAE
        MachineCompareKind::Ge => Cc::AE,
    }
}

#[derive(Clone, Copy)]
pub(super) enum ParallelSource {
    Reg {
        reg: MachineReg,
        float_width: Option<MachineFloatWidth>,
    },
    Imm(u64),
    GpTemp,
    FpTemp(MachineFloatWidth),
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

use crate::vm::raw_value::{as_f32, as_f64, from_i32, from_i64};
use crate::vm::runtime::context::NativeContext;

/// Return type for trapping truncation helpers.
/// On System V AMD64, a 2-field repr(C) struct of u64s is returned in RAX and RDX.
#[repr(C)]
pub(crate) struct TruncResult {
    pub status: u64,
    pub value: u64,
}

/// Trapping truncation helper called from generated code.
/// Returns status in RAX (0 = ok) and result in RDX via struct return.
pub(crate) unsafe extern "C" fn x86_64_trapping_trunc(
    ctx: *mut NativeContext,
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

/// Saturating truncation helper called from generated code.
/// Returns result in RAX (no error possible for sat).
pub(crate) unsafe extern "C" fn x86_64_saturating_trunc(src_bits: u64, op_code: u64) -> u64 {
    match op_code {
        8 => trunc_sat_f32_to_i32_s(src_bits as u32),
        9 => trunc_sat_f32_to_i32_u(src_bits as u32),
        10 => trunc_sat_f64_to_i32_s(src_bits),
        11 => trunc_sat_f64_to_i32_u(src_bits),
        12 => trunc_sat_f32_to_i64_s(src_bits as u32),
        13 => trunc_sat_f32_to_i64_u(src_bits as u32),
        14 => trunc_sat_f64_to_i64_s(src_bits),
        15 => trunc_sat_f64_to_i64_u(src_bits),
        _ => 0,
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

// Saturating truncation implementations

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0_f32 {
        return from_i32(i32::MAX);
    }
    if value < -2147483648.0_f32 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 4294967296.0_f32 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 2147483648.0 {
        return from_i32(i32::MAX);
    }
    if value <= -2147483649.0 {
        return from_i32(i32::MIN);
    }
    from_i32(value as i32)
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 4294967296.0 {
        return u64::from(u32::MAX);
    }
    u64::from(value as u32)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0_f32 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0_f32 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= -1.0_f32 {
        return 0;
    }
    if value >= 18446744073709551616.0_f32 {
        return u64::MAX;
    }
    value as u64
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        return 0;
    }
    if value >= 9223372036854775808.0 {
        return from_i64(i64::MAX);
    }
    if value < -9223372036854775808.0 {
        return from_i64(i64::MIN);
    }
    from_i64(value as i64)
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= -1.0 {
        return 0;
    }
    if value >= 18446744073709551616.0 {
        return u64::MAX;
    }
    value as u64
}
