pub(super) mod abi;
pub(crate) mod backend;
pub mod compile;
mod control;
mod enc;
mod inst;
mod operands;
mod reg;
mod select;

use alloc::vec;

use crate::{
    constants::MAX_STACK_SIZE,
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        machine::machine_ir::MachineFunctionRuntime,
        raw_value::{as_f32, as_f64, from_f32, from_f64, from_i32, from_i64},
        runtime::{
            code::{CompiledNativeModule, NativeCode},
            context::NativeContext,
        },
        result_buffer::ResultBuffer,
        store::Store,
        value::Value,
    },
};

#[cfg(feature = "function-trace")]
use crate::vm::debug::function_trace;

const MAX_STACK_SLOTS: usize = MAX_STACK_SIZE / core::mem::size_of::<u64>();

unsafe extern "C" {
    fn ceilf(x: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn truncf(x: f32) -> f32;
    fn ceil(x: f64) -> f64;
    fn floor(x: f64) -> f64;
    fn trunc(x: f64) -> f64;
}

pub fn eval(
    spec: &FunctionSpec,
    code: &NativeCode,
    store: &mut Store,
    args: &[Value],
    backend: &'static str,
) -> Result<ResultBuffer, WasmError> {
    let func_type = spec.func_type();
    if args.len() != func_type.params().len() {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            func_type.params().len()
        )));
    }

    let compiled = code.compiled();
    let func_id = code.func_id();
    let runtime = compiled
        .runtime()
        .functions
        .get(func_id.0 as usize)
        .ok_or_else(|| {
            WasmError::internal("native entry function is missing runtime metadata".into())
        })?;
    let entry = code.native_entry().ok_or_else(|| {
        WasmError::internal("armv7a native entry is missing finalized code".into())
    })?;
    let root_return = code.native_root_return().ok_or_else(|| {
        WasmError::internal("armv7a native root return continuation is missing".into())
    })?;

    let mut stack = vec![0u64; MAX_STACK_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_STACK_SLOTS) };

    unsafe {
        for (index, arg) in args.iter().enumerate() {
            *stack_base.add(index) = arg.to_raw();
        }
        if runtime.frame_prefix_slots as usize > args.len() {
            core::ptr::write_bytes(
                stack_base.add(args.len()),
                0,
                runtime.frame_prefix_slots as usize - args.len(),
            );
        }
    }
    ensure_stack_capacity(stack_base, stack_end, runtime.total_frame_slots)?;

    let mut ctx = NativeContext::new(store as *mut Store, stack_end);
    seed_root_call_link(compiled, runtime, stack_base, root_return)?;
    #[cfg(feature = "function-trace")]
    {
        function_trace::init_from_env();
        function_trace::native_root_entry(&mut ctx, spec, backend);
    }

    #[cfg(has_guard_pages)]
    {
        use crate::vm::machine::{runtime::context::ctx_offset, trap_signal};
        trap_signal::install_signal_handler();
        trap_signal::set_trap_kind_offset(ctx_offset::TRAP_KIND as usize);
        trap_signal::reset_debug_state();
        ctx.trap_kind = 0;
    }

    let status = unsafe { entry(&mut ctx, stack_base) };

    #[cfg(has_guard_pages)]
    if ctx.trap_kind != 0 {
        let error = WasmError::trap("out of bounds memory access".into());
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    if status != 0 {
        let error = ctx.error.take().unwrap_or_else(|| {
            WasmError::internal("armv7a root entry failed without setting an error".into())
        });
        #[cfg(feature = "function-trace")]
        function_trace::native_trap_current(&mut ctx, &error);
        return Err(error);
    }

    let results_len = func_type.results().len();
    let mut out = ResultBuffer::with_exact_capacity(results_len);
    unsafe {
        for index in 0..results_len {
            out.push(*stack_base.add(index));
        }
    }
    #[cfg(feature = "function-trace")]
    {
        let results = unsafe { core::slice::from_raw_parts(stack_base, results_len) };
        function_trace::native_root_exit(&mut ctx, spec, results);
    }
    Ok(out)
}

fn seed_root_call_link(
    compiled: &CompiledNativeModule,
    runtime: &MachineFunctionRuntime,
    fp: *mut u64,
    root_return: *const u8,
) -> Result<(), WasmError> {
    let call_scratch = runtime.call_scratch.ok_or_else(|| {
        WasmError::internal("armv7a root entry requires call scratch for unified return".into())
    })?;
    let layout = compiled.runtime().call_link;
    unsafe {
        *fp.add(call_scratch.base_slot as usize + (layout.continuation_offset / 8) as usize) =
            root_return as u64;
        *fp.add(call_scratch.base_slot as usize + (layout.caller_frame_offset / 8) as usize) =
            fp as u64;
        *fp.add(
            call_scratch.base_slot as usize + (layout.caller_result_base_offset / 8) as usize,
        ) = 0;
    }
    Ok(())
}

fn ensure_stack_capacity(
    fp: *mut u64,
    stack_end: *mut u64,
    total_frame_slots: u16,
) -> Result<(), WasmError> {
    let end =
        (fp as usize).saturating_add(total_frame_slots as usize * core::mem::size_of::<u64>());
    if end > stack_end as usize {
        return Err(WasmError::exhaustion("stack overflow".into()));
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn armv7a_raise_trap(
    ctx: *mut NativeContext,
    kind: u32,
    site: u32,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = match kind {
        0 => WasmError::trap(trap_message("unreachable executed", site)),
        1 => WasmError::trap(trap_message("out of bounds memory access", site)),
        2 => WasmError::trap(trap_message("out of bounds table access", site)),
        3 => WasmError::trap(trap_message("invalid function reference", site)),
        4 => WasmError::trap(trap_message("indirect call type mismatch", site)),
        5 => WasmError::trap(trap_message("integer divide by zero", site)),
        6 => WasmError::trap(trap_message("integer overflow", site)),
        7 => WasmError::exhaustion(trap_message("stack overflow", site)),
        _ => WasmError::trap(trap_message("native helper failed", site)),
    };
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

fn trap_message(base: &str, site: u32) -> alloc::string::String {
    if !trap_site_debug_enabled() {
        return base.into();
    }
    let func_idx = site >> 12;
    let block_idx = site & 0x0fff;
    if block_idx == 0x0fff {
        alloc::format!("{base} [armv7a site func{func_idx}]")
    } else {
        alloc::format!("{base} [armv7a site func{func_idx} block{block_idx}]")
    }
}

fn trap_site_debug_enabled() -> bool {
    #[cfg(any(feature = "wasi", feature = "std", test))]
    {
        std::env::var_os("SF_ARMV7_TRAP_SITE").is_some()
    }
    #[cfg(not(any(feature = "wasi", feature = "std", test)))]
    {
        false
    }
}

// ─── Software integer division helpers (ARMv7-A has no UDIV/SDIV) ───────────

/// Unsigned 32-bit division. Returns quotient.
pub(crate) extern "C" fn armv7a_udiv(num: u32, den: u32) -> u32 {
    // Caller guarantees den != 0 (JIT emits trap check before calling)
    num / den
}

/// Signed 32-bit division. Returns quotient.
pub(crate) extern "C" fn armv7a_sdiv(num: i32, den: i32) -> i32 {
    // The backend emits the Wasm overflow trap for i32.div_s before calling
    // this helper. Use wrapping semantics here anyway so helper-backed
    // remainder paths can safely compute INT_MIN / -1 as an intermediate.
    num.wrapping_div(den)
}

/// Unsigned 32x32 -> 64 multiply.
pub(crate) extern "C" fn armv7a_umul_wide(lhs: u32, rhs: u32) -> u64 {
    u64::from(lhs) * u64::from(rhs)
}

/// Signed 32x32 -> 64 multiply.
pub(crate) extern "C" fn armv7a_smul_wide(lhs: i32, rhs: i32) -> i64 {
    i64::from(lhs) * i64::from(rhs)
}

#[inline]
fn join_u64(lo: u32, hi: u32) -> u64 {
    u64::from(lo) | (u64::from(hi) << 32)
}

/// Count leading zeros in a 64-bit value passed as lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_clz(lo: u32, hi: u32) -> u64 {
    u64::from(join_u64(lo, hi).leading_zeros())
}

/// Count trailing zeros in a 64-bit value passed as lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_ctz(lo: u32, hi: u32) -> u64 {
    u64::from(join_u64(lo, hi).trailing_zeros())
}

/// Count set bits in a 64-bit value passed as lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_popcnt(lo: u32, hi: u32) -> u64 {
    u64::from(join_u64(lo, hi).count_ones())
}

/// Shift a 64-bit value left by the low 6 bits of `count`.
pub(crate) extern "C" fn armv7a_i64_shl(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi) << (count & 63)
}

/// Arithmetic right shift of a 64-bit value by the low 6 bits of `count`.
pub(crate) extern "C" fn armv7a_i64_shr_s(lo: u32, hi: u32, count: u32) -> i64 {
    ((join_u64(lo, hi)) as i64) >> (count & 63)
}

/// Logical right shift of a 64-bit value by the low 6 bits of `count`.
pub(crate) extern "C" fn armv7a_i64_shr_u(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi) >> (count & 63)
}

/// Rotate a 64-bit value left by the low 6 bits of `count`.
pub(crate) extern "C" fn armv7a_i64_rotl(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi).rotate_left(count & 63)
}

/// Rotate a 64-bit value right by the low 6 bits of `count`.
pub(crate) extern "C" fn armv7a_i64_rotr(lo: u32, hi: u32, count: u32) -> u64 {
    join_u64(lo, hi).rotate_right(count & 63)
}

/// Add two 64-bit values passed as lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_mul(lhs_lo: u32, lhs_hi: u32, rhs_lo: u32, rhs_hi: u32) -> u64 {
    join_u64(lhs_lo, lhs_hi).wrapping_mul(join_u64(rhs_lo, rhs_hi))
}

/// Signed 64-bit division over split lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_div_s(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_div(rhs) as u64
}

/// Unsigned 64-bit division over split lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_div_u(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    join_u64(lhs_lo, lhs_hi) / join_u64(rhs_lo, rhs_hi)
}

/// Signed 64-bit remainder over split lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_rem_s(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    let lhs = join_u64(lhs_lo, lhs_hi) as i64;
    let rhs = join_u64(rhs_lo, rhs_hi) as i64;
    lhs.wrapping_rem(rhs) as u64
}

/// Unsigned 64-bit remainder over split lo/hi halves.
pub(crate) extern "C" fn armv7a_i64_rem_u(
    lhs_lo: u32,
    lhs_hi: u32,
    rhs_lo: u32,
    rhs_hi: u32,
) -> u64 {
    join_u64(lhs_lo, lhs_hi) % join_u64(rhs_lo, rhs_hi)
}

// ─── i64 ↔ float conversion helpers ─────────────────────────────────────────

/// Convert signed i64 (passed as lo/hi halves) to f64.
pub(crate) extern "C" fn armv7a_i64s_to_f64(lo: u32, hi: u32) -> f64 {
    let val = (lo as u64) | ((hi as u64) << 32);
    (val as i64) as f64
}

/// Convert unsigned i64 (passed as lo/hi halves) to f64.
pub(crate) extern "C" fn armv7a_i64u_to_f64(lo: u32, hi: u32) -> f64 {
    let val = (lo as u64) | ((hi as u64) << 32);
    val as f64
}

/// Convert signed i64 (passed as lo/hi halves) to f32.
pub(crate) extern "C" fn armv7a_i64s_to_f32(lo: u32, hi: u32) -> f32 {
    let val = (lo as u64) | ((hi as u64) << 32);
    (val as i64) as f32
}

/// Convert unsigned i64 (passed as lo/hi halves) to f32.
pub(crate) extern "C" fn armv7a_i64u_to_f32(lo: u32, hi: u32) -> f32 {
    let val = (lo as u64) | ((hi as u64) << 32);
    val as f32
}

/// Saturating float-to-i64 conversion helper for legalized pair results.
pub(crate) extern "C" fn armv7a_saturating_trunc(src_bits: u64, op_code: u32) -> u64 {
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

/// Trapping float-to-i64 conversion helper for legalized pair results.
pub(crate) unsafe extern "C" fn armv7a_trapping_trunc(
    src_bits: u64,
    op_code: u32,
    ctx: *mut NativeContext,
    out: *mut u64,
) -> u32 {
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
        Ok(value) => {
            if let Some(out) = unsafe { out.as_mut() } {
                *out = value;
            }
            0
        }
        Err(err) => {
            if let Some(ctx) = unsafe { ctx.as_mut() } {
                ctx.error = Some(err);
            }
            1
        }
    }
}

pub(crate) extern "C" fn armv7a_f32_ceil(val: f32) -> f32 {
    ceil_f32(val)
}

pub(crate) extern "C" fn armv7a_f32_floor(val: f32) -> f32 {
    floor_f32(val)
}

pub(crate) extern "C" fn armv7a_f32_trunc(val: f32) -> f32 {
    trunc_f32(val)
}

pub(crate) extern "C" fn armv7a_f32_nearest_bits(bits: u32) -> u32 {
    wasm_f32_nearest_bits(bits) as u32
}

pub(crate) extern "C" fn armv7a_f64_ceil(val: f64) -> f64 {
    ceil_f64(val)
}

pub(crate) extern "C" fn armv7a_f64_floor(val: f64) -> f64 {
    floor_f64(val)
}

pub(crate) extern "C" fn armv7a_f64_trunc(val: f64) -> f64 {
    trunc_f64(val)
}

pub(crate) extern "C" fn armv7a_f64_nearest_bits(bits: u64) -> u64 {
    wasm_f64_nearest_bits(bits)
}

pub(crate) unsafe extern "C" fn armv7a_raise_unsupported(
    ctx: *mut NativeContext,
    func_id: u32,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = WasmError::invalid(alloc::format!(
        "armv7a backend has not finalized machine function {} yet",
        func_id
    ));
    #[cfg(feature = "function-trace")]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

#[inline]
fn ceil_f32(value: f32) -> f32 {
    unsafe { ceilf(value) }
}

#[inline]
fn floor_f32(value: f32) -> f32 {
    unsafe { floorf(value) }
}

#[inline]
fn trunc_f32(value: f32) -> f32 {
    unsafe { truncf(value) }
}

#[inline]
fn ceil_f64(value: f64) -> f64 {
    unsafe { ceil(value) }
}

#[inline]
fn floor_f64(value: f64) -> f64 {
    unsafe { floor(value) }
}

#[inline]
fn trunc_f64(value: f64) -> f64 {
    unsafe { trunc(value) }
}

fn wasm_f32_nearest_bits(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if !value.is_finite() {
        return u64::from(bits);
    }
    let floor = floor_f32(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    from_f32(rounded)
}

fn wasm_f64_nearest_bits(bits: u64) -> u64 {
    let value = as_f64(bits);
    if !value.is_finite() {
        return bits;
    }
    let floor = floor_f64(value);
    let diff = value - floor;
    let rounded = if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    from_f64(rounded)
}

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
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -9223372036854777856.0 || value >= 9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value.is_infinite() || value <= -1.0 || value >= 18446744073709551616.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

fn trunc_sat_f32_to_i64_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
    }
}

fn trunc_sat_f32_to_i64_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64) as f64;
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn trunc_sat_f64_to_i64_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i64::MIN as f64 {
        from_i64(i64::MIN)
    } else if value >= i64::MAX as f64 {
        from_i64(i64::MAX)
    } else {
        from_i64(value as i64)
    }
}

fn trunc_sat_f64_to_i64_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value as u64
    }
}

fn trunc_sat_f32_to_i32_s(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f32 {
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f32 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
    }
}

fn trunc_sat_f32_to_i32_u(bits: u32) -> u64 {
    let value = as_f32(bits as u64);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f32 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}

fn trunc_sat_f64_to_i32_s(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() {
        0
    } else if value <= i32::MIN as f64 {
        from_i32(i32::MIN)
    } else if value >= i32::MAX as f64 {
        from_i32(i32::MAX)
    } else {
        from_i32(value as i32)
    }
}

fn trunc_sat_f64_to_i32_u(bits: u64) -> u64 {
    let value = as_f64(bits);
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u64::from(u32::MAX)
    } else {
        u64::from(value as u32)
    }
}
