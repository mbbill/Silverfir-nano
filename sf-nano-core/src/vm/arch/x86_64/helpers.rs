//! x86_64 runtime helpers called from generated code.

use crate::error::WasmError;
use crate::vm::raw_value::{as_f32, as_f64, from_i32, from_i64};
use crate::vm::runtime::context::NativeContext;

#[cfg(sf_call_trace)]
use crate::vm::debug::function_trace;

// ── raise_trap ───────────────────────────────────────────────────────────────

pub(crate) unsafe extern "C" fn x86_64_raise_trap(ctx: *mut NativeContext, kind: u64) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = match kind {
        0 => WasmError::trap("unreachable executed".into()),
        1 => WasmError::trap("out of bounds memory access".into()),
        2 => WasmError::trap("out of bounds table access".into()),
        3 => WasmError::trap("invalid function reference".into()),
        4 => WasmError::trap("indirect call type mismatch".into()),
        5 => WasmError::trap("integer divide by zero".into()),
        6 => WasmError::trap("integer overflow".into()),
        7 => WasmError::exhaustion("stack overflow".into()),
        _ => WasmError::trap("native helper failed".into()),
    };
    #[cfg(sf_call_trace)]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

pub(crate) unsafe extern "C" fn x86_64_raise_unsupported(
    ctx: *mut NativeContext,
    func_id: u64,
) -> u32 {
    let Some(ctx) = (unsafe { ctx.as_mut() }) else {
        return 1;
    };
    let error = WasmError::invalid(alloc::format!(
        "x86_64 backend has not finalized machine function {} yet",
        func_id
    ));
    #[cfg(sf_call_trace)]
    function_trace::native_trap_current(ctx, &error);
    ctx.error = Some(error);
    1
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_frame(func_id: u64, fp: *const u64) {
    if std::env::var_os("SF_DEBUG_FRAME_TRACE").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[frame] f{func_id} fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 10) };
    std::eprintln!(
        "[frame] f{func_id} slots0-9 = {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        slots[3] as u32,
        slots[4] as u32,
        slots[5] as u32,
        slots[6] as u32,
        slots[7] as u32,
        slots[8] as u32,
        slots[9] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_return(
    func_id: u64,
    fp: *const u64,
    result_slot: u64,
) {
    let Some(spec) = std::env::var_os("SF_DEBUG_RETURN_TRACE") else {
        return;
    };
    let trace_this = if spec == "*" {
        true
    } else {
        spec.to_string_lossy()
            .split(',')
            .filter_map(|part| part.trim().parse::<u64>().ok())
            .any(|id| id == func_id)
    };
    if !trace_this {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[ret] f{func_id} fp=null slot={result_slot}");
        return;
    }
    let result_slot = result_slot as usize;
    let slots = unsafe { core::slice::from_raw_parts(fp, result_slot.saturating_add(1)) };
    let slot = |index: usize| -> u32 { slots.get(index).copied().unwrap_or_default() as u32 };
    let result = slots.get(result_slot).copied().unwrap_or_default() as u32;
    std::eprintln!(
        "[ret] f{func_id} p0={:08x} p1={:08x} p2={:08x} p3={:08x} p4={:08x} p5={:08x} slot{}={:08x}",
        slot(0),
        slot(1),
        slot(2),
        slot(3),
        slot(4),
        slot(5),
        result_slot,
        result,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f721_b559(
    fp: *const u64,
    lane0: u64,
    lane1: u64,
    lane2: u64,
    lane3: u64,
) {
    if std::env::var_os("SF_DEBUG_B559").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!(
            "[b559] fp=null lane0={:08x} lane1={:08x} lane2={:08x} lane3={:08x}",
            lane0 as u32,
            lane1 as u32,
            lane2 as u32,
            lane3 as u32,
        );
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 24) };
    std::eprintln!(
        "[b559] lane0={:08x} lane1={:08x} lane2={:08x} lane3={:08x} fp5={:08x} fp9={:08x} fp16={:08x} fp18={:08x} fp23={:08x}",
        lane0 as u32,
        lane1 as u32,
        lane2 as u32,
        lane3 as u32,
        slots[5] as u32,
        slots[9] as u32,
        slots[16] as u32,
        slots[18] as u32,
        slots[23] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_indirect_call(
    caller_func_id: u64,
    callee_func_id: u64,
    fp: *const u64,
) {
    if std::env::var_os("SF_DEBUG_INDIRECT_TRACE").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[indirect] f{caller_func_id} -> f{callee_func_id} fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 6) };
    std::eprintln!(
        "[indirect] f{caller_func_id} -> f{callee_func_id} args = {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        slots[3] as u32,
        slots[4] as u32,
        slots[5] as u32,
    );
}

fn debug_load_u8(mem0_base: *const u8, addr: u32) -> u8 {
    unsafe { core::ptr::read_unaligned(mem0_base.add(addr as usize) as *const u8) }
}

fn debug_load_u16(mem0_base: *const u8, addr: u32) -> u16 {
    unsafe { core::ptr::read_unaligned(mem0_base.add(addr as usize) as *const u16) }
}

fn debug_load_u32(mem0_base: *const u8, addr: u32) -> u32 {
    unsafe { core::ptr::read_unaligned(mem0_base.add(addr as usize) as *const u32) }
}

fn debug_eval_f58(mem0_base: *const u8, p0: u32, mut p1: u32) -> u32 {
    let mut l2 = debug_load_u32(mem0_base, p0.wrapping_add(20));
    if (p1 as i32) >= 1 {
        p1 = debug_load_u32(mem0_base, l2).wrapping_add(p1.wrapping_shl(4));
        if p1 >= debug_load_u32(mem0_base, p0.wrapping_add(12)) {
            p1 = debug_load_u32(mem0_base, p0.wrapping_add(16)).wrapping_add(56);
        }
    } else if (p1 as i32) >= -1_000_999 {
        p1 = debug_load_u32(mem0_base, p0.wrapping_add(12)).wrapping_add(p1.wrapping_shl(4));
    } else if p1 == (-1_001_000i32) as u32 {
        p1 = debug_load_u32(mem0_base, p0.wrapping_add(16)).wrapping_add(40);
    } else {
        l2 = debug_load_u32(mem0_base, l2);
        if debug_load_u8(mem0_base, l2.wrapping_add(8)) == 102 {
            let distance = ((-1_001_000i32) as u32).wrapping_sub(p1) as i32;
            let limit = debug_load_u8(mem0_base, l2.wrapping_add(6)) as i32;
            if distance <= limit {
                p1 = l2
                    .wrapping_add(((-1_001_001i32) as u32).wrapping_sub(p1).wrapping_shl(4))
                    .wrapping_add(16);
            } else {
                p1 = debug_load_u32(mem0_base, p0.wrapping_add(16)).wrapping_add(56);
            }
        } else {
            p1 = debug_load_u32(mem0_base, p0.wrapping_add(16)).wrapping_add(56);
        }
    }

    l2 = debug_load_u8(mem0_base, p1.wrapping_add(8)) as u32;
    let mut p0_result = (l2 & 63).wrapping_add(u32::MAX);
    if p0_result <= 20 {
        match p0_result {
            0 | 5 | 21 => {
                p0_result = 0;
                let tail = (l2 & 15).wrapping_add(u32::MAX);
                if tail <= 5 {
                    match tail {
                        0 | 6 => p0_result = debug_load_u32(mem0_base, p1),
                        5 => {
                            p1 = debug_load_u32(mem0_base, p1);
                            let size = debug_load_u16(mem0_base, p1.wrapping_add(6)) as u32;
                            let offset = if size != 0 {
                                size.wrapping_shl(4).wrapping_add(24)
                            } else {
                                16
                            };
                            p0_result = p1.wrapping_add(offset);
                        }
                        _ => {}
                    }
                }
            }
            20 => p0_result = debug_load_u32(mem0_base, p1),
            _ => {
                p0_result = 0;
                if (l2 & 64) != 0 {
                    p0_result = debug_load_u32(mem0_base, p1);
                }
            }
        }
    } else {
        p0_result = 0;
        if (l2 & 64) != 0 {
            p0_result = debug_load_u32(mem0_base, p1);
        }
    }
    p0_result
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f418_return(fp: *const u64, mem0_base: *const u8) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() || mem0_base.is_null() {
        std::eprintln!("[f418-ret] fp={fp:p} mem0={mem0_base:p}");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 14) };
    let p0 = slots[0] as u32;
    let arg1 = slots[1] as u32;
    let pair = slots[3] as u32;
    let flag = slots[4] as u32;
    let ptr_slot = slots[5] as u32;
    let pair_ptr = debug_load_u32(mem0_base, pair);
    let pair_len = debug_load_u32(mem0_base, pair.wrapping_add(4));
    let pair_obj_tag = debug_load_u8(mem0_base, pair_ptr.wrapping_add(4));
    let pair_obj_small_len = debug_load_u8(mem0_base, pair_ptr.wrapping_add(7));
    let pair_obj_len = debug_load_u32(mem0_base, pair_ptr.wrapping_add(12));
    let final_l3 = debug_load_u32(mem0_base, p0.wrapping_add(12)).wrapping_sub(16);
    let final_kind = (debug_load_u8(mem0_base, final_l3.wrapping_add(8)) & 15).wrapping_add(253);
    let final_obj = debug_load_u32(mem0_base, final_l3);
    let final_obj_tag = debug_load_u8(mem0_base, final_obj.wrapping_add(4));
    let final_obj_small_len = debug_load_u8(mem0_base, final_obj.wrapping_add(7));
    let final_obj_len = debug_load_u32(mem0_base, final_obj.wrapping_add(12));
    let ret_ptr = slots[11] as u32;
    let len_ptr = slots[2] as u32;
    let eval_f58 = debug_eval_f58(mem0_base, p0, arg1);
    let len = unsafe { core::ptr::read_unaligned(mem0_base.add(len_ptr as usize) as *const u32) };
    let bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(ret_ptr as usize), 32) };
    let pair_bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(pair_ptr as usize), 32) };
    let pair_preview = pair_bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    let preview = bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    std::eprintln!(
        "[f418-ret] p0={p0:08x} p1={arg1:08x} pair={pair:08x} pair_ptr={pair_ptr:08x} pair_len={pair_len:08x} pair_tag={pair_obj_tag:02x} pair_small_len={pair_obj_small_len:02x} pair_obj_len={pair_obj_len:08x} flag={flag:08x} ptr_slot={ptr_slot:08x} eval_f58={eval_f58:08x} f54_l3={final_l3:08x} f54_kind={final_kind:08x} f54_obj={final_obj:08x} f54_tag={final_obj_tag:02x} f54_small_len={final_obj_small_len:02x} f54_len={final_obj_len:08x} ret_ptr={ret_ptr:08x} len_ptr={len_ptr:08x} len={len:08x} slot11={:08x} slot12={:08x} slot13={:08x} ret_bytes={:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} ret_text={preview} pair_text={pair_preview}",
        slots[11] as u32,
        slots[12] as u32,
        slots[13] as u32,
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
        bytes[16],
        bytes[17],
        bytes[18],
        bytes[19],
        bytes[20],
        bytes[21],
        bytes[22],
        bytes[23],
        bytes[24],
        bytes[25],
        bytes[26],
        bytes[27],
        bytes[28],
        bytes[29],
        bytes[30],
        bytes[31],
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f205_return(fp: *const u64, mem0_base: *const u8) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() || mem0_base.is_null() {
        std::eprintln!("[f205-ret] fp={fp:p} mem0={mem0_base:p}");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 18) };
    let ret_ptr = slots[15] as u32;
    let obj = ret_ptr.wrapping_sub(16);
    let tag = debug_load_u8(mem0_base, obj.wrapping_add(4));
    let small_len = debug_load_u8(mem0_base, obj.wrapping_add(7));
    let len = debug_load_u32(mem0_base, obj.wrapping_add(12));
    let bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(ret_ptr as usize), 32) };
    let preview = bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    std::eprintln!(
        "[f205-ret] p0={:08x} p1={:08x} p2={:08x} ret_ptr={ret_ptr:08x} obj={obj:08x} tag={tag:02x} small_len={small_len:02x} len={len:08x} bytes={:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} text={preview}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f316_return(fp: *const u64, mem0_base: *const u8) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() || mem0_base.is_null() {
        std::eprintln!("[f316-ret] fp={fp:p} mem0={mem0_base:p}");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 14) };
    let obj = slots[11] as u32;
    let ret_ptr = obj.wrapping_add(16);
    let tag = debug_load_u8(mem0_base, obj.wrapping_add(4));
    let small_len = debug_load_u8(mem0_base, obj.wrapping_add(7));
    let len = debug_load_u32(mem0_base, obj.wrapping_add(12));
    let bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(ret_ptr as usize), 16) };
    let preview = bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    std::eprintln!(
        "[f316-ret] p0={:08x} p1={:08x} p2={:08x} ret_ptr={ret_ptr:08x} obj={obj:08x} tag={tag:02x} small_len={small_len:02x} len={len:08x} text={preview}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f717_return(fp: *const u64) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[f717-ret] fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 12) };
    std::eprintln!(
        "[f717-ret] p0={:08x} p1={:08x} p2={:08x} p3={:08x} ret={:08x}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        slots[3] as u32,
        slots[8] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f720_return(fp: *const u64) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[f720-ret] fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 16) };
    std::eprintln!(
        "[f720-ret] p0={:08x} p1={:08x} p2={:08x} s4={:08x} s5={:08x} ret={:08x}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        slots[4] as u32,
        slots[5] as u32,
        slots[9] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f721_return(fp: *const u64) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[f721-ret] fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 50) };
    std::eprintln!(
        "[f721-ret] p0={:08x} p1={:08x} p2={:08x} p3={:08x} p4={:08x} s15={:08x} s16={:08x} s17={:08x} s18={:08x} s22={:08x} ret={:08x}",
        slots[0] as u32,
        slots[1] as u32,
        slots[2] as u32,
        slots[3] as u32,
        slots[4] as u32,
        slots[15] as u32,
        slots[16] as u32,
        slots[17] as u32,
        slots[18] as u32,
        slots[22] as u32,
        slots[43] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f418_pre_f65(fp: *const u64, mem0_base: *const u8) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() || mem0_base.is_null() {
        std::eprintln!("[f418-pre-f65] fp={fp:p} mem0={mem0_base:p}");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 13) };
    let ptr = slots[5] as u32;
    let len = slots[12] as u32;
    let bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(ptr as usize), 8) };
    let preview = bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    std::eprintln!(
        "[f418-pre-f65] ptr={ptr:08x} len={len:08x} bytes={:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} text={preview}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_f418_f58_frame(phase: u64, fp: *const u64) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if fp.is_null() {
        std::eprintln!("[f418-f58] fp=null");
        return;
    }
    let slots = unsafe { core::slice::from_raw_parts(fp, 14) };
    let phase = if phase == 0 { "pre" } else { "post" };
    std::eprintln!(
        "[f418-f58-{phase}] pair={:08x} slot12={:08x} slot13={:08x}",
        slots[11] as u32,
        slots[12] as u32,
        slots[13] as u32,
    );
}

pub(crate) unsafe extern "C" fn x86_64_debug_dump_pair(pair_ptr: u64, mem0_base: *const u8) {
    if std::env::var_os("SF_DEBUG_TOSTRING_INT").is_none() {
        return;
    }
    if mem0_base.is_null() {
        std::eprintln!("[pair] pair={pair_ptr:08x} mem0={mem0_base:p}");
        return;
    }
    let pair = pair_ptr as u32;
    let ptr = unsafe { core::ptr::read_unaligned(mem0_base.add(pair as usize) as *const u32) };
    let len =
        unsafe { core::ptr::read_unaligned(mem0_base.add(pair as usize + 4) as *const u32) };
    let bytes = unsafe { core::slice::from_raw_parts(mem0_base.add(ptr as usize), 32) };
    let preview = bytes
        .iter()
        .map(|byte| if byte.is_ascii_graphic() || *byte == b' ' { *byte as char } else { '.' })
        .collect::<alloc::string::String>();
    std::eprintln!(
        "[pair] pair={pair:08x} ptr={ptr:08x} len={len:08x} bytes={:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} text={preview}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
        bytes[16],
        bytes[17],
        bytes[18],
        bytes[19],
        bytes[20],
        bytes[21],
        bytes[22],
        bytes[23],
        bytes[24],
        bytes[25],
        bytes[26],
        bytes[27],
        bytes[28],
        bytes[29],
        bytes[30],
        bytes[31],
    );
}

// ── Trapping truncation ──────────────────────────────────────────────────────

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

pub(super) fn trunc_f32_to_i32_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0_f32 || value < -2147483648.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

pub(super) fn trunc_f32_to_i32_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

pub(super) fn trunc_f64_to_i32_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 2147483648.0 || value <= -2147483649.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i32(value as i32))
}

pub(super) fn trunc_f64_to_i32_u(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 4294967296.0 || value <= -1.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(u64::from(value as u32))
}

pub(super) fn trunc_f32_to_i64_s(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0_f32 || value < -9223372036854775808.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

pub(super) fn trunc_f32_to_i64_u(bits: u32) -> Result<u64, WasmError> {
    let value = as_f32(bits as u64);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 18446744073709551616.0_f32 || value <= -1.0_f32 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(value as u64)
}

pub(super) fn trunc_f64_to_i64_s(bits: u64) -> Result<u64, WasmError> {
    let value = as_f64(bits);
    if value.is_nan() {
        return Err(WasmError::trap("invalid conversion to integer".into()));
    }
    if value >= 9223372036854775808.0 || value < -9223372036854775808.0 {
        return Err(WasmError::trap("integer overflow".into()));
    }
    Ok(from_i64(value as i64))
}

pub(super) fn trunc_f64_to_i64_u(bits: u64) -> Result<u64, WasmError> {
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
