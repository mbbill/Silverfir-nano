//! Cold helper boundary for the native backend.
//!
//! Hot native code stays in the native VM ABI. Cold wrappers cross into Rust
//! helpers through one normal-ABI boundary with one metadata pointer.

use alloc::{string::ToString, vec::Vec};
use core::arch::global_asm;

use crate::error::WasmError;
use crate::vm::entities::{Caller, FunctionInst, MemInst};
use crate::vm::interp::raw_value::{raw_to_value, value_to_raw};
use crate::vm::lowered::{IrOpKind, SlotRef, stack_effect};
use crate::vm::store::Store;
use crate::vm::value::Value;
use crate::vm::value::RefHandle as VmRefHandle;

use super::context::Context;
use super::instruction::{NativeEntry, NativeInst, EMPTY_TOS_SLOTS};
use super::runtime::term_entry;

pub type NativeHelper = unsafe extern "C" fn(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HelperResult {
    pub next_entry: NativeEntry,
    pub next_tos_slots: u64,
}

impl HelperResult {
    #[inline]
    pub const fn new(next_entry: NativeEntry, next_tos_slots: u64) -> Self {
        Self {
            next_entry,
            next_tos_slots,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ColdHelperKind {
    CallExternal,
    CallInternal,
    CallIndirect,
    GlobalGet,
    GlobalSet,
    MemoryGrow,
    MemoryCopy,
    MemoryFill,
    MemoryInit,
    DataDrop,
    I32Popcnt,
    I64Popcnt,
    F32Copysign,
    F64Copysign,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    TableGet,
    TableSet,
    TableSize,
    TableGrow,
    TableFill,
    TableCopy,
    TableInit,
    ElemDrop,
    RefNull,
    RefIsNull,
    RefFunc,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct HelperMetadata {
    pub helper: NativeHelper,
    pub next_entry: u64,
    pub next_tos_slots: u64,
    pub branch_entry: u64,
    pub branch_tos_slots: u64,
    pub stack_slot: u64,
    pub data0: u64,
    pub data1: u64,
    pub data2: u64,
}

impl HelperMetadata {
    #[inline]
    pub fn new(
        helper: NativeHelper,
        next_entry: u64,
        next_tos_slots: u64,
        branch_entry: u64,
        branch_tos_slots: u64,
        stack_slot: u64,
        data0: u64,
        data1: u64,
        data2: u64,
    ) -> Self {
        Self {
            helper,
            next_entry,
            next_tos_slots,
            branch_entry,
            branch_tos_slots,
            stack_slot,
            data0,
            data1,
            data2,
        }
    }
}

pub mod meta_offset {
    use super::HelperMetadata;

    pub const HELPER: u32 = core::mem::offset_of!(HelperMetadata, helper) as u32;
    pub const NEXT_ENTRY: u32 = core::mem::offset_of!(HelperMetadata, next_entry) as u32;
    pub const NEXT_TOS_SLOTS: u32 = core::mem::offset_of!(HelperMetadata, next_tos_slots) as u32;
    pub const BRANCH_ENTRY: u32 = core::mem::offset_of!(HelperMetadata, branch_entry) as u32;
    pub const BRANCH_TOS_SLOTS: u32 = core::mem::offset_of!(HelperMetadata, branch_tos_slots) as u32;
    pub const STACK_SLOT: u32 = core::mem::offset_of!(HelperMetadata, stack_slot) as u32;
    pub const DATA0: u32 = core::mem::offset_of!(HelperMetadata, data0) as u32;
    pub const DATA1: u32 = core::mem::offset_of!(HelperMetadata, data1) as u32;
    pub const DATA2: u32 = core::mem::offset_of!(HelperMetadata, data2) as u32;
}

const _: [(); 0] = [(); meta_offset::HELPER as usize];
const _: [(); 8] = [(); meta_offset::NEXT_ENTRY as usize];
const _: [(); 16] = [(); meta_offset::NEXT_TOS_SLOTS as usize];
const _: [(); 24] = [(); meta_offset::BRANCH_ENTRY as usize];
const _: [(); 32] = [(); meta_offset::BRANCH_TOS_SLOTS as usize];
const _: [(); 40] = [(); meta_offset::STACK_SLOT as usize];
const _: [(); 48] = [(); meta_offset::DATA0 as usize];
const _: [(); 56] = [(); meta_offset::DATA1 as usize];
const _: [(); 64] = [(); meta_offset::DATA2 as usize];
const _: [(); 72] = [(); core::mem::size_of::<HelperMetadata>()];

#[inline]
pub fn cold_helper_kind(kind: &IrOpKind) -> Option<ColdHelperKind> {
    match kind {
        IrOpKind::CallExternal { .. } => Some(ColdHelperKind::CallExternal),
        IrOpKind::CallInternal { .. } => Some(ColdHelperKind::CallInternal),
        IrOpKind::CallIndirect { .. } => Some(ColdHelperKind::CallIndirect),
        IrOpKind::GlobalGet { .. } => Some(ColdHelperKind::GlobalGet),
        IrOpKind::GlobalSet { .. } => Some(ColdHelperKind::GlobalSet),
        IrOpKind::MemoryGrow { .. } => Some(ColdHelperKind::MemoryGrow),
        IrOpKind::MemoryCopy { .. } => Some(ColdHelperKind::MemoryCopy),
        IrOpKind::MemoryFill { .. } => Some(ColdHelperKind::MemoryFill),
        IrOpKind::MemoryInit { .. } => Some(ColdHelperKind::MemoryInit),
        IrOpKind::DataDrop { .. } => Some(ColdHelperKind::DataDrop),
        IrOpKind::I32Popcnt => Some(ColdHelperKind::I32Popcnt),
        IrOpKind::I64Popcnt => Some(ColdHelperKind::I64Popcnt),
        IrOpKind::F32Copysign => Some(ColdHelperKind::F32Copysign),
        IrOpKind::F64Copysign => Some(ColdHelperKind::F64Copysign),
        IrOpKind::I32TruncF32S => Some(ColdHelperKind::I32TruncF32S),
        IrOpKind::I32TruncF32U => Some(ColdHelperKind::I32TruncF32U),
        IrOpKind::I32TruncF64S => Some(ColdHelperKind::I32TruncF64S),
        IrOpKind::I32TruncF64U => Some(ColdHelperKind::I32TruncF64U),
        IrOpKind::I64TruncF32S => Some(ColdHelperKind::I64TruncF32S),
        IrOpKind::I64TruncF32U => Some(ColdHelperKind::I64TruncF32U),
        IrOpKind::I64TruncF64S => Some(ColdHelperKind::I64TruncF64S),
        IrOpKind::I64TruncF64U => Some(ColdHelperKind::I64TruncF64U),
        IrOpKind::TableGet { .. } => Some(ColdHelperKind::TableGet),
        IrOpKind::TableSet { .. } => Some(ColdHelperKind::TableSet),
        IrOpKind::TableSize { .. } => Some(ColdHelperKind::TableSize),
        IrOpKind::TableGrow { .. } => Some(ColdHelperKind::TableGrow),
        IrOpKind::TableFill { .. } => Some(ColdHelperKind::TableFill),
        IrOpKind::TableCopy { .. } => Some(ColdHelperKind::TableCopy),
        IrOpKind::TableInit { .. } => Some(ColdHelperKind::TableInit),
        IrOpKind::ElemDrop { .. } => Some(ColdHelperKind::ElemDrop),
        IrOpKind::RefNull => Some(ColdHelperKind::RefNull),
        IrOpKind::RefIsNull => Some(ColdHelperKind::RefIsNull),
        IrOpKind::RefFunc { .. } => Some(ColdHelperKind::RefFunc),
        _ => None,
    }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn native_helper_entry();
    fn native_helper_entry_call_internal();
    fn native_helper_merge_tos_from_x1();
    fn native_helper_entry_write_t0();
    fn native_helper_entry_write_t1();
    fn native_helper_entry_write_t2();
    fn native_helper_entry_write_t3();
    fn native_helper_entry_read_t0();
    fn native_helper_entry_read_t1();
    fn native_helper_entry_read_t2();
    fn native_helper_entry_read_t3();
    fn native_helper_entry_read_top2_d1();
    fn native_helper_entry_read_top2_d2();
    fn native_helper_entry_read_top2_d3();
    fn native_helper_entry_read_top2_d4();
    fn native_helper_entry_read_top3_d1();
    fn native_helper_entry_read_top3_d2();
    fn native_helper_entry_read_top3_d3();
    fn native_helper_entry_read_top3_d4();
}

#[inline]
pub fn helper_entry() -> NativeEntry {
    native_helper_entry
}

#[inline]
pub fn helper_entry_call_internal() -> NativeEntry {
    native_helper_entry_call_internal
}

#[inline]
pub fn helper_entry_write_t0() -> NativeEntry {
    native_helper_entry_write_t0
}

#[inline]
pub fn helper_entry_write_t1() -> NativeEntry {
    native_helper_entry_write_t1
}

#[inline]
pub fn helper_entry_write_t2() -> NativeEntry {
    native_helper_entry_write_t2
}

#[inline]
pub fn helper_entry_write_t3() -> NativeEntry {
    native_helper_entry_write_t3
}

#[inline]
pub fn helper_entry_read_t0() -> NativeEntry {
    native_helper_entry_read_t0
}

#[inline]
pub fn helper_entry_read_t1() -> NativeEntry {
    native_helper_entry_read_t1
}

#[inline]
pub fn helper_entry_read_t2() -> NativeEntry {
    native_helper_entry_read_t2
}

#[inline]
pub fn helper_entry_read_t3() -> NativeEntry {
    native_helper_entry_read_t3
}

#[inline]
pub fn helper_entry_read_top2_d1() -> NativeEntry {
    native_helper_entry_read_top2_d1
}

#[inline]
pub fn helper_entry_read_top2_d2() -> NativeEntry {
    native_helper_entry_read_top2_d2
}

#[inline]
pub fn helper_entry_read_top2_d3() -> NativeEntry {
    native_helper_entry_read_top2_d3
}

#[inline]
pub fn helper_entry_read_top2_d4() -> NativeEntry {
    native_helper_entry_read_top2_d4
}

#[inline]
pub fn helper_entry_read_top3_d1() -> NativeEntry {
    native_helper_entry_read_top3_d1
}

#[inline]
pub fn helper_entry_read_top3_d2() -> NativeEntry {
    native_helper_entry_read_top3_d2
}

#[inline]
pub fn helper_entry_read_top3_d3() -> NativeEntry {
    native_helper_entry_read_top3_d3
}

#[inline]
pub fn helper_entry_read_top3_d4() -> NativeEntry {
    native_helper_entry_read_top3_d4
}

#[cfg(target_arch = "aarch64")]
global_asm!(
    r#"
    .text
    .p2align 2
    .global _native_helper_reload_tos_from_x1
_native_helper_reload_tos_from_x1:
    mov x11, x1
    mov x25, xzr
    mov x26, xzr
    mov x27, xzr
    mov x28, xzr
    movz x12, #0xffff
    and x9, x11, #0xffff
    cmp x9, x12
    b.eq 0f
    ldr x25, [x21, x9, lsl #3]
0:
    ubfx x9, x11, #16, #16
    cmp x9, x12
    b.eq 1f
    ldr x26, [x21, x9, lsl #3]
1:
    ubfx x9, x11, #32, #16
    cmp x9, x12
    b.eq 2f
    ldr x27, [x21, x9, lsl #3]
2:
    ubfx x9, x11, #48, #16
    cmp x9, x12
    b.eq 3f
    ldr x28, [x21, x9, lsl #3]
3:
    ret

    .p2align 2
    .global _native_helper_merge_tos_from_x1
_native_helper_merge_tos_from_x1:
    mov x11, x1
    movz x12, #0xffff
    and x9, x11, #0xffff
    cmp x9, x12
    b.eq 0f
    ldr x25, [x21, x9, lsl #3]
0:
    ubfx x9, x11, #16, #16
    cmp x9, x12
    b.eq 1f
    ldr x26, [x21, x9, lsl #3]
1:
    ubfx x9, x11, #32, #16
    cmp x9, x12
    b.eq 2f
    ldr x27, [x21, x9, lsl #3]
2:
    ubfx x9, x11, #48, #16
    cmp x9, x12
    b.eq 3f
    ldr x28, [x21, x9, lsl #3]
3:
    ret

    .macro save_hot
    stp x22, x23, [x21, #0]
    str x24, [x21, #16]
    .endm

    .macro call_helper
    mov x0, x19
    mov x1, x21
    mov x2, x20
    ldr x16, [x20, #0]
    blr x16
    ldr x21, [x19, #48]
    ldp x22, x23, [x21, #0]
    ldr x24, [x21, #16]
    cmn x1, #1
    b.eq 0f
    bl _native_helper_merge_tos_from_x1
0:
    br x0
    .endm

    .p2align 2
    .global _native_helper_entry
_native_helper_entry:
    save_hot
    call_helper

    .p2align 2
    .global _native_helper_entry_call_internal
_native_helper_entry_call_internal:
    save_hot
    ldr x9, [x20, #64]
    tbz x9, #63, 9f
    ldr x16, [x20, #48]
    cbz x16, 9f
    ldr x10, [x20, #56]
    ubfx x11, x9, #0, #16
    ubfx x12, x9, #16, #16
    add x13, x21, x10, lsl #3
    add x14, x11, x12
    add x15, x13, x14, lsl #3
    add x15, x15, #24
    ldr x17, [x19, #0]
    cmp x15, x17
    b.hi 9f
    ldr x17, [x19, #8]
    cmp x17, #300
    b.hs 9f
    add x17, x17, #1
    str x17, [x19, #8]
    cbz x12, 1f
    add x15, x13, x11, lsl #3
    mov x17, x12
0:
    cmp x17, #2
    b.lo 2f
    stp xzr, xzr, [x15], #16
    sub x17, x17, #2
    cbnz x17, 0b
    b 1f
2:
    str xzr, [x15]
1:
    add x15, x13, x14, lsl #3
    ldr x17, [x20, #8]
    str x17, [x15]
    str x21, [x15, #8]
    ldr x17, [x20, #16]
    str x17, [x15, #16]
    mov x21, x13
    mov x25, xzr
    mov x26, xzr
    mov x27, xzr
    mov x28, xzr
    br x16
9:
    call_helper

    .p2align 2
    .global _native_helper_entry_write_t0
_native_helper_entry_write_t0:
    save_hot
    call_helper

    .p2align 2
    .global _native_helper_entry_write_t1
_native_helper_entry_write_t1:
    save_hot
    call_helper

    .p2align 2
    .global _native_helper_entry_write_t2
_native_helper_entry_write_t2:
    save_hot
    call_helper

    .p2align 2
    .global _native_helper_entry_write_t3
_native_helper_entry_write_t3:
    save_hot
    call_helper

    .p2align 2
    .global _native_helper_entry_read_t0
_native_helper_entry_read_t0:
    save_hot
    ldr x9, [x20, #40]
    str x25, [x21, x9, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_t1
_native_helper_entry_read_t1:
    save_hot
    ldr x9, [x20, #40]
    str x26, [x21, x9, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_t2
_native_helper_entry_read_t2:
    save_hot
    ldr x9, [x20, #40]
    str x27, [x21, x9, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_t3
_native_helper_entry_read_t3:
    save_hot
    ldr x9, [x20, #40]
    str x28, [x21, x9, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top2_d1
_native_helper_entry_read_top2_d1:
    save_hot
    ldr x9, [x20, #40]
    str x25, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x28, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top2_d2
_native_helper_entry_read_top2_d2:
    save_hot
    ldr x9, [x20, #40]
    str x26, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x25, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top2_d3
_native_helper_entry_read_top2_d3:
    save_hot
    ldr x9, [x20, #40]
    str x27, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x26, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top2_d4
_native_helper_entry_read_top2_d4:
    save_hot
    ldr x9, [x20, #40]
    str x28, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x27, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top3_d1
_native_helper_entry_read_top3_d1:
    save_hot
    ldr x9, [x20, #40]
    str x25, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x28, [x21, x10, lsl #3]
    sub x10, x9, #2
    str x27, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top3_d2
_native_helper_entry_read_top3_d2:
    save_hot
    ldr x9, [x20, #40]
    str x26, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x25, [x21, x10, lsl #3]
    sub x10, x9, #2
    str x28, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top3_d3
_native_helper_entry_read_top3_d3:
    save_hot
    ldr x9, [x20, #40]
    str x27, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x26, [x21, x10, lsl #3]
    sub x10, x9, #2
    str x25, [x21, x10, lsl #3]
    call_helper

    .p2align 2
    .global _native_helper_entry_read_top3_d4
_native_helper_entry_read_top3_d4:
    save_hot
    ldr x9, [x20, #40]
    str x28, [x21, x9, lsl #3]
    sub x10, x9, #1
    str x27, [x21, x10, lsl #3]
    sub x10, x9, #2
    str x26, [x21, x10, lsl #3]
    call_helper
"#
);

#[inline]
fn resume_at(
    ctx: *mut Context,
    fp: *mut u64,
    next_entry: NativeEntry,
    next_tos_slots: u64,
) -> HelperResult {
    unsafe {
        (*ctx).hot.resume_fp = fp;
    }
    HelperResult::new(next_entry, next_tos_slots)
}

#[inline]
fn trap_with(ctx: *mut Context, fp: *mut u64, error: WasmError) -> HelperResult {
    unsafe {
        if (*ctx).error.is_none() {
            (*ctx).error = Some(error);
        }
    }
    resume_at(ctx, fp, term_entry(), EMPTY_TOS_SLOTS)
}

#[inline]
fn frame_read(fp: *mut u64, offset: usize) -> u64 {
    unsafe { *fp.add(offset) }
}

#[inline]
fn frame_write(fp: *mut u64, offset: usize, value: u64) {
    unsafe {
        *fp.add(offset) = value;
    }
}

#[inline]
fn operand_read(fp: *mut u64, operand_base_offset: usize, index: usize) -> u64 {
    unsafe {
        let operand_base = fp.add(operand_base_offset / 8);
        *operand_base.add(index)
    }
}

#[inline]
fn operand_base(fp: *mut u64, operand_base_offset: usize) -> *mut u64 {
    unsafe { fp.add(operand_base_offset / 8) }
}

#[inline]
fn refresh_mem0(ctx: *mut Context, store: &Store) {
    unsafe {
        if store.module().memories.is_empty() {
            (*ctx).hot.mem0_base = core::ptr::null_mut();
            (*ctx).hot.mem0_size = 0;
        } else {
            let mem = &store.module().memories[0];
            (*ctx).hot.mem0_base = mem.data.as_ptr() as *mut u8;
            (*ctx).hot.mem0_size = mem.data.len() as u64;
        }
    }
}

fn debug_helper_origin(store: &Store, meta: *const HelperMetadata) -> Option<(usize, usize)> {
    for (func_idx, func) in store.module().functions.iter().enumerate() {
        let FunctionInst::Local { spec, .. } = func else {
            continue;
        };
        let Some(native) = spec.get_native_code() else {
            continue;
        };
        let metas = native.helper_metadata();
        let start = metas.as_ptr() as usize;
        let end = start + metas.len() * core::mem::size_of::<HelperMetadata>();
        let ptr = meta as usize;
        if ptr >= start && ptr < end {
            let meta_idx = (ptr - start) / core::mem::size_of::<HelperMetadata>();
            return Some((func_idx, meta_idx));
        }
    }
    None
}

fn debug_function_index(store: &Store, func: *const FunctionInst) -> Option<usize> {
    store
        .module()
        .functions
        .iter()
        .enumerate()
        .find_map(|(func_idx, candidate)| {
            let candidate_ptr = candidate as *const FunctionInst;
            core::ptr::eq(candidate_ptr, func).then_some(func_idx)
        })
}

fn debug_helper_site(store: &Store, func_idx: usize, meta_idx: usize) -> Option<alloc::string::String> {
    let func = store.module().functions.get(func_idx)?;
    let (spec, type_index) = match func {
        FunctionInst::Local { spec, type_index } => (spec, *type_index as usize),
        FunctionInst::External { .. } => return None,
    };

    let func_type = spec.func_type();
    let params_count = func_type.params().len();
    let locals_count = spec.locals().len();
    let results_count = func_type.results().len();
    let frame_size = params_count + locals_count;
    let config = crate::vm::backend::BackendKind::Native.compile_config();
    let plan = crate::vm::planner::CompilePlan::for_config(spec.code(), frame_size, config);
    let hot_local_plan = plan.hot_locals();
    let ctx = crate::vm::compile::CompileContext::new(
        Some(store.module().types.as_slice()),
        store,
        store.module(),
        results_count,
    );
    let mut stack = crate::vm::compile::StackTracker::new(
        plan.config(),
        params_count,
        locals_count,
        results_count,
    );
    let ir_ops = crate::vm::lowered::lower_to_ir(spec.code(), &ctx, &mut stack, hot_local_plan).ok()?;
    let resolved = crate::vm::native::resolve_backend(
        &ir_ops,
        stack.operand_base(),
        plan.config().tos_register_count,
        store.module(),
        hot_local_plan.hot_mask(),
        func_idx as u32,
    )
    .ok()?;
    let (op_idx, op) = resolved
        .iter()
        .enumerate()
        .filter(|(_, op)| op.cold_helper.is_some())
        .nth(meta_idx)?;

    let mut inverse = alloc::vec![0u32; frame_size];
    for original in 0..frame_size as u32 {
        let remapped = hot_local_plan.remap_local(original) as usize;
        if remapped < inverse.len() {
            inverse[remapped] = original;
        }
    }

    let ir_site = ir_ops
        .iter()
        .enumerate()
        .filter(|(_, op)| cold_helper_kind(&op.kind).is_some())
        .nth(meta_idx);
    let mut resolved_window = alloc::string::String::new();
    let window_start = op_idx.saturating_sub(8);
    let window_end = core::cmp::min(op_idx + 9, resolved.len());
    for (idx, candidate) in resolved
        .iter()
        .enumerate()
        .skip(window_start)
        .take(window_end - window_start)
    {
        if !resolved_window.is_empty() {
            resolved_window.push_str(" | ");
        }
        let marker = if idx == op_idx { "*" } else { "" };
        resolved_window.push_str(&alloc::format!(
            "{}{}:{:?}/h{}",
            marker, idx, candidate.kind, candidate.pre_height
        ));
    }

    let ir_summary = if let Some((ir_idx, ir_op)) = ir_site {
        let ir_window_start = ir_idx.saturating_sub(12);
        let ir_window_end = core::cmp::min(ir_idx + 13, ir_ops.len());
        let mut ir_window = alloc::string::String::new();
        for (idx, candidate) in ir_ops
            .iter()
            .enumerate()
            .skip(ir_window_start)
            .take(ir_window_end - ir_window_start)
        {
            if !ir_window.is_empty() {
                ir_window.push_str(" | ");
            }
            let marker = if idx == ir_idx { "*" } else { "" };
            ir_window.push_str(&alloc::format!(
                "{}{}:{:?}/h{}",
                marker, idx, candidate.kind, candidate.pre_height
            ));
        }
        let tracked_local = inverse.get(6).copied();
        let local_write = tracked_local.and_then(|original_local| {
            let remapped = hot_local_plan.remap_local(original_local) as u16;
            ir_ops
                .iter()
                .enumerate()
                .take(ir_idx)
                .rev()
                .find(|(_, candidate)| {
                    matches!(
                        candidate.kind,
                        crate::vm::lowered::IrOpKind::LocalSetFrame { idx }
                            | crate::vm::lowered::IrOpKind::LocalTeeFrame { idx }
                            if idx == remapped
                    )
                })
        });
        let local_write_summary = if let Some((write_idx, write_op)) = local_write {
            alloc::format!(" last_write_local0={} {:?}", write_idx, write_op.kind)
        } else {
            " last_write_local0=?".into()
        };
        alloc::format!(
            " ir_idx={} ir_kind={:?} ir_window=[{}]{}",
            ir_idx, ir_op.kind, ir_window, local_write_summary
        )
    } else {
        " ir_idx=?".into()
    };

    Some(alloc::format!(
        "func={} type={} frame={} raw_hot={:?} eff_hot={:?} inverse={:?} cold_meta={} op_idx={} pre_h={} kind={:?} window=[{}]{}",
        func_idx,
        type_index,
        frame_size,
        hot_local_plan.raw(),
        hot_local_plan.effective(),
        inverse,
        meta_idx,
        op_idx,
        op.pre_height,
        op.kind,
        resolved_window,
        ir_summary,
    ))
}

const MAX_CALL_DEPTH: u64 = 300;

#[inline]
fn invoke_external_callee(
    callee: &FunctionInst,
    store_ref: &mut Store,
    fp: *mut u64,
    delta: usize,
) -> Result<(), WasmError> {
    let (func_type, callback) = match callee {
        FunctionInst::External { func_type, callback } => (func_type, callback),
        _ => return Err(WasmError::internal("expected external function".into())),
    };

    let params = func_type.params();
    let results = func_type.results();
    let results_len = results.len();

    let args: Vec<Value> = params
        .iter()
        .enumerate()
        .map(|(i, ty)| raw_to_value(frame_read(fp, delta + i), *ty))
        .collect();

    let mut ret_vals = alloc::vec![Value::default(); results_len];
    let mem_slice = if !store_ref.module().memories.is_empty() {
        let mem = &store_ref.module().memories[0] as *const MemInst as *mut MemInst;
        unsafe { Some((*mem).data.as_mut_slice()) }
    } else {
        None
    };
    let mut caller = Caller::new(mem_slice);
    callback(&mut caller, &args, &mut ret_vals)?;

    if ret_vals.len() != results_len {
        return Err(WasmError::internal("result count mismatch".into()));
    }

    for (i, v) in ret_vals.into_iter().enumerate() {
        frame_write(fp, delta + i, value_to_raw(v));
    }

    Ok(())
}

#[inline]
fn ensure_native_entry(
    callee: &FunctionInst,
    store_ref: &mut Store,
) -> Result<NativeEntry, WasmError> {
    let spec = match callee {
        FunctionInst::Local { spec, .. } => spec,
        FunctionInst::External { .. } => {
            return Err(WasmError::internal(
                "expected local function for native call".into(),
            ));
        }
    };

    if !spec.has_native_code() {
        crate::vm::native::precompile::precompile_module(store_ref)?;
    }

    if !spec.has_native_code() {
        let module = store_ref.module();
        if let Some((func_idx, _)) =
            module
                .functions
                .iter()
                .enumerate()
                .find(|(_, candidate)| match candidate {
                    FunctionInst::Local { spec: candidate_spec, .. } => {
                        core::ptr::eq(candidate_spec as *const _, spec as *const _)
                    }
                    FunctionInst::External { .. } => false,
                })
        {
            crate::vm::native::compiler::build_for_function(
                spec,
                Some(&module.types),
                store_ref,
                module,
                func_idx as u32,
            )?;
        }
    }

    if !spec.has_native_code() {
        return Err(WasmError::invalid(
            "native backend unavailable for function".into(),
        ));
    }

    Ok(spec.native_cache().entry())
}

#[inline]
fn enter_unified_callee(
    ctx: *mut Context,
    fp: *mut u64,
    store_ref: &mut Store,
    callee: &FunctionInst,
    delta: usize,
    return_entry: NativeEntry,
    return_tos_slots: u64,
) -> Result<HelperResult, WasmError> {
    let spec = match callee {
        FunctionInst::Local { spec, .. } => spec,
        FunctionInst::External { .. } => {
            return Err(WasmError::trap("unexpected external function".into()));
        }
    };

    let entry = ensure_native_entry(callee, store_ref)?;
    let params_count = spec.func_type().params().len();
    let locals_count = spec.locals().len();
    let frame_size = params_count + locals_count;
    let callee_fp = unsafe { fp.add(delta) };
    let new_stack_top = unsafe { callee_fp.add(frame_size + 3) };

    unsafe {
        if new_stack_top > (*ctx).hot.stack_end {
            return Err(WasmError::exhaustion("stack overflow".into()));
        }
        if (*ctx).hot.call_depth >= MAX_CALL_DEPTH {
            return Err(WasmError::exhaustion("call stack exhausted".into()));
        }
        (*ctx).hot.call_depth += 1;
    }

    if locals_count > 0 {
        unsafe {
            let mut local_dst = callee_fp.add(params_count);
            let mut remaining = locals_count;
            while remaining >= 2 {
                *local_dst = 0;
                *local_dst.add(1) = 0;
                local_dst = local_dst.add(2);
                remaining -= 2;
            }
            if remaining != 0 {
                *local_dst = 0;
            }
        }
    }

    unsafe {
        *callee_fp.add(frame_size) = return_entry as usize as u64;
        *callee_fp.add(frame_size + 1) = fp as u64;
        *callee_fp.add(frame_size + 2) = return_tos_slots;
    }

    Ok(resume_at(ctx, callee_fp, entry, EMPTY_TOS_SLOTS))
}

#[inline]
unsafe fn store_mut<'a>(ctx: *mut Context) -> &'a mut Store {
    &mut *(*ctx).store
}

#[inline]
fn meta_next(_ctx: *mut Context, fp: *mut u64, meta: *const HelperMetadata) -> HelperResult {
    resume_at(
        _ctx,
        fp,
        unsafe { core::mem::transmute::<u64, NativeEntry>((*meta).next_entry) },
        unsafe { (*meta).next_tos_slots },
    )
}

fn pack_helper_result_tos(pre_height: u16, kind: &IrOpKind, result_slot: Option<u64>) -> u64 {
    let Some(result_slot) = result_slot else {
        return EMPTY_TOS_SLOTS;
    };
    let (pop, push) = stack_effect(kind);
    debug_assert!(push <= 1, "cold helper result pack only supports arity <= 1");
    if push == 0 {
        return EMPTY_TOS_SLOTS;
    }
    let post_height = (pre_height as usize).saturating_sub(pop as usize) + push as usize;
    let variant = crate::vm::native::arm64::depth_variant(post_height);
    let top_reg = crate::vm::native::arm64::tos_reg(variant, 1);
    let reg_idx = match top_reg {
        crate::vm::native::arm64::reg::Reg::T0 => 0,
        crate::vm::native::arm64::reg::Reg::T1 => 1,
        crate::vm::native::arm64::reg::Reg::T2 => 2,
        crate::vm::native::arm64::reg::Reg::T3 => 3,
        _ => unreachable!("top result must map to a TOS register"),
    };
    let mut packed = EMPTY_TOS_SLOTS;
    let shift = reg_idx * 16;
    packed &= !(0xffffu64 << shift);
    packed |= result_slot << shift;
    packed
}

fn helper_result_slot(helper_kind: ColdHelperKind, stack_slot: u64) -> Option<u64> {
    match helper_kind {
        ColdHelperKind::GlobalGet
        | ColdHelperKind::MemoryGrow
        | ColdHelperKind::I32Popcnt
        | ColdHelperKind::I64Popcnt
        | ColdHelperKind::I32TruncF32S
        | ColdHelperKind::I32TruncF32U
        | ColdHelperKind::I32TruncF64S
        | ColdHelperKind::I32TruncF64U
        | ColdHelperKind::I64TruncF32S
        | ColdHelperKind::I64TruncF32U
        | ColdHelperKind::I64TruncF64S
        | ColdHelperKind::I64TruncF64U
        | ColdHelperKind::TableGet
        | ColdHelperKind::TableSize
        | ColdHelperKind::RefNull
        | ColdHelperKind::RefIsNull
        | ColdHelperKind::RefFunc => Some(stack_slot),
        ColdHelperKind::F32Copysign
        | ColdHelperKind::F64Copysign
        | ColdHelperKind::TableGrow => Some(stack_slot.saturating_sub(1)),
        ColdHelperKind::CallExternal
        | ColdHelperKind::CallInternal
        | ColdHelperKind::CallIndirect
        | ColdHelperKind::GlobalSet
        | ColdHelperKind::MemoryCopy
        | ColdHelperKind::MemoryFill
        | ColdHelperKind::MemoryInit
        | ColdHelperKind::DataDrop
        | ColdHelperKind::TableSet
        | ColdHelperKind::TableFill
        | ColdHelperKind::TableCopy
        | ColdHelperKind::TableInit
        | ColdHelperKind::ElemDrop => None,
    }
}

#[inline]
pub fn build_metadata<F: Fn(SlotRef) -> u16>(
    helper_kind: ColdHelperKind,
    kind: &IrOpKind,
    pre_height: u16,
    current: *mut NativeInst,
    next: *mut NativeInst,
    branch_target: Option<*mut NativeInst>,
    stack_slot: Option<SlotRef>,
    fix_slot: &F,
) -> HelperMetadata {
    let stack_slot = stack_slot.map_or(0, |slot| fix_slot(slot) as u64);
    let next_entry = unsafe { (*next).entry as usize as u64 };
    let next_tos_slots = match helper_kind {
        ColdHelperKind::CallExternal | ColdHelperKind::CallInternal | ColdHelperKind::CallIndirect => {
            unsafe { (*next).tos_slots }
        }
        _ => pack_helper_result_tos(pre_height, kind, helper_result_slot(helper_kind, stack_slot)),
    };
    let (branch_entry, branch_tos_slots) = match branch_target {
        Some(target) => unsafe { ((*target).entry as usize as u64, (*target).tos_slots) },
        None => (0, EMPTY_TOS_SLOTS),
    };
    match (helper_kind, kind) {
        (ColdHelperKind::CallExternal, IrOpKind::CallExternal { func_idx, delta, .. }) => {
            HelperMetadata::new(
                call_external_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
                *func_idx as u64,
                fix_slot(*delta) as u64,
                0,
            )
        }
        (ColdHelperKind::CallInternal, IrOpKind::CallInternal { callee, delta, .. }) => {
            HelperMetadata::new(
                call_internal_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
                *callee,
                fix_slot(*delta) as u64,
                0,
            )
        }
        (
            ColdHelperKind::CallIndirect,
            IrOpKind::CallIndirect {
                type_idx,
                table_idx,
                delta,
                operand_base_offset,
                height,
            },
        ) => HelperMetadata::new(
            call_indirect_helper,
            next_entry,
            next_tos_slots,
            branch_entry,
            branch_tos_slots,
            stack_slot,
            ((*type_idx as u64) << 32) | (*table_idx as u64),
            fix_slot(*delta) as u64,
            ((*operand_base_offset as u64) << 32) | (*height as u64),
        ),
        (ColdHelperKind::GlobalGet, IrOpKind::GlobalGet { idx }) => HelperMetadata::new(
                global_get_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::GlobalSet, IrOpKind::GlobalSet { idx }) => HelperMetadata::new(
                global_set_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::MemoryGrow, IrOpKind::MemoryGrow { mem_idx }) => HelperMetadata::new(
                memory_grow_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *mem_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::MemoryCopy, IrOpKind::MemoryCopy { imm0, imm1 }) => HelperMetadata::new(
                memory_copy_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            *imm1 as u64,
            0,
        ),
        (ColdHelperKind::MemoryFill, IrOpKind::MemoryFill { imm0, imm1: _ }) => HelperMetadata::new(
                memory_fill_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            0,
            0,
        ),
        (ColdHelperKind::MemoryInit, IrOpKind::MemoryInit { imm0, imm1 }) => HelperMetadata::new(
                memory_init_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            *imm1 as u64,
            0,
        ),
        (ColdHelperKind::DataDrop, IrOpKind::DataDrop { data_idx }) => HelperMetadata::new(
                data_drop_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *data_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::I32Popcnt, IrOpKind::I32Popcnt) => HelperMetadata::new(
                i32_popcnt_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I64Popcnt, IrOpKind::I64Popcnt) => HelperMetadata::new(
                i64_popcnt_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::F32Copysign, IrOpKind::F32Copysign) => HelperMetadata::new(
                f32_copysign_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::F64Copysign, IrOpKind::F64Copysign) => HelperMetadata::new(
                f64_copysign_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I32TruncF32S, IrOpKind::I32TruncF32S) => HelperMetadata::new(
                i32_trunc_f32_s_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I32TruncF32U, IrOpKind::I32TruncF32U) => HelperMetadata::new(
                i32_trunc_f32_u_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I32TruncF64S, IrOpKind::I32TruncF64S) => HelperMetadata::new(
                i32_trunc_f64_s_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I32TruncF64U, IrOpKind::I32TruncF64U) => HelperMetadata::new(
                i32_trunc_f64_u_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I64TruncF32S, IrOpKind::I64TruncF32S) => HelperMetadata::new(
                i64_trunc_f32_s_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I64TruncF32U, IrOpKind::I64TruncF32U) => HelperMetadata::new(
                i64_trunc_f32_u_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I64TruncF64S, IrOpKind::I64TruncF64S) => HelperMetadata::new(
                i64_trunc_f64_s_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::I64TruncF64U, IrOpKind::I64TruncF64U) => HelperMetadata::new(
                i64_trunc_f64_u_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::TableGet, IrOpKind::TableGet { table_idx }) => HelperMetadata::new(
                table_get_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *table_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::TableSet, IrOpKind::TableSet { table_idx }) => HelperMetadata::new(
                table_set_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *table_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::TableSize, IrOpKind::TableSize { table_idx }) => HelperMetadata::new(
                table_size_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *table_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::TableGrow, IrOpKind::TableGrow { table_idx }) => HelperMetadata::new(
                table_grow_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *table_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::TableFill, IrOpKind::TableFill { imm0, .. }) => HelperMetadata::new(
                table_fill_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            0,
            0,
        ),
        (ColdHelperKind::TableCopy, IrOpKind::TableCopy { imm0, imm1 }) => HelperMetadata::new(
                table_copy_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            *imm1 as u64,
            0,
        ),
        (ColdHelperKind::TableInit, IrOpKind::TableInit { imm0, imm1 }) => HelperMetadata::new(
                table_init_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *imm0 as u64,
            *imm1 as u64,
            0,
        ),
        (ColdHelperKind::ElemDrop, IrOpKind::ElemDrop { elem_idx }) => HelperMetadata::new(
                elem_drop_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *elem_idx as u64,
            0,
            0,
        ),
        (ColdHelperKind::RefNull, IrOpKind::RefNull) => HelperMetadata::new(
                ref_null_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::RefIsNull, IrOpKind::RefIsNull) => HelperMetadata::new(
                ref_is_null_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            0,
            0,
            0,
        ),
        (ColdHelperKind::RefFunc, IrOpKind::RefFunc { func_idx }) => HelperMetadata::new(
                ref_func_helper,
                next_entry,
                next_tos_slots,
                branch_entry,
                branch_tos_slots,
                stack_slot,
            *func_idx as u64,
            0,
            0,
        ),
        _ => unreachable!("cold helper kind/op mismatch"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn call_external_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let func_idx = (*meta).data0 as usize;
    let delta = (*meta).data1 as usize;
    let store = store_mut(ctx);
    let callee = store.function(func_idx) as *const FunctionInst;

    match invoke_external_callee(&*callee, store, fp, delta) {
        Ok(()) => {
            refresh_mem0(ctx, store);
            meta_next(ctx, fp, meta)
        }
        Err(e) => trap_with(ctx, fp, e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn call_internal_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let callee = (*meta).data0 as *const FunctionInst;
    let delta = (*meta).data1 as usize;
    let store = store_mut(ctx);
    match enter_unified_callee(
        ctx,
        fp,
        store,
        &*callee,
        delta,
        core::mem::transmute::<u64, NativeEntry>((*meta).next_entry),
        (*meta).next_tos_slots,
    ) {
        Ok(result) => result,
        Err(e) => trap_with(ctx, fp, e),
    }
}

#[no_mangle]
pub unsafe extern "C" fn call_indirect_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let type_idx = ((*meta).data0 >> 32) as usize;
    let table_idx = ((*meta).data0 & 0xffff_ffff) as usize;
    let delta = (*meta).data1 as usize;
    let operand_base_offset = ((*meta).data2 >> 32) as usize;
    let height = ((*meta).data2 & 0xffff_ffff) as usize;

    let store = store_mut(ctx);
    let elem_index = operand_read(fp, operand_base_offset, height - 1) as usize;
    let expected_ty = match store.module().get_type(type_idx as u32) {
        Some(t) => t.clone(),
        None => return trap_with(ctx, fp, WasmError::trap("indirect call type error".into())),
    };

    let table = store.table(table_idx);
    if elem_index >= table.elements.len() {
        return trap_with(
            ctx,
            fp,
            WasmError::trap(alloc::format!(
                "undefined element: table={} index={} len={}",
                table_idx,
                elem_index,
                table.elements.len()
            )),
        );
    }
    let func_ref = table.elements[elem_index];
    if func_ref.is_null() {
        let origin = debug_helper_origin(store, meta)
            .map(|(func_idx, meta_idx)| alloc::format!(" func={} helper_meta={}", func_idx, meta_idx))
            .unwrap_or_default();
        let site = if let Some((func_idx, meta_idx)) = debug_helper_origin(store, meta) {
            debug_helper_site(store, func_idx, meta_idx)
                .map(|s| alloc::format!(" site={}", s))
                .unwrap_or_default()
        } else {
            alloc::string::String::new()
        };
        let arg0 = frame_read(fp, delta);
        let arg1 = frame_read(fp, delta + 1);
        let arg2 = frame_read(fp, delta + 2);
        let table_dump = table
            .elements
            .iter()
            .enumerate()
            .map(|(i, r)| {
                if r.is_null() {
                    alloc::format!("{}=null", i)
                } else {
                    alloc::format!("{}={}", i, r.raw_value())
                }
            })
            .collect::<alloc::vec::Vec<_>>()
            .join(",");
        return trap_with(
            ctx,
            fp,
            WasmError::trap(alloc::format!(
                "uninitialized element: table={} index={} len={} elems=[{}] height={} delta={} operand_base_offset={} args=[{:#x},{:#x},{:#x}]{}{}",
                table_idx,
                elem_index,
                table.elements.len(),
                table_dump,
                height,
                delta,
                operand_base_offset,
                arg0,
                arg1,
                arg2,
                origin,
                site,
            )),
        );
    }

    let func_index = func_ref.raw_value();
    if func_index >= store.module().functions.len() {
        return trap_with(
            ctx,
            fp,
            WasmError::trap(alloc::format!(
                "invalid element target: table={} index={} func_index={} func_count={}",
                table_idx,
                elem_index,
                func_index,
                store.module().functions.len()
            )),
        );
    }
    let callee = store.function(func_index) as *const FunctionInst;

    let actual_type = (&*callee).func_type();
    if *actual_type != *expected_ty {
        let mut types_equivalent = false;
        for (idx, ft) in store.module().types.as_slice().iter().enumerate() {
            if **ft == *actual_type {
                types_equivalent = store.module().types.types_equivalent(idx as u32, type_idx as u32);
                break;
            }
        }
        if !types_equivalent {
            return trap_with(ctx, fp, WasmError::trap("indirect call type mismatch".into()));
        }
    }

    if (&*callee).is_external() {
        match invoke_external_callee(&*callee, store, fp, delta) {
            Ok(()) => {
                refresh_mem0(ctx, store);
                meta_next(ctx, fp, meta)
            }
            Err(e) => trap_with(ctx, fp, e),
        }
    } else {
        match enter_unified_callee(
            ctx,
            fp,
            store,
            &*callee,
            delta,
            core::mem::transmute::<u64, NativeEntry>((*meta).next_entry),
            (*meta).next_tos_slots,
        ) {
            Ok(result) => result,
            Err(e) => trap_with(ctx, fp, e),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn global_get_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let idx = (*meta).data0 as usize;
    let slot = (*meta).stack_slot as usize;
    let store = store_mut(ctx);
    let global = store.global(idx);
    frame_write(fp, slot, value_to_raw(global.value));
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn global_set_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let idx = (*meta).data0 as usize;
    let slot = (*meta).stack_slot as usize;
    let raw = frame_read(fp, slot);
    let store = store_mut(ctx);
    let global = store.global_mut(idx);
    global.value = raw_to_value(raw, global.value_type);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn memory_copy_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let dst_idx = (*meta).data0 as usize;
    let src_idx = (*meta).data1 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let src = frame_read(fp, top - 1) as usize;
    let dst = frame_read(fp, top - 2) as usize;
    let store = store_mut(ctx);

    if dst_idx == src_idx {
        let mem_data = &mut store.memory_mut(dst_idx).data;
        if src.saturating_add(size) > mem_data.len()
            || dst.saturating_add(size) > mem_data.len()
        {
            return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
        }
        mem_data.copy_within(src..src + size, dst);
        return meta_next(ctx, fp, meta);
    }

    let module = store.module_mut();
    let (src_mem, dst_mem) = if src_idx < dst_idx {
        let (left, right) = module.memories.split_at_mut(dst_idx);
        (&left[src_idx], &mut right[0])
    } else {
        let (left, right) = module.memories.split_at_mut(src_idx);
        (&right[0] as &MemInst, &mut left[dst_idx])
    };

    if src.saturating_add(size) > src_mem.data.len()
        || dst.saturating_add(size) > dst_mem.data.len()
    {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }

    dst_mem.data[dst..dst + size].copy_from_slice(&src_mem.data[src..src + size]);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn memory_grow_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let mem_idx = (*meta).data0 as usize;
    let slot = (*meta).stack_slot as usize;
    let delta_pages_raw = frame_read(fp, slot);
    let store = store_mut(ctx);
    let mem = store.memory_mut(mem_idx);
    let is_64 = mem.limits.is64;
    let error_value = if is_64 { u64::MAX } else { u32::MAX as u64 };

    let delta_pages = if is_64 {
        delta_pages_raw as i64 as usize
    } else {
        let d = delta_pages_raw as i32;
        if d < 0 {
            frame_write(fp, slot, error_value);
            return meta_next(ctx, fp, meta);
        }
        d as usize
    };

    let old_pages = mem.current_pages();
    let new_pages = old_pages.saturating_add(delta_pages);
    let max_pages = mem.limits.get_max();
    let result = if new_pages > max_pages {
        error_value
    } else {
        let new_len_bytes = new_pages * crate::constants::WASM_PAGE_SIZE;
        mem.data.resize(new_len_bytes, 0);
        old_pages as u64
    };

    frame_write(fp, slot, result);
    refresh_mem0(ctx, store);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn memory_fill_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let mem_idx = (*meta).data0 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let value = frame_read(fp, top - 1) as u8;
    let dst = frame_read(fp, top - 2) as usize;
    let store = store_mut(ctx);
    let mem_data = &mut store.memory_mut(mem_idx).data;

    if dst.saturating_add(size) > mem_data.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }

    mem_data[dst..dst + size].fill(value);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn memory_init_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let mem_idx = (*meta).data0 as usize;
    let data_idx = (*meta).data1 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let src = frame_read(fp, top - 1) as usize;
    let dst = frame_read(fp, top - 2) as usize;

    let store = store_mut(ctx);
    let module = store.module_mut();
    if data_idx >= module.data.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }

    let data_bytes = &module.data[data_idx].bytes;
    let data_dropped = module.data[data_idx].is_dropped();

    if size == 0 {
        if src > data_bytes.len() || dst > module.memories[mem_idx].data.len() {
            return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
        }
        return meta_next(ctx, fp, meta);
    }
    if data_dropped {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }
    if src.saturating_add(size) > data_bytes.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }

    let src_ptr = data_bytes[src..].as_ptr();
    let mem_data = &mut module.memories[mem_idx].data;
    if dst.saturating_add(size) > mem_data.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds memory access".into()));
    }
    core::ptr::copy_nonoverlapping(src_ptr, mem_data[dst..].as_mut_ptr(), size);

    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn data_drop_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let data_idx = (*meta).data0 as usize;
    let store = store_mut(ctx);
    let module = store.module_mut();
    if data_idx < module.data.len() {
        module.data[data_idx].drop_segment();
    }
    meta_next(ctx, fp, meta)
}

#[inline]
fn read_f32_bits(raw: u64) -> f32 {
    f32::from_bits(raw as u32)
}

#[inline]
fn read_f64_bits(raw: u64) -> f64 {
    f64::from_bits(raw)
}

#[inline]
fn write_f32_bits(fp: *mut u64, slot: usize, value: f32) {
    frame_write(fp, slot, value.to_bits() as u64);
}

#[inline]
fn write_f64_bits(fp: *mut u64, slot: usize, value: f64) {
    frame_write(fp, slot, value.to_bits());
}

#[no_mangle]
pub unsafe extern "C" fn i32_popcnt_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    frame_write(fp, slot, (frame_read(fp, slot) as u32).count_ones() as u64);
    meta_next(_ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i64_popcnt_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    frame_write(fp, slot, frame_read(fp, slot).count_ones() as u64);
    meta_next(_ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn f32_copysign_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let top = (*meta).stack_slot as usize;
    let rhs = read_f32_bits(frame_read(fp, top));
    let lhs = read_f32_bits(frame_read(fp, top - 1));
    write_f32_bits(fp, top - 1, lhs.copysign(rhs));
    meta_next(_ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn f64_copysign_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let top = (*meta).stack_slot as usize;
    let rhs = read_f64_bits(frame_read(fp, top));
    let lhs = read_f64_bits(frame_read(fp, top - 1));
    write_f64_bits(fp, top - 1, lhs.copysign(rhs));
    meta_next(_ctx, fp, meta)
}

#[inline]
fn trap_integer_overflow(ctx: *mut Context, fp: *mut u64) -> HelperResult {
    trap_with(ctx, fp, WasmError::trap("integer overflow".into()))
}

#[inline]
fn trap_invalid_conversion(ctx: *mut Context, fp: *mut u64) -> HelperResult {
    trap_with(ctx, fp, WasmError::trap("invalid conversion to integer".into()))
}

#[no_mangle]
pub unsafe extern "C" fn i32_trunc_f32_s_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f32_bits(frame_read(fp, slot)) as f64;
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -2147483649.0 || f >= 2147483648.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, (f as i32 as u32) as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i32_trunc_f32_u_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f32_bits(frame_read(fp, slot)) as f64;
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -1.0 || f >= 4294967296.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, (f as u32) as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i32_trunc_f64_s_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f64_bits(frame_read(fp, slot));
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -2147483649.0 || f >= 2147483648.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, (f as i32 as u32) as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i32_trunc_f64_u_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f64_bits(frame_read(fp, slot));
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -1.0 || f >= 4294967296.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, (f as u32) as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i64_trunc_f32_s_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f32_bits(frame_read(fp, slot)) as f64;
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -9223372036854777856.0 || f >= 9223372036854775808.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, f as i64 as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i64_trunc_f32_u_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f32_bits(frame_read(fp, slot)) as f64;
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -1.0 || f >= 18446744073709551616.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, f as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i64_trunc_f64_s_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f64_bits(frame_read(fp, slot));
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -9223372036854777856.0 || f >= 9223372036854775808.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, f as i64 as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn i64_trunc_f64_u_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let f = read_f64_bits(frame_read(fp, slot));
    if f.is_nan() {
        return trap_invalid_conversion(ctx, fp);
    }
    if f.is_infinite() {
        return trap_integer_overflow(ctx, fp);
    }
    if f <= -1.0 || f >= 18446744073709551616.0 {
        return trap_integer_overflow(ctx, fp);
    }
    frame_write(fp, slot, f as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_get_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let slot = (*meta).stack_slot as usize;
    let elem_index = frame_read(fp, slot) as usize;
    let store = store_mut(ctx);
    let table = store.table(table_idx);
    if elem_index >= table.elements.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }
    frame_write(fp, slot, usize::from(table.elements[elem_index]) as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_set_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let top = (*meta).stack_slot as usize;
    let value = VmRefHandle::new(frame_read(fp, top) as usize);
    let elem_index = frame_read(fp, top - 1) as usize;
    let store = store_mut(ctx);
    let table = store.table_mut(table_idx);
    if elem_index >= table.elements.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }
    table.elements[elem_index] = value;
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_size_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let slot = (*meta).stack_slot as usize;
    let store = store_mut(ctx);
    frame_write(fp, slot, store.table(table_idx).size() as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_grow_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let top = (*meta).stack_slot as usize;
    let delta = frame_read(fp, top) as usize;
    let value = VmRefHandle::new(frame_read(fp, top - 1) as usize);
    let store = store_mut(ctx);
    let table = store.table_mut(table_idx);
    let new_size = table.elements.len().checked_add(delta).unwrap_or(usize::MAX);
    if new_size > table.limits.get_max() {
        frame_write(fp, top - 1, u32::MAX as u64);
        return meta_next(ctx, fp, meta);
    }
    let old_size = table.elements.len();
    table.elements.resize_with(old_size + delta, || value);
    frame_write(fp, top - 1, old_size as u64);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_fill_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let value = VmRefHandle::new(frame_read(fp, top - 1) as usize);
    let start = frame_read(fp, top - 2) as usize;
    let store = store_mut(ctx);
    let table = store.table_mut(table_idx);
    if start.saturating_add(size) > table.elements.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }
    table.elements[start..start + size].fill(value);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_init_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let table_idx = (*meta).data0 as usize;
    let elem_idx = (*meta).data1 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let src = frame_read(fp, top - 1) as usize;
    let dst = frame_read(fp, top - 2) as usize;
    let store = store_mut(ctx);
    let module = store.module_mut();

    if elem_idx >= module.elements.len() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }

    let elem_inst = &module.elements[elem_idx];
    if size == 0 {
        if src > elem_inst.refs.len() || dst > module.tables[table_idx].elements.len() {
            return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
        }
        return meta_next(ctx, fp, meta);
    }
    if src.saturating_add(size) > elem_inst.refs.len()
        || dst.saturating_add(size) > module.tables[table_idx].elements.len()
    {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }
    if elem_inst.is_dropped() {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }

    for i in 0..size {
        module.tables[table_idx].elements[dst + i] = module.elements[elem_idx].refs[src + i];
    }
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn elem_drop_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let elem_idx = (*meta).data0 as usize;
    let store = store_mut(ctx);
    let module = store.module_mut();
    if elem_idx < module.elements.len() {
        module.elements[elem_idx].drop_segment();
    }
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn table_copy_helper(
    ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let dstidx = (*meta).data0 as usize;
    let srcidx = (*meta).data1 as usize;
    let top = (*meta).stack_slot as usize;
    let size = frame_read(fp, top) as usize;
    let src = frame_read(fp, top - 1) as usize;
    let dst = frame_read(fp, top - 2) as usize;
    let store = store_mut(ctx);

    if dstidx == srcidx {
        let table = store.table_mut(dstidx);
        if src.saturating_add(size) > table.elements.len()
            || dst.saturating_add(size) > table.elements.len()
        {
            return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
        }
        table.elements.copy_within(src..src + size, dst);
        return meta_next(ctx, fp, meta);
    }

    let module = store.module_mut();
    let (src_table, dst_table) = if srcidx < dstidx {
        let (left, right) = module.tables.split_at_mut(dstidx);
        (&left[srcidx], &mut right[0])
    } else {
        let (left, right) = module.tables.split_at_mut(srcidx);
        (&right[0], &mut left[dstidx])
    };
    if src.saturating_add(size) > src_table.elements.len()
        || dst.saturating_add(size) > dst_table.elements.len()
    {
        return trap_with(ctx, fp, WasmError::trap("out of bounds table access".into()));
    }
    dst_table.elements[dst..dst + size].copy_from_slice(&src_table.elements[src..src + size]);
    meta_next(ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn ref_null_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    frame_write(fp, (*meta).stack_slot as usize, usize::MAX as u64);
    meta_next(_ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn ref_is_null_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    let slot = (*meta).stack_slot as usize;
    let value = frame_read(fp, slot) as usize;
    frame_write(fp, slot, if value == usize::MAX { 1 } else { 0 });
    meta_next(_ctx, fp, meta)
}

#[no_mangle]
pub unsafe extern "C" fn ref_func_helper(
    _ctx: *mut Context,
    fp: *mut u64,
    meta: *const HelperMetadata,
) -> HelperResult {
    frame_write(fp, (*meta).stack_slot as usize, (*meta).data0);
    meta_next(_ctx, fp, meta)
}
