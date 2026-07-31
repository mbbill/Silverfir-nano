//! Target runtime ABI layout helpers for MachineIR lowering.
//!
//! The Rust host structs in [`context`] use the host pointer width. MachineIR
//! lowering, however, must target the selected backend ABI rather than the
//! host compiler ABI. These helpers describe the machine-visible subset of the
//! runtime layout in terms of the backend GP budget unit size.

use core::mem::{offset_of, size_of};

use super::context::NativeContext;
use super::dispatch_view::{
    CallDispatchView, NativeFixedCallTableEntry, NativeFixedCallTableView, NativeLocalCallInfo32,
    NativeLocalCallInfo64,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PointerLenAbiLayout {
    pub base_offset: u32,
    pub len_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FunctionViewAbiLayout {
    pub kind_offset: u32,
    pub type_canon_offset: u32,
    pub local_target_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalCallInfoAbiLayout {
    pub entry_offset: u32,
    pub total_frame_bytes_offset: u32,
    pub frame_prefix_slots_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedCallTableViewAbiLayout {
    pub entry_base_offset: u32,
    pub len_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedCallTableEntryAbiLayout {
    pub type_canon_offset: u32,
    pub local_target_offset: u32,
    pub entry_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeContextAbiLayout {
    pub stack_end_offset: u32,
    pub mem0_base_offset: u32,
    pub mem0_size_offset: u32,
    pub memory_views_base_offset: u32,
    pub memory_views_len_offset: u32,
    pub table_views_base_offset: u32,
    pub table_views_len_offset: u32,
    pub function_views_base_offset: u32,
    pub function_views_len_offset: u32,
    pub local_call_infos_base_offset: u32,
    pub local_call_infos_len_offset: u32,
    pub fixed_call_table_views_base_offset: u32,
    pub fixed_call_table_views_len_offset: u32,
    pub type_canon_base_offset: u32,
    pub type_canon_len_offset: u32,
    pub globals_len_offset: u32,
    pub store_offset: u32,
    pub current_module_offset: u32,
    pub self_abs_base_offset: u32,
    pub self_local_by_abs_base_offset: u32,
    pub self_local_by_abs_len_offset: u32,
    /// Offset of the trailing inline raw-ptr array (`[*mut u64; globals_len]`)
    /// within the runtime context. JIT-emitted `global.get`/`global.set` uses
    /// this as the base for a single indexed load into the tail, followed by
    /// a dereference of the loaded pointer.
    pub globals_ptrs_inline_offset: u32,
    /// Size of the fixed header. The actual per-instance allocation is
    /// `size + globals_len * ptr_stride`; this field is the offset of the
    /// inline ptr array as well (it starts immediately after the header).
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeRuntimeAbiLayout {
    pub gp_unit_bytes: u8,
    pub pointer_len_view: PointerLenAbiLayout,
    pub function_view: FunctionViewAbiLayout,
    pub local_call_info: LocalCallInfoAbiLayout,
    pub fixed_call_table_view: FixedCallTableViewAbiLayout,
    pub fixed_call_table_entry: FixedCallTableEntryAbiLayout,
    pub context: NativeContextAbiLayout,
    pub ref_value_stride: u32,
}

#[inline]
const fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

#[inline]
pub(crate) const fn pointer_len_abi_layout(gp_unit_bytes: u8) -> PointerLenAbiLayout {
    let ptr = gp_unit_bytes as u32;
    PointerLenAbiLayout {
        base_offset: 0,
        len_offset: ptr,
        stride: ptr * 2,
    }
}

#[inline]
pub(crate) const fn function_view_abi_layout() -> FunctionViewAbiLayout {
    FunctionViewAbiLayout {
        kind_offset: offset_of!(CallDispatchView, kind) as u32,
        type_canon_offset: offset_of!(CallDispatchView, type_canon) as u32,
        local_target_offset: offset_of!(CallDispatchView, local_target) as u32,
        stride: size_of::<CallDispatchView>() as u32,
    }
}

#[inline]
pub(crate) const fn local_call_info_abi_layout(gp_unit_bytes: u8) -> LocalCallInfoAbiLayout {
    match gp_unit_bytes {
        4 => LocalCallInfoAbiLayout {
            entry_offset: offset_of!(NativeLocalCallInfo32, entry) as u32,
            total_frame_bytes_offset: offset_of!(NativeLocalCallInfo32, total_frame_bytes) as u32,
            frame_prefix_slots_offset: offset_of!(NativeLocalCallInfo32, frame_prefix_slots) as u32,
            stride: size_of::<NativeLocalCallInfo32>() as u32,
        },
        8 => LocalCallInfoAbiLayout {
            entry_offset: offset_of!(NativeLocalCallInfo64, entry) as u32,
            total_frame_bytes_offset: offset_of!(NativeLocalCallInfo64, total_frame_bytes) as u32,
            frame_prefix_slots_offset: offset_of!(NativeLocalCallInfo64, frame_prefix_slots) as u32,
            stride: size_of::<NativeLocalCallInfo64>() as u32,
        },
        _ => panic!("unsupported GP unit size"),
    }
}

#[inline]
pub(crate) const fn fixed_call_table_view_abi_layout() -> FixedCallTableViewAbiLayout {
    FixedCallTableViewAbiLayout {
        entry_base_offset: offset_of!(NativeFixedCallTableView, entry_base) as u32,
        len_offset: offset_of!(NativeFixedCallTableView, len) as u32,
        stride: size_of::<NativeFixedCallTableView>() as u32,
    }
}

#[inline]
pub(crate) const fn fixed_call_table_entry_abi_layout() -> FixedCallTableEntryAbiLayout {
    FixedCallTableEntryAbiLayout {
        type_canon_offset: offset_of!(NativeFixedCallTableEntry, type_canon) as u32,
        local_target_offset: offset_of!(NativeFixedCallTableEntry, local_target) as u32,
        entry_offset: offset_of!(NativeFixedCallTableEntry, entry) as u32,
        stride: size_of::<NativeFixedCallTableEntry>() as u32,
    }
}

#[inline]
pub(crate) const fn native_runtime_abi_layout(gp_unit_bytes: u8) -> NativeRuntimeAbiLayout {
    match gp_unit_bytes {
        4 | 8 => {}
        _ => panic!("unsupported GP unit size"),
    }

    let ptr = gp_unit_bytes as u32;
    let pointer_len_view = pointer_len_abi_layout(gp_unit_bytes);
    let function_view = function_view_abi_layout();
    let local_call_info = local_call_info_abi_layout(gp_unit_bytes);
    let fixed_call_table_view = fixed_call_table_view_abi_layout();
    let fixed_call_table_entry = fixed_call_table_entry_abi_layout();

    let stack_end_offset = 0;
    let mem0_base_offset = align_up(stack_end_offset + ptr, ptr);
    let mem0_size_offset = align_up(mem0_base_offset + ptr, 8);
    let memory_views_base_offset = mem0_size_offset + 8;
    let memory_views_len_offset = memory_views_base_offset + ptr;
    let table_views_base_offset = memory_views_len_offset + ptr;
    let table_views_len_offset = table_views_base_offset + ptr;
    let function_views_base_offset = table_views_len_offset + ptr;
    let function_views_len_offset = function_views_base_offset + ptr;
    let local_call_infos_base_offset = function_views_len_offset + ptr;
    let local_call_infos_len_offset = local_call_infos_base_offset + ptr;
    let fixed_call_table_views_base_offset = local_call_infos_len_offset + ptr;
    let fixed_call_table_views_len_offset = fixed_call_table_views_base_offset + ptr;
    let type_canon_base_offset = fixed_call_table_views_len_offset + ptr;
    let type_canon_len_offset = type_canon_base_offset + ptr;
    let globals_len_offset = type_canon_len_offset + ptr;
    let store_offset = globals_len_offset + ptr;
    let current_module_offset = store_offset + ptr;
    let self_abs_base_offset = current_module_offset + ptr;
    let self_local_by_abs_base_offset = self_abs_base_offset + ptr;
    let self_local_by_abs_len_offset = self_local_by_abs_base_offset + ptr;
    // The inline raw-ptr tail is pinned to the host struct's tail offset
    // (equivalent to `offset_of!(NativeContext, globals_ptrs_tail)` since the
    // marker is the last field of the `#[repr(C)]` struct). This keeps
    // JIT-emitted `[runtime_base + globals_ptrs_inline_offset + idx*ptr]`
    // reads landing on the real raw_ptr slots (target = host).
    let globals_ptrs_inline_offset = size_of::<NativeContext>() as u32;
    let size = globals_ptrs_inline_offset;

    NativeRuntimeAbiLayout {
        gp_unit_bytes,
        pointer_len_view,
        function_view,
        local_call_info,
        fixed_call_table_view,
        fixed_call_table_entry,
        context: NativeContextAbiLayout {
            stack_end_offset,
            mem0_base_offset,
            mem0_size_offset,
            memory_views_base_offset,
            memory_views_len_offset,
            table_views_base_offset,
            table_views_len_offset,
            function_views_base_offset,
            function_views_len_offset,
            local_call_infos_base_offset,
            local_call_infos_len_offset,
            fixed_call_table_views_base_offset,
            fixed_call_table_views_len_offset,
            type_canon_base_offset,
            type_canon_len_offset,
            globals_len_offset,
            store_offset,
            current_module_offset,
            self_abs_base_offset,
            self_local_by_abs_base_offset,
            self_local_by_abs_len_offset,
            globals_ptrs_inline_offset,
            size,
        },
        ref_value_stride: ptr,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        fixed_call_table_entry_abi_layout, fixed_call_table_view_abi_layout,
        function_view_abi_layout, local_call_info_abi_layout, native_runtime_abi_layout,
        pointer_len_abi_layout,
    };
    use crate::vm::jit::runtime::context::{
        ctx_offset, fixed_call_table_view_offset, function_view_offset, memory_view_offset,
        table_view_offset, NativeContext,
    };
    use crate::vm::jit::runtime::dispatch_view::{
        NativeFixedCallTableEntry, NativeFixedCallTableView, NativeLocalCallInfo32,
        NativeLocalCallInfo64,
    };

    #[test]
    fn host_width_runtime_layout_matches_host_struct_offsets() {
        let layout = native_runtime_abi_layout(core::mem::size_of::<usize>() as u8);
        assert_eq!(layout.context.stack_end_offset, ctx_offset::STACK_END);
        assert_eq!(layout.context.mem0_base_offset, ctx_offset::MEM0_BASE);
        assert_eq!(layout.context.mem0_size_offset, ctx_offset::MEM0_SIZE);
        assert_eq!(
            layout.context.globals_ptrs_inline_offset as usize,
            core::mem::size_of::<NativeContext>()
        );
        assert_eq!(
            layout.context.memory_views_base_offset,
            ctx_offset::MEMORY_VIEWS_BASE
        );
        assert_eq!(
            layout.context.memory_views_len_offset,
            ctx_offset::MEMORY_VIEWS_LEN
        );
        assert_eq!(
            layout.context.table_views_base_offset,
            ctx_offset::TABLE_VIEWS_BASE
        );
        assert_eq!(
            layout.context.table_views_len_offset,
            ctx_offset::TABLE_VIEWS_LEN
        );
        assert_eq!(
            layout.context.function_views_base_offset,
            ctx_offset::FUNCTION_VIEWS_BASE
        );
        assert_eq!(
            layout.context.function_views_len_offset,
            ctx_offset::FUNCTION_VIEWS_LEN
        );
        assert_eq!(
            layout.context.local_call_infos_base_offset,
            ctx_offset::LOCAL_CALL_INFOS_BASE
        );
        assert_eq!(
            layout.context.local_call_infos_len_offset,
            ctx_offset::LOCAL_CALL_INFOS_LEN
        );
        assert_eq!(
            layout.context.fixed_call_table_views_base_offset,
            ctx_offset::FIXED_CALL_TABLE_VIEWS_BASE
        );
        assert_eq!(
            layout.context.fixed_call_table_views_len_offset,
            ctx_offset::FIXED_CALL_TABLE_VIEWS_LEN
        );
        assert_eq!(
            layout.context.type_canon_base_offset,
            ctx_offset::TYPE_CANON_BASE
        );
        assert_eq!(
            layout.context.type_canon_len_offset,
            ctx_offset::TYPE_CANON_LEN
        );
        assert_eq!(
            layout.context.self_abs_base_offset,
            ctx_offset::SELF_ABS_BASE
        );
        assert_eq!(
            layout.context.self_local_by_abs_base_offset,
            ctx_offset::SELF_LOCAL_BY_ABS_BASE
        );
        assert_eq!(
            layout.context.self_local_by_abs_len_offset,
            ctx_offset::SELF_LOCAL_BY_ABS_LEN
        );
    }

    #[test]
    fn pointer_len_abi_layout_matches_host_view_offsets() {
        let layout = pointer_len_abi_layout(core::mem::size_of::<usize>() as u8);
        assert_eq!(layout.base_offset, memory_view_offset::BASE);
        assert_eq!(layout.len_offset, memory_view_offset::LEN);
        assert_eq!(layout.base_offset, table_view_offset::ELEMENTS_BASE);
        assert_eq!(layout.len_offset, table_view_offset::ELEMENTS_LEN);
    }

    #[test]
    fn function_view_abi_layout_matches_host_offsets() {
        let layout = function_view_abi_layout();
        assert_eq!(layout.kind_offset, function_view_offset::KIND);
        assert_eq!(layout.type_canon_offset, function_view_offset::TYPE_CANON);
        assert_eq!(
            layout.local_target_offset,
            function_view_offset::LOCAL_TARGET
        );
    }

    #[test]
    fn fixed_call_table_view_abi_layout_matches_host_offsets() {
        let layout = fixed_call_table_view_abi_layout();
        assert_eq!(
            layout.entry_base_offset,
            fixed_call_table_view_offset::ENTRY_BASE
        );
        assert_eq!(layout.len_offset, fixed_call_table_view_offset::LEN);
        assert_eq!(
            layout.stride,
            core::mem::size_of::<NativeFixedCallTableView>() as u32
        );
    }

    #[test]
    fn fixed_call_table_entry_abi_layout_matches_host_offsets() {
        let layout = fixed_call_table_entry_abi_layout();
        assert_eq!(
            layout.type_canon_offset,
            core::mem::offset_of!(NativeFixedCallTableEntry, type_canon) as u32
        );
        assert_eq!(
            layout.local_target_offset,
            core::mem::offset_of!(NativeFixedCallTableEntry, local_target) as u32
        );
        assert_eq!(
            layout.entry_offset,
            core::mem::offset_of!(NativeFixedCallTableEntry, entry) as u32
        );
        assert_eq!(
            layout.stride,
            core::mem::size_of::<NativeFixedCallTableEntry>() as u32
        );
    }

    #[test]
    fn thirty_two_bit_local_call_info_layout_matches_record() {
        let layout = local_call_info_abi_layout(4);
        assert_eq!(
            layout.entry_offset,
            core::mem::offset_of!(NativeLocalCallInfo32, entry) as u32
        );
        assert_eq!(
            layout.total_frame_bytes_offset,
            core::mem::offset_of!(NativeLocalCallInfo32, total_frame_bytes) as u32
        );
        assert_eq!(
            layout.frame_prefix_slots_offset,
            core::mem::offset_of!(NativeLocalCallInfo32, frame_prefix_slots) as u32
        );
        assert_eq!(
            layout.stride,
            core::mem::size_of::<NativeLocalCallInfo32>() as u32
        );
    }

    #[test]
    fn sixty_four_bit_local_call_info_layout_matches_record() {
        let layout = local_call_info_abi_layout(8);
        assert_eq!(
            layout.entry_offset,
            core::mem::offset_of!(NativeLocalCallInfo64, entry) as u32
        );
        assert_eq!(
            layout.total_frame_bytes_offset,
            core::mem::offset_of!(NativeLocalCallInfo64, total_frame_bytes) as u32
        );
        assert_eq!(
            layout.frame_prefix_slots_offset,
            core::mem::offset_of!(NativeLocalCallInfo64, frame_prefix_slots) as u32
        );
        assert_eq!(
            layout.stride,
            core::mem::size_of::<NativeLocalCallInfo64>() as u32
        );
    }

    #[test]
    fn thirty_two_bit_runtime_layout_uses_pointer_sized_views() {
        let layout = native_runtime_abi_layout(4);
        assert_eq!(layout.pointer_len_view.stride, 8);
        assert_eq!(layout.context.mem0_base_offset, 4);
        assert_eq!(layout.context.mem0_size_offset, 8);
        // With globals_view collapsed into an inline tail, the fixed
        // ABI-visible prefix begins memory-views directly after mem0_size.
        assert_eq!(layout.context.memory_views_base_offset, 16);
        assert_eq!(layout.context.memory_views_len_offset, 20);
        assert_eq!(layout.context.table_views_base_offset, 24);
        assert_eq!(layout.context.function_views_base_offset, 32);
        assert_eq!(layout.context.local_call_infos_base_offset, 40);
        assert_eq!(layout.context.local_call_infos_len_offset, 44);
        assert_eq!(layout.context.fixed_call_table_views_base_offset, 48);
        assert_eq!(layout.context.fixed_call_table_views_len_offset, 52);
        assert_eq!(layout.context.type_canon_base_offset, 56);
        assert_eq!(layout.context.globals_len_offset, 64);
        assert_eq!(layout.ref_value_stride, 4);
    }
}
