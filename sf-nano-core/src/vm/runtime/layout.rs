//! Target runtime ABI layout helpers shared by lowering and the emulator.
//!
//! The Rust host structs in [`context`] use the host pointer width. MachineIR
//! lowering, however, must target the selected backend ABI rather than the
//! host compiler ABI. These helpers describe the machine-visible subset of the
//! runtime layout in terms of the backend GP budget unit size.

use core::mem::{offset_of, size_of};

use super::dispatch_view::{CallDispatchView, NativeLocalCallInfo32, NativeLocalCallInfo64};

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
    pub call_scratch_base_slot_offset: u32,
    pub stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeContextAbiLayout {
    pub stack_end_offset: u32,
    pub mem0_base_offset: u32,
    pub mem0_size_offset: u32,
    pub globals_view_offset: u32,
    pub globals_view_base_offset: u32,
    pub globals_view_len_offset: u32,
    pub memory_views_base_offset: u32,
    pub memory_views_len_offset: u32,
    pub table_views_base_offset: u32,
    pub table_views_len_offset: u32,
    pub function_views_base_offset: u32,
    pub function_views_len_offset: u32,
    pub local_call_infos_base_offset: u32,
    pub local_call_infos_len_offset: u32,
    pub type_canon_base_offset: u32,
    pub type_canon_len_offset: u32,
    pub store_offset: u32,
    pub current_module_offset: u32,
    pub size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeRuntimeAbiLayout {
    pub gp_unit_bytes: u8,
    pub pointer_len_view: PointerLenAbiLayout,
    pub function_view: FunctionViewAbiLayout,
    pub local_call_info: LocalCallInfoAbiLayout,
    pub context: NativeContextAbiLayout,
    pub ref_handle_stride: u32,
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
            call_scratch_base_slot_offset: offset_of!(NativeLocalCallInfo32, call_scratch_base_slot)
                as u32,
            stride: size_of::<NativeLocalCallInfo32>() as u32,
        },
        8 => LocalCallInfoAbiLayout {
            entry_offset: offset_of!(NativeLocalCallInfo64, entry) as u32,
            total_frame_bytes_offset: offset_of!(NativeLocalCallInfo64, total_frame_bytes) as u32,
            frame_prefix_slots_offset: offset_of!(NativeLocalCallInfo64, frame_prefix_slots) as u32,
            call_scratch_base_slot_offset: offset_of!(NativeLocalCallInfo64, call_scratch_base_slot)
                as u32,
            stride: size_of::<NativeLocalCallInfo64>() as u32,
        },
        _ => panic!("unsupported GP unit size"),
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

    let stack_end_offset = 0;
    let mem0_base_offset = align_up(stack_end_offset + ptr, ptr);
    let mem0_size_offset = align_up(mem0_base_offset + ptr, 8);
    let globals_view_offset = mem0_size_offset + 8;
    let globals_view_base_offset = globals_view_offset + pointer_len_view.base_offset;
    let globals_view_len_offset = globals_view_offset + pointer_len_view.len_offset;
    let memory_views_base_offset = globals_view_offset + pointer_len_view.stride;
    let memory_views_len_offset = memory_views_base_offset + ptr;
    let table_views_base_offset = memory_views_len_offset + ptr;
    let table_views_len_offset = table_views_base_offset + ptr;
    let function_views_base_offset = table_views_len_offset + ptr;
    let function_views_len_offset = function_views_base_offset + ptr;
    let local_call_infos_base_offset = function_views_len_offset + ptr;
    let local_call_infos_len_offset = local_call_infos_base_offset + ptr;
    let type_canon_base_offset = local_call_infos_len_offset + ptr;
    let type_canon_len_offset = type_canon_base_offset + ptr;
    let store_offset = type_canon_len_offset + ptr;
    let current_module_offset = store_offset + ptr;
    let size = current_module_offset + ptr;

    NativeRuntimeAbiLayout {
        gp_unit_bytes,
        pointer_len_view,
        function_view,
        local_call_info,
        context: NativeContextAbiLayout {
            stack_end_offset,
            mem0_base_offset,
            mem0_size_offset,
            globals_view_offset,
            globals_view_base_offset,
            globals_view_len_offset,
            memory_views_base_offset,
            memory_views_len_offset,
            table_views_base_offset,
            table_views_len_offset,
            function_views_base_offset,
            function_views_len_offset,
            local_call_infos_base_offset,
            local_call_infos_len_offset,
            type_canon_base_offset,
            type_canon_len_offset,
            store_offset,
            current_module_offset,
            size,
        },
        ref_handle_stride: ptr,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        function_view_abi_layout, local_call_info_abi_layout, native_runtime_abi_layout,
        pointer_len_abi_layout,
    };
    use crate::vm::runtime::context::{
        ctx_offset, function_view_offset, globals_view_offset, memory_view_offset,
        table_view_offset,
    };
    use crate::vm::runtime::dispatch_view::{NativeLocalCallInfo32, NativeLocalCallInfo64};

    #[test]
    fn host_width_runtime_layout_matches_host_struct_offsets() {
        let layout = native_runtime_abi_layout(core::mem::size_of::<usize>() as u8);
        assert_eq!(layout.context.stack_end_offset, ctx_offset::STACK_END);
        assert_eq!(layout.context.mem0_base_offset, ctx_offset::MEM0_BASE);
        assert_eq!(layout.context.mem0_size_offset, ctx_offset::MEM0_SIZE);
        assert_eq!(
            layout.context.globals_view_base_offset,
            ctx_offset::GLOBALS_VIEW + globals_view_offset::BASE
        );
        assert_eq!(
            layout.context.globals_view_len_offset,
            ctx_offset::GLOBALS_VIEW + globals_view_offset::LEN
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
            layout.context.type_canon_base_offset,
            ctx_offset::TYPE_CANON_BASE
        );
        assert_eq!(
            layout.context.type_canon_len_offset,
            ctx_offset::TYPE_CANON_LEN
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
            layout.call_scratch_base_slot_offset,
            core::mem::offset_of!(NativeLocalCallInfo32, call_scratch_base_slot) as u32
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
            layout.call_scratch_base_slot_offset,
            core::mem::offset_of!(NativeLocalCallInfo64, call_scratch_base_slot) as u32
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
        assert_eq!(layout.context.memory_views_base_offset, 24);
        assert_eq!(layout.context.table_views_base_offset, 32);
        assert_eq!(layout.context.function_views_base_offset, 40);
        assert_eq!(layout.context.type_canon_base_offset, 48);
        assert_eq!(layout.ref_handle_stride, 4);
    }
}
