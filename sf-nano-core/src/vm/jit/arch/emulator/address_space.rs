use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::MachineMemWidth,
        runtime::{
            code::CompiledNativeModule,
            context::NativeContext,
            layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
        },
        value_encoding::{as_ref, machine_raw_to_ref, ref_to_machine_raw},
    },
};

const STACK_BASE_32: u64 = 0x0100_0000;
const CTX_BASE_32: u64 = 0x1000_0000;
const MEMORY_VIEWS_BASE_32: u64 = 0x1100_0000;
const TABLE_VIEWS_BASE_32: u64 = 0x1200_0000;
const FUNCTION_VIEWS_BASE_32: u64 = 0x1300_0000;
const LOCAL_CALL_INFOS_BASE_32: u64 = 0x1400_0000;
const GLOBALS_RAW_BASE_32: u64 = 0x1500_0000;
const TYPE_CANON_BASE_32: u64 = 0x1600_0000;
const MEMORY_BASE_32: u64 = 0x4000_0000;
const MEMORY_WINDOW_32: u64 = 0x0400_0000;
const TABLE_ELEMENTS_BASE_32: u64 = 0xC000_0000;
const TABLE_ELEMENTS_WINDOW_32: u64 = 0x0400_0000;
const MEMORY_VIEWS_WINDOW_32: u64 = TABLE_VIEWS_BASE_32 - MEMORY_VIEWS_BASE_32;
const TABLE_VIEWS_WINDOW_32: u64 = FUNCTION_VIEWS_BASE_32 - TABLE_VIEWS_BASE_32;
const FUNCTION_VIEWS_WINDOW_32: u64 = LOCAL_CALL_INFOS_BASE_32 - FUNCTION_VIEWS_BASE_32;
const LOCAL_CALL_INFOS_WINDOW_32: u64 = GLOBALS_RAW_BASE_32 - LOCAL_CALL_INFOS_BASE_32;
/// Per-global storage in the synthetic globals-raw window. Values are u64
/// regardless of target pointer width; 32-bit targets access the low/high
/// halves through this 8-byte stride.
const GLOBALS_RAW_STRIDE_32: u64 = 8;
const GLOBALS_RAW_WINDOW_32: u64 = TYPE_CANON_BASE_32 - GLOBALS_RAW_BASE_32;
const TYPE_CANON_WINDOW_32: u64 = MEMORY_BASE_32 - TYPE_CANON_BASE_32;
const MAX_MEMORY_COUNT_32: usize =
    ((TABLE_ELEMENTS_BASE_32 - MEMORY_BASE_32) / MEMORY_WINDOW_32) as usize;
const MAX_TABLE_COUNT_32: usize =
    (((1u64 << 32) - TABLE_ELEMENTS_BASE_32) / TABLE_ELEMENTS_WINDOW_32) as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EmulatorAddressSpace {
    Host,
    Target32(Target32AddressSpace),
}

impl EmulatorAddressSpace {
    pub(super) fn new(
        compiled: &CompiledNativeModule,
        stack_base: *mut u64,
        stack_end: *mut u64,
    ) -> Self {
        if compiled.backend().gp_unit_bytes == 4 && core::mem::size_of::<usize>() > 4 {
            Self::Target32(Target32AddressSpace::new(
                stack_base.cast::<u8>(),
                stack_end.cast::<u8>(),
            ))
        } else {
            Self::Host
        }
    }

    #[inline]
    pub(super) fn runtime_base_value(self, ctx: &NativeContext) -> u64 {
        match self {
            Self::Host => ctx as *const NativeContext as u64,
            Self::Target32(space) => space.ctx_base,
        }
    }

    #[inline]
    pub(super) fn frame_base_value(self, fp: *mut u64) -> Result<u64, WasmError> {
        match self {
            Self::Host => Ok(fp as u64),
            Self::Target32(space) => space.stack_addr(fp.cast::<u8>()),
        }
    }

    #[inline]
    pub(super) fn mem0_base_value(self, ctx: &NativeContext) -> u64 {
        match self {
            Self::Host => ctx.mem0_base as u64,
            Self::Target32(space) => {
                if ctx.mem0_size == 0 {
                    0
                } else {
                    space.memory_base(0)
                }
            }
        }
    }

    #[inline]
    pub(super) fn host_stack_ptr(self, addr: u64) -> Result<*mut u64, WasmError> {
        match self {
            Self::Host => Ok(addr as *mut u64),
            Self::Target32(space) => Ok(space.host_stack_ptr(addr)?.cast::<u64>()),
        }
    }

    #[inline]
    pub(super) fn validate_runtime_shape(self, ctx: &NativeContext) -> Result<(), WasmError> {
        match self {
            Self::Host => Ok(()),
            Self::Target32(space) => space.validate_runtime_shape(ctx),
        }
    }

    #[inline]
    pub(super) fn load(
        self,
        ctx: &NativeContext,
        addr: u64,
        width: MachineMemWidth,
    ) -> Option<Result<u64, WasmError>> {
        match self {
            Self::Host => None,
            Self::Target32(space) => space.load(ctx, addr, width),
        }
    }

    #[inline]
    pub(super) fn store(
        self,
        ctx: &NativeContext,
        addr: u64,
        width: MachineMemWidth,
        value: u64,
    ) -> Option<Result<(), WasmError>> {
        match self {
            Self::Host => None,
            Self::Target32(space) => space.store(ctx, addr, width, value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Target32AddressSpace {
    stack_base: u64,
    stack_len_bytes: u32,
    host_stack_base: *mut u8,
    ctx_base: u64,
    memory_views_base: u64,
    table_views_base: u64,
    function_views_base: u64,
    local_call_infos_base: u64,
    /// Synthetic base for the u64 raw-value window. Entries are 8 bytes apart;
    /// 32-bit targets access low/high halves through `GLOBALS_RAW_STRIDE_32`.
    globals_raw_base: u64,
    type_canon_base: u64,
    layout: NativeRuntimeAbiLayout,
}

impl Target32AddressSpace {
    fn new(host_stack_base: *mut u8, host_stack_end: *mut u8) -> Self {
        let stack_len_bytes = host_stack_end.addr().saturating_sub(host_stack_base.addr());
        Self {
            stack_base: STACK_BASE_32,
            stack_len_bytes: u32::try_from(stack_len_bytes).unwrap_or(u32::MAX),
            host_stack_base,
            ctx_base: CTX_BASE_32,
            memory_views_base: MEMORY_VIEWS_BASE_32,
            table_views_base: TABLE_VIEWS_BASE_32,
            function_views_base: FUNCTION_VIEWS_BASE_32,
            local_call_infos_base: LOCAL_CALL_INFOS_BASE_32,
            globals_raw_base: GLOBALS_RAW_BASE_32,
            type_canon_base: TYPE_CANON_BASE_32,
            layout: native_runtime_abi_layout(4),
        }
    }

    /// Total size of the synthetic runtime-context window, including the
    /// inline globals-ptr tail. JIT-emitted addresses inside the tail must
    /// fall within this window so [`load`] routes them to [`load_context`].
    #[inline]
    fn ctx_window_size(self, globals_len: usize) -> u32 {
        let tail_bytes = (globals_len as u32).saturating_mul(self.layout.gp_unit_bytes as u32);
        self.layout.context.size.saturating_add(tail_bytes)
    }

    #[inline]
    fn stack_end(self) -> u64 {
        self.stack_base + u64::from(self.stack_len_bytes)
    }

    fn validate_runtime_shape(self, ctx: &NativeContext) -> Result<(), WasmError> {
        ensure_window_fits(
            checked_region_bytes(
                ctx.memory_views_len,
                self.layout.pointer_len_view.stride as usize,
            )?,
            MEMORY_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            checked_region_bytes(
                ctx.table_views_len,
                self.layout.pointer_len_view.stride as usize,
            )?,
            TABLE_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            checked_region_bytes(
                ctx.function_views_len,
                self.layout.function_view.stride as usize,
            )?,
            FUNCTION_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            checked_region_bytes(
                ctx.local_call_infos_len,
                self.layout.local_call_info.stride as usize,
            )?,
            LOCAL_CALL_INFOS_WINDOW_32,
        )?;
        ensure_window_fits(
            checked_region_bytes(ctx.globals_len, GLOBALS_RAW_STRIDE_32 as usize)?,
            GLOBALS_RAW_WINDOW_32,
        )?;
        ensure_window_fits(
            checked_region_bytes(ctx.type_canon_len, core::mem::size_of::<u32>())?,
            TYPE_CANON_WINDOW_32,
        )?;

        if ctx.memory_views_len > MAX_MEMORY_COUNT_32 {
            return Err(WasmError::internal(
                "emu32 synthetic address space supports at most 32 memories",
            ));
        }
        for mem_index in 0..ctx.memory_views_len {
            let view = unsafe { *ctx.memory_views_base.add(mem_index) };
            ensure_window_fits(view.len, MEMORY_WINDOW_32)?;
        }

        if ctx.table_views_len > MAX_TABLE_COUNT_32 {
            return Err(WasmError::internal(
                "emu32 synthetic address space supports at most 16 tables",
            ));
        }
        for table_index in 0..ctx.table_views_len {
            let view = unsafe { *ctx.table_views_base.add(table_index) };
            let table_bytes =
                checked_region_bytes(view.elements_len, self.layout.ref_handle_stride as usize)?;
            ensure_window_fits(table_bytes, TABLE_ELEMENTS_WINDOW_32)?;
        }

        Ok(())
    }

    #[inline]
    fn stack_addr(self, host_ptr: *mut u8) -> Result<u64, WasmError> {
        let offset = host_ptr.addr().saturating_sub(self.host_stack_base.addr());
        if offset > self.stack_len_bytes as usize {
            return Err(WasmError::internal(
                "host frame pointer is outside the emulator stack window".into(),
            ));
        }
        Ok(self.stack_base + offset as u64)
    }

    #[inline]
    fn host_stack_ptr(self, addr: u64) -> Result<*mut u8, WasmError> {
        let offset = self.offset_in_region(addr, self.stack_base, self.stack_len_bytes)?;
        Ok(self.host_stack_base.wrapping_add(offset as usize))
    }

    fn load(
        self,
        ctx: &NativeContext,
        addr: u64,
        width: MachineMemWidth,
    ) -> Option<Result<u64, WasmError>> {
        if self.contains(addr, self.stack_base, self.stack_len_bytes) {
            return Some(self.load_stack(addr, width));
        }
        if self.contains(addr, self.ctx_base, self.ctx_window_size(ctx.globals_len)) {
            return Some(self.load_context(ctx, addr - self.ctx_base, width));
        }
        if ctx.memory_views_len != 0 {
            let size =
                (ctx.memory_views_len as u32).saturating_mul(self.layout.pointer_len_view.stride);
            if self.contains(addr, self.memory_views_base, size) {
                return Some(self.load_memory_view(ctx, addr - self.memory_views_base, width));
            }
        }
        if ctx.table_views_len != 0 {
            let size =
                (ctx.table_views_len as u32).saturating_mul(self.layout.pointer_len_view.stride);
            if self.contains(addr, self.table_views_base, size) {
                return Some(self.load_table_view(ctx, addr - self.table_views_base, width));
            }
        }
        if ctx.function_views_len != 0 {
            let size =
                (ctx.function_views_len as u32).saturating_mul(self.layout.function_view.stride);
            if self.contains(addr, self.function_views_base, size) {
                return Some(self.load_function_view(ctx, addr - self.function_views_base, width));
            }
        }
        if ctx.local_call_infos_len != 0 {
            let size = (ctx.local_call_infos_len as u32)
                .saturating_mul(self.layout.local_call_info.stride);
            if self.contains(addr, self.local_call_infos_base, size) {
                return Some(self.load_local_call_info(
                    ctx,
                    addr - self.local_call_infos_base,
                    width,
                ));
            }
        }
        if ctx.globals_len != 0 {
            let raw_size = u32::try_from(
                ctx.globals_len
                    .saturating_mul(GLOBALS_RAW_STRIDE_32 as usize),
            )
            .unwrap_or(u32::MAX);
            if self.contains(addr, self.globals_raw_base, raw_size) {
                return Some(self.load_global_raw(ctx, addr - self.globals_raw_base, width));
            }
        }
        if ctx.type_canon_len != 0 {
            let size = u32::try_from(
                ctx.type_canon_len
                    .saturating_mul(core::mem::size_of::<u32>()),
            )
            .unwrap_or(u32::MAX);
            if self.contains(addr, self.type_canon_base, size) {
                return Some(self.load_type_canon(ctx, addr - self.type_canon_base, width));
            }
        }
        for mem_index in 0..ctx.memory_views_len {
            let view = unsafe { *ctx.memory_views_base.add(mem_index) };
            let size = u32::try_from(view.len).unwrap_or(u32::MAX);
            let base = self.memory_base(mem_index);
            if size != 0 && self.contains(addr, base, size) {
                return Some(self.load_memory(ctx, mem_index, addr - base, width));
            }
        }
        for table_index in 0..ctx.table_views_len {
            let view = unsafe { *ctx.table_views_base.add(table_index) };
            let size = view
                .elements_len
                .saturating_mul(self.layout.ref_handle_stride as usize);
            let size = u32::try_from(size).unwrap_or(u32::MAX);
            let base = self.table_elements_base(table_index);
            if size != 0 && self.contains(addr, base, size) {
                return Some(self.load_table_element(ctx, table_index, addr - base, width));
            }
        }
        None
    }

    fn store(
        self,
        ctx: &NativeContext,
        addr: u64,
        width: MachineMemWidth,
        value: u64,
    ) -> Option<Result<(), WasmError>> {
        if self.contains(addr, self.stack_base, self.stack_len_bytes) {
            return Some(self.store_stack(addr, width, value));
        }
        for mem_index in 0..ctx.memory_views_len {
            let view = unsafe { *ctx.memory_views_base.add(mem_index) };
            let size = u32::try_from(view.len).unwrap_or(u32::MAX);
            let base = self.memory_base(mem_index);
            if size != 0 && self.contains(addr, base, size) {
                return Some(self.store_memory(ctx, mem_index, addr - base, width, value));
            }
        }
        for table_index in 0..ctx.table_views_len {
            let view = unsafe { *ctx.table_views_base.add(table_index) };
            let size = view
                .elements_len
                .saturating_mul(self.layout.ref_handle_stride as usize);
            let size = u32::try_from(size).unwrap_or(u32::MAX);
            let base = self.table_elements_base(table_index);
            if size != 0 && self.contains(addr, base, size) {
                return Some(self.store_table_element(ctx, table_index, addr - base, value));
            }
        }
        if ctx.globals_len != 0 {
            let raw_size = u32::try_from(
                ctx.globals_len
                    .saturating_mul(GLOBALS_RAW_STRIDE_32 as usize),
            )
            .unwrap_or(u32::MAX);
            if self.contains(addr, self.globals_raw_base, raw_size) {
                return Some(self.store_global_raw(
                    ctx,
                    addr - self.globals_raw_base,
                    width,
                    value,
                ));
            }
        }
        if self.is_synthetic_container(addr) {
            return Some(Err(WasmError::internal(
                "machine store into synthetic runtime metadata is unsupported".into(),
            )));
        }
        None
    }

    #[inline]
    fn contains(self, addr: u64, base: u64, size: u32) -> bool {
        addr >= base && addr < base.saturating_add(u64::from(size))
    }

    #[inline]
    fn offset_in_region(self, addr: u64, base: u64, size: u32) -> Result<u32, WasmError> {
        if !self.contains(addr, base, size) {
            return Err(WasmError::internal(
                "synthetic address is out of range".into(),
            ));
        }
        Ok((addr - base) as u32)
    }

    #[inline]
    fn is_synthetic_container(self, addr: u64) -> bool {
        // The ctx container check uses the fixed header size; the inline-ptr
        // tail is handled by `load`/`store` dispatch via `ctx_window_size` on
        // live ctx, not here (this helper runs without a ctx reference).
        self.contains(addr, self.ctx_base, self.layout.context.size)
            || self.contains(
                addr,
                self.memory_views_base,
                u32::MAX - self.memory_views_base as u32,
            )
            || self.contains(
                addr,
                self.table_views_base,
                u32::MAX - self.table_views_base as u32,
            )
            || self.contains(
                addr,
                self.function_views_base,
                u32::MAX - self.function_views_base as u32,
            )
            || self.contains(
                addr,
                self.local_call_infos_base,
                u32::MAX - self.local_call_infos_base as u32,
            )
    }

    #[inline]
    fn memory_base(self, mem_index: usize) -> u64 {
        MEMORY_BASE_32 + mem_index as u64 * MEMORY_WINDOW_32
    }

    #[inline]
    fn table_elements_base(self, table_index: usize) -> u64 {
        TABLE_ELEMENTS_BASE_32 + table_index as u64 * TABLE_ELEMENTS_WINDOW_32
    }

    #[inline]
    fn global_raw_base(self, global_index: usize) -> u64 {
        self.globals_raw_base + global_index as u64 * GLOBALS_RAW_STRIDE_32
    }

    fn load_stack(self, addr: u64, width: MachineMemWidth) -> Result<u64, WasmError> {
        let ptr = self.host_stack_ptr(addr)?;
        Ok(unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::read_unaligned(ptr.cast::<u8>()) as u64,
                MachineMemWidth::U16 => core::ptr::read_unaligned(ptr.cast::<u16>()) as u64,
                MachineMemWidth::U32 => core::ptr::read_unaligned(ptr.cast::<u32>()) as u64,
                MachineMemWidth::U64 => core::ptr::read_unaligned(ptr.cast::<u64>()),
            }
        })
    }

    fn store_stack(self, addr: u64, width: MachineMemWidth, value: u64) -> Result<(), WasmError> {
        let ptr = self.host_stack_ptr(addr)?;
        unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::write_unaligned(ptr.cast::<u8>(), value as u8),
                MachineMemWidth::U16 => core::ptr::write_unaligned(ptr.cast::<u16>(), value as u16),
                MachineMemWidth::U32 => core::ptr::write_unaligned(ptr.cast::<u32>(), value as u32),
                MachineMemWidth::U64 => core::ptr::write_unaligned(ptr.cast::<u64>(), value),
            }
        }
        Ok(())
    }

    fn load_context(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let ctx_layout = self.layout.context;
        let offset = offset as u32;
        if offset == ctx_layout.stack_end_offset {
            return Ok(self.stack_end());
        }
        if offset == ctx_layout.mem0_base_offset {
            return Ok(if ctx.mem0_size == 0 {
                0
            } else {
                self.memory_base(0)
            });
        }
        if offset == ctx_layout.mem0_size_offset {
            return self.read_scalar(width, ctx.mem0_size);
        }
        if offset == ctx_layout.globals_len_offset {
            return self.read_scalar(width, ctx.globals_len as u64);
        }
        // Inline globals-ptr tail: any offset inside
        // [globals_ptrs_inline_offset, globals_ptrs_inline_offset + n*ptr)
        // resolves to the synthetic raw-value address for that index, letting
        // the JIT's subsequent deref hit `load_global_raw` / `store_global_raw`.
        if ctx.globals_len != 0 {
            let inline_base = ctx_layout.globals_ptrs_inline_offset;
            let stride = self.layout.gp_unit_bytes as u32;
            let inline_len = (ctx.globals_len as u32).saturating_mul(stride);
            if offset >= inline_base && offset < inline_base.saturating_add(inline_len) {
                let rel = offset - inline_base;
                if rel % stride != 0 {
                    return Err(WasmError::internal(
                        "synthetic globals-ptr tail load is misaligned",
                    ));
                }
                let index = (rel / stride) as usize;
                return Ok(self.global_raw_base(index));
            }
        }
        if offset == ctx_layout.memory_views_base_offset {
            return Ok(if ctx.memory_views_len == 0 {
                0
            } else {
                self.memory_views_base
            });
        }
        if offset == ctx_layout.memory_views_len_offset {
            return self.read_scalar(width, ctx.memory_views_len as u64);
        }
        if offset == ctx_layout.table_views_base_offset {
            return Ok(if ctx.table_views_len == 0 {
                0
            } else {
                self.table_views_base
            });
        }
        if offset == ctx_layout.table_views_len_offset {
            return self.read_scalar(width, ctx.table_views_len as u64);
        }
        if offset == ctx_layout.function_views_base_offset {
            return Ok(if ctx.function_views_len == 0 {
                0
            } else {
                self.function_views_base
            });
        }
        if offset == ctx_layout.function_views_len_offset {
            return self.read_scalar(width, ctx.function_views_len as u64);
        }
        if offset == ctx_layout.local_call_infos_base_offset {
            return Ok(if ctx.local_call_infos_len == 0 {
                0
            } else {
                self.local_call_infos_base
            });
        }
        if offset == ctx_layout.local_call_infos_len_offset {
            return self.read_scalar(width, ctx.local_call_infos_len as u64);
        }
        if offset == ctx_layout.type_canon_base_offset {
            return Ok(if ctx.type_canon_len == 0 {
                0
            } else {
                self.type_canon_base
            });
        }
        if offset == ctx_layout.type_canon_len_offset {
            return self.read_scalar(width, ctx.type_canon_len as u64);
        }
        if offset == ctx_layout.store_offset {
            return Ok(ctx.store as u64);
        }
        if offset == ctx_layout.current_module_offset {
            return Ok(ctx.current_module as u64);
        }
        Err(WasmError::internal(
            "synthetic 32-bit runtime context load uses unsupported offset",
        ))
    }

    fn load_memory_view(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let entry = self.pointer_len_entry(ctx.memory_views_len, offset, "memory view")?;
        let view = unsafe { *ctx.memory_views_base.add(entry.index) };
        if entry.field_offset == self.layout.pointer_len_view.base_offset {
            Ok(if view.len == 0 {
                0
            } else {
                self.memory_base(entry.index)
            })
        } else if entry.field_offset == self.layout.pointer_len_view.len_offset {
            self.read_scalar(width, view.len as u64)
        } else {
            Err(WasmError::internal(
                "synthetic 32-bit memory view load uses unsupported field offset",
            ))
        }
    }

    fn load_table_view(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let entry = self.pointer_len_entry(ctx.table_views_len, offset, "table view")?;
        let view = unsafe { *ctx.table_views_base.add(entry.index) };
        if entry.field_offset == self.layout.pointer_len_view.base_offset {
            Ok(if view.elements_len == 0 {
                0
            } else {
                self.table_elements_base(entry.index)
            })
        } else if entry.field_offset == self.layout.pointer_len_view.len_offset {
            self.read_scalar(width, view.elements_len as u64)
        } else {
            Err(WasmError::internal(
                "synthetic 32-bit table view load uses unsupported field offset",
            ))
        }
    }

    fn load_function_view(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let stride = u64::from(self.layout.function_view.stride);
        let index = (offset / stride) as usize;
        if index >= ctx.function_views_len {
            return Err(WasmError::internal(
                "synthetic 32-bit function view load is out of range: entry >=",
            ));
        }
        let field_offset = (offset % stride) as u32;
        let view = unsafe { *ctx.function_views_base.add(index) };
        let value = if field_offset == self.layout.function_view.kind_offset {
            u64::from(view.kind)
        } else if field_offset == self.layout.function_view.type_canon_offset {
            u64::from(view.type_canon)
        } else if field_offset == self.layout.function_view.local_target_offset {
            u64::from(view.local_target)
        } else {
            return Err(WasmError::internal(
                "synthetic 32-bit function view load uses unsupported field offset",
            ));
        };
        self.read_scalar(width, value)
    }

    fn load_local_call_info(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let stride = u64::from(self.layout.local_call_info.stride);
        let index = (offset / stride) as usize;
        if index >= ctx.local_call_infos_len {
            return Err(WasmError::internal(
                "synthetic 32-bit local call info load is out of range: entry >=",
            ));
        }
        let field_offset = (offset % stride) as usize;
        let ptr = ctx
            .local_call_infos_base
            .wrapping_add(index.saturating_mul(self.layout.local_call_info.stride as usize))
            .wrapping_add(field_offset);
        Ok(unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::read_unaligned(ptr.cast::<u8>()) as u64,
                MachineMemWidth::U16 => core::ptr::read_unaligned(ptr.cast::<u16>()) as u64,
                MachineMemWidth::U32 => core::ptr::read_unaligned(ptr.cast::<u32>()) as u64,
                MachineMemWidth::U64 => core::ptr::read_unaligned(ptr.cast::<u64>()),
            }
        })
    }

    /// Resolve a per-global synthetic address into the host-side raw u64 slot
    /// plus the owning global's value type. The slot is reached via the
    /// inline raw-ptr tail (`ctx.globals_ptrs_base()`), but we still consult
    /// the store for `value_type` to distinguish ref globals.
    fn resolve_global_slot(
        self,
        ctx: &NativeContext,
        index: usize,
    ) -> Result<(*mut u64, crate::value_type::ValueType), WasmError> {
        if index >= ctx.globals_len {
            return Err(WasmError::internal(
                "synthetic 32-bit global slot is out of range",
            ));
        }
        // SAFETY: the inline tail was allocated with `ctx.globals_len` slots
        // and populated by `refresh_globals_ptrs`; `globals_ptrs_base()`
        // returns a valid pointer into the same allocation.
        let raw_ptr = unsafe { *ctx.globals_ptrs_base().add(index) };
        if raw_ptr.is_null() {
            return Err(WasmError::internal(
                "synthetic 32-bit global slot raw_ptr is null (refresh missed?)",
            ));
        }
        // SAFETY: while the context is live the owning `Store` is also live,
        // and `refresh_cached_views` only runs between invocations. The
        // pointer dereference here is read-only metadata (value_type).
        let store = unsafe { ctx.store.as_ref() }
            .ok_or_else(|| WasmError::internal("emu32 global access without a store"))?;
        let value_type = store
            .module()
            .globals
            .get(index)
            .ok_or_else(|| WasmError::internal("emu32 global index outside store module"))?
            .value_type;
        Ok((raw_ptr, value_type))
    }

    fn load_global_raw(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let stride = GLOBALS_RAW_STRIDE_32;
        let index = (offset / stride) as usize;
        let field_offset = (offset % stride) as usize;
        if field_offset + width.bytes() as usize > stride as usize {
            return Err(WasmError::internal(
                "synthetic 32-bit global raw load is out of range",
            ));
        }
        let (raw_ptr, value_type) = self.resolve_global_slot(ctx, index)?;
        // SAFETY: raw_ptr came from `GlobalInst::raw_ptr()`, which points at
        // a live `UnsafeCell<u64>` owned by the (refcounted) shared cell for
        // the life of the owning GlobalInst.
        let raw = unsafe { *raw_ptr };

        if value_type.is_ref() {
            if field_offset != 0 || !matches!(width, MachineMemWidth::U32 | MachineMemWidth::U64) {
                return Err(WasmError::internal(
                    "synthetic 32-bit ref global raw load uses unsupported access shape",
                ));
            }
            return self.read_scalar(
                width,
                ref_to_machine_raw(as_ref(raw), self.layout.gp_unit_bytes),
            );
        }

        self.read_scalar(width, raw >> (field_offset * 8))
    }

    fn store_global_raw(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
        value: u64,
    ) -> Result<(), WasmError> {
        let stride = GLOBALS_RAW_STRIDE_32;
        let index = (offset / stride) as usize;
        let field_offset = (offset % stride) as usize;
        let width_bytes = width.bytes() as usize;
        if field_offset + width_bytes > stride as usize {
            return Err(WasmError::internal(
                "synthetic 32-bit global raw store is out of range",
            ));
        }
        let (raw_ptr, value_type) = self.resolve_global_slot(ctx, index)?;

        if value_type.is_ref() {
            if field_offset != 0 || !matches!(width, MachineMemWidth::U32 | MachineMemWidth::U64) {
                return Err(WasmError::internal(
                    "synthetic 32-bit ref global raw store uses unsupported access shape",
                ));
            }
            // SAFETY: see `resolve_global_slot`.
            unsafe {
                *raw_ptr = machine_raw_to_ref(value, self.layout.gp_unit_bytes).encoded() as u64;
            }
            return Ok(());
        }

        // SAFETY: see `resolve_global_slot`.
        let current = unsafe { *raw_ptr };
        let shift = field_offset * 8;
        let bits = width_bytes * 8;
        let mask = if bits == 64 {
            u64::MAX
        } else {
            ((1u64 << bits) - 1) << shift
        };
        let merged = (current & !mask) | ((value << shift) & mask);
        // SAFETY: same origin as `*raw_ptr` above.
        unsafe { *raw_ptr = merged };
        Ok(())
    }

    fn load_type_canon(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let index = (offset / core::mem::size_of::<u32>() as u64) as usize;
        if index >= ctx.type_canon_len {
            return Err(WasmError::internal(
                "synthetic 32-bit type canon load is out of range: entry >=",
            ));
        }
        if offset % core::mem::size_of::<u32>() as u64 != 0 {
            return Err(WasmError::internal(
                "synthetic 32-bit type canon load uses unsupported byte offset",
            ));
        }
        let value = unsafe { u64::from(*ctx.type_canon_base.add(index)) };
        self.read_scalar(width, value)
    }

    fn load_memory(
        self,
        ctx: &NativeContext,
        mem_index: usize,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let view = unsafe { *ctx.memory_views_base.add(mem_index) };
        let size = width.bytes() as usize;
        if offset as usize + size > view.len {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        let ptr = unsafe { view.base.add(offset as usize) };
        Ok(unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::read_unaligned(ptr.cast::<u8>()) as u64,
                MachineMemWidth::U16 => core::ptr::read_unaligned(ptr.cast::<u16>()) as u64,
                MachineMemWidth::U32 => core::ptr::read_unaligned(ptr.cast::<u32>()) as u64,
                MachineMemWidth::U64 => core::ptr::read_unaligned(ptr.cast::<u64>()),
            }
        })
    }

    fn store_memory(
        self,
        ctx: &NativeContext,
        mem_index: usize,
        offset: u64,
        width: MachineMemWidth,
        value: u64,
    ) -> Result<(), WasmError> {
        let view = unsafe { *ctx.memory_views_base.add(mem_index) };
        let size = width.bytes() as usize;
        if offset as usize + size > view.len {
            return Err(WasmError::trap("out of bounds memory access"));
        }
        let ptr = unsafe { view.base.add(offset as usize) };
        unsafe {
            match width {
                MachineMemWidth::U8 => core::ptr::write_unaligned(ptr.cast::<u8>(), value as u8),
                MachineMemWidth::U16 => core::ptr::write_unaligned(ptr.cast::<u16>(), value as u16),
                MachineMemWidth::U32 => core::ptr::write_unaligned(ptr.cast::<u32>(), value as u32),
                MachineMemWidth::U64 => core::ptr::write_unaligned(ptr.cast::<u64>(), value),
            }
        }
        Ok(())
    }

    fn load_table_element(
        self,
        ctx: &NativeContext,
        table_index: usize,
        offset: u64,
        _width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let entry = self.ref_handle_entry(ctx, table_index, offset)?;
        let view = unsafe { *ctx.table_views_base.add(table_index) };
        let handle = unsafe { *view.elements_base.add(entry) };
        Ok(ref_to_machine_raw(handle, self.layout.gp_unit_bytes))
    }

    fn store_table_element(
        self,
        ctx: &NativeContext,
        table_index: usize,
        offset: u64,
        value: u64,
    ) -> Result<(), WasmError> {
        let entry = self.ref_handle_entry(ctx, table_index, offset)?;
        let view = unsafe { *ctx.table_views_base.add(table_index) };
        unsafe {
            *view.elements_base.add(entry) = machine_raw_to_ref(value, self.layout.gp_unit_bytes);
        }
        Ok(())
    }

    fn pointer_len_entry(
        self,
        len: usize,
        offset: u64,
        _label: &'static str,
    ) -> Result<PointerLenEntry, WasmError> {
        let stride = u64::from(self.layout.pointer_len_view.stride);
        let index = (offset / stride) as usize;
        if index >= len {
            return Err(WasmError::internal(
                "synthetic 32-bit load is out of range: entry >=",
            ));
        }
        Ok(PointerLenEntry {
            index,
            field_offset: (offset % stride) as u32,
        })
    }

    fn ref_handle_entry(
        self,
        ctx: &NativeContext,
        table_index: usize,
        offset: u64,
    ) -> Result<usize, WasmError> {
        let view = unsafe { *ctx.table_views_base.add(table_index) };
        let stride = u64::from(self.layout.ref_handle_stride);
        let entry = (offset / stride) as usize;
        if entry >= view.elements_len {
            return Err(WasmError::internal(
                "synthetic 32-bit table element access is out of range: entry >=",
            ));
        }
        if offset % stride != 0 {
            return Err(WasmError::internal(
                "synthetic 32-bit table element access uses unsupported byte offset",
            ));
        }
        Ok(entry)
    }

    fn read_scalar(self, width: MachineMemWidth, value: u64) -> Result<u64, WasmError> {
        Ok(match width {
            MachineMemWidth::U8 => u64::from(value as u8),
            MachineMemWidth::U16 => u64::from(value as u16),
            MachineMemWidth::U32 => u64::from(value as u32),
            MachineMemWidth::U64 => value,
        })
    }
}

fn checked_region_bytes(count: usize, stride: usize) -> Result<usize, WasmError> {
    count
        .checked_mul(stride)
        .ok_or_else(|| WasmError::internal("emu32 synthetic region size overflows usize"))
}

fn ensure_window_fits(bytes: usize, window_bytes: u64) -> Result<(), WasmError> {
    if u64::try_from(bytes).unwrap_or(u64::MAX) > window_bytes {
        return Err(WasmError::internal(
            "emu32 reserved address window is too small for synthetic region",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PointerLenEntry {
    index: usize,
    field_offset: u32,
}
