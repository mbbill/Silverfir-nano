use crate::{
    error::WasmError,
    vm::{
        entities::{global_offset, GlobalInst},
        machine::mir::MachineMemWidth,
        runtime::{
            code::CompiledNativeModule,
            context::NativeContext,
            layout::{native_runtime_abi_layout, NativeRuntimeAbiLayout},
        },
        value::RefHandle,
    },
};

const STACK_BASE_32: u64 = 0x0100_0000;
const CTX_BASE_32: u64 = 0x1000_0000;
const MEMORY_VIEWS_BASE_32: u64 = 0x1100_0000;
const TABLE_VIEWS_BASE_32: u64 = 0x1200_0000;
const FUNCTION_VIEWS_BASE_32: u64 = 0x1300_0000;
const GLOBALS_BASE_32: u64 = 0x1400_0000;
const TYPE_CANON_BASE_32: u64 = 0x1500_0000;
const MEMORY_BASE_32: u64 = 0x4000_0000;
const MEMORY_WINDOW_32: u64 = 0x1000_0000;
const TABLE_ELEMENTS_BASE_32: u64 = 0xC000_0000;
const TABLE_ELEMENTS_WINDOW_32: u64 = 0x0400_0000;
const MEMORY_VIEWS_WINDOW_32: u64 = TABLE_VIEWS_BASE_32 - MEMORY_VIEWS_BASE_32;
const TABLE_VIEWS_WINDOW_32: u64 = FUNCTION_VIEWS_BASE_32 - TABLE_VIEWS_BASE_32;
const FUNCTION_VIEWS_WINDOW_32: u64 = GLOBALS_BASE_32 - FUNCTION_VIEWS_BASE_32;
const GLOBALS_WINDOW_32: u64 = TYPE_CANON_BASE_32 - GLOBALS_BASE_32;
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
    globals_base: u64,
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
            globals_base: GLOBALS_BASE_32,
            type_canon_base: TYPE_CANON_BASE_32,
            layout: native_runtime_abi_layout(4),
        }
    }

    #[inline]
    fn stack_end(self) -> u64 {
        self.stack_base + u64::from(self.stack_len_bytes)
    }

    fn validate_runtime_shape(self, ctx: &NativeContext) -> Result<(), WasmError> {
        ensure_window_fits(
            "emu32 synthetic memory-view metadata",
            checked_region_bytes(
                "emu32 synthetic memory-view metadata",
                ctx.memory_views_len,
                self.layout.pointer_len_view.stride as usize,
            )?,
            MEMORY_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            "emu32 synthetic table-view metadata",
            checked_region_bytes(
                "emu32 synthetic table-view metadata",
                ctx.table_views_len,
                self.layout.pointer_len_view.stride as usize,
            )?,
            TABLE_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            "emu32 synthetic function-view metadata",
            checked_region_bytes(
                "emu32 synthetic function-view metadata",
                ctx.function_views_len,
                self.layout.function_view.stride as usize,
            )?,
            FUNCTION_VIEWS_WINDOW_32,
        )?;
        ensure_window_fits(
            "emu32 synthetic globals metadata",
            checked_region_bytes(
                "emu32 synthetic globals metadata",
                ctx.globals_view.len,
                core::mem::size_of::<GlobalInst>(),
            )?,
            GLOBALS_WINDOW_32,
        )?;
        ensure_window_fits(
            "emu32 synthetic type-canon metadata",
            checked_region_bytes(
                "emu32 synthetic type-canon metadata",
                ctx.type_canon_len,
                core::mem::size_of::<u32>(),
            )?,
            TYPE_CANON_WINDOW_32,
        )?;

        if ctx.memory_views_len > MAX_MEMORY_COUNT_32 {
            return Err(WasmError::internal(alloc::format!(
                "emu32 synthetic address space supports at most {} memories, found {}",
                MAX_MEMORY_COUNT_32,
                ctx.memory_views_len,
            )));
        }
        for mem_index in 0..ctx.memory_views_len {
            let view = unsafe { *ctx.memory_views_base.add(mem_index) };
            ensure_window_fits(
                &alloc::format!("emu32 synthetic memory {} contents", mem_index),
                view.len,
                MEMORY_WINDOW_32,
            )?;
        }

        if ctx.table_views_len > MAX_TABLE_COUNT_32 {
            return Err(WasmError::internal(alloc::format!(
                "emu32 synthetic address space supports at most {} tables, found {}",
                MAX_TABLE_COUNT_32,
                ctx.table_views_len,
            )));
        }
        for table_index in 0..ctx.table_views_len {
            let view = unsafe { *ctx.table_views_base.add(table_index) };
            let table_bytes = checked_region_bytes(
                &alloc::format!("emu32 synthetic table {} contents", table_index),
                view.elements_len,
                self.layout.ref_handle_stride as usize,
            )?;
            ensure_window_fits(
                &alloc::format!("emu32 synthetic table {} contents", table_index),
                table_bytes,
                TABLE_ELEMENTS_WINDOW_32,
            )?;
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
        if self.contains(addr, self.ctx_base, self.layout.context.size) {
            return Some(self.load_context(ctx, addr - self.ctx_base, width));
        }
        if ctx.globals_view.len != 0 {
            let size = u32::try_from(
                ctx.globals_view
                    .len
                    .saturating_mul(core::mem::size_of::<GlobalInst>()),
            )
            .unwrap_or(u32::MAX);
            if self.contains(addr, self.globals_base, size) {
                return Some(self.load_global(ctx, addr - self.globals_base, width));
            }
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
        if ctx.globals_view.len != 0 {
            let size = u32::try_from(
                ctx.globals_view
                    .len
                    .saturating_mul(core::mem::size_of::<GlobalInst>()),
            )
            .unwrap_or(u32::MAX);
            if self.contains(addr, self.globals_base, size) {
                return Some(self.store_global(ctx, addr - self.globals_base, width, value));
            }
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
    }

    #[inline]
    fn memory_base(self, mem_index: usize) -> u64 {
        MEMORY_BASE_32 + mem_index as u64 * MEMORY_WINDOW_32
    }

    #[inline]
    fn table_elements_base(self, table_index: usize) -> u64 {
        TABLE_ELEMENTS_BASE_32 + table_index as u64 * TABLE_ELEMENTS_WINDOW_32
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
        if offset == ctx_layout.globals_view_base_offset {
            return Ok(if ctx.globals_view.len == 0 {
                0
            } else {
                self.globals_base
            });
        }
        if offset == ctx_layout.globals_view_len_offset {
            return self.read_scalar(width, ctx.globals_view.len as u64);
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
        Err(WasmError::internal(alloc::format!(
            "synthetic 32-bit runtime context load uses unsupported offset {}",
            offset
        )))
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
            Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit memory view load uses unsupported field offset {}",
                entry.field_offset
            )))
        }
    }

    fn load_global(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let stride = core::mem::size_of::<GlobalInst>() as u64;
        let index = (offset / stride) as usize;
        if index >= ctx.globals_view.len {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit global load is out of range: entry {} >= {}",
                index,
                ctx.globals_view.len
            )));
        }
        let field_offset = (offset % stride) as u32;
        let global = unsafe { &*ctx.globals_view.base.add(index) };
        let width_bytes = mem_width_bytes(width) as u32;
        if field_offset < global_offset::RAW
            || field_offset
                .saturating_add(width_bytes)
                .saturating_sub(global_offset::RAW)
                > core::mem::size_of::<u64>() as u32
        {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit global load uses unsupported field offset {}",
                field_offset
            )));
        }
        let bit_shift = (field_offset - global_offset::RAW) * 8;
        let raw = global.raw() >> bit_shift;
        self.read_scalar(width, raw)
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
            Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit table view load uses unsupported field offset {}",
                entry.field_offset
            )))
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
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit function view load is out of range: entry {} >= {}",
                index,
                ctx.function_views_len
            )));
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
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit function view load uses unsupported field offset {}",
                field_offset
            )));
        };
        self.read_scalar(width, value)
    }

    fn load_type_canon(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
    ) -> Result<u64, WasmError> {
        let index = (offset / core::mem::size_of::<u32>() as u64) as usize;
        if index >= ctx.type_canon_len {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit type canon load is out of range: entry {} >= {}",
                index,
                ctx.type_canon_len
            )));
        }
        if offset % core::mem::size_of::<u32>() as u64 != 0 {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit type canon load uses unsupported byte offset {}",
                offset
            )));
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
        let size = mem_width_bytes(width);
        if offset as usize + size > view.len {
            return Err(WasmError::trap("out of bounds memory access".into()));
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
        let size = mem_width_bytes(width);
        if offset as usize + size > view.len {
            return Err(WasmError::trap("out of bounds memory access".into()));
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

    fn store_global(
        self,
        ctx: &NativeContext,
        offset: u64,
        width: MachineMemWidth,
        value: u64,
    ) -> Result<(), WasmError> {
        let stride = core::mem::size_of::<GlobalInst>() as u64;
        let index = (offset / stride) as usize;
        if index >= ctx.globals_view.len {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit global store is out of range: entry {} >= {}",
                index,
                ctx.globals_view.len
            )));
        }
        let field_offset = (offset % stride) as u32;
        let global = unsafe { &mut *ctx.globals_view.base.add(index) };
        let width_bytes = mem_width_bytes(width) as u32;
        if field_offset < global_offset::RAW
            || field_offset
                .saturating_add(width_bytes)
                .saturating_sub(global_offset::RAW)
                > core::mem::size_of::<u64>() as u32
        {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit global store uses unsupported field offset {}",
                field_offset
            )));
        }
        let raw = match width {
            MachineMemWidth::U8 => u64::from(value as u8),
            MachineMemWidth::U16 => u64::from(value as u16),
            MachineMemWidth::U32 => u64::from(value as u32),
            MachineMemWidth::U64 => value,
        };
        let bit_shift = (field_offset - global_offset::RAW) * 8;
        let mask = match width {
            MachineMemWidth::U8 => 0xffu64,
            MachineMemWidth::U16 => 0xffffu64,
            MachineMemWidth::U32 => 0xffff_ffffu64,
            MachineMemWidth::U64 => u64::MAX,
        } << bit_shift;
        let merged = (global.raw() & !mask) | ((raw << bit_shift) & mask);
        global.set_raw(merged);
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
        Ok(handle.0 as u64)
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
            *view.elements_base.add(entry) = RefHandle(value as usize);
        }
        Ok(())
    }

    fn pointer_len_entry(
        self,
        len: usize,
        offset: u64,
        label: &'static str,
    ) -> Result<PointerLenEntry, WasmError> {
        let stride = u64::from(self.layout.pointer_len_view.stride);
        let index = (offset / stride) as usize;
        if index >= len {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit {label} load is out of range: entry {} >= {}",
                index,
                len
            )));
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
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit table element access is out of range: entry {} >= {}",
                entry,
                view.elements_len
            )));
        }
        if offset % stride != 0 {
            return Err(WasmError::internal(alloc::format!(
                "synthetic 32-bit table element access uses unsupported byte offset {}",
                offset
            )));
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

fn checked_region_bytes(label: &str, count: usize, stride: usize) -> Result<usize, WasmError> {
    count.checked_mul(stride).ok_or_else(|| {
        WasmError::internal(alloc::format!(
            "{label} size overflowed usize during emu32 validation"
        ))
    })
}

fn ensure_window_fits(label: &str, bytes: usize, window_bytes: u64) -> Result<(), WasmError> {
    if u64::try_from(bytes).unwrap_or(u64::MAX) > window_bytes {
        return Err(WasmError::internal(alloc::format!(
            "{label} requires {} bytes, but emu32 reserves only {} bytes",
            bytes,
            window_bytes,
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PointerLenEntry {
    index: usize,
    field_offset: u32,
}

#[inline]
fn mem_width_bytes(width: MachineMemWidth) -> usize {
    match width {
        MachineMemWidth::U8 => 1,
        MachineMemWidth::U16 => 2,
        MachineMemWidth::U32 => 4,
        MachineMemWidth::U64 => 8,
    }
}
