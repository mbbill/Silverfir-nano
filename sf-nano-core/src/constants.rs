pub const WASM_PAGE_SIZE: usize = 65536;

// maximum number of pages a memory can have (32-bit: 4GB limit)
pub const MAX_MEM_PAGES: usize = 65536;

// maximum number of pages a 64-bit memory can have
#[cfg(target_pointer_width = "64")]
pub const MAX_MEM_PAGES_64: usize = 1 << 48;
#[cfg(target_pointer_width = "32")]
pub const MAX_MEM_PAGES_64: usize = MAX_MEM_PAGES;

// maximum number of elements a table can have (32-bit)
pub const MAX_TABLE_SIZE: usize = u32::MAX as usize;

// maximum number of elements a 64-bit table can have
pub const MAX_TABLE_SIZE_64: usize = usize::MAX;

// Maximum number of locals allowed.
pub const MAX_LOCALS: u32 = 4096;

// Maximum total stack memory in bytes.
pub const MAX_STACK_SIZE: usize = 1024 * 1024 * 2;
