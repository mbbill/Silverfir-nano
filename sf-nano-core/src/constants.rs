pub const WASM_PAGE_SIZE: usize = 65536;

// specification maximum number of pages a 32-bit memory can have
pub(crate) const MAX_MEM_PAGES_SPEC: usize = 65536;

// maximum page count representable by the current usize-byte backing model
const MAX_MEM_PAGES_REPR: usize = usize::MAX / WASM_PAGE_SIZE;

// maximum number of pages a 32-bit memory can have in the current in-memory
// representation
pub const MAX_MEM_PAGES: usize = if MAX_MEM_PAGES_SPEC > MAX_MEM_PAGES_REPR {
    MAX_MEM_PAGES_REPR
} else {
    MAX_MEM_PAGES_SPEC
};

// specification maximum number of pages a 64-bit memory can have
pub const MAX_MEM_PAGES_64_SPEC: u64 = 1u64 << 48;

// maximum number of pages a 64-bit memory can have in the current in-memory
// representation
pub const MAX_MEM_PAGES_64: usize = if MAX_MEM_PAGES_64_SPEC > MAX_MEM_PAGES_REPR as u64 {
    MAX_MEM_PAGES_REPR
} else {
    MAX_MEM_PAGES_64_SPEC as usize
};

// maximum number of elements a table can have (32-bit)
pub const MAX_TABLE_SIZE: usize = u32::MAX as usize;

// maximum number of elements a 64-bit table can have
pub const MAX_TABLE_SIZE_64: usize = usize::MAX;

// Maximum number of locals allowed.
pub const MAX_LOCALS: u32 = 4096;
