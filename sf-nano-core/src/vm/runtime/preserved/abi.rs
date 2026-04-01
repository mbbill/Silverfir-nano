//! ABI-visible constants for the preserved-helper entry.

/// Fixed native-stack I/O slot indices used by the preserved-helper entry.
pub(crate) mod io {
    /// First immediate (mem_idx, table_idx, data_idx, elem_idx).
    pub(crate) const IMM0: usize = 0;
    /// Second immediate (src_mem_idx, src_table_idx) — zero when unused.
    pub(crate) const IMM1: usize = 1;
    /// First Wasm stack operand.
    pub(crate) const ARG0: usize = 2;
    /// Second Wasm stack operand.
    pub(crate) const ARG1: usize = 3;
    /// Third Wasm stack operand.
    pub(crate) const ARG2: usize = 4;
    /// Result written by the helper.
    pub(crate) const RET0: usize = 5;
    /// Total number of 8-byte slots in the I/O area.
    pub(crate) const SLOT_COUNT: usize = 8;
}

/// Op-code constants for the preserved-helper entry.
pub(crate) mod op {
    pub(crate) const MEMORY_GROW: u32 = 0;
    pub(crate) const MEMORY_FILL: u32 = 1;
    pub(crate) const MEMORY_COPY: u32 = 2;
    pub(crate) const MEMORY_INIT: u32 = 3;
    pub(crate) const DATA_DROP: u32 = 4;
    pub(crate) const TABLE_GROW: u32 = 5;
    pub(crate) const TABLE_FILL: u32 = 6;
    pub(crate) const TABLE_COPY: u32 = 7;
    pub(crate) const TABLE_INIT: u32 = 8;
    pub(crate) const ELEM_DROP: u32 = 9;
}
