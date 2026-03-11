//! ARM64 emission helpers and templates.

use super::reg::Arm64Reg;

/// Minimal ARM64 emitter interface.
pub trait Arm64Emit {
    fn mov_reg(&mut self, _dst: Arm64Reg, _src: Arm64Reg);
    fn load_frame_u64(&mut self, _dst: Arm64Reg, _slot: u16);
    fn store_frame_u64(&mut self, _src: Arm64Reg, _slot: u16);
    fn branch_entry(&mut self, _entry_addr: usize);
    fn branch_cold_wrapper(&mut self, _entry_addr: usize);
}
