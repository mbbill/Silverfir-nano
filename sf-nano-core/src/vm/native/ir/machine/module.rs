use alloc::vec::Vec;

use super::types::{MachineBlockId, MachineConstId, MachineFuncId};
use crate::vm::native::ir::runtime::MachineExternBinding;

/// One read-only sidecar constant record referenced from machine IR.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineConstData {
    pub id: MachineConstId,
    pub align: u32,
    pub bytes: Vec<u8>,
}

/// Full machine program for one function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineProgram {
    pub entry: MachineBlockId,
    pub reg_count: u16,
    pub blocks: Vec<super::cfg::MachineBlock>,
}

/// One machine function inside a machine module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineFunction {
    pub id: MachineFuncId,
    pub program: MachineProgram,
}

/// One full machine module.
///
/// This is the shared allocation domain for function ids and sidecar constant
/// ids used by the machine IR, plus opaque external target ids used by helper
/// calls and other out-of-line native targets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MachineModule {
    pub functions: Vec<MachineFunction>,
    pub consts: Vec<MachineConstData>,
    pub externs: Vec<MachineExternBinding>,
}

