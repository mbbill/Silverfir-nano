use super::super::machine::MachineExternId;

/// One closed helper symbol that may remain as a true runtime boundary.
///
/// The backend resolves this symbol to the real helper wrapper address for the
/// active ISA/runtime implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MachineHelperSymbol {
    CallExternal,
    MemoryGrow,
    TableGrow,
    MemoryInit,
    DataDrop,
    TableInit,
    ElemDrop,
}

/// One opaque external binding referenced from machine IR.
///
/// The machine layer uses only the id. Sidecar runtime data owns the meaning
/// of each external target and how it resolves during finalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MachineExternBinding {
    pub id: MachineExternId,
    pub symbol: MachineHelperSymbol,
}
