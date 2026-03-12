//! LIR-visible runtime-boundary operations.
//!
//! These ops remain semantic runtime boundaries above machine IR. They must not
//! be folded back into generic leaf ops.

use crate::vm::wasm::primitive_op::PrimitiveOpKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LirRuntimeOp {
    MemoryGrow { mem_idx: u32 },
    TableGrow { table_idx: u32 },
    MemoryInit { data_idx: u32, mem_idx: u32 },
    DataDrop { data_idx: u32 },
    TableInit { elem_idx: u32, table_idx: u32 },
    ElemDrop { elem_idx: u32 },
}

impl LirRuntimeOp {
    pub fn from_primitive(kind: PrimitiveOpKind) -> Option<Self> {
        match kind {
            PrimitiveOpKind::MemoryGrow { mem_idx } => Some(Self::MemoryGrow { mem_idx }),
            PrimitiveOpKind::TableGrow { table_idx } => Some(Self::TableGrow { table_idx }),
            PrimitiveOpKind::MemoryInit { imm0, imm1 } => Some(Self::MemoryInit {
                data_idx: imm0,
                mem_idx: imm1,
            }),
            PrimitiveOpKind::DataDrop { data_idx } => Some(Self::DataDrop { data_idx }),
            PrimitiveOpKind::TableInit { imm0, imm1 } => Some(Self::TableInit {
                elem_idx: imm0,
                table_idx: imm1,
            }),
            PrimitiveOpKind::ElemDrop { elem_idx } => Some(Self::ElemDrop { elem_idx }),
            _ => None,
        }
    }
}

#[inline]
pub fn is_runtime_boundary_primitive(kind: &PrimitiveOpKind) -> bool {
    matches!(
        kind,
        PrimitiveOpKind::MemoryGrow { .. }
            | PrimitiveOpKind::TableGrow { .. }
            | PrimitiveOpKind::MemoryInit { .. }
            | PrimitiveOpKind::DataDrop { .. }
            | PrimitiveOpKind::TableInit { .. }
            | PrimitiveOpKind::ElemDrop { .. }
    )
}
