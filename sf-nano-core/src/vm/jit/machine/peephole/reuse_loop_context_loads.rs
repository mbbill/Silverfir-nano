//! Reuse invariant runtime-context loads across loop backedges.
//!
//! The block-local load-reuse pass cannot see that a dynamic register still
//! contains the same context field on every incoming edge of a loop. This
//! pass carries that register explicitly as a block parameter when every
//! predecessor already provides the exact same load and nothing can
//! invalidate it along the edge.

use crate::collections;
use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineInstKind,
    MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner, MachineStorageType,
    MachineTerminator, MachineValue, MACHINE_CTX_REG,
};

use super::helpers::{inst_defines, store_may_alias};

#[derive(Clone, Copy)]
struct ContextLoad {
    owner: MachineRegOwner,
    ty: MachineStorageType,
    dst: MachineReg,
    addr: MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
}

pub(super) fn reuse_loop_context_loads(blocks: &mut [MachineBlock], entry: MachineBlockId) {
    for block_index in 0..blocks.len() {
        // The root-to-entry transfer is implicit and cannot supply an edge
        // argument introduced here.
        if blocks[block_index].id == entry {
            continue;
        }
        let Some(load) = blocks[block_index].ops.first().and_then(context_load) else {
            continue;
        };
        if load.addr.base != MACHINE_CTX_REG
            || load.owner != MachineRegOwner::LinearValue
            || load.ty != MachineStorageType::GpWord
            || blocks[block_index]
                .params
                .iter()
                .any(|param| param.reg == load.dst)
        {
            continue;
        }

        let target = blocks[block_index].id;
        // This transform requires a self edge. Reject other blocks before
        // searching the whole function for entry predecessors; context loads
        // at ordinary block entries are much more common than self loops.
        if !terminator_targets(&blocks[block_index].terminator, target) {
            continue;
        }
        let mut predecessors = collections::Vec::new();
        for (predecessor_index, block) in blocks.iter().enumerate() {
            if terminator_targets(&block.terminator, target) {
                predecessors.push(predecessor_index);
            }
        }
        let has_entry_edge = predecessors.iter().any(|&index| index != block_index);
        if !has_entry_edge {
            continue;
        }
        if !predecessors
            .iter()
            .all(|&index| load_available_at_end(&blocks[index], load))
        {
            continue;
        }

        blocks[block_index].ops.remove(0);
        blocks[block_index].params.push(MachineBlockParam {
            reg: load.dst,
            ty: load.ty,
            owner: load.owner,
        });
        for predecessor_index in predecessors {
            append_edge_arg(
                &mut blocks[predecessor_index].terminator,
                target,
                MachineValue::Reg(load.dst),
            );
        }
    }
}

fn context_load(kind: &crate::vm::jit::machine::machine_ir::MachineInst) -> Option<ContextLoad> {
    let MachineInstKind::Load {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    } = kind.kind
    else {
        return None;
    };
    Some(ContextLoad {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    })
}

fn load_available_at_end(block: &MachineBlock, load: ContextLoad) -> bool {
    let Some(definition_index) = block
        .ops
        .iter()
        .rposition(|inst| inst_defines(&inst.kind, load.dst))
    else {
        return false;
    };
    if !same_load(&block.ops[definition_index].kind, load) {
        return false;
    }
    block.ops[definition_index + 1..]
        .iter()
        .all(|inst| preserves_load(&inst.kind, load))
}

fn same_load(kind: &MachineInstKind, load: ContextLoad) -> bool {
    matches!(
        kind,
        MachineInstKind::Load {
            owner,
            ty,
            dst,
            addr,
            width,
            extension,
        } if *owner == load.owner
            && *ty == load.ty
            && *dst == load.dst
            && *addr == load.addr
            && *width == load.width
            && *extension == load.extension
    )
}

fn preserves_load(kind: &MachineInstKind, load: ContextLoad) -> bool {
    if inst_defines(kind, load.dst) {
        return false;
    }
    match kind {
        MachineInstKind::Store { addr, width, .. } => {
            !store_may_alias(load.addr, load.width, *addr, *width)
        }
        MachineInstKind::CallRuntime(_)
        | MachineInstKind::MemoryGrow { .. }
        | MachineInstKind::TableGrow { .. }
        | MachineInstKind::RefFunc { .. }
        | MachineInstKind::RefAsNonNull { .. }
        | MachineInstKind::RefAbsolutize { .. }
        | MachineInstKind::RefEq { .. }
        | MachineInstKind::RefI31 { .. }
        | MachineInstKind::I31GetS { .. }
        | MachineInstKind::I31GetU { .. }
        | MachineInstKind::AnyConvertExtern { .. }
        | MachineInstKind::ExternConvertAny { .. }
        | MachineInstKind::RefTest { .. }
        | MachineInstKind::RefCast { .. }
        | MachineInstKind::StructNew { .. }
        | MachineInstKind::StructNewDefault { .. }
        | MachineInstKind::ArrayNew { .. }
        | MachineInstKind::ArrayNewDefault { .. }
        | MachineInstKind::ArrayNewFixed { .. }
        | MachineInstKind::ArrayNewData { .. }
        | MachineInstKind::ArrayNewElem { .. }
        | MachineInstKind::EhAllocExnRef { .. } => false,
        _ => true,
    }
}

fn terminator_targets(terminator: &MachineTerminator, target: MachineBlockId) -> bool {
    match terminator {
        MachineTerminator::Jump(edge) => edge.target == target,
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => then_edge.target == target || else_edge.target == target,
        MachineTerminator::JumpTable { entries, .. } => {
            entries.iter().any(|edge| edge.target == target)
        }
        MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => false,
    }
}

fn append_edge_arg(terminator: &mut MachineTerminator, target: MachineBlockId, arg: MachineValue) {
    let append = |edge: &mut crate::vm::jit::machine::machine_ir::MachineEdge| {
        if edge.target == target {
            edge.args.push(arg);
        }
    };
    match terminator {
        MachineTerminator::Jump(edge) => append(edge),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            append(then_edge);
            append(else_edge);
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                append(edge);
            }
        }
        MachineTerminator::Call { .. }
        | MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{MachineBranchCond, MachineEdge, MachineInst};

    fn edge(target: u32) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: collections::Vec::new(),
        }
    }

    fn context_pointer_load(dst: MachineReg) -> MachineInst {
        MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst,
                addr: MachineAddr {
                    base: MACHINE_CTX_REG,
                    offset: 16,
                },
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
            },
        }
    }

    #[test]
    fn does_not_reuse_context_load_at_function_entry() {
        let pointer = MachineReg(4);
        let mut blocks = collections::vec![
            MachineBlock {
                id: MachineBlockId(0),
                params: collections::Vec::new(),
                ops: collections::vec![context_pointer_load(pointer)],
                terminator: MachineTerminator::Branch {
                    cond: MachineBranchCond::Value(MachineValue::Imm64(1)),
                    then_edge: edge(0),
                    else_edge: edge(2),
                },
            },
            MachineBlock {
                id: MachineBlockId(1),
                params: collections::Vec::new(),
                ops: collections::vec![context_pointer_load(pointer)],
                terminator: MachineTerminator::Jump(edge(0)),
            },
            MachineBlock {
                id: MachineBlockId(2),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Return,
            },
        ];
        let before = blocks.clone();

        reuse_loop_context_loads(&mut blocks, MachineBlockId(0));

        assert_eq!(blocks, before);
    }
}
