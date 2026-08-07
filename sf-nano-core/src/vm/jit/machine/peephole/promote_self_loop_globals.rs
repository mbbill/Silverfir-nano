//! Keep a simple mutable-global update in a register across a self loop.
//!
//! Lowering deliberately treats globals as memory because imported globals
//! can alias storage owned by another instance.  A loop such as
//! `global.get; i32.sub; global.set; global.get; br_if`, however, has no
//! observer between iterations.  For the narrow, non-trapping shape matched
//! here, carry the just-stored value around the self edge and materialize the
//! final store in the dedicated exit block.
//!
//! This pass is intentionally strict.  It accepts exactly one non-trapping
//! integer update between the load and store, a unique plain-jump preheader,
//! and a dedicated exit that reloads the same global pointer from the same
//! runtime-context slot.  Anything involving calls, traps, multiple blocks,
//! aliases, or ambiguous pointer provenance remains in memory form.

use crate::vm::jit::machine::machine_ir::{
    MachineAddr, MachineBlock, MachineBlockId, MachineBlockParam, MachineInst, MachineInstKind,
    MachineIntBinaryOp, MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner,
    MachineStorageType, MachineTerminator, MachineValue, MACHINE_CTX_REG,
};
use crate::vm::jit::{backend::BackendConfig, runtime::layout::native_runtime_abi_layout};

use super::helpers::{inst_defines, store_may_alias};
use super::hoist_loop_address_bases::{block_index_for_id, LoopGraph};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalarLoad {
    owner: MachineRegOwner,
    ty: MachineStorageType,
    dst: MachineReg,
    addr: MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PointerLoad {
    owner: MachineRegOwner,
    ty: MachineStorageType,
    dst: MachineReg,
    addr: MachineAddr,
    width: MachineMemWidth,
    extension: MachineLoadExtension,
}

pub(super) fn promote_self_loop_globals(
    blocks: &mut [MachineBlock],
    loop_graph: &LoopGraph,
    entry: MachineBlockId,
    config: BackendConfig,
) {
    if config.gp_unit_bytes != 8 {
        return;
    }
    for header in 0..blocks.len() {
        // The root shim supplies entry params outside the explicit CFG. Never
        // infer a seed solely from explicit predecessors of that block.
        if blocks[header].id == entry {
            continue;
        }
        try_promote(blocks, header, loop_graph, entry, config);
    }
}

fn try_promote(
    blocks: &mut [MachineBlock],
    header: usize,
    loop_graph: &LoopGraph,
    entry: MachineBlockId,
    config: BackendConfig,
) {
    if loop_graph.latches_by_header[header].as_slice() != [header] {
        return;
    }

    let predecessors = &loop_graph.predecessors[header];
    if predecessors.len() != 2 || !predecessors.contains(&header) {
        return;
    }
    let Some(preheader) = predecessors.iter().copied().find(|&index| index != header) else {
        return;
    };

    let Some((load, updated)) = loop_update(&blocks[header]) else {
        return;
    };
    if blocks[header]
        .params
        .iter()
        .any(|param| param.reg == load.dst)
    {
        return;
    }

    let Some(exit) = self_loop_exit(blocks, header, load.dst, load.addr.base) else {
        return;
    };
    if blocks[exit].id == entry
        || loop_graph.predecessors[exit].as_slice() != [header]
        || blocks[exit]
            .params
            .iter()
            .any(|param| param.reg == load.dst)
    {
        return;
    }

    let Some((seed, pointer_source)) = preheader_seed(blocks, preheader, header, load, config)
    else {
        return;
    };
    let Some(exit_reload_index) = exit_reload(blocks, exit, load, pointer_source) else {
        return;
    };

    // Every condition above is read-only.  Mutate only after the complete
    // shape and provenance proof has succeeded.
    blocks[header].ops.clear();
    blocks[header].ops.push(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: updated.width,
            op: updated.op,
            dst: load.dst,
            lhs: MachineValue::Reg(load.dst),
            rhs: updated.rhs,
        },
    });
    blocks[header].params.push(MachineBlockParam {
        reg: load.dst,
        ty: load.ty,
        owner: load.owner,
    });

    let MachineTerminator::Jump(preheader_edge) = &mut blocks[preheader].terminator else {
        unreachable!("validated preheader jump");
    };
    preheader_edge.args.push(seed);

    let MachineTerminator::Branch {
        then_edge,
        else_edge,
        ..
    } = &mut blocks[header].terminator
    else {
        unreachable!("validated self-loop branch");
    };
    then_edge.args.push(MachineValue::Reg(load.dst));
    else_edge.args.push(MachineValue::Reg(load.dst));

    blocks[exit].params.push(MachineBlockParam {
        reg: load.dst,
        ty: load.ty,
        owner: load.owner,
    });
    blocks[exit].ops[exit_reload_index] = MachineInst {
        kind: MachineInstKind::Store {
            ty: load.ty,
            addr: load.addr,
            width: load.width,
            src: MachineValue::Reg(load.dst),
        },
    };
}

#[derive(Clone, Copy)]
struct IntegerUpdate {
    width: crate::vm::jit::machine::machine_ir::MachineIntWidth,
    op: MachineIntBinaryOp,
    rhs: MachineValue,
}

fn loop_update(block: &MachineBlock) -> Option<(ScalarLoad, IntegerUpdate)> {
    let [load_inst, update_inst, store_inst] = block.ops.as_slice() else {
        return None;
    };
    let load = scalar_load(load_inst)?;
    let MachineInstKind::IntBinary {
        width,
        op,
        dst,
        lhs,
        rhs,
    } = update_inst.kind
    else {
        return None;
    };
    if dst != load.dst
        || load.dst == load.addr.base
        || lhs != MachineValue::Reg(load.dst)
        || !matches!(rhs, MachineValue::Imm64(_))
        || !non_trapping_integer_op(op)
        || load.owner != MachineRegOwner::LinearValue
        || load.ty != MachineStorageType::GpWord
        || load.width != MachineMemWidth::U64
        || load.extension != MachineLoadExtension::None
        || width != crate::vm::jit::machine::machine_ir::MachineIntWidth::I32
    {
        return None;
    }
    let MachineInstKind::Store {
        ty,
        addr,
        width: store_width,
        src,
    } = store_inst.kind
    else {
        return None;
    };
    if ty != load.ty
        || addr != load.addr
        || store_width != load.width
        || src != MachineValue::Reg(load.dst)
    {
        return None;
    }
    Some((load, IntegerUpdate { width, op, rhs }))
}

fn non_trapping_integer_op(op: MachineIntBinaryOp) -> bool {
    matches!(
        op,
        MachineIntBinaryOp::Add
            | MachineIntBinaryOp::Sub
            | MachineIntBinaryOp::Mul
            | MachineIntBinaryOp::And
            | MachineIntBinaryOp::Or
            | MachineIntBinaryOp::Xor
            | MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr
    )
}

fn self_loop_exit(
    blocks: &[MachineBlock],
    header: usize,
    value: MachineReg,
    address_base: MachineReg,
) -> Option<usize> {
    let MachineTerminator::Branch {
        cond,
        then_edge,
        else_edge,
    } = &blocks[header].terminator
    else {
        return None;
    };
    if *cond
        != crate::vm::jit::machine::machine_ir::MachineBranchCond::Value(MachineValue::Reg(value))
    {
        return None;
    }
    let header_id = blocks[header].id;
    let (self_edge, exit_id) = if then_edge.target == header_id && else_edge.target != header_id {
        (then_edge, else_edge.target)
    } else if else_edge.target == header_id && then_edge.target != header_id {
        (else_edge, then_edge.target)
    } else {
        return None;
    };
    if !edge_preserves_base(&blocks[header], self_edge, address_base) {
        return None;
    }
    block_index_for_id(blocks, exit_id)
}

fn preheader_seed(
    blocks: &[MachineBlock],
    preheader: usize,
    header: usize,
    load: ScalarLoad,
    config: BackendConfig,
) -> Option<(MachineValue, PointerLoad)> {
    let MachineTerminator::Jump(edge) = &blocks[preheader].terminator else {
        return None;
    };
    if edge.target != blocks[header].id
        || !edge_preserves_base(&blocks[header], edge, load.addr.base)
    {
        return None;
    }

    let seed_store_index = blocks[preheader].ops.iter().rposition(|inst| {
        matches!(
            inst.kind,
            MachineInstKind::Store { ty, addr, width, .. }
                if ty == load.ty && addr == load.addr && width == load.width
        )
    })?;
    let MachineInstKind::Store { src: seed, .. } = blocks[preheader].ops[seed_store_index].kind
    else {
        unreachable!("matched seed store");
    };

    let pointer_definition = blocks[preheader].ops[..seed_store_index]
        .iter()
        .rposition(|inst| inst_defines(&inst.kind, load.addr.base))?;
    let pointer_source = pointer_load(&blocks[preheader].ops[pointer_definition], load.addr.base)?;
    if pointer_source.addr.base != MACHINE_CTX_REG
        || pointer_source.owner != MachineRegOwner::LinearValue
        || pointer_source.ty != MachineStorageType::GpWord
        || pointer_source.width != MachineMemWidth::U64
        || pointer_source.extension != MachineLoadExtension::None
        || !is_global_pointer_slot(pointer_source.addr, config)
    {
        return None;
    }

    for inst in &blocks[preheader].ops[seed_store_index + 1..] {
        if inst_defines(&inst.kind, load.addr.base)
            || matches!(seed, MachineValue::Reg(reg) if inst_defines(&inst.kind, reg))
            || !preserves_seed_store(&inst.kind, load)
        {
            return None;
        }
    }
    Some((seed, pointer_source))
}

fn is_global_pointer_slot(addr: MachineAddr, config: BackendConfig) -> bool {
    let globals_base = native_runtime_abi_layout(config.gp_unit_bytes)
        .context
        .globals_ptrs_inline_offset as i32;
    addr.offset >= globals_base
        && (addr.offset - globals_base) % i32::from(config.gp_unit_bytes) == 0
}

fn edge_preserves_base(
    header: &MachineBlock,
    edge: &crate::vm::jit::machine::machine_ir::MachineEdge,
    base: MachineReg,
) -> bool {
    let Some(index) = header.params.iter().position(|param| param.reg == base) else {
        return false;
    };
    edge.args.get(index) == Some(&MachineValue::Reg(base))
}

fn preserves_seed_store(kind: &MachineInstKind, load: ScalarLoad) -> bool {
    match kind {
        MachineInstKind::Store { addr, width, .. } => {
            !store_may_alias(load.addr, load.width, *addr, *width)
        }
        _ => false,
    }
}

fn exit_reload(
    blocks: &[MachineBlock],
    exit: usize,
    load: ScalarLoad,
    pointer_source: PointerLoad,
) -> Option<usize> {
    let [pointer_inst, reload_inst, ..] = blocks[exit].ops.as_slice() else {
        return None;
    };
    if pointer_load(pointer_inst, load.addr.base)? != pointer_source {
        return None;
    }
    let reload = scalar_load(reload_inst)?;
    (reload == load).then_some(1)
}

fn scalar_load(inst: &MachineInst) -> Option<ScalarLoad> {
    let MachineInstKind::Load {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    } = inst.kind
    else {
        return None;
    };
    Some(ScalarLoad {
        owner,
        ty,
        dst,
        addr,
        width,
        extension,
    })
}

fn pointer_load(inst: &MachineInst, dst: MachineReg) -> Option<PointerLoad> {
    let MachineInstKind::Load {
        owner,
        ty,
        dst: actual_dst,
        addr,
        width,
        extension,
    } = inst.kind
    else {
        return None;
    };
    (actual_dst == dst).then_some(PointerLoad {
        owner,
        ty,
        dst: actual_dst,
        addr,
        width,
        extension,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collections;
    use crate::vm::jit::machine::machine_ir::{
        MachineBlockId, MachineBranchCond, MachineEdge, MachineIntWidth, MachineProgram,
        MachineResultSrc, MachineReturnValue, MACHINE_FP_REG,
    };

    fn edge(target: u32, args: &[MachineReg]) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: args.iter().copied().map(MachineValue::Reg).collect(),
        }
    }

    fn counter_program(exit_pointer_delta: i32) -> MachineProgram {
        let value = MachineReg(4);
        let pointer = MachineReg(5);
        let global_pointer_offset = native_runtime_abi_layout(8)
            .context
            .globals_ptrs_inline_offset as i32;
        let global_addr = MachineAddr {
            base: pointer,
            offset: 0,
        };
        let pointer_addr = |offset| MachineAddr {
            base: MACHINE_CTX_REG,
            offset,
        };
        MachineProgram {
            entry: MachineBlockId(0),
            fp_reg_init_widths: collections::Vec::new(),
            blocks: collections::vec![
                MachineBlock {
                    id: MachineBlockId(0),
                    params: collections::Vec::new(),
                    ops: collections::vec![
                        MachineInst {
                            kind: MachineInstKind::Load {
                                owner: MachineRegOwner::LinearValue,
                                ty: MachineStorageType::GpWord,
                                dst: pointer,
                                addr: pointer_addr(global_pointer_offset),
                                width: MachineMemWidth::U64,
                                extension: MachineLoadExtension::None,
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: global_addr,
                                width: MachineMemWidth::U64,
                                src: MachineValue::Reg(value),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: MachineAddr {
                                    base: MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: MachineMemWidth::U64,
                                src: MachineValue::Reg(value),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Jump(edge(1, &[pointer])),
                },
                MachineBlock {
                    id: MachineBlockId(1),
                    params: collections::vec![MachineBlockParam::gp_word(pointer)],
                    ops: collections::vec![
                        MachineInst {
                            kind: MachineInstKind::Load {
                                owner: MachineRegOwner::LinearValue,
                                ty: MachineStorageType::GpWord,
                                dst: value,
                                addr: global_addr,
                                width: MachineMemWidth::U64,
                                extension: MachineLoadExtension::None,
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::IntBinary {
                                width: MachineIntWidth::I32,
                                op: MachineIntBinaryOp::Sub,
                                dst: value,
                                lhs: MachineValue::Reg(value),
                                rhs: MachineValue::Imm64(1),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: global_addr,
                                width: MachineMemWidth::U64,
                                src: MachineValue::Reg(value),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Branch {
                        cond: MachineBranchCond::Value(MachineValue::Reg(value)),
                        then_edge: edge(1, &[pointer]),
                        else_edge: edge(2, &[]),
                    },
                },
                MachineBlock {
                    id: MachineBlockId(2),
                    params: collections::Vec::new(),
                    ops: collections::vec![
                        MachineInst {
                            kind: MachineInstKind::Load {
                                owner: MachineRegOwner::LinearValue,
                                ty: MachineStorageType::GpWord,
                                dst: pointer,
                                addr: pointer_addr(global_pointer_offset + exit_pointer_delta),
                                width: MachineMemWidth::U64,
                                extension: MachineLoadExtension::None,
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Load {
                                owner: MachineRegOwner::LinearValue,
                                ty: MachineStorageType::GpWord,
                                dst: value,
                                addr: global_addr,
                                width: MachineMemWidth::U64,
                                extension: MachineLoadExtension::None,
                            },
                        },
                    ],
                    terminator: MachineTerminator::ReturnScalar {
                        value: MachineReturnValue::ScalarGp {
                            ty: MachineStorageType::GpWord,
                            src: MachineResultSrc::Reg(value),
                        },
                    },
                },
            ],
        }
    }

    fn run_promotion(program: &mut MachineProgram, config: BackendConfig) {
        let entry = program.entry;
        let graph =
            super::super::hoist_loop_address_bases::analyze_loop_graph(&program.blocks, entry);
        promote_self_loop_globals(&mut program.blocks, &graph, entry, config);
    }

    #[test]
    fn promotes_exact_global_counter_loop() {
        let mut program = counter_program(0);
        let config = BackendConfig::new(8, 8, 0, 0);
        run_promotion(&mut program, config);
        program.validate(config).unwrap();

        assert_eq!(program.blocks[1].ops.len(), 1);
        assert!(matches!(
            program.blocks[1].ops[0].kind,
            MachineInstKind::IntBinary {
                op: MachineIntBinaryOp::Sub,
                ..
            }
        ));
        assert_eq!(
            program.blocks[1].params.last().map(|param| param.reg),
            Some(MachineReg(4))
        );
        let MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } = &program.blocks[1].terminator
        else {
            panic!("expected branch");
        };
        assert_eq!(
            then_edge.args.last(),
            Some(&MachineValue::Reg(MachineReg(4)))
        );
        assert_eq!(
            else_edge.args.last(),
            Some(&MachineValue::Reg(MachineReg(4)))
        );
        assert!(matches!(
            program.blocks[2].ops[1].kind,
            MachineInstKind::Store {
                src: MachineValue::Reg(MachineReg(4)),
                ..
            }
        ));
    }

    #[test]
    fn rejects_exit_that_loads_a_different_global_pointer() {
        let mut program = counter_program(8);
        let before = program.clone();
        run_promotion(&mut program, BackendConfig::new(8, 8, 0, 0));
        assert_eq!(program, before);
    }

    #[test]
    fn rejects_non_dedicated_exit() {
        let mut program = counter_program(0);
        program.blocks.push(MachineBlock {
            id: MachineBlockId(3),
            params: collections::Vec::new(),
            ops: collections::Vec::new(),
            terminator: MachineTerminator::Jump(edge(2, &[])),
        });
        let before = program.clone();
        run_promotion(&mut program, BackendConfig::new(8, 8, 0, 0));
        assert_eq!(program, before);
    }

    #[test]
    fn rejects_non_store_after_the_seed() {
        let mut program = counter_program(0);
        program.blocks[0].ops.push(MachineInst {
            kind: MachineInstKind::IntBinary {
                width: MachineIntWidth::I32,
                op: MachineIntBinaryOp::Add,
                dst: MachineReg(6),
                lhs: MachineValue::Imm64(1),
                rhs: MachineValue::Imm64(1),
            },
        });
        let before = program.clone();
        run_promotion(&mut program, BackendConfig::new(8, 8, 0, 0));
        assert_eq!(program, before);
    }

    #[test]
    fn rejects_entry_header_with_unreachable_preheader() {
        let mut program = counter_program(0);
        program.entry = MachineBlockId(1);
        let before = program.clone();
        run_promotion(&mut program, BackendConfig::new(8, 8, 0, 0));

        assert_eq!(program, before);
    }

    #[test]
    fn rejects_entry_as_the_implicit_exit_predecessor() {
        let mut program = counter_program(0);
        program.entry = MachineBlockId(2);
        program.blocks[2].terminator = MachineTerminator::Jump(edge(0, &[]));
        let before = program.clone();
        run_promotion(&mut program, BackendConfig::new(8, 8, 0, 0));

        assert_eq!(program, before);
    }
}
