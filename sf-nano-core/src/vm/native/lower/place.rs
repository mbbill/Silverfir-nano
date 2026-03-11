//! Placement onto the native VM ABI.
//!
//! This phase owns:
//! - mapping transient values onto `Tos*`, `Hot*`, `Tmp*`, and `fp[...]`
//! - edge parallel-copy construction
//! - hot-local and frame-local alias tracking
//! - call clobber preservation
//!
//! ISA backends should consume the placed result mechanically.

use alloc::vec::Vec;

use crate::vm::{
    backend::BackendConfig,
    lir::ir::LirValue,
    native::{
        abi::{BlockAbi, NativeAbi, NativeLocation, NativePlace, NativeValue},
        ir::{
            NativeBlock, NativeEdge, NativeInst, NativeInstKind, NativeMove, NativeProgram,
            NativeTerminator,
        },
    },
    plan::PlannedProgram,
};

use super::{
    select::{
        SelectedBlock, SelectedEdge, SelectedInst, SelectedInstKind, SelectedProgram,
        SelectedSource, SelectedTarget, SelectedTerminator,
    },
    state::{PlacedValue, PlacementState},
};

pub(super) fn place_program(
    selected: &SelectedProgram,
    planned: &PlannedProgram,
    backend_config: BackendConfig,
) -> NativeProgram {
    let abi = NativeAbi::from(backend_config);
    let mut state = PlacementState::new(value_count(selected));
    seed_block_params(selected, &mut state);

    let blocks = selected
        .blocks
        .iter()
        .map(|block| {
            let mut placer = BlockPlacer::new(&mut state, abi, block);
            let mut ops = Vec::new();
            for inst in &block.ops {
                placer.place_inst(inst, &mut ops);
            }

            NativeBlock {
                id: block.id,
                entry_abi: BlockAbi {
                    tos_width: block.tos_params.len() as u8,
                },
                ops,
                terminator: placer.place_terminator(&block.terminator),
            }
        })
        .collect();

    NativeProgram {
        entry: selected.entry,
        blocks,
        abi,
        frame: planned.frame,
        spill_slots: state.spill_slots(),
        groups: planned.groups.clone(),
        hot_locals: planned.hot_locals.clone(),
    }
}

struct BlockPlacer<'a> {
    state: &'a mut PlacementState,
    abi: NativeAbi,
    remaining_uses: Vec<u32>,
}

impl<'a> BlockPlacer<'a> {
    fn new(state: &'a mut PlacementState, abi: NativeAbi, block: &SelectedBlock) -> Self {
        Self {
            state,
            abi,
            remaining_uses: count_uses(block),
        }
    }

    fn place_inst(&mut self, inst: &SelectedInst, out: &mut Vec<NativeInst>) {
        match &inst.kind {
            SelectedInstKind::Copy { dst, src } => self.place_copy(*dst, *src, out),
            SelectedInstKind::Leaf { op, args, results } => {
                let args = args
                    .iter()
                    .copied()
                    .map(|value| self.consume_value(value))
                    .collect();
                let result_places = self.allocate_results(results.len());
                for (value, place) in results.iter().copied().zip(result_places.iter().copied()) {
                    self.state.assign(value, place);
                }
                out.push(NativeInst {
                    kind: NativeInstKind::Leaf {
                        op: op.clone(),
                        args,
                        results: result_places,
                    },
                });
            }
            SelectedInstKind::CallExternal {
                func_idx,
                args,
                results,
            } => {
                let args = args
                    .iter()
                    .copied()
                    .map(|value| self.consume_value(value))
                    .collect();
                self.preserve_call_clobbers(out);
                let result_places = self.allocate_results(results.len());
                for (value, place) in results.iter().copied().zip(result_places.iter().copied()) {
                    self.state.assign(value, place);
                }
                out.push(NativeInst {
                    kind: NativeInstKind::CallExternal {
                        func_idx: *func_idx,
                        args,
                        results: result_places,
                    },
                });
            }
            SelectedInstKind::CallLocal {
                callee,
                args,
                results,
            } => {
                let args = args
                    .iter()
                    .copied()
                    .map(|value| self.consume_value(value))
                    .collect();
                self.preserve_call_clobbers(out);
                let result_places = self.allocate_results(results.len());
                for (value, place) in results.iter().copied().zip(result_places.iter().copied()) {
                    self.state.assign(value, place);
                }
                out.push(NativeInst {
                    kind: NativeInstKind::CallLocal {
                        callee: *callee,
                        args,
                        results: result_places,
                    },
                });
            }
            SelectedInstKind::CallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                results,
            } => {
                let index = self.consume_value(*index);
                let args = args
                    .iter()
                    .copied()
                    .map(|value| self.consume_value(value))
                    .collect();
                self.preserve_call_clobbers(out);
                let result_places = self.allocate_results(results.len());
                for (value, place) in results.iter().copied().zip(result_places.iter().copied()) {
                    self.state.assign(value, place);
                }
                out.push(NativeInst {
                    kind: NativeInstKind::CallIndirect {
                        type_idx: *type_idx,
                        table_idx: *table_idx,
                        index,
                        args,
                        results: result_places,
                    },
                });
            }
        }
    }

    fn place_terminator(&mut self, term: &SelectedTerminator) -> NativeTerminator {
        match term {
            SelectedTerminator::Goto(edge) => NativeTerminator::Goto(self.place_edge(edge)),
            SelectedTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => NativeTerminator::Branch {
                cond: self.consume_value(*cond),
                then_edge: self.place_edge(then_edge),
                else_edge: self.place_edge(else_edge),
            },
            SelectedTerminator::BrTable { index, entries } => NativeTerminator::BrTable {
                index: self.consume_value(*index),
                entries: entries.iter().map(|edge| self.place_edge(edge)).collect(),
            },
            SelectedTerminator::Return { values } => NativeTerminator::Return {
                values: values
                    .iter()
                    .copied()
                    .map(|value| self.consume_value(value))
                    .collect(),
            },
            SelectedTerminator::TrapUnreachable => NativeTerminator::TrapUnreachable,
        }
    }

    fn place_copy(&mut self, dst: SelectedTarget, src: SelectedSource, out: &mut Vec<NativeInst>) {
        match dst {
            SelectedTarget::Value(dst_value) => {
                let placed = self.alias_source(src);
                self.state.assign_value(dst_value, placed);
            }
            SelectedTarget::Hot(reg) => {
                let src = self.consume_source(src);
                self.place_store(
                    NativePlace::Location(NativeLocation::Hot(reg)),
                    src,
                    out,
                );
            }
            SelectedTarget::Operand(slot) | SelectedTarget::FrameLocal(slot) => {
                let src = self.consume_source(src);
                self.place_store(NativePlace::Frame(slot), src, out);
            }
        }
    }

    fn place_store(&mut self, dst: NativePlace, src: NativeValue, out: &mut Vec<NativeInst>) {
        if matches!(src, NativeValue::Place(src_place) if src_place == dst) {
            return;
        }

        self.preserve_future_aliases(dst, out);
        out.push(NativeInst {
            kind: NativeInstKind::Move(NativeMove { dst, src }),
        });
    }

    fn place_edge(&mut self, edge: &SelectedEdge) -> NativeEdge {
        let mut copies = Vec::new();
        for (lane, value) in edge.tos.iter().copied().enumerate() {
            let dst = NativePlace::Location(NativeLocation::Tos(lane as u8));
            let src = self.consume_value(value);
            if matches!(src, NativeValue::Place(src_place) if src_place == dst) {
                continue;
            }
            copies.push(NativeMove { dst, src });
        }

        NativeEdge {
            target: edge.target,
            copies,
        }
    }

    fn consume_source(&mut self, source: SelectedSource) -> NativeValue {
        self.alias_source(source).as_native_value()
    }

    fn alias_source(&mut self, source: SelectedSource) -> PlacedValue {
        match source {
            SelectedSource::Value(value) => {
                let placed = self.placed(value);
                self.consume_use(value);
                placed
            }
            SelectedSource::Imm64(value) => PlacedValue::Imm64(value),
            SelectedSource::Hot(reg) => {
                PlacedValue::Place(NativePlace::Location(NativeLocation::Hot(reg)))
            }
            SelectedSource::Operand(slot) | SelectedSource::FrameLocal(slot) => {
                PlacedValue::Place(NativePlace::Frame(slot))
            }
        }
    }

    fn consume_value(&mut self, value: LirValue) -> NativeValue {
        let placed = self.placed(value);
        self.consume_use(value);
        placed.as_native_value()
    }

    fn consume_use(&mut self, value: LirValue) {
        let index = value.0 as usize;
        let remaining = self
            .remaining_uses
            .get_mut(index)
            .unwrap_or_else(|| panic!("native placement use out of range for value {}", value.0));
        assert!(*remaining > 0, "native placement consumed dead value {}", value.0);
        *remaining -= 1;
    }

    fn home(&self, value: LirValue) -> NativePlace {
        self.state
            .home(value)
            .unwrap_or_else(|| panic!("native placement missing home for value {}", value.0))
    }

    fn placed(&self, value: LirValue) -> PlacedValue {
        self.state
            .placed(value)
            .unwrap_or_else(|| panic!("native placement missing home for value {}", value.0))
    }

    fn allocate_results(&mut self, count: usize) -> Vec<NativePlace> {
        let mut reserved = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(place) = self.allocate_free_tmp(&reserved) {
                reserved.push(place);
            } else if let Some(place) = self.allocate_free_tos(&reserved) {
                reserved.push(place);
            } else {
                reserved.push(self.state.snapshot_spill());
            }
        }
        reserved
    }

    fn allocate_free_tmp(&self, reserved: &[NativePlace]) -> Option<NativePlace> {
        (0..self.abi.tmp_register_count).find_map(|reg| {
            let place = NativePlace::Location(NativeLocation::Tmp(reg));
            (!reserved.contains(&place) && !self.place_live_after_current_op(place)).then_some(place)
        })
    }

    fn allocate_free_tos(&self, reserved: &[NativePlace]) -> Option<NativePlace> {
        (0..self.abi.tos_register_count).find_map(|lane| {
            let place = NativePlace::Location(NativeLocation::Tos(lane));
            (!reserved.contains(&place) && !self.place_live_after_current_op(place)).then_some(place)
        })
    }

    fn place_live_after_current_op(&self, place: NativePlace) -> bool {
        self.remaining_uses
            .iter()
            .enumerate()
            .any(|(index, &uses)| uses > 0 && self.state.home(LirValue(index as u32)) == Some(place))
    }

    fn preserve_future_aliases(&mut self, place: NativePlace, out: &mut Vec<NativeInst>) {
        let live_values = self
            .remaining_uses
            .iter()
            .enumerate()
            .filter_map(|(index, &uses)| {
                (uses > 0 && self.state.home(LirValue(index as u32)) == Some(place))
                    .then_some(LirValue(index as u32))
            })
            .collect::<Vec<_>>();

        if live_values.is_empty() {
            return;
        }

        let snapshot = self.state.snapshot_spill();
        out.push(NativeInst {
            kind: NativeInstKind::Move(NativeMove {
                dst: snapshot,
                src: NativeValue::Place(place),
            }),
        });
        for value in live_values {
            self.state.assign(value, snapshot);
        }
    }

    fn preserve_call_clobbers(&mut self, out: &mut Vec<NativeInst>) {
        let mut places = Vec::new();
        for (index, &uses) in self.remaining_uses.iter().enumerate() {
            if uses == 0 {
                continue;
            }
            let Some(place) = self.state.home(LirValue(index as u32)) else {
                continue;
            };
            if clobbered_by_call(place) && !places.contains(&place) {
                places.push(place);
            }
        }

        for place in places {
            self.preserve_future_aliases(place, out);
        }
    }
}

fn clobbered_by_call(place: NativePlace) -> bool {
    matches!(
        place,
        NativePlace::Location(NativeLocation::Hot(_))
            | NativePlace::Location(NativeLocation::Tmp(_))
            | NativePlace::Location(NativeLocation::Tos(_))
    )
}

fn seed_block_params(selected: &SelectedProgram, state: &mut PlacementState) {
    for block in &selected.blocks {
        for (lane, value) in block.tos_params.iter().copied().enumerate() {
            state.assign(value, NativePlace::Location(NativeLocation::Tos(lane as u8)));
        }
    }
}

fn count_uses(block: &SelectedBlock) -> Vec<u32> {
    let mut next = 0usize;
    let mut counts = Vec::new();

    let mut mark = |value: LirValue| {
        let index = value.0 as usize;
        if index >= counts.len() {
            counts.resize(index + 1, 0);
        }
        counts[index] += 1;
        next = next.max(index + 1);
    };

    for inst in &block.ops {
        match &inst.kind {
            SelectedInstKind::Copy { src, .. } => {
                if let SelectedSource::Value(value) = src {
                    mark(*value);
                }
            }
            SelectedInstKind::Leaf { args, .. }
            | SelectedInstKind::CallExternal { args, .. }
            | SelectedInstKind::CallLocal { args, .. } => {
                for value in args {
                    mark(*value);
                }
            }
            SelectedInstKind::CallIndirect { index, args, .. } => {
                mark(*index);
                for value in args {
                    mark(*value);
                }
            }
        }
    }

    match &block.terminator {
        SelectedTerminator::Goto(edge) => {
            for value in &edge.tos {
                mark(*value);
            }
        }
        SelectedTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            mark(*cond);
            for value in &then_edge.tos {
                mark(*value);
            }
            for value in &else_edge.tos {
                mark(*value);
            }
        }
        SelectedTerminator::BrTable { index, entries } => {
            mark(*index);
            for edge in entries {
                for value in &edge.tos {
                    mark(*value);
                }
            }
        }
        SelectedTerminator::Return { values } => {
            for value in values {
                mark(*value);
            }
        }
        SelectedTerminator::TrapUnreachable => {}
    }

    counts.resize(next, 0);
    counts
}

fn value_count(selected: &SelectedProgram) -> usize {
    let mut next = 0usize;
    for block in &selected.blocks {
        for value in &block.tos_params {
            next = next.max(value.0 as usize + 1);
        }
        for inst in &block.ops {
            match &inst.kind {
                SelectedInstKind::Copy { dst, src } => {
                    next = next.max(target_value(dst));
                    next = next.max(source_value(src));
                }
                SelectedInstKind::Leaf { args, results, .. }
                | SelectedInstKind::CallExternal { args, results, .. }
                | SelectedInstKind::CallLocal { args, results, .. } => {
                    for value in args {
                        next = next.max(value.0 as usize + 1);
                    }
                    for value in results {
                        next = next.max(value.0 as usize + 1);
                    }
                }
                SelectedInstKind::CallIndirect {
                    index,
                    args,
                    results,
                    ..
                } => {
                    next = next.max(index.0 as usize + 1);
                    for value in args {
                        next = next.max(value.0 as usize + 1);
                    }
                    for value in results {
                        next = next.max(value.0 as usize + 1);
                    }
                }
            }
        }
        match &block.terminator {
            SelectedTerminator::Goto(edge) => next = next.max(edge_value_count(edge)),
            SelectedTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                next = next.max(cond.0 as usize + 1);
                next = next.max(edge_value_count(then_edge));
                next = next.max(edge_value_count(else_edge));
            }
            SelectedTerminator::BrTable { index, entries } => {
                next = next.max(index.0 as usize + 1);
                for edge in entries {
                    next = next.max(edge_value_count(edge));
                }
            }
            SelectedTerminator::Return { values } => {
                for value in values {
                    next = next.max(value.0 as usize + 1);
                }
            }
            SelectedTerminator::TrapUnreachable => {}
        }
    }
    next
}

fn edge_value_count(edge: &SelectedEdge) -> usize {
    edge.tos
        .iter()
        .map(|value| value.0 as usize + 1)
        .max()
        .unwrap_or(0)
}

fn source_value(source: &SelectedSource) -> usize {
    match source {
        SelectedSource::Value(value) => value.0 as usize + 1,
        _ => 0,
    }
}

fn target_value(target: &SelectedTarget) -> usize {
    match target {
        SelectedTarget::Value(value) => value.0 as usize + 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::place_program;
    use crate::vm::{
        backend::BackendConfig,
        lir::{
            ir::LirValue,
            leaf::LirLeafOp,
            slot::FrameSlot,
        },
        native::{
            abi::{NativeLocation, NativePlace, NativeValue},
            ir::{NativeBlockId, NativeInstKind, NativeTerminator},
        },
        plan::{
            frame::{FrameLayoutPlan, FramePlanner},
            group::GroupPlan,
            PlannedProgram,
        },
    };

    use crate::vm::native::lower::select::{
        SelectedBlock, SelectedInst, SelectedInstKind, SelectedProgram, SelectedSource,
        SelectedTarget, SelectedTerminator,
    };

    fn test_backend(tmp: u8, hot: u8, tos: u8) -> BackendConfig {
        BackendConfig {
            ctx_register_count: 1,
            fp_register_count: 1,
            tmp_register_count: tmp,
            hot_local_count: hot,
            tos_register_count: tos,
        }
    }

    fn test_frame(local_count: u16, operand_count: u16) -> FrameLayoutPlan {
        let planner = FramePlanner::new(local_count);
        let (planner, _) = planner.reserve_operands(operand_count);
        planner.finish()
    }

    fn planned(frame: FrameLayoutPlan) -> PlannedProgram {
        PlannedProgram {
            frame,
            hot_locals: None,
            ops: alloc::vec![],
            groups: GroupPlan::default(),
        }
    }

    #[test]
    fn hot_local_round_trip_is_zero_code() {
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![],
                ops: alloc::vec![
                    SelectedInst {
                        kind: SelectedInstKind::Copy {
                            dst: SelectedTarget::Value(LirValue(0)),
                            src: SelectedSource::Hot(0),
                        },
                    },
                    SelectedInst {
                        kind: SelectedInstKind::Copy {
                            dst: SelectedTarget::Hot(0),
                            src: SelectedSource::Value(LirValue(0)),
                        },
                    },
                ],
                terminator: SelectedTerminator::Return { values: alloc::vec![] },
            }],
        };

        let program = place_program(&selected, &planned(test_frame(1, 0)), test_backend(1, 1, 1));

        assert!(program.blocks[0].ops.is_empty());
    }

    #[test]
    fn call_local_preserves_live_hot_alias() {
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![],
                ops: alloc::vec![
                    SelectedInst {
                        kind: SelectedInstKind::Copy {
                            dst: SelectedTarget::Value(LirValue(0)),
                            src: SelectedSource::Hot(0),
                        },
                    },
                    SelectedInst {
                        kind: SelectedInstKind::CallLocal {
                            callee: 3,
                            args: alloc::vec![],
                            results: alloc::vec![],
                        },
                    },
                ],
                terminator: SelectedTerminator::Return {
                    values: alloc::vec![LirValue(0)],
                },
            }],
        };

        let program = place_program(&selected, &planned(test_frame(1, 0)), test_backend(1, 1, 1));
        let block = &program.blocks[0];

        assert_eq!(program.spill_slots, 1);
        assert_eq!(block.ops.len(), 2);
        assert_eq!(
            block.ops[0].kind,
            NativeInstKind::Move(crate::vm::native::ir::NativeMove {
                dst: NativePlace::Spill(crate::vm::native::abi::NativeSpillSlot(0)),
                src: NativeValue::Place(NativePlace::Location(NativeLocation::Hot(0))),
            })
        );
        assert!(matches!(block.ops[1].kind, NativeInstKind::CallLocal { callee: 3, .. }));
        assert_eq!(
            block.terminator,
            NativeTerminator::Return {
                values: alloc::vec![NativeValue::Place(NativePlace::Spill(
                    crate::vm::native::abi::NativeSpillSlot(0)
                ))],
            }
        );
    }

    #[test]
    fn frame_store_preserves_live_frame_alias() {
        let slot = FrameSlot(0);
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![],
                ops: alloc::vec![
                    SelectedInst {
                        kind: SelectedInstKind::Copy {
                            dst: SelectedTarget::Value(LirValue(0)),
                            src: SelectedSource::FrameLocal(slot),
                        },
                    },
                    SelectedInst {
                        kind: SelectedInstKind::Copy {
                            dst: SelectedTarget::FrameLocal(slot),
                            src: SelectedSource::Hot(0),
                        },
                    },
                ],
                terminator: SelectedTerminator::Return {
                    values: alloc::vec![LirValue(0)],
                },
            }],
        };

        let program = place_program(&selected, &planned(test_frame(1, 0)), test_backend(1, 1, 1));
        let block = &program.blocks[0];

        assert_eq!(program.spill_slots, 1);
        assert_eq!(block.ops.len(), 2);
        assert_eq!(
            block.ops[0].kind,
            NativeInstKind::Move(crate::vm::native::ir::NativeMove {
                dst: NativePlace::Spill(crate::vm::native::abi::NativeSpillSlot(0)),
                src: NativeValue::Place(NativePlace::Frame(slot)),
            })
        );
        assert_eq!(
            block.ops[1].kind,
            NativeInstKind::Move(crate::vm::native::ir::NativeMove {
                dst: NativePlace::Frame(slot),
                src: NativeValue::Place(NativePlace::Location(NativeLocation::Hot(0))),
            })
        );
        assert_eq!(
            block.terminator,
            NativeTerminator::Return {
                values: alloc::vec![NativeValue::Place(NativePlace::Spill(
                    crate::vm::native::abi::NativeSpillSlot(0)
                ))],
            }
        );
    }

    #[test]
    fn immediate_value_flows_through_return_without_move() {
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![],
                ops: alloc::vec![SelectedInst {
                    kind: SelectedInstKind::Copy {
                        dst: SelectedTarget::Value(LirValue(0)),
                        src: SelectedSource::Imm64(9),
                    },
                }],
                terminator: SelectedTerminator::Return {
                    values: alloc::vec![LirValue(0)],
                },
            }],
        };

        let program = place_program(&selected, &planned(test_frame(0, 0)), test_backend(1, 0, 1));

        assert!(program.blocks[0].ops.is_empty());
        assert_eq!(
            program.blocks[0].terminator,
            NativeTerminator::Return {
                values: alloc::vec![NativeValue::Imm64(9)],
            }
        );
    }

    #[test]
    fn results_prefer_tmp_then_tos_then_spill() {
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![],
                ops: alloc::vec![SelectedInst {
                    kind: SelectedInstKind::CallExternal {
                        func_idx: 0,
                        args: alloc::vec![],
                        results: alloc::vec![LirValue(0), LirValue(1), LirValue(2)],
                    },
                }],
                terminator: SelectedTerminator::Return {
                    values: alloc::vec![LirValue(0), LirValue(1), LirValue(2)],
                },
            }],
        };

        let program = place_program(&selected, &planned(test_frame(0, 0)), test_backend(1, 0, 1));
        let block = &program.blocks[0];

        assert_eq!(program.spill_slots, 1);
        assert_eq!(
            block.ops[0].kind,
            NativeInstKind::CallExternal {
                func_idx: 0,
                args: alloc::vec![],
                results: alloc::vec![
                    NativePlace::Location(NativeLocation::Tmp(0)),
                    NativePlace::Location(NativeLocation::Tos(0)),
                    NativePlace::Spill(crate::vm::native::abi::NativeSpillSlot(0)),
                ],
            }
        );
    }

    #[test]
    fn leaf_result_uses_target_independent_places() {
        let selected = SelectedProgram {
            entry: NativeBlockId(0),
            blocks: alloc::vec![SelectedBlock {
                id: NativeBlockId(0),
                tos_params: alloc::vec![LirValue(0), LirValue(1)],
                ops: alloc::vec![SelectedInst {
                    kind: SelectedInstKind::Leaf {
                        op: LirLeafOp::I32Add,
                        args: alloc::vec![LirValue(0), LirValue(1)],
                        results: alloc::vec![LirValue(2)],
                    },
                }],
                terminator: SelectedTerminator::Return {
                    values: alloc::vec![LirValue(2)],
                },
            }],
        };
        let frame = test_frame(0, 0);
        let program = place_program(&selected, &planned(frame), test_backend(1, 0, 1));

        assert!(matches!(
            &program.blocks[0].ops[0].kind,
            NativeInstKind::Leaf {
                results,
                ..
            } if results == &alloc::vec![NativePlace::Location(NativeLocation::Tmp(0))]
        ));
    }
}
