//! Lower SSA-IR into the interpreter instruction stream.
//!
//! The interpreter still executes a small handler machine with implicit TOS
//! registers, so this pass assigns stable frame homes for SSA values and emits
//! the handler sequence that materializes each SSA-IR block.

use alloc::vec::Vec;

use crate::error::WasmError;
use crate::vm::{
    entities::{FunctionInst, ModuleInst},
    interp::{
        build::InterpreterBuildBundle,
        encoding,
        fast_code::FastCode,
        frame_layout,
        instruction::Instruction,
        resolve::{self, IrOpKind},
    },
    lir::{
        ir::{SsaEdge, SsaInstKind, SsaProgram, SsaTerminator, SsaValue},
        leaf::SsaLeafOp,
        slot::FrameSlot,
        target::SsaTarget,
    },
    plan::{frame::FrameLayoutPlan, hot_local::HotLocalPlan},
};

#[derive(Clone, Debug, Default)]
pub struct InterpreterFinalized {
    pub code: FastCode,
    pub stack_slots_used: usize,
}

pub fn finalize_interpreter(
    bundle: &InterpreterBuildBundle,
    module: &ModuleInst,
) -> Result<InterpreterFinalized, WasmError> {
    bundle.planned.validate(&bundle.semantic, bundle.config)?;
    bundle.lir.validate()?;
    let layout = FinalLayout::new(&bundle.lir, bundle.planned.frame)?;
    let mut emitter = Finalizer::new(bundle, module, layout);
    emitter.emit_program()?;
    let stack_slots_used = emitter.layout.stack_slots_used;
    let code = emitter.finish()?;
    Ok(InterpreterFinalized {
        code,
        stack_slots_used,
    })
}

#[derive(Clone, Debug)]
struct FinalLayout {
    frame: FrameLayoutPlan,
    value_home_base: u16,
    value_count: u16,
    call_base: u16,
    move_temp_slot: u16,
    stack_slots_used: usize,
    hot_local_slots: Vec<Option<u16>>,
}

impl FinalLayout {
    fn new(program: &SsaProgram, frame: FrameLayoutPlan) -> Result<Self, WasmError> {
        let value_count = max_value_index(program).map(|max| max + 1).unwrap_or(0);
        let value_home_base = physical_slot(frame, frame.operands.end())?;
        let value_count_u16 = u16::try_from(value_count)
            .map_err(|_| WasmError::internal("interpreter value-home count exceeds u16".into()))?;
        let call_base = value_home_base
            .checked_add(value_count_u16)
            .ok_or_else(|| WasmError::internal("interpreter frame slots exceed u16".into()))?;
        let move_temp_slot = call_base;
        let call_base = call_base
            .checked_add(1)
            .ok_or_else(|| WasmError::internal("interpreter move-temp slot exceeds u16".into()))?;
        let scratch_width = max_scratch_width(program);
        let stack_slots_used = core::cmp::max(
            frame_layout::operand_stack_base(frame.frame_prefix_size as usize),
            usize::from(call_base) + scratch_width,
        );

        Ok(Self {
            frame,
            value_home_base,
            value_count: value_count_u16,
            call_base,
            move_temp_slot,
            stack_slots_used,
            hot_local_slots: build_hot_local_slots(program.hot_locals.as_ref(), frame)?,
        })
    }

    #[inline]
    fn frame_slot(&self, slot: FrameSlot) -> Result<u16, WasmError> {
        physical_slot(self.frame, slot)
    }

    #[inline]
    fn value_home(&self, value: SsaValue) -> Result<u16, WasmError> {
        let offset = u16::try_from(value.0)
            .map_err(|_| WasmError::internal("SSA-IR value id exceeds u16".into()))?;
        if offset >= self.value_count {
            return Err(WasmError::internal(
                "SSA-IR value id exceeds finalized home-slot range".into(),
            ));
        }
        self.value_home_base
            .checked_add(offset)
            .ok_or_else(|| WasmError::internal("interpreter value-home slot exceeds u16".into()))
    }

    #[inline]
    fn hot_local_slot(&self, reg: u8) -> Result<u16, WasmError> {
        self.hot_local_slots
            .get(reg as usize)
            .copied()
            .flatten()
            .ok_or_else(|| WasmError::internal("missing hot-local mapping in interpreter".into()))
    }
}

#[derive(Clone, Copy, Debug)]
enum DirectTarget {
    Block(SsaTarget),
    Index(usize),
}

#[derive(Clone, Copy, Debug)]
struct DirectPatch {
    inst_index: usize,
    target: DirectTarget,
}

#[derive(Clone, Copy, Debug)]
struct BrTablePatch {
    data_index: usize,
    source_index: usize,
    target: DirectTarget,
}

struct Finalizer<'a> {
    bundle: &'a InterpreterBuildBundle,
    module: &'a ModuleInst,
    layout: FinalLayout,
    instructions: Vec<Instruction>,
    transient_depth: usize,
    block_starts: Vec<Option<usize>>,
    direct_patches: Vec<DirectPatch>,
    br_table_patches: Vec<BrTablePatch>,
}

impl<'a> Finalizer<'a> {
    fn new(
        bundle: &'a InterpreterBuildBundle,
        module: &'a ModuleInst,
        layout: FinalLayout,
    ) -> Self {
        Self {
            bundle,
            module,
            layout,
            instructions: Vec::new(),
            transient_depth: 0,
            block_starts: alloc::vec![None; bundle.lir.blocks.len()],
            direct_patches: Vec::new(),
            br_table_patches: Vec::new(),
        }
    }

    fn emit_program(&mut self) -> Result<(), WasmError> {
        let entry = self.bundle.lir.entry;
        if self.bundle.lir.blocks.get(entry.as_usize()).is_none()
            && !self.bundle.lir.blocks.is_empty()
        {
            return Err(WasmError::internal(
                "interpreter finalizer entry block is out of range".into(),
            ));
        }

        let mut visited = alloc::vec![false; self.bundle.lir.blocks.len()];
        let mut pending = alloc::vec![entry];

        while let Some(target) = pending.pop() {
            let index = target.as_usize();
            if visited.get(index).copied().unwrap_or(false) {
                continue;
            }
            let Some(block) = self.bundle.lir.blocks.get(index).cloned() else {
                return Err(WasmError::internal(
                    "reachable interpreter block is out of range".into(),
                ));
            };

            visited[index] = true;
            self.emit_block(&block)?;

            for successor in block_successors(&block.terminator).into_iter().rev() {
                if successor.as_usize() < visited.len() && !visited[successor.as_usize()] {
                    pending.push(successor);
                }
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<FastCode, WasmError> {
        let term = Instruction::new_handler_only(self.term_handler());
        self.instructions.push(term);
        self.instructions.push(term);

        self.patch_br_tables()?;

        let instructions = core::mem::take(&mut self.instructions);
        let mut code = FastCode::from_instructions(instructions);
        self.patch_direct_targets(&mut code.instructions)?;
        Ok(code)
    }

    fn emit_block(
        &mut self,
        block: &crate::vm::middle::ssa_ir::ir::SsaBlock,
    ) -> Result<(), WasmError> {
        self.block_starts[block.id.as_usize()] = Some(self.instructions.len());

        for op in &block.ops {
            self.emit_inst_kind(&op.kind)?;
        }

        self.emit_terminator(&block.terminator)
    }

    fn emit_inst_kind(&mut self, kind: &SsaInstKind) -> Result<(), WasmError> {
        match kind {
            SsaInstKind::Leaf { op, args, results } => self.emit_leaf(op, args, results),
            SsaInstKind::WriteOperandSlot { slot, src } => self.emit_copy(
                self.layout.value_home(*src)?,
                self.layout.frame_slot(*slot)?,
            ),
            SsaInstKind::ReadOperandSlot { slot, dst } => self.emit_copy(
                self.layout.frame_slot(*slot)?,
                self.layout.value_home(*dst)?,
            ),
            SsaInstKind::ReadHotLocal { reg, dst } => self.emit_copy(
                self.layout.hot_local_slot(*reg)?,
                self.layout.value_home(*dst)?,
            ),
            SsaInstKind::WriteHotLocal { reg, src } => self.emit_copy(
                self.layout.value_home(*src)?,
                self.layout.hot_local_slot(*reg)?,
            ),
            SsaInstKind::ReadFrameLocal { frame_slot, dst } => self.emit_copy(
                self.layout.frame_slot(*frame_slot)?,
                self.layout.value_home(*dst)?,
            ),
            SsaInstKind::WriteFrameLocal { frame_slot, src } => self.emit_copy(
                self.layout.value_home(*src)?,
                self.layout.frame_slot(*frame_slot)?,
            ),
            SsaInstKind::CallExternal {
                func_idx,
                args,
                results,
            } => self.emit_call_external(*func_idx, args, results),
            SsaInstKind::CallInternal {
                callee,
                args,
                results,
            } => self.emit_call_internal(*callee, args, results),
            SsaInstKind::CallIndirect {
                type_idx,
                table_idx,
                index,
                args,
                results,
            } => self.emit_call_indirect(*type_idx, *table_idx, *index, args, results),
        }
    }

    fn emit_leaf(
        &mut self,
        op: &SsaLeafOp,
        args: &[SsaValue],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        for arg in args {
            self.emit_local_get(self.layout.value_home(*arg)?)?;
        }

        let (imm0, imm1, imm2) = encode_leaf(op);
        let kind = IrOpKind::from(op);
        let variant = resolve::variant_for_leaf(op, self.transient_depth);
        self.push_instruction(kind, variant, imm0, imm1, imm2);
        let (pop, push) = resolve::leaf_stack_effect(op);
        self.transient_depth = self
            .transient_depth
            .saturating_sub(pop as usize)
            .saturating_add(push as usize);

        match results {
            [] => Ok(()),
            [result] => self.emit_local_set(self.layout.value_home(*result)?),
            _ => Err(WasmError::internal(
                "interpreter leaf lowering only supports single-result leaf ops".into(),
            )),
        }
    }

    fn emit_call_external(
        &mut self,
        func_idx: u32,
        args: &[SsaValue],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        self.materialize_call_args(args)?;
        let delta = self.layout.call_base;
        let (imm0, imm1, imm2) = encoding::call_external::encode(func_idx as u64, delta);
        self.push_instruction(
            IrOpKind::CallExternal { func_idx, delta },
            0,
            imm0,
            imm1,
            imm2,
        );
        self.capture_call_results(results)
    }

    fn emit_call_internal(
        &mut self,
        callee: u32,
        args: &[SsaValue],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        self.materialize_call_args(args)?;
        let delta = self.layout.call_base;
        let callee_ptr = module_function_ptr(self.module, callee);
        let (imm0, imm1, imm2) = encoding::call_internal::encode(callee_ptr, delta);
        self.push_instruction(
            IrOpKind::CallInternal {
                callee: callee_ptr,
                delta,
            },
            0,
            imm0,
            imm1,
            imm2,
        );
        self.capture_call_results(results)
    }

    fn emit_call_indirect(
        &mut self,
        type_idx: u32,
        table_idx: u32,
        index: SsaValue,
        args: &[SsaValue],
        results: &[SsaValue],
    ) -> Result<(), WasmError> {
        self.materialize_call_args(args)?;
        let index_slot =
            self.layout
                .call_base
                .checked_add(u16::try_from(args.len()).map_err(|_| {
                    WasmError::internal("call_indirect arg count exceeds u16".into())
                })?)
                .ok_or_else(|| WasmError::internal("call_indirect scratch slot overflow".into()))?;
        self.emit_copy(self.layout.value_home(index)?, index_slot)?;

        let delta = self.layout.call_base;
        let (imm0, imm1, imm2) =
            encoding::call_indirect::encode(type_idx as u64, table_idx as u64, delta, index_slot);
        self.push_instruction(
            IrOpKind::CallIndirect {
                type_idx,
                table_idx,
                delta,
                index_slot,
            },
            0,
            imm0,
            imm1,
            imm2,
        );
        self.capture_call_results(results)
    }

    fn materialize_call_args(&mut self, args: &[SsaValue]) -> Result<(), WasmError> {
        for (index, value) in args.iter().enumerate() {
            let dst = self
                .layout
                .call_base
                .checked_add(
                    u16::try_from(index)
                        .map_err(|_| WasmError::internal("call arg count exceeds u16".into()))?,
                )
                .ok_or_else(|| WasmError::internal("call arg slot overflow".into()))?;
            self.emit_copy(self.layout.value_home(*value)?, dst)?;
        }
        Ok(())
    }

    fn capture_call_results(&mut self, results: &[SsaValue]) -> Result<(), WasmError> {
        for (index, value) in results.iter().enumerate() {
            let src = self
                .layout
                .call_base
                .checked_add(
                    u16::try_from(index)
                        .map_err(|_| WasmError::internal("call result count exceeds u16".into()))?,
                )
                .ok_or_else(|| WasmError::internal("call result slot overflow".into()))?;
            self.emit_copy(src, self.layout.value_home(*value)?)?;
        }
        Ok(())
    }

    fn emit_terminator(&mut self, term: &SsaTerminator) -> Result<(), WasmError> {
        match term {
            SsaTerminator::Goto(edge) => {
                self.emit_edge_moves(edge)?;
                self.emit_br_to_block(edge.target);
                Ok(())
            }
            SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                self.emit_local_get(self.layout.value_home(*cond)?)?;
                let (imm0, imm1, imm2) = encoding::if_::encode(0);
                let if_index = self.push_instruction(
                    IrOpKind::If,
                    resolve::variant_for_depth(self.transient_depth),
                    imm0,
                    imm1,
                    imm2,
                );
                self.transient_depth = self.transient_depth.saturating_sub(1);
                self.emit_edge_moves(then_edge)?;
                self.emit_br_to_block(then_edge.target);
                let else_index = self.instructions.len();
                self.direct_patches.push(DirectPatch {
                    inst_index: if_index,
                    target: DirectTarget::Index(else_index),
                });
                self.emit_edge_moves(else_edge)?;
                self.emit_br_to_block(else_edge.target);
                Ok(())
            }
            SsaTerminator::BrTable { index, entries } => {
                self.emit_local_get(self.layout.value_home(*index)?)?;
                let (imm0, imm1, imm2) =
                    encoding::br_table::encode(entries.len() as u64, entries.len() as u64);
                let br_table_index = self.push_instruction(
                    IrOpKind::BrTable {
                        entry_count: entries.len() as u32,
                        data_slot_count: entries.len() as u32,
                    },
                    resolve::variant_for_depth(self.transient_depth),
                    imm0,
                    imm1,
                    imm2,
                );
                self.transient_depth = self.transient_depth.saturating_sub(1);

                let first_data = self.instructions.len();
                for _ in entries {
                    self.instructions
                        .push(Instruction::new_handler_only(self.data_handler()));
                }

                for (entry_index, entry) in entries.iter().enumerate() {
                    let stub_index = self.instructions.len();
                    let data_index = first_data + entry_index;
                    self.br_table_patches.push(BrTablePatch {
                        data_index,
                        source_index: br_table_index,
                        target: DirectTarget::Index(stub_index),
                    });
                    self.emit_edge_moves(entry)?;
                    self.emit_br_to_block(entry.target);
                }
                Ok(())
            }
            SsaTerminator::Return { values } => {
                for (index, value) in values.iter().enumerate() {
                    let dst =
                        self.layout
                            .call_base
                            .checked_add(u16::try_from(index).map_err(|_| {
                                WasmError::internal("return arity exceeds u16".into())
                            })?)
                            .ok_or_else(|| WasmError::internal("return scratch overflow".into()))?;
                    self.emit_copy(self.layout.value_home(*value)?, dst)?;
                }

                let frame_size = self.bundle.planned.frame.frame_prefix_size;
                match values.len() {
                    0 => {
                        let (imm0, imm1, imm2) = encoding::return_void::encode(frame_size);
                        self.push_instruction(
                            IrOpKind::ReturnVoid { frame_size },
                            0,
                            imm0,
                            imm1,
                            imm2,
                        );
                    }
                    1 => {
                        let (imm0, imm1, imm2) =
                            encoding::return_one::encode(frame_size, self.layout.call_base);
                        self.push_instruction(
                            IrOpKind::ReturnOne {
                                frame_size,
                                result_slot: self.layout.call_base,
                            },
                            0,
                            imm0,
                            imm1,
                            imm2,
                        );
                    }
                    arity => {
                        let arity = u16::try_from(arity)
                            .map_err(|_| WasmError::internal("return arity exceeds u16".into()))?;
                        let (imm0, imm1, imm2) =
                            encoding::r#return::encode(arity, frame_size, self.layout.call_base);
                        self.push_instruction(
                            IrOpKind::Return {
                                arity,
                                frame_size,
                                result_slot: self.layout.call_base,
                            },
                            0,
                            imm0,
                            imm1,
                            imm2,
                        );
                    }
                }
                Ok(())
            }
            SsaTerminator::TrapUnreachable => {
                self.push_instruction(IrOpKind::Unreachable, 0, 0, 0, 0);
                Ok(())
            }
        }
    }

    fn emit_edge_moves(&mut self, edge: &SsaEdge) -> Result<(), WasmError> {
        let target = self
            .bundle
            .lir
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| WasmError::internal("edge target block out of range".into()))?;
        if edge.tos.len() != target.params.tos.len() {
            return Err(WasmError::internal(
                alloc::format!(
                    "edge TOS arity does not match target block params: target=b{} edge_tos={} target_params={}",
                    edge.target.as_usize(),
                    edge.tos.len(),
                    target.params.tos.len(),
                ),
            ));
        }
        let mut pending = Vec::with_capacity(edge.tos.len());
        for (src, dst) in edge.tos.iter().zip(target.params.tos.iter()) {
            let src = self.layout.value_home(*src)?;
            let dst = self.layout.value_home(*dst)?;
            if src != dst {
                pending.push((src, dst));
            }
        }
        self.emit_parallel_copies(&pending)?;
        Ok(())
    }

    fn emit_parallel_copies(&mut self, pending: &[(u16, u16)]) -> Result<(), WasmError> {
        for (src, dst) in schedule_parallel_copies(pending, self.layout.move_temp_slot) {
            self.emit_copy(src, dst)?;
        }
        Ok(())
    }

    fn emit_copy(&mut self, src: u16, dst: u16) -> Result<(), WasmError> {
        if src == dst {
            return Ok(());
        }
        self.emit_local_get(src)?;
        self.emit_local_set(dst)
    }

    fn emit_local_get(&mut self, slot: u16) -> Result<(), WasmError> {
        let (imm0, imm1, imm2) = encoding::local_get::encode(slot);
        self.push_instruction(
            IrOpKind::LocalGetFrame { idx: slot },
            resolve::variant_for_depth(self.transient_depth + 1),
            imm0,
            imm1,
            imm2,
        );
        self.transient_depth += 1;
        Ok(())
    }

    fn emit_local_set(&mut self, slot: u16) -> Result<(), WasmError> {
        let (imm0, imm1, imm2) = encoding::local_set::encode(slot);
        self.push_instruction(
            IrOpKind::LocalSetFrame { idx: slot },
            resolve::variant_for_depth(self.transient_depth),
            imm0,
            imm1,
            imm2,
        );
        self.transient_depth = self.transient_depth.saturating_sub(1);
        Ok(())
    }

    fn emit_br_to_block(&mut self, target: SsaTarget) {
        let (imm0, imm1, imm2) = encoding::br::encode(0, 0, 0, 0);
        let inst_index =
            self.push_instruction(IrOpKind::Br { has_fixup: false }, 0, imm0, imm1, imm2);
        self.direct_patches.push(DirectPatch {
            inst_index,
            target: DirectTarget::Block(target),
        });
    }

    #[inline]
    fn push_instruction(
        &mut self,
        kind: IrOpKind,
        variant: u8,
        imm0: u64,
        imm1: u64,
        imm2: u64,
    ) -> usize {
        let handler = resolve::resolve_handler_for_kind(&kind, variant);
        let index = self.instructions.len();
        self.instructions
            .push(Instruction::new(handler, imm0, imm1, imm2));
        index
    }

    #[inline]
    fn term_handler(&self) -> crate::vm::interp::handlers::OpHandler {
        resolve::resolve_handler_for_kind(&IrOpKind::Term, 0)
    }

    #[inline]
    fn data_handler(&self) -> crate::vm::interp::handlers::OpHandler {
        resolve::resolve_handler_for_kind(
            &IrOpKind::Data {
                imm0: 0,
                imm1: 0,
                imm2: 0,
            },
            0,
        )
    }

    fn patch_br_tables(&mut self) -> Result<(), WasmError> {
        let data_handler = self.data_handler();
        for patch in &self.br_table_patches {
            let target_index = self.resolve_target_index(patch.target)?;
            let rel = target_index as i32 - patch.source_index as i32;
            let inst = self
                .instructions
                .get_mut(patch.data_index)
                .ok_or_else(|| WasmError::internal("br_table data index out of range".into()))?;
            inst.handler = data_handler;
            inst.imm0 = rel as i64 as u64;
            inst.imm1 = 0;
            inst.imm2 = 0;
        }
        Ok(())
    }

    fn patch_direct_targets(&self, instructions: &mut [Instruction]) -> Result<(), WasmError> {
        let base = instructions.as_ptr() as usize;
        for patch in &self.direct_patches {
            let target_index = self.resolve_target_index(patch.target)?;
            let inst = instructions
                .get_mut(patch.inst_index)
                .ok_or_else(|| WasmError::internal("branch patch index out of range".into()))?;
            inst.imm0 = (base + target_index * core::mem::size_of::<Instruction>()) as u64;
        }
        Ok(())
    }

    fn resolve_target_index(&self, target: DirectTarget) -> Result<usize, WasmError> {
        match target {
            DirectTarget::Index(index) => Ok(index),
            DirectTarget::Block(block) => self
                .block_starts
                .get(block.as_usize())
                .and_then(|index| *index)
                .ok_or_else(|| WasmError::internal("missing block start during patching".into())),
        }
    }
}

fn max_value_index(program: &SsaProgram) -> Option<u32> {
    let mut max = None;
    for block in &program.blocks {
        for value in &block.params.tos {
            update_max(&mut max, *value);
        }
        for op in &block.ops {
            match &op.kind {
                SsaInstKind::Leaf { args, results, .. } => {
                    for value in args.iter().chain(results.iter()) {
                        update_max(&mut max, *value);
                    }
                }
                SsaInstKind::WriteOperandSlot { src, .. }
                | SsaInstKind::WriteHotLocal { src, .. }
                | SsaInstKind::WriteFrameLocal { src, .. } => update_max(&mut max, *src),
                SsaInstKind::ReadOperandSlot { dst, .. }
                | SsaInstKind::ReadHotLocal { dst, .. }
                | SsaInstKind::ReadFrameLocal { dst, .. } => update_max(&mut max, *dst),
                SsaInstKind::CallExternal { args, results, .. }
                | SsaInstKind::CallInternal { args, results, .. } => {
                    for value in args.iter().chain(results.iter()) {
                        update_max(&mut max, *value);
                    }
                }
                SsaInstKind::CallIndirect {
                    index,
                    args,
                    results,
                    ..
                } => {
                    update_max(&mut max, *index);
                    for value in args.iter().chain(results.iter()) {
                        update_max(&mut max, *value);
                    }
                }
            }
        }
        match &block.terminator {
            SsaTerminator::Goto(edge) => {
                for value in &edge.tos {
                    update_max(&mut max, *value);
                }
            }
            SsaTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                update_max(&mut max, *cond);
                for value in then_edge.tos.iter().chain(else_edge.tos.iter()) {
                    update_max(&mut max, *value);
                }
            }
            SsaTerminator::BrTable { index, entries } => {
                update_max(&mut max, *index);
                for edge in entries {
                    for value in &edge.tos {
                        update_max(&mut max, *value);
                    }
                }
            }
            SsaTerminator::Return { values } => {
                for value in values {
                    update_max(&mut max, *value);
                }
            }
            SsaTerminator::TrapUnreachable => {}
        }
    }
    max
}

fn block_successors(term: &SsaTerminator) -> Vec<SsaTarget> {
    match term {
        SsaTerminator::Goto(edge) => alloc::vec![edge.target],
        SsaTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => alloc::vec![then_edge.target, else_edge.target],
        SsaTerminator::BrTable { entries, .. } => {
            entries.iter().map(|entry| entry.target).collect()
        }
        SsaTerminator::Return { .. } | SsaTerminator::TrapUnreachable => Vec::new(),
    }
}

fn max_scratch_width(program: &SsaProgram) -> usize {
    let mut width = 0usize;
    for block in &program.blocks {
        for op in &block.ops {
            match &op.kind {
                SsaInstKind::CallExternal { args, results, .. }
                | SsaInstKind::CallInternal { args, results, .. } => {
                    width = width.max(args.len().max(results.len()));
                }
                SsaInstKind::CallIndirect { args, results, .. } => {
                    width = width.max((args.len() + 1).max(results.len()));
                }
                _ => {}
            }
        }
        if let SsaTerminator::Return { values } = &block.terminator {
            width = width.max(values.len());
        }
    }
    width
}

#[inline]
fn update_max(max: &mut Option<u32>, value: SsaValue) {
    *max = Some(max.map_or(value.0, |current| current.max(value.0)));
}

fn build_hot_local_slots(
    hot_locals: Option<&HotLocalPlan>,
    frame: FrameLayoutPlan,
) -> Result<Vec<Option<u16>>, WasmError> {
    let Some(hot_locals) = hot_locals else {
        return Ok(Vec::new());
    };

    let mut slots = Vec::with_capacity(hot_locals.slot_count());
    for mapped in hot_locals.mapping() {
        let slot = mapped
            .map(|local_idx| physical_slot(frame, frame.local_slot(local_idx as u16)))
            .transpose()?;
        slots.push(slot);
    }
    Ok(slots)
}

fn schedule_parallel_copies(pending: &[(u16, u16)], temp: u16) -> Vec<(u16, u16)> {
    let mut pending = pending.to_vec();
    let mut out = Vec::with_capacity(pending.len().saturating_add(1));

    while !pending.is_empty() {
        let mut progressed = false;
        let mut index = 0usize;
        while index < pending.len() {
            let (_, dst) = pending[index];
            let read_later = pending
                .iter()
                .enumerate()
                .any(|(other_index, (other_src, _))| other_index != index && *other_src == dst);
            if !read_later {
                out.push(pending.remove(index));
                progressed = true;
            } else {
                index += 1;
            }
        }

        if progressed {
            continue;
        }

        let (src, dst) = pending.remove(0);
        out.push((src, temp));
        for (pending_src, _) in &mut pending {
            if *pending_src == src {
                *pending_src = temp;
            }
        }
        pending.push((temp, dst));
    }

    out
}

fn encode_leaf(kind: &SsaLeafOp) -> (u64, u64, u64) {
    match kind {
        SsaLeafOp::I32Const { value } => encoding::r#const::encode(*value as u64),
        SsaLeafOp::I64Const { value } => encoding::r#const::encode(*value),
        SsaLeafOp::F32Const { value } => encoding::r#const::encode(*value as u64),
        SsaLeafOp::F64Const { value } => encoding::r#const::encode(*value),

        SsaLeafOp::I32Load { offset, memidx }
        | SsaLeafOp::I64Load { offset, memidx }
        | SsaLeafOp::F32Load { offset, memidx }
        | SsaLeafOp::F64Load { offset, memidx }
        | SsaLeafOp::I32Load8S { offset, memidx }
        | SsaLeafOp::I32Load8U { offset, memidx }
        | SsaLeafOp::I32Load16S { offset, memidx }
        | SsaLeafOp::I32Load16U { offset, memidx }
        | SsaLeafOp::I64Load8S { offset, memidx }
        | SsaLeafOp::I64Load8U { offset, memidx }
        | SsaLeafOp::I64Load16S { offset, memidx }
        | SsaLeafOp::I64Load16U { offset, memidx }
        | SsaLeafOp::I64Load32S { offset, memidx }
        | SsaLeafOp::I64Load32U { offset, memidx } => encoding::load::encode(*offset, *memidx),

        SsaLeafOp::I32Store { offset, memidx }
        | SsaLeafOp::I64Store { offset, memidx }
        | SsaLeafOp::F32Store { offset, memidx }
        | SsaLeafOp::F64Store { offset, memidx }
        | SsaLeafOp::I32Store8 { offset, memidx }
        | SsaLeafOp::I32Store16 { offset, memidx }
        | SsaLeafOp::I64Store8 { offset, memidx }
        | SsaLeafOp::I64Store16 { offset, memidx }
        | SsaLeafOp::I64Store32 { offset, memidx } => encoding::store::encode(*offset, *memidx),

        SsaLeafOp::GlobalGet { idx } | SsaLeafOp::GlobalSet { idx } => {
            encoding::global::encode(*idx as u64)
        }
        SsaLeafOp::MemorySize { mem_idx } => encoding::memory_size::encode(*mem_idx as u64),
        SsaLeafOp::MemoryGrow { mem_idx } => encoding::memory_grow::encode(*mem_idx as u64),
        SsaLeafOp::MemoryFill { imm0, imm1 }
        | SsaLeafOp::MemoryCopy { imm0, imm1 }
        | SsaLeafOp::MemoryInit { imm0, imm1 }
        | SsaLeafOp::TableFill { imm0, imm1 }
        | SsaLeafOp::TableCopy { imm0, imm1 }
        | SsaLeafOp::TableInit { imm0, imm1 } => {
            encoding::ternary::encode(*imm0 as u64, *imm1 as u64)
        }
        SsaLeafOp::DataDrop { data_idx } => encoding::drop_op::encode(*data_idx as u64),
        SsaLeafOp::TableGet { table_idx } => encoding::table_get::encode(*table_idx as u64),
        SsaLeafOp::TableSet { table_idx } => encoding::table_set::encode(*table_idx as u64),
        SsaLeafOp::TableSize { table_idx } => encoding::table_size::encode(*table_idx as u64),
        SsaLeafOp::TableGrow { table_idx } => encoding::table_grow::encode(*table_idx as u64),
        SsaLeafOp::ElemDrop { elem_idx } => encoding::drop_op::encode(*elem_idx as u64),
        SsaLeafOp::RefFunc { func_idx } => encoding::ref_func::encode(*func_idx as u64),

        _ => (0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::schedule_parallel_copies;

    fn verify_parallel_copies(original: &[(u16, u16)], scheduled: &[(u16, u16)], temp: u16) {
        let max_slot = original
            .iter()
            .chain(scheduled.iter())
            .flat_map(|(src, dst)| [*src as usize, *dst as usize])
            .chain(core::iter::once(temp as usize))
            .max()
            .unwrap_or(0);
        let mut values = (0..=max_slot as u16).collect::<alloc::vec::Vec<_>>();

        for (src, dst) in scheduled {
            values[*dst as usize] = values[*src as usize];
        }

        for (src, dst) in original {
            assert_eq!(values[*dst as usize], *src);
        }
    }

    #[test]
    fn schedules_acyclic_parallel_copies_without_temp() {
        let original = [(1, 3), (2, 4)];
        let scheduled = schedule_parallel_copies(&original, 9);

        assert_eq!(scheduled.len(), 2);
        verify_parallel_copies(&original, &scheduled, 9);
    }

    #[test]
    fn breaks_copy_cycles_with_temp_slot() {
        let original = [(1, 2), (2, 3), (3, 1)];
        let scheduled = schedule_parallel_copies(&original, 9);

        assert!(scheduled.iter().any(|copy| *copy == (1, 9)));
        verify_parallel_copies(&original, &scheduled, 9);
    }
}

fn module_function_ptr(module: &ModuleInst, func_idx: u32) -> u64 {
    module
        .functions
        .get(func_idx as usize)
        .map(|func| func as *const FunctionInst as u64)
        .unwrap_or(0)
}

fn physical_slot(frame: FrameLayoutPlan, slot: FrameSlot) -> Result<u16, WasmError> {
    if slot.0 < frame.operand_base {
        Ok(slot.0)
    } else {
        let operand_base = u16::try_from(frame_layout::operand_stack_base(
            frame.frame_prefix_size as usize,
        ))
        .map_err(|_| WasmError::internal("operand stack base exceeds u16".into()))?;
        operand_base
            .checked_add(slot.0 - frame.operand_base)
            .ok_or_else(|| WasmError::internal("physical interpreter slot exceeds u16".into()))
    }
}
