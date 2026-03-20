use alloc::{collections::VecDeque, string::String, vec, vec::Vec};

use crate::error::WasmError;

use super::{
    machine_ptr_width, machine_word_int_width, MachineBlockId, MachineBlockParam,
    MachineBranchCond, MachineCompareKind, MachineConvertOp, MachineEdge, MachineFloatWidth,
    MachineFunction, MachineInst, MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp,
    MachineIntWidth, MachineMemWidth, MachineModule, MachineProgram, MachineReg, MachineSign,
    MachineStorageType, MachineTerminator, MachineValue, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT,
    MACHINE_FP_REG, MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MachineStorageFact {
    Undefined,
    Known(MachineStorageType),
    Conflict,
}

impl MachineStorageFact {
    #[inline]
    pub(crate) const fn known(self) -> Option<MachineStorageType> {
        match self {
            Self::Known(ty) => Some(ty),
            Self::Undefined | Self::Conflict => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MachineStorageFlow {
    pub(crate) block_entry: Vec<Option<Vec<MachineStorageFact>>>,
    pub(crate) block_exit: Vec<Option<Vec<MachineStorageFact>>>,
}

impl MachineModule {
    /// Rewrite true GP i64 values into paired GP-word operations for 32-bit
    /// targets.
    pub fn legalize(&mut self, gp_reg_width: u8) -> Result<(), WasmError> {
        if gp_reg_width != 4 {
            return Ok(());
        }

        for MachineFunction { id, program } in &mut self.functions {
            let flow = program.analyze_32bit_storage_flow().map_err(|err| {
                WasmError::internal(alloc::format!(
                    "machine function {} failed 32-bit storage-flow validation: {}",
                    id.0,
                    err
                ))
            })?;
            let Some(mut legalizer) = Legalize32BitGpI64::new(program, &flow)? else {
                continue;
            };
            legalizer.rewrite_program(program, &flow).map_err(|err| {
                WasmError::internal(alloc::format!(
                    "machine function {} failed 32-bit i64 legalization: {}",
                    id.0,
                    err
                ))
            })?;
        }

        Ok(())
    }

    /// Compact the legalized GP bank back into the backend's fixed GP budget.
    ///
    /// The 32-bit legalizer may temporarily introduce sparse GP register ids
    /// for high halves and scratch values. After a second peephole pass removes
    /// dead copies, the remaining live GP ids can usually be packed back into
    /// the original lowering budget without changing semantics.
    pub fn compact_32bit_gp_bank(&mut self, max_gp_regs: u16) -> Result<(), WasmError> {
        for MachineFunction { id, program } in &mut self.functions {
            program.compact_32bit_gp_bank(max_gp_regs).map_err(|err| {
                WasmError::internal(alloc::format!(
                    "machine function {} failed 32-bit gp-bank compaction: {}",
                    id.0,
                    err
                ))
            })?;
        }
        Ok(())
    }

    /// Final cleanup step for legalized 32-bit GP MachineIR.
    ///
    /// The legalizer may temporarily introduce sparse GP ids and small
    /// copy-heavy sequences. A final peephole pass removes redundant moves,
    /// then GP compaction packs the surviving legalized GP regs back into the
    /// backend's fixed budget.
    pub fn finalize_32bit_gp_legalization(
        &mut self,
        first_transient: u16,
        max_gp_regs: u16,
    ) -> Result<(), WasmError> {
        self.optimize(first_transient, 4);
        self.compact_32bit_gp_bank(max_gp_regs)
    }
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GpExtraRegDebugInfo {
    pub(crate) reg: MachineReg,
    pub(crate) candidate_slots: Vec<MachineReg>,
    pub(crate) pair_low_regs: Vec<MachineReg>,
    pub(crate) samples: Vec<String>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GpCompactionDebugInfo {
    pub(crate) used_gp_regs: Vec<MachineReg>,
    pub(crate) used_degrees: Vec<(MachineReg, usize)>,
    pub(crate) peak_live_regs: usize,
    pub(crate) peak_live_point: GpPeakLivePoint,
    pub(crate) estimated_pressure: usize,
    pub(crate) extra_regs: Vec<GpExtraRegDebugInfo>,
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GpPeakLivePoint {
    pub(crate) block: MachineBlockId,
    pub(crate) stage: String,
    pub(crate) live_regs: Vec<MachineReg>,
}

struct GpLiveness {
    live_in: Vec<Vec<bool>>,
    live_out: Vec<Vec<bool>>,
}

struct Legalize32BitGpI64 {
    original_reg_count: u16,
    original_first_fp_reg: u16,
    persistent_hi: Vec<Option<MachineReg>>,
    next_reg: u16,
}

impl Legalize32BitGpI64 {
    fn new(
        program: &MachineProgram,
        _flow: &MachineStorageFlow,
    ) -> Result<Option<Self>, WasmError> {
        if !program_needs_gp_i64_legalization(program) {
            return Ok(None);
        }

        let mut next_reg = program.reg_count;
        let mut persistent_hi = vec![None; program.reg_count as usize];
        let mut needs_hi = vec![false; program.reg_count as usize];
        collect_legalized_i64_regs(program, &mut needs_hi)?;
        for reg_index in 0..program.first_fp_reg {
            if !needs_hi[reg_index as usize] {
                continue;
            }
            persistent_hi[reg_index as usize] = Some(MachineReg(next_reg));
            next_reg = next_reg.checked_add(1).ok_or_else(|| {
                WasmError::internal("32-bit legalizer overflowed the machine register count".into())
            })?;
        }

        Ok(Some(Self {
            original_reg_count: program.reg_count,
            original_first_fp_reg: program.first_fp_reg,
            persistent_hi,
            next_reg,
        }))
    }

    fn rewrite_program(
        &mut self,
        program: &mut MachineProgram,
        flow: &MachineStorageFlow,
    ) -> Result<(), WasmError> {
        let original_params: Vec<Vec<MachineBlockParam>> = program
            .blocks
            .iter()
            .map(|block| block.params.clone())
            .collect();
        let gp_word_mem_width = machine_ptr_width(4);
        let gp_word_int_width = machine_word_int_width(4);

        for block_index in 0..program.blocks.len() {
            let block_id = MachineBlockId(block_index as u32);
            let mut state = match &flow.block_entry[block_index] {
                Some(entry) => entry.clone(),
                None => {
                    let mut seeded = program.base_32bit_storage_state()?;
                    seed_block_params(
                        &mut seeded,
                        &original_params[block_index],
                        "legalize unreachable block entry",
                    )?;
                    seeded
                }
            };

            let block = &mut program.blocks[block_index];
            let mut current_hi = vec![None; self.original_first_fp_reg as usize];
            for reg in 0..self.original_first_fp_reg {
                if matches!(
                    state.get(reg as usize).and_then(|fact| fact.known()),
                    Some(MachineStorageType::GpI64)
                ) {
                    current_hi[reg as usize] = self.persistent_hi[reg as usize];
                }
            }
            block.params =
                self.rewrite_block_params(&original_params[block_index], &mut current_hi)?;

            let old_ops = core::mem::take(&mut block.ops);
            let mut new_ops = Vec::with_capacity(old_ops.len() * 4);
            for inst in old_ops {
                self.rewrite_inst(&inst.kind, &state, &mut current_hi, &mut new_ops)?;
                simulate_inst(
                    &mut state,
                    &inst.kind,
                    self.original_first_fp_reg,
                    gp_word_mem_width,
                    gp_word_int_width,
                )
                .map_err(|err| {
                    WasmError::internal(alloc::format!(
                        "machine block {} failed during 32-bit legalization state simulation: {}",
                        block_id.0,
                        err
                    ))
                })?;
            }

            let old_term = block.terminator.clone();
            block.terminator = self.rewrite_term(
                old_term,
                &state,
                &current_hi,
                &original_params,
                &mut new_ops,
            )?;
            block.ops = new_ops;
        }

        program.reg_count = self.next_reg;
        program.finalize_32bit_gp_expansion(
            self.original_reg_count,
            self.next_reg - self.original_reg_count,
        )?;
        program.validate()?;
        Ok(())
    }

    fn rewrite_block_params(
        &mut self,
        params: &[MachineBlockParam],
        current_hi: &mut [Option<MachineReg>],
    ) -> Result<Vec<MachineBlockParam>, WasmError> {
        let mut rewritten = Vec::with_capacity(params.len() * 2);
        for param in params {
            match param.ty {
                MachineStorageType::GpI64 => {
                    let hi =
                        self.alloc_hi_reg(current_hi, param.reg, "block parameter high half")?;
                    rewritten.push(MachineBlockParam::gp_word(param.reg));
                    rewritten.push(MachineBlockParam::gp_word(hi));
                }
                _ => rewritten.push(*param),
            }
        }
        Ok(rewritten)
    }

    fn rewrite_inst(
        &mut self,
        inst: &MachineInstKind,
        state: &[MachineStorageFact],
        current_hi: &mut [Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        match inst {
            MachineInstKind::Move { ty, dst, src } => match ty {
                MachineStorageType::GpI64 => {
                    let (src_lo, src_hi) =
                        self.split_move_like_i64_source(state, current_hi, *src, "move source")?;
                    let dst_hi =
                        self.alloc_hi_reg(current_hi, *dst, "move destination high half")?;
                    emit_move(out, MachineStorageType::GpWord, *dst, src_lo);
                    emit_move(out, MachineStorageType::GpWord, dst_hi, src_hi);
                }
                _ => {
                    self.clear_hi_binding(current_hi, *dst);
                    emit_move(
                        out,
                        *ty,
                        *dst,
                        self.rewrite_single_move_like_value(*ty, state, *src, "move source")?,
                    );
                }
            },
            MachineInstKind::FloatConst { .. } | MachineInstKind::Lea { .. } => {
                out.push(MachineInst { kind: inst.clone() });
            }
            MachineInstKind::Load {
                ty,
                dst,
                addr,
                width,
                extension,
            } => match ty {
                MachineStorageType::GpI64 => {
                    self.rewrite_i64_load(*dst, *addr, *width, *extension, current_hi, out)?
                }
                _ => {
                    self.clear_hi_binding(current_hi, *dst);
                    out.push(MachineInst {
                        kind: MachineInstKind::Load {
                            ty: *ty,
                            dst: *dst,
                            addr: *addr,
                            width: *width,
                            extension: *extension,
                        },
                    })
                }
            },
            MachineInstKind::Store {
                ty,
                addr,
                width,
                src,
            } => match ty {
                MachineStorageType::GpI64 => {
                    self.rewrite_i64_store(state, current_hi, *addr, *width, *src, out)?
                }
                _ => out.push(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: *ty,
                        addr: *addr,
                        width: *width,
                        src: self.rewrite_single_move_like_value(
                            *ty,
                            state,
                            *src,
                            "store source",
                        )?,
                    },
                }),
            },
            MachineInstKind::IntUnary {
                width,
                op,
                dst,
                src,
            } => {
                if matches!(width, MachineIntWidth::I64) {
                    self.rewrite_i64_unary(*op, *dst, *src, state, current_hi, out)?;
                } else {
                    self.clear_hi_binding(current_hi, *dst);
                    out.push(MachineInst { kind: inst.clone() });
                }
            }
            MachineInstKind::IntBinary {
                width,
                op,
                dst,
                lhs,
                rhs,
            } => {
                if matches!(width, MachineIntWidth::I64) {
                    self.rewrite_i64_binary(*op, *dst, *lhs, *rhs, state, current_hi, out)?;
                } else {
                    self.clear_hi_binding(current_hi, *dst);
                    out.push(MachineInst { kind: inst.clone() });
                }
            }
            MachineInstKind::IntMulWide { .. } => {
                out.push(MachineInst { kind: inst.clone() });
            }
            MachineInstKind::Int64PairBinary { .. }
            | MachineInstKind::Int64PairUnary { .. }
            | MachineInstKind::Int64PairDivRem { .. }
            | MachineInstKind::Int64PairShift { .. }
            | MachineInstKind::Int64PairCompare { .. }
            | MachineInstKind::ConvertI64PairToFloat { .. }
            | MachineInstKind::ConvertFloatToI64Pair { .. }
            | MachineInstKind::ReinterpretF64ToI64Pair { .. }
            | MachineInstKind::ReinterpretI64PairToF64 { .. } => {
                out.push(MachineInst { kind: inst.clone() });
            }
            MachineInstKind::IntCompare {
                width,
                kind,
                sign,
                dst,
                lhs,
                rhs,
            } => {
                if matches!(width, MachineIntWidth::I64) {
                    self.clear_hi_binding(current_hi, *dst);
                    self.emit_i64_compare_result(
                        *kind, *sign, *dst, *lhs, *rhs, state, current_hi, out,
                    )?;
                } else {
                    self.clear_hi_binding(current_hi, *dst);
                    out.push(MachineInst { kind: inst.clone() });
                }
            }
            MachineInstKind::FloatUnary { .. }
            | MachineInstKind::FloatBinary { .. }
            | MachineInstKind::FloatCompare { .. } => {
                out.push(MachineInst { kind: inst.clone() });
            }
            MachineInstKind::Convert { op, dst, src } => {
                self.rewrite_convert(*op, *dst, *src, state, current_hi, out)?
            }
            MachineInstKind::Select {
                ty,
                dst,
                on_true,
                on_false,
                cond,
            } => match ty {
                MachineStorageType::GpI64 => {
                    let (true_lo, true_hi) = self.split_move_like_i64_source(
                        state,
                        current_hi,
                        *on_true,
                        "select on_true",
                    )?;
                    let (false_lo, false_hi) = self.split_move_like_i64_source(
                        state,
                        current_hi,
                        *on_false,
                        "select on_false",
                    )?;
                    let dst_hi =
                        self.alloc_hi_reg(current_hi, *dst, "select destination high half")?;
                    out.push(MachineInst {
                        kind: MachineInstKind::Select {
                            ty: MachineStorageType::GpWord,
                            dst: *dst,
                            on_true: true_lo,
                            on_false: false_lo,
                            cond: *cond,
                        },
                    });
                    out.push(MachineInst {
                        kind: MachineInstKind::Select {
                            ty: MachineStorageType::GpWord,
                            dst: dst_hi,
                            on_true: true_hi,
                            on_false: false_hi,
                            cond: *cond,
                        },
                    });
                }
                _ => {
                    self.clear_hi_binding(current_hi, *dst);
                    out.push(MachineInst {
                        kind: MachineInstKind::Select {
                            ty: *ty,
                            dst: *dst,
                            on_true: self.rewrite_single_move_like_value(
                                *ty,
                                state,
                                *on_true,
                                "select on_true",
                            )?,
                            on_false: self.rewrite_single_move_like_value(
                                *ty,
                                state,
                                *on_false,
                                "select on_false",
                            )?,
                            cond: *cond,
                        },
                    })
                }
            },
            MachineInstKind::TrapIf { kind, cond } => {
                let cond = self.rewrite_branch_cond(*cond, state, current_hi, out)?;
                out.push(MachineInst {
                    kind: MachineInstKind::TrapIf { kind: *kind, cond },
                });
            }
            MachineInstKind::CallHelper(_) => out.push(MachineInst { kind: inst.clone() }),
        }
        Ok(())
    }

    fn rewrite_term(
        &mut self,
        term: MachineTerminator,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        original_params: &[Vec<MachineBlockParam>],
        out: &mut Vec<MachineInst>,
    ) -> Result<MachineTerminator, WasmError> {
        Ok(match term {
            MachineTerminator::Jump(mut edge) => {
                let target_index = edge.target.as_usize();
                self.rewrite_edge(&mut edge, state, current_hi, &original_params[target_index])?;
                MachineTerminator::Jump(edge)
            }
            MachineTerminator::Branch {
                cond,
                mut then_edge,
                mut else_edge,
            } => {
                let cond = self.rewrite_branch_cond(cond, state, current_hi, out)?;
                let then_target = then_edge.target.as_usize();
                self.rewrite_edge(
                    &mut then_edge,
                    state,
                    current_hi,
                    &original_params[then_target],
                )?;
                let else_target = else_edge.target.as_usize();
                self.rewrite_edge(
                    &mut else_edge,
                    state,
                    current_hi,
                    &original_params[else_target],
                )?;
                MachineTerminator::Branch {
                    cond,
                    then_edge,
                    else_edge,
                }
            }
            MachineTerminator::JumpTable { index, mut entries } => {
                let index = self.rewrite_single_move_like_value(
                    MachineStorageType::GpWord,
                    state,
                    index,
                    "jump table index",
                )?;
                for edge in &mut entries {
                    self.rewrite_edge(
                        edge,
                        state,
                        current_hi,
                        &original_params[edge.target.as_usize()],
                    )?;
                }
                MachineTerminator::JumpTable { index, entries }
            }
            MachineTerminator::CallIndirect {
                callee_target,
                callee_frame_base,
                arg_slots,
                caller_result_base,
                continuation,
            } => MachineTerminator::CallIndirect {
                callee_target: self.rewrite_single_move_like_value(
                    MachineStorageType::GpWord,
                    state,
                    callee_target,
                    "indirect callee target",
                )?,
                callee_frame_base,
                arg_slots,
                caller_result_base,
                continuation,
            },
            other => other,
        })
    }

    fn rewrite_edge(
        &mut self,
        edge: &mut MachineEdge,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        target_params: &[MachineBlockParam],
    ) -> Result<(), WasmError> {
        let mut args = Vec::with_capacity(edge.args.len() * 2);
        for (index, (arg, param)) in edge
            .args
            .iter()
            .copied()
            .zip(target_params.iter())
            .enumerate()
        {
            match param.ty {
                MachineStorageType::GpI64 => {
                    let (lo, hi) = self.split_move_like_i64_source(
                        state,
                        current_hi,
                        arg,
                        &alloc::format!("edge arg {} to gpi64 param r{}", index, param.reg.0),
                    )?;
                    args.push(lo);
                    args.push(hi);
                }
                _ => args.push(self.rewrite_single_move_like_value(
                    param.ty,
                    state,
                    arg,
                    &alloc::format!("edge arg {} to param r{}", index, param.reg.0),
                )?),
            }
        }
        edge.args = args;
        Ok(())
    }

    fn rewrite_i64_load(
        &mut self,
        dst: MachineReg,
        addr: super::MachineAddr,
        width: MachineMemWidth,
        extension: super::MachineLoadExtension,
        current_hi: &mut [Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        let dst_hi = self.alloc_hi_reg(current_hi, dst, "load destination high half")?;
        let load_addr = if addr.base == dst || addr.base == dst_hi {
            // A pair load cannot clobber the address base between the low-word
            // and high-word loads. Preserve the base in a scratch register when
            // the legalized low half reuses that same GP register.
            let preserved_base = self.alloc_temp_gp_reg()?;
            emit_move(
                out,
                MachineStorageType::GpWord,
                preserved_base,
                MachineValue::Reg(addr.base),
            );
            super::MachineAddr {
                base: preserved_base,
                offset: addr.offset,
            }
        } else {
            addr
        };
        match width {
            MachineMemWidth::U64 => {
                if !matches!(extension, super::MachineLoadExtension::None) {
                    return unsupported_32bit_i64("i64 load with U64 extension");
                }
                out.push(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst,
                        addr: load_addr,
                        width: MachineMemWidth::U32,
                        extension: super::MachineLoadExtension::None,
                    },
                });
                out.push(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst: dst_hi,
                        addr: addr_with_offset(load_addr, 4)?,
                        width: MachineMemWidth::U32,
                        extension: super::MachineLoadExtension::None,
                    },
                });
            }
            MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
                out.push(MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst,
                        addr: load_addr,
                        width,
                        extension,
                    },
                });
                match extension {
                    super::MachineLoadExtension::SignExtend => {
                        emit_i32_shift_right_signed(out, dst_hi, MachineValue::Reg(dst), 31)
                    }
                    super::MachineLoadExtension::ZeroExtend | super::MachineLoadExtension::None => {
                        emit_move(
                            out,
                            MachineStorageType::GpWord,
                            dst_hi,
                            MachineValue::Imm64(0),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn rewrite_i64_store(
        &self,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        addr: super::MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        match width {
            MachineMemWidth::U64 => {
                let (src_lo, src_hi) =
                    self.split_move_like_i64_source(state, current_hi, src, "store source")?;
                out.push(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr,
                        width: MachineMemWidth::U32,
                        src: src_lo,
                    },
                });
                out.push(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: addr_with_offset(addr, 4)?,
                        width: MachineMemWidth::U32,
                        src: src_hi,
                    },
                });
            }
            MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
                out.push(MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr,
                        width,
                        src: self.low_word_value(src),
                    },
                })
            }
        }
        Ok(())
    }

    fn rewrite_i64_unary(
        &mut self,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
        state: &[MachineStorageFact],
        current_hi: &mut [Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        match op {
            MachineIntUnaryOp::Eqz => {
                let (src_lo, src_hi) =
                    self.split_exact_i64_value(state, current_hi, src, "i64.eqz source")?;
                let scratch0 = self.alloc_temp_gp_reg()?;
                let scratch1 = self.alloc_temp_gp_reg()?;
                emit_i32_compare(
                    out,
                    scratch0,
                    MachineCompareKind::Eq,
                    MachineSign::Unsigned,
                    src_lo,
                    MachineValue::Imm64(0),
                );
                emit_i32_compare(
                    out,
                    scratch1,
                    MachineCompareKind::Eq,
                    MachineSign::Unsigned,
                    src_hi,
                    MachineValue::Imm64(0),
                );
                emit_i32_binary(
                    out,
                    MachineIntBinaryOp::And,
                    dst,
                    MachineValue::Reg(scratch0),
                    MachineValue::Reg(scratch1),
                );
                Ok(())
            }
            MachineIntUnaryOp::Extend8S | MachineIntUnaryOp::Extend16S => {
                let (src_lo, _) =
                    self.split_exact_i64_value(state, current_hi, src, "i64 extend source")?;
                let dst_hi =
                    self.alloc_hi_reg(current_hi, dst, "i64 extend destination high half")?;
                emit_i32_unary(out, op, dst, src_lo);
                emit_i32_shift_right_signed(out, dst_hi, MachineValue::Reg(dst), 31);
                Ok(())
            }
            MachineIntUnaryOp::Extend32S => {
                let (src_lo, _) =
                    self.split_exact_i64_value(state, current_hi, src, "i64 extend source")?;
                let dst_hi =
                    self.alloc_hi_reg(current_hi, dst, "i64 extend destination high half")?;
                emit_move(out, MachineStorageType::GpWord, dst, src_lo);
                emit_i32_shift_right_signed(out, dst_hi, MachineValue::Reg(dst), 31);
                Ok(())
            }
            MachineIntUnaryOp::Clz | MachineIntUnaryOp::Ctz | MachineIntUnaryOp::Popcnt => {
                let (src_lo, src_hi) =
                    self.split_exact_i64_value(state, current_hi, src, "i64 unary source")?;
                let dst_hi =
                    self.alloc_hi_reg(current_hi, dst, "i64 unary destination high half")?;
                out.push(MachineInst {
                    kind: MachineInstKind::Int64PairUnary {
                        op,
                        dst_lo: dst,
                        dst_hi,
                        src_lo,
                        src_hi,
                    },
                });
                Ok(())
            }
        }
    }

    fn rewrite_i64_binary(
        &mut self,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
        state: &[MachineStorageFact],
        current_hi: &mut [Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        let (lhs_lo, lhs_hi) =
            self.split_exact_i64_value(state, current_hi, lhs, "i64 binary lhs")?;
        let (rhs_lo, rhs_hi) =
            self.split_exact_i64_value(state, current_hi, rhs, "i64 binary rhs")?;
        let dst_hi = self.alloc_hi_reg(current_hi, dst, "i64 binary destination high half")?;
        match op {
            MachineIntBinaryOp::Add | MachineIntBinaryOp::Sub | MachineIntBinaryOp::Mul => {
                out.push(MachineInst {
                    kind: MachineInstKind::Int64PairBinary {
                        op,
                        dst_lo: dst,
                        dst_hi,
                        lhs_lo,
                        lhs_hi,
                        rhs_lo,
                        rhs_hi,
                    },
                });
                Ok(())
            }
            MachineIntBinaryOp::And | MachineIntBinaryOp::Or | MachineIntBinaryOp::Xor => {
                emit_i32_binary(out, op, dst, lhs_lo, rhs_lo);
                emit_i32_binary(out, op, dst_hi, lhs_hi, rhs_hi);
                Ok(())
            }
            MachineIntBinaryOp::DivS
            | MachineIntBinaryOp::DivU
            | MachineIntBinaryOp::RemS
            | MachineIntBinaryOp::RemU => {
                out.push(MachineInst {
                    kind: MachineInstKind::Int64PairDivRem {
                        sign: match op {
                            MachineIntBinaryOp::DivS | MachineIntBinaryOp::RemS => {
                                MachineSign::Signed
                            }
                            MachineIntBinaryOp::DivU | MachineIntBinaryOp::RemU => {
                                MachineSign::Unsigned
                            }
                            _ => unreachable!(),
                        },
                        rem: matches!(op, MachineIntBinaryOp::RemS | MachineIntBinaryOp::RemU),
                        dst_lo: dst,
                        dst_hi,
                        lhs_lo,
                        lhs_hi,
                        rhs_lo,
                        rhs_hi,
                    },
                });
                Ok(())
            }
            MachineIntBinaryOp::Shl
            | MachineIntBinaryOp::ShrS
            | MachineIntBinaryOp::ShrU
            | MachineIntBinaryOp::Rotl
            | MachineIntBinaryOp::Rotr => {
                out.push(MachineInst {
                    kind: MachineInstKind::Int64PairShift {
                        op,
                        dst_lo: dst,
                        dst_hi,
                        lhs_lo,
                        lhs_hi,
                        rhs: rhs_lo,
                    },
                });
                Ok(())
            }
        }
    }

    fn rewrite_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
        state: &[MachineStorageFact],
        current_hi: &mut [Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        match op {
            MachineConvertOp::I32WrapI64 => {
                self.clear_hi_binding(current_hi, dst);
                emit_move(
                    out,
                    MachineStorageType::GpWord,
                    dst,
                    self.low_word_value(src),
                );
                Ok(())
            }
            MachineConvertOp::I64ExtendI32U => {
                let dst_hi =
                    self.alloc_hi_reg(current_hi, dst, "i64.extend_i32_u destination high half")?;
                emit_move(
                    out,
                    MachineStorageType::GpWord,
                    dst,
                    self.low_word_value(src),
                );
                emit_move(
                    out,
                    MachineStorageType::GpWord,
                    dst_hi,
                    MachineValue::Imm64(0),
                );
                Ok(())
            }
            MachineConvertOp::I64ExtendI32S => {
                let dst_hi =
                    self.alloc_hi_reg(current_hi, dst, "i64.extend_i32_s destination high half")?;
                emit_move(
                    out,
                    MachineStorageType::GpWord,
                    dst,
                    self.low_word_value(src),
                );
                emit_i32_shift_right_signed(out, dst_hi, MachineValue::Reg(dst), 31);
                Ok(())
            }
            MachineConvertOp::I64ReinterpretF64 => {
                let dst_hi = self.alloc_hi_reg(
                    current_hi,
                    dst,
                    "i64.reinterpret_f64 destination high half",
                )?;
                out.push(MachineInst {
                    kind: MachineInstKind::ReinterpretF64ToI64Pair {
                        dst_lo: dst,
                        dst_hi,
                        src: self.rewrite_single_move_like_value(
                            MachineStorageType::Fp64,
                            state,
                            src,
                            "i64.reinterpret_f64 source",
                        )?,
                    },
                });
                Ok(())
            }
            MachineConvertOp::F64ReinterpretI64 => {
                let (src_lo, src_hi) = self.split_exact_i64_value(
                    state,
                    current_hi,
                    src,
                    "f64.reinterpret_i64 source",
                )?;
                self.clear_hi_binding(current_hi, dst);
                out.push(MachineInst {
                    kind: MachineInstKind::ReinterpretI64PairToF64 {
                        dst,
                        src_lo,
                        src_hi,
                    },
                });
                Ok(())
            }
            _ => {
                if matches!(
                    op,
                    MachineConvertOp::I64TruncF32S
                        | MachineConvertOp::I64TruncF32U
                        | MachineConvertOp::I64TruncF64S
                        | MachineConvertOp::I64TruncF64U
                        | MachineConvertOp::I64TruncSatF32S
                        | MachineConvertOp::I64TruncSatF32U
                        | MachineConvertOp::I64TruncSatF64S
                        | MachineConvertOp::I64TruncSatF64U
                ) {
                    let dst_hi = self.alloc_hi_reg(
                        current_hi,
                        dst,
                        "float-to-i64 convert destination high half",
                    )?;
                    let (src_ty, _) = storage_types_for_convert(op);
                    out.push(MachineInst {
                        kind: MachineInstKind::ConvertFloatToI64Pair {
                            op,
                            dst_lo: dst,
                            dst_hi,
                            src: self.rewrite_single_move_like_value(
                                src_ty,
                                state,
                                src,
                                "float-to-i64 convert source",
                            )?,
                        },
                    });
                    return Ok(());
                }
                if matches!(
                    op,
                    MachineConvertOp::F32ConvertI64S
                        | MachineConvertOp::F32ConvertI64U
                        | MachineConvertOp::F64ConvertI64S
                        | MachineConvertOp::F64ConvertI64U
                ) {
                    let (src_lo, src_hi) = self.split_exact_i64_value(
                        state,
                        current_hi,
                        src,
                        "i64-to-float convert source",
                    )?;
                    self.clear_hi_binding(current_hi, dst);
                    out.push(MachineInst {
                        kind: MachineInstKind::ConvertI64PairToFloat {
                            width: match op {
                                MachineConvertOp::F32ConvertI64S
                                | MachineConvertOp::F32ConvertI64U => MachineFloatWidth::F32,
                                MachineConvertOp::F64ConvertI64S
                                | MachineConvertOp::F64ConvertI64U => MachineFloatWidth::F64,
                                _ => unreachable!(),
                            },
                            sign: match op {
                                MachineConvertOp::F32ConvertI64S
                                | MachineConvertOp::F64ConvertI64S => MachineSign::Signed,
                                MachineConvertOp::F32ConvertI64U
                                | MachineConvertOp::F64ConvertI64U => MachineSign::Unsigned,
                                _ => unreachable!(),
                            },
                            dst,
                            src_lo,
                            src_hi,
                        },
                    });
                    return Ok(());
                }
                let (src_ty, dst_ty) = storage_types_for_convert(op);
                if matches!(src_ty, MachineStorageType::GpI64)
                    || matches!(dst_ty, MachineStorageType::GpI64)
                {
                    return unsupported_32bit_i64(&alloc::format!("i64 conversion op {:?}", op));
                }
                self.clear_hi_binding(current_hi, dst);
                out.push(MachineInst {
                    kind: MachineInstKind::Convert { op, dst, src },
                });
                let _ = state;
                Ok(())
            }
        }
    }

    fn rewrite_branch_cond(
        &mut self,
        cond: MachineBranchCond,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<MachineBranchCond, WasmError> {
        Ok(match cond {
            MachineBranchCond::Value(value) => {
                MachineBranchCond::Value(self.rewrite_single_move_like_value(
                    MachineStorageType::GpWord,
                    state,
                    value,
                    "branch value",
                )?)
            }
            MachineBranchCond::IntCompare {
                width: MachineIntWidth::I64,
                kind,
                sign,
                lhs,
                rhs,
            } => {
                let scratch = self.alloc_temp_gp_reg()?;
                self.emit_i64_compare_result(
                    kind, sign, scratch, lhs, rhs, state, current_hi, out,
                )?;
                MachineBranchCond::Value(MachineValue::Reg(scratch))
            }
            other => other,
        })
    }

    fn emit_i64_compare_result(
        &self,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        out: &mut Vec<MachineInst>,
    ) -> Result<(), WasmError> {
        let (lhs_lo, lhs_hi) =
            self.split_exact_i64_value(state, current_hi, lhs, "i64 compare lhs")?;
        let (rhs_lo, rhs_hi) =
            self.split_exact_i64_value(state, current_hi, rhs, "i64 compare rhs")?;
        out.push(MachineInst {
            kind: MachineInstKind::Int64PairCompare {
                kind,
                sign,
                dst,
                lhs_lo,
                lhs_hi,
                rhs_lo,
                rhs_hi,
            },
        });
        Ok(())
    }

    fn rewrite_single_move_like_value(
        &self,
        ty: MachineStorageType,
        state: &[MachineStorageFact],
        value: MachineValue,
        context: &str,
    ) -> Result<MachineValue, WasmError> {
        let MachineValue::Reg(reg) = value else {
            return Ok(match ty {
                MachineStorageType::GpWord => self.low_word_value(value),
                _ => value,
            });
        };
        let actual = require_known_reg_type(state, reg, context)?;
        match ty {
            MachineStorageType::GpWord => match actual {
                MachineStorageType::GpWord
                | MachineStorageType::GpI64
                | MachineStorageType::Fp32 => Ok(MachineValue::Reg(reg)),
                MachineStorageType::Fp64 => Err(WasmError::internal(alloc::format!(
                    "{} cannot consume fp64 as a gp word on 32-bit",
                    context,
                ))),
            },
            MachineStorageType::Fp32 => match actual {
                MachineStorageType::Fp32 | MachineStorageType::GpWord => Ok(MachineValue::Reg(reg)),
                MachineStorageType::GpI64 | MachineStorageType::Fp64 => Err(WasmError::internal(
                    alloc::format!("{} requires fp32-compatible source on 32-bit", context),
                )),
            },
            MachineStorageType::Fp64 => match actual {
                MachineStorageType::Fp64 => Ok(MachineValue::Reg(reg)),
                MachineStorageType::GpI64 => Err(WasmError::internal(
                    "gp<->fp64 raw bit move on legalized 32-bit MachineIR is not implemented yet"
                        .into(),
                )),
                MachineStorageType::GpWord | MachineStorageType::Fp32 => Err(WasmError::internal(
                    alloc::format!("{} requires fp64-compatible source on 32-bit", context),
                )),
            },
            MachineStorageType::GpI64 => Err(WasmError::internal(alloc::format!(
                "{} unexpectedly requested a single-value gpi64 rewrite",
                context,
            ))),
        }
    }

    fn split_exact_i64_value(
        &self,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        value: MachineValue,
        context: &str,
    ) -> Result<(MachineValue, MachineValue), WasmError> {
        match value {
            MachineValue::Imm64(bits) => Ok(split_i64_immediate(bits)),
            MachineValue::Reg(reg) => match require_known_reg_type(state, reg, context)? {
                MachineStorageType::GpI64 => Ok((
                    MachineValue::Reg(reg),
                    MachineValue::Reg(self.current_hi_reg(current_hi, reg, context)?),
                )),
                actual => Err(WasmError::internal(alloc::format!(
                    "{} requires an exact gpi64 source on 32-bit, found {:?} in r{}",
                    context,
                    actual,
                    reg.0,
                ))),
            },
        }
    }

    fn split_move_like_i64_source(
        &self,
        state: &[MachineStorageFact],
        current_hi: &[Option<MachineReg>],
        value: MachineValue,
        context: &str,
    ) -> Result<(MachineValue, MachineValue), WasmError> {
        match value {
            MachineValue::Imm64(bits) => Ok(split_i64_immediate(bits)),
            MachineValue::Reg(reg) => match require_known_reg_type(state, reg, context)? {
                MachineStorageType::GpI64 => Ok((
                    MachineValue::Reg(reg),
                    MachineValue::Reg(self.current_hi_reg(current_hi, reg, context)?),
                )),
                MachineStorageType::GpWord => Ok((MachineValue::Reg(reg), MachineValue::Imm64(0))),
                MachineStorageType::Fp64 => Err(WasmError::internal(
                    "gp<->fp64 raw bit move on legalized 32-bit MachineIR is not implemented yet"
                        .into(),
                )),
                MachineStorageType::Fp32 => Err(WasmError::internal(alloc::format!(
                    "{} requires a gp-compatible source on 32-bit, found fp32 in r{}",
                    context,
                    reg.0,
                ))),
            },
        }
    }

    fn low_word_value(&self, value: MachineValue) -> MachineValue {
        match value {
            MachineValue::Imm64(bits) => MachineValue::Imm64(u64::from(bits as u32)),
            MachineValue::Reg(reg) => MachineValue::Reg(reg),
        }
    }

    fn alloc_temp_gp_reg(&mut self) -> Result<MachineReg, WasmError> {
        let reg = MachineReg(self.next_reg);
        self.next_reg = self.next_reg.checked_add(1).ok_or_else(|| {
            WasmError::internal("32-bit legalizer overflowed the machine register count".into())
        })?;
        Ok(reg)
    }

    fn alloc_hi_reg(
        &mut self,
        current_hi: &mut [Option<MachineReg>],
        reg: MachineReg,
        context: &str,
    ) -> Result<MachineReg, WasmError> {
        if reg.0 >= self.original_first_fp_reg {
            return Err(WasmError::internal(alloc::format!(
                "{} requires a GP destination register, found r{}",
                context,
                reg.0,
            )));
        }
        let hi = self.persistent_hi[reg.0 as usize].unwrap_or(self.alloc_temp_gp_reg()?);
        current_hi[reg.0 as usize] = Some(hi);
        Ok(hi)
    }

    fn current_hi_reg(
        &self,
        current_hi: &[Option<MachineReg>],
        reg: MachineReg,
        context: &str,
    ) -> Result<MachineReg, WasmError> {
        current_hi
            .get(reg.0 as usize)
            .and_then(|mapped| *mapped)
            .or_else(|| {
                self.persistent_hi
                    .get(reg.0 as usize)
                    .and_then(|mapped| *mapped)
            })
            .ok_or_else(|| {
                WasmError::internal(alloc::format!(
                    "{} requires a live legalized high-half register for r{}",
                    context,
                    reg.0,
                ))
            })
    }

    fn clear_hi_binding(&self, current_hi: &mut [Option<MachineReg>], reg: MachineReg) {
        if reg.0 < self.original_first_fp_reg {
            current_hi[reg.0 as usize] = None;
        }
    }
}

fn program_needs_gp_i64_legalization(program: &MachineProgram) -> bool {
    program.blocks.iter().any(|block| {
        block
            .params
            .iter()
            .any(|param| matches!(param.ty, MachineStorageType::GpI64))
            || block
                .ops
                .iter()
                .any(|inst| inst_needs_gp_i64_legalization(&inst.kind))
            || terminator_needs_gp_i64_legalization(&block.terminator)
    })
}

fn inst_needs_gp_i64_legalization(inst: &MachineInstKind) -> bool {
    match inst {
        MachineInstKind::Move { ty, .. }
        | MachineInstKind::Load { ty, .. }
        | MachineInstKind::Store { ty, .. }
        | MachineInstKind::Select { ty, .. } => matches!(ty, MachineStorageType::GpI64),
        MachineInstKind::IntUnary { width, .. }
        | MachineInstKind::IntBinary { width, .. }
        | MachineInstKind::IntCompare { width, .. } => matches!(width, MachineIntWidth::I64),
        MachineInstKind::Convert { op, .. } => {
            let (src_ty, dst_ty) = storage_types_for_convert(*op);
            matches!(src_ty, MachineStorageType::GpI64)
                || matches!(dst_ty, MachineStorageType::GpI64)
        }
        MachineInstKind::TrapIf { cond, .. } => match cond {
            MachineBranchCond::IntCompare { width, .. } => matches!(width, MachineIntWidth::I64),
            _ => false,
        },
        MachineInstKind::ConvertI64PairToFloat { .. }
        | MachineInstKind::ConvertFloatToI64Pair { .. }
        | MachineInstKind::Int64PairBinary { .. }
        | MachineInstKind::Int64PairUnary { .. }
        | MachineInstKind::Int64PairDivRem { .. }
        | MachineInstKind::Int64PairShift { .. }
        | MachineInstKind::Int64PairCompare { .. }
        | MachineInstKind::ReinterpretF64ToI64Pair { .. }
        | MachineInstKind::ReinterpretI64PairToF64 { .. }
        | MachineInstKind::IntMulWide { .. } => true,
        MachineInstKind::FloatConst { .. }
        | MachineInstKind::Lea { .. }
        | MachineInstKind::FloatBinary { .. }
        | MachineInstKind::FloatUnary { .. }
        | MachineInstKind::FloatCompare { .. }
        | MachineInstKind::CallHelper(_) => false,
    }
}

fn terminator_needs_gp_i64_legalization(terminator: &MachineTerminator) -> bool {
    match terminator {
        MachineTerminator::Branch { cond, .. } => match cond {
            MachineBranchCond::IntCompare { width, .. } => matches!(width, MachineIntWidth::I64),
            _ => false,
        },
        MachineTerminator::Jump { .. }
        | MachineTerminator::Return
        | MachineTerminator::Trap { .. }
        | MachineTerminator::CallDirect { .. }
        | MachineTerminator::CallIndirect { .. }
        | MachineTerminator::JumpTable { .. } => false,
    }
}

fn collect_legalized_i64_regs(
    program: &MachineProgram,
    needs_hi: &mut [bool],
) -> Result<(), WasmError> {
    let target_params: Vec<Vec<MachineBlockParam>> = program
        .blocks
        .iter()
        .map(|block| block.params.clone())
        .collect();
    for block in &program.blocks {
        for param in &block.params {
            if matches!(param.ty, MachineStorageType::GpI64) {
                mark_gp_i64_reg(needs_hi, program.first_fp_reg, param.reg);
            }
        }
        for inst in &block.ops {
            match &inst.kind {
                MachineInstKind::Move { ty, dst, .. }
                | MachineInstKind::Load { ty, dst, .. }
                | MachineInstKind::Select { ty, dst, .. } => {
                    if matches!(ty, MachineStorageType::GpI64) {
                        mark_gp_i64_reg(needs_hi, program.first_fp_reg, *dst);
                    }
                }
                MachineInstKind::IntUnary {
                    width,
                    op,
                    dst,
                    src,
                } => {
                    if matches!(width, MachineIntWidth::I64) {
                        if !matches!(op, MachineIntUnaryOp::Eqz) {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *dst);
                            if let MachineValue::Reg(reg) = src {
                                mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                            }
                        } else if let MachineValue::Reg(reg) = src {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                    }
                }
                MachineInstKind::IntBinary {
                    width,
                    dst,
                    lhs,
                    rhs,
                    ..
                } => {
                    if matches!(width, MachineIntWidth::I64) {
                        mark_gp_i64_reg(needs_hi, program.first_fp_reg, *dst);
                        if let MachineValue::Reg(reg) = lhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                        if let MachineValue::Reg(reg) = rhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                    }
                }
                MachineInstKind::IntMulWide { .. } => {}
                MachineInstKind::Int64PairBinary { .. }
                | MachineInstKind::Int64PairUnary { .. } => {}
                MachineInstKind::Int64PairDivRem { .. }
                | MachineInstKind::Int64PairShift { .. }
                | MachineInstKind::Int64PairCompare { .. }
                | MachineInstKind::ConvertI64PairToFloat { .. }
                | MachineInstKind::ConvertFloatToI64Pair { .. }
                | MachineInstKind::ReinterpretF64ToI64Pair { .. }
                | MachineInstKind::ReinterpretI64PairToF64 { .. } => {}
                MachineInstKind::IntCompare {
                    width, lhs, rhs, ..
                } => {
                    if matches!(width, MachineIntWidth::I64) {
                        if let MachineValue::Reg(reg) = lhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                        if let MachineValue::Reg(reg) = rhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                    }
                }
                MachineInstKind::Convert { op, dst, src } => {
                    let (src_ty, dst_ty) = storage_types_for_convert(*op);
                    if matches!(dst_ty, MachineStorageType::GpI64) {
                        mark_gp_i64_reg(needs_hi, program.first_fp_reg, *dst);
                    }
                    if matches!(src_ty, MachineStorageType::GpI64) {
                        if let MachineValue::Reg(reg) = src {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                    }
                }
                MachineInstKind::TrapIf { cond, .. } => {
                    if let MachineBranchCond::IntCompare {
                        width: MachineIntWidth::I64,
                        lhs,
                        rhs,
                        ..
                    } = cond
                    {
                        if let MachineValue::Reg(reg) = lhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                        if let MachineValue::Reg(reg) = rhs {
                            mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                        }
                    }
                }
                MachineInstKind::FloatConst { .. }
                | MachineInstKind::Lea { .. }
                | MachineInstKind::Store { .. }
                | MachineInstKind::FloatUnary { .. }
                | MachineInstKind::FloatBinary { .. }
                | MachineInstKind::FloatCompare { .. }
                | MachineInstKind::CallHelper(_) => {}
            }
        }
        match &block.terminator {
            MachineTerminator::Jump(edge) => {
                mark_edge_i64_regs(needs_hi, program.first_fp_reg, edge, &target_params)?;
            }
            MachineTerminator::Branch {
                cond,
                then_edge,
                else_edge,
            } => {
                if let MachineBranchCond::IntCompare {
                    width: MachineIntWidth::I64,
                    lhs,
                    rhs,
                    ..
                } = cond
                {
                    if let MachineValue::Reg(reg) = lhs {
                        mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                    }
                    if let MachineValue::Reg(reg) = rhs {
                        mark_gp_i64_reg(needs_hi, program.first_fp_reg, *reg);
                    }
                }
                mark_edge_i64_regs(needs_hi, program.first_fp_reg, then_edge, &target_params)?;
                mark_edge_i64_regs(needs_hi, program.first_fp_reg, else_edge, &target_params)?;
            }
            MachineTerminator::JumpTable { entries, .. } => {
                for edge in entries {
                    mark_edge_i64_regs(needs_hi, program.first_fp_reg, edge, &target_params)?;
                }
            }
            MachineTerminator::CallDirect { .. }
            | MachineTerminator::CallIndirect { .. }
            | MachineTerminator::Return
            | MachineTerminator::Trap { .. } => {}
        }
    }
    Ok(())
}

fn mark_edge_i64_regs(
    needs_hi: &mut [bool],
    first_fp_reg: u16,
    edge: &MachineEdge,
    target_params: &[Vec<MachineBlockParam>],
) -> Result<(), WasmError> {
    let params = target_params.get(edge.target.as_usize()).ok_or_else(|| {
        WasmError::internal("machine edge target is out of range during i64 legalizer scan".into())
    })?;
    for (arg, param) in edge.args.iter().zip(params.iter()) {
        if matches!(param.ty, MachineStorageType::GpI64) {
            if let MachineValue::Reg(reg) = arg {
                mark_gp_i64_reg(needs_hi, first_fp_reg, *reg);
            }
            mark_gp_i64_reg(needs_hi, first_fp_reg, param.reg);
        }
    }
    Ok(())
}

fn mark_gp_i64_reg(needs_hi: &mut [bool], first_fp_reg: u16, reg: MachineReg) {
    if reg.0 < first_fp_reg {
        needs_hi[reg.0 as usize] = true;
    }
}

fn split_i64_immediate(bits: u64) -> (MachineValue, MachineValue) {
    (
        MachineValue::Imm64(u64::from(bits as u32)),
        MachineValue::Imm64(u64::from((bits >> 32) as u32)),
    )
}

fn addr_with_offset(
    addr: super::MachineAddr,
    offset: i32,
) -> Result<super::MachineAddr, WasmError> {
    Ok(super::MachineAddr {
        base: addr.base,
        offset: addr.offset.checked_add(offset).ok_or_else(|| {
            WasmError::internal("32-bit legalizer address offset overflowed".into())
        })?,
    })
}

fn emit_move(
    out: &mut Vec<MachineInst>,
    ty: MachineStorageType,
    dst: MachineReg,
    src: MachineValue,
) {
    out.push(MachineInst {
        kind: MachineInstKind::Move { ty, dst, src },
    });
}

fn emit_i32_unary(
    out: &mut Vec<MachineInst>,
    op: MachineIntUnaryOp,
    dst: MachineReg,
    src: MachineValue,
) {
    out.push(MachineInst {
        kind: MachineInstKind::IntUnary {
            width: MachineIntWidth::I32,
            op,
            dst,
            src,
        },
    });
}

fn emit_i32_binary(
    out: &mut Vec<MachineInst>,
    op: MachineIntBinaryOp,
    dst: MachineReg,
    lhs: MachineValue,
    rhs: MachineValue,
) {
    out.push(MachineInst {
        kind: MachineInstKind::IntBinary {
            width: MachineIntWidth::I32,
            op,
            dst,
            lhs,
            rhs,
        },
    });
}

fn emit_i32_compare(
    out: &mut Vec<MachineInst>,
    dst: MachineReg,
    kind: MachineCompareKind,
    sign: MachineSign,
    lhs: MachineValue,
    rhs: MachineValue,
) {
    out.push(MachineInst {
        kind: MachineInstKind::IntCompare {
            width: MachineIntWidth::I32,
            kind,
            sign,
            dst,
            lhs,
            rhs,
        },
    });
}

fn emit_i32_shift_right_signed(
    out: &mut Vec<MachineInst>,
    dst: MachineReg,
    src: MachineValue,
    shift: u32,
) {
    emit_i32_binary(
        out,
        MachineIntBinaryOp::ShrS,
        dst,
        src,
        MachineValue::Imm64(u64::from(shift)),
    );
}

fn unsupported_32bit_i64(context: &str) -> Result<(), WasmError> {
    Err(WasmError::internal(alloc::format!(
        "{} is not implemented in the first 32-bit MachineIR legalization slice",
        context,
    )))
}

impl MachineProgram {
    pub(crate) fn compact_32bit_gp_bank(&mut self, max_gp_regs: u16) -> Result<(), WasmError> {
        if self.first_fp_reg <= max_gp_regs {
            return Ok(());
        }

        let original_first_fp_reg = self.first_fp_reg;
        let fp_reg_count = self
            .reg_count
            .checked_sub(original_first_fp_reg)
            .ok_or_else(|| {
                WasmError::internal(
                    "machine reg_count underflowed the fp bank during 32-bit gp compaction".into(),
                )
            })?;
        let mut used_gp = vec![false; original_first_fp_reg as usize];
        collect_used_gp_regs(self, &mut used_gp)?;

        let mut gp_map = vec![None; original_first_fp_reg as usize];
        // The original lowering GP bank `[0, max_gp_regs)` already fits the
        // backend budget. Compaction only needs to pack the extra legalized GP
        // regs back into that existing dynamic-bank color space; recoloring the
        // original non-SSA GP bank would only add false pressure.
        for reg in 0..max_gp_regs.min(original_first_fp_reg) {
            let reg = MachineReg(reg);
            gp_map[reg.0 as usize] = Some(reg);
        }
        let interference = build_gp_interference(self, original_first_fp_reg)?;
        let mut free_gp_slots =
            Vec::with_capacity(max_gp_regs.saturating_sub(MACHINE_FIXED_REG_COUNT) as usize);
        for reg in MACHINE_FIXED_REG_COUNT..max_gp_regs {
            let reg = MachineReg(reg);
            free_gp_slots.push(reg);
        }

        if !assign_compacted_gp_colors(&interference, &used_gp, &mut gp_map, &free_gp_slots) {
            let required_gp_regs = max_live_gp_regs(&interference, &used_gp, &gp_map);
            return Err(WasmError::invalid(alloc::format!(
                "32-bit gp-bank compaction requires more than {} GP regs (estimated live pressure {})",
                max_gp_regs,
                required_gp_regs
            )));
        }

        // Keep the FP bank anchored at the backend's configured boundary.
        //
        // The legalizer may prove that fewer GP regs are actually needed than
        // the backend budget, but native backends still use a fixed machine-reg
        // partition chosen by lowering/ABI setup. Letting first_fp_reg drift
        // downward would make post-legalization MachineIR disagree with that
        // fixed physical mapping contract.
        let new_first_fp_reg = max_gp_regs;
        remap_program_regs(self, |reg| {
            if reg.0 < original_first_fp_reg {
                return gp_map
                    .get(reg.0 as usize)
                    .and_then(|mapped| *mapped)
                    .ok_or_else(|| {
                        WasmError::internal(alloc::format!(
                            "32-bit gp-bank compaction missing remap for gp reg {}",
                            reg.0
                        ))
                    });
            }
            Ok(MachineReg(
                new_first_fp_reg + (reg.0 - original_first_fp_reg),
            ))
        })?;

        self.first_fp_reg = new_first_fp_reg;
        self.reg_count = new_first_fp_reg.checked_add(fp_reg_count).ok_or_else(|| {
            WasmError::internal(
                "32-bit gp-bank compaction overflowed the machine register count".into(),
            )
        })?;
        self.validate()?;
        Ok(())
    }

    pub(crate) fn finalize_32bit_gp_expansion(
        &mut self,
        original_reg_count: u16,
        added_gp_regs: u16,
    ) -> Result<(), WasmError> {
        if added_gp_regs == 0 {
            return Ok(());
        }

        let original_first_fp_reg = self.first_fp_reg;
        if original_first_fp_reg > original_reg_count {
            return Err(WasmError::internal(alloc::format!(
                "machine first_fp_reg {} exceeds original register count {} during 32-bit gp expansion",
                original_first_fp_reg,
                original_reg_count,
            )));
        }

        let expected_reg_count =
            original_reg_count
                .checked_add(added_gp_regs)
                .ok_or_else(|| {
                    WasmError::internal(
                        "32-bit gp expansion overflowed the machine register count".into(),
                    )
                })?;
        if self.reg_count != expected_reg_count {
            return Err(WasmError::internal(alloc::format!(
                "32-bit gp expansion expected provisional reg_count {}, found {}",
                expected_reg_count,
                self.reg_count,
            )));
        }

        let new_first_fp_reg = original_first_fp_reg
            .checked_add(added_gp_regs)
            .ok_or_else(|| {
                WasmError::internal(
                    "32-bit gp expansion overflowed the first_fp_reg boundary".into(),
                )
            })?;

        remap_program_regs(self, |reg| {
            if reg.0 < original_first_fp_reg {
                return Ok(reg);
            }
            if reg.0 < original_reg_count {
                return Ok(MachineReg(reg.0 + added_gp_regs));
            }
            if reg.0 < expected_reg_count {
                return Ok(MachineReg(
                    original_first_fp_reg + (reg.0 - original_reg_count),
                ));
            }
            Err(WasmError::internal(alloc::format!(
                "machine register {} is out of range during 32-bit gp expansion remap",
                reg.0,
            )))
        })?;

        self.first_fp_reg = new_first_fp_reg;
        self.reg_count = expected_reg_count;
        Ok(())
    }

    pub(crate) fn analyze_32bit_storage_flow(&self) -> Result<MachineStorageFlow, WasmError> {
        let mut block_entry = vec![None; self.blocks.len()];
        let mut block_exit = vec![None; self.blocks.len()];
        if self.blocks.is_empty() {
            return Ok(MachineStorageFlow {
                block_entry,
                block_exit,
            });
        }

        let gp_word_mem_width = machine_ptr_width(4);
        let gp_word_int_width = machine_word_int_width(4);
        let mut entry_state = self.base_32bit_storage_state()?;
        seed_block_params(
            &mut entry_state,
            &self.blocks[self.entry.as_usize()].params,
            "entry",
        )?;
        block_entry[self.entry.as_usize()] = Some(entry_state);

        let mut queued = vec![false; self.blocks.len()];
        let mut worklist = VecDeque::new();
        worklist.push_back(self.entry);
        queued[self.entry.as_usize()] = true;

        while let Some(block_id) = worklist.pop_front() {
            queued[block_id.as_usize()] = false;
            let in_state = block_entry[block_id.as_usize()].clone().ok_or_else(|| {
                WasmError::internal("reachable machine block is missing storage-flow state".into())
            })?;
            let block = self
                .blocks
                .get(block_id.as_usize())
                .ok_or_else(|| WasmError::internal("machine block id is out of range".into()))?;
            let out_state = self.simulate_block(
                block_id,
                block,
                in_state,
                gp_word_mem_width,
                gp_word_int_width,
            )?;
            block_exit[block_id.as_usize()] = Some(out_state.clone());

            for successor in self.successor_states(
                block_id,
                &block.terminator,
                &out_state,
                gp_word_mem_width,
                gp_word_int_width,
            )? {
                let (target, candidate) = successor;
                if merge_block_state(&mut block_entry[target.as_usize()], candidate)?
                    && !queued[target.as_usize()]
                {
                    worklist.push_back(target);
                    queued[target.as_usize()] = true;
                }
            }
        }

        Ok(MachineStorageFlow {
            block_entry,
            block_exit,
        })
    }

    fn base_32bit_storage_state(&self) -> Result<Vec<MachineStorageFact>, WasmError> {
        let mut state = vec![MachineStorageFact::Undefined; self.reg_count as usize];

        for reg in [
            MACHINE_CTX_REG,
            MACHINE_FP_REG,
            MACHINE_MEM0_BASE_REG,
            MACHINE_MEM0_SIZE_REG,
        ] {
            if reg.0 < self.reg_count {
                write_reg_type(
                    &mut state,
                    reg,
                    MachineStorageType::GpWord,
                    "fixed register",
                )?;
            }
        }

        for (index, width) in self.fp_reg_init_widths.iter().copied().enumerate() {
            let Some(width) = width else {
                continue;
            };
            let reg = MachineReg(self.first_fp_reg.checked_add(index as u16).ok_or_else(|| {
                WasmError::internal(
                    "machine FP init register index overflowed register space".into(),
                )
            })?);
            if reg.0 >= self.reg_count {
                return Err(WasmError::internal(alloc::format!(
                    "machine fp_reg_init_widths entry {} references out-of-range register {}",
                    index,
                    reg.0,
                )));
            }
            write_reg_type(
                &mut state,
                reg,
                float_storage_type(width),
                "FP init storage",
            )?;
        }

        Ok(state)
    }

    fn simulate_block(
        &self,
        block_id: MachineBlockId,
        block: &super::MachineBlock,
        mut state: Vec<MachineStorageFact>,
        gp_word_mem_width: MachineMemWidth,
        gp_word_int_width: MachineIntWidth,
    ) -> Result<Vec<MachineStorageFact>, WasmError> {
        require_block_params(&state, &block.params, "block entry")?;

        for (inst_index, inst) in block.ops.iter().enumerate() {
            simulate_inst(
                &mut state,
                &inst.kind,
                self.first_fp_reg,
                gp_word_mem_width,
                gp_word_int_width,
            )
            .map_err(|err| {
                WasmError::internal(alloc::format!(
                    "machine block {} op{} ({:?}) failed 32-bit storage-flow validation: {}\nblock={:?}",
                    block_id.0,
                    inst_index,
                    inst.kind,
                    err,
                    block
                ))
            })?;
        }

        simulate_term(
            &mut state,
            &block.terminator,
            self.first_fp_reg,
            gp_word_mem_width,
            gp_word_int_width,
        )
        .map_err(|err| {
            WasmError::internal(alloc::format!(
                "machine block {} terminator ({:?}) failed 32-bit storage-flow validation: {}\nblock={:?}",
                block_id.0,
                block.terminator,
                err,
                block
            ))
        })?;

        Ok(state)
    }

    fn successor_states(
        &self,
        block_id: MachineBlockId,
        term: &MachineTerminator,
        out_state: &[MachineStorageFact],
        gp_word_mem_width: MachineMemWidth,
        gp_word_int_width: MachineIntWidth,
    ) -> Result<Vec<(MachineBlockId, Vec<MachineStorageFact>)>, WasmError> {
        let mut successors = Vec::new();
        match term {
            MachineTerminator::Jump(edge) => {
                successors.push((
                    edge.target,
                    self.apply_edge_state(
                        out_state,
                        edge,
                        gp_word_mem_width,
                        gp_word_int_width,
                        block_id,
                    )?,
                ));
            }
            MachineTerminator::Branch {
                then_edge,
                else_edge,
                ..
            } => {
                successors.push((
                    then_edge.target,
                    self.apply_edge_state(
                        out_state,
                        then_edge,
                        gp_word_mem_width,
                        gp_word_int_width,
                        block_id,
                    )?,
                ));
                successors.push((
                    else_edge.target,
                    self.apply_edge_state(
                        out_state,
                        else_edge,
                        gp_word_mem_width,
                        gp_word_int_width,
                        block_id,
                    )?,
                ));
            }
            MachineTerminator::JumpTable { entries, .. } => {
                for edge in entries {
                    successors.push((
                        edge.target,
                        self.apply_edge_state(
                            out_state,
                            edge,
                            gp_word_mem_width,
                            gp_word_int_width,
                            block_id,
                        )?,
                    ));
                }
            }
            MachineTerminator::CallDirect { continuation, .. }
            | MachineTerminator::CallIndirect { continuation, .. } => {
                successors.push((*continuation, out_state.to_vec()));
            }
            MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
        }
        Ok(successors)
    }

    fn apply_edge_state(
        &self,
        out_state: &[MachineStorageFact],
        edge: &MachineEdge,
        gp_word_mem_width: MachineMemWidth,
        gp_word_int_width: MachineIntWidth,
        source_block: MachineBlockId,
    ) -> Result<Vec<MachineStorageFact>, WasmError> {
        let mut next = out_state.to_vec();
        let source = self.blocks.get(source_block.as_usize()).ok_or_else(|| {
            WasmError::internal(alloc::format!(
                "machine block {} is out of range during 32-bit storage-flow validation",
                source_block.0,
            ))
        })?;
        let target_params = self
            .blocks
            .get(edge.target.as_usize())
            .ok_or_else(|| {
                WasmError::internal(alloc::format!(
                    "machine block {} edge target {} is out of range during 32-bit storage-flow validation",
                    source_block.0,
                    edge.target.0,
                ))
            })?
            .params
            .clone();

        for (index, (arg, param)) in edge.args.iter().zip(target_params.iter()).enumerate() {
            require_move_like_value_type(
                out_state,
                *arg,
                param.ty,
                self.first_fp_reg,
                &alloc::format!(
                    "edge {} -> {} arg {} for param r{}",
                    source_block.0,
                    edge.target.0,
                    index,
                    param.reg.0
                ),
            )
            .map_err(|err| {
                WasmError::internal(alloc::format!(
                    "{}\nsource_block={:?}\ntarget_block={:?}",
                    err,
                    source,
                    self.blocks
                        .get(edge.target.as_usize())
                        .expect("edge target checked above")
                ))
            })?;
            write_reg_type(&mut next, param.reg, param.ty, "edge parameter destination")?;
        }

        let _ = (gp_word_mem_width, gp_word_int_width);
        Ok(next)
    }
}

fn remap_program_regs(
    program: &mut MachineProgram,
    mut remap: impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    for block in &mut program.blocks {
        for param in &mut block.params {
            param.reg = remap(param.reg)?;
        }
        for inst in &mut block.ops {
            remap_inst_regs(&mut inst.kind, &mut remap)?;
        }
        remap_term_regs(&mut block.terminator, &mut remap)?;
    }
    Ok(())
}

fn collect_used_gp_regs(program: &MachineProgram, used: &mut [bool]) -> Result<(), WasmError> {
    for block in &program.blocks {
        for param in &block.params {
            mark_gp_used(used, param.reg)?;
        }
        for inst in &block.ops {
            collect_inst_gp_regs(&inst.kind, used)?;
        }
        collect_term_gp_regs(&block.terminator, used)?;
    }
    Ok(())
}

fn build_gp_interference(
    program: &MachineProgram,
    first_fp_reg: u16,
) -> Result<Vec<Vec<bool>>, WasmError> {
    let liveness = build_gp_liveness(program, first_fp_reg)?;

    let gp_reg_count = first_fp_reg as usize;
    let mut interference = vec![vec![false; gp_reg_count]; gp_reg_count];
    for block in &program.blocks {
        let block_index = block.id.as_usize();
        let mut live = liveness.live_out[block_index].clone();
        let block_param_regs: Vec<MachineReg> =
            block.params.iter().map(|param| param.reg).collect();
        add_gp_group_interference(&mut interference, &block_param_regs, first_fp_reg);

        let mut term_uses = Vec::new();
        collect_term_gp_uses(&block.terminator, first_fp_reg, &mut term_uses)?;
        for reg in term_uses {
            mark_gp_live(&mut live, reg, first_fp_reg);
        }

        for inst in block.ops.iter().rev() {
            let mut uses = Vec::new();
            let mut defs = Vec::new();
            collect_inst_gp_uses_defs(&inst.kind, first_fp_reg, &mut uses, &mut defs)?;
            add_gp_group_interference(&mut interference, &defs, first_fp_reg);
            for reg in defs {
                add_gp_interference(
                    &mut interference,
                    reg,
                    &live,
                    first_fp_reg,
                    gp_copy_coalescing_partner(&inst.kind, reg, first_fp_reg),
                );
                clear_gp_live(&mut live, reg, first_fp_reg);
            }
            for reg in uses {
                mark_gp_live(&mut live, reg, first_fp_reg);
            }
        }

        for param in block.params.iter().rev() {
            add_gp_interference(&mut interference, param.reg, &live, first_fp_reg, None);
            clear_gp_live(&mut live, param.reg, first_fp_reg);
        }
    }

    Ok(interference)
}

fn build_gp_liveness(program: &MachineProgram, first_fp_reg: u16) -> Result<GpLiveness, WasmError> {
    let gp_reg_count = first_fp_reg as usize;
    let block_count = program.blocks.len();
    let mut block_use = vec![vec![false; gp_reg_count]; block_count];
    let mut block_def = vec![vec![false; gp_reg_count]; block_count];

    for block in &program.blocks {
        let block_index = block.id.as_usize();
        for param in &block.params {
            mark_gp_def(&mut block_def[block_index], param.reg, first_fp_reg);
        }

        for inst in &block.ops {
            let mut uses = Vec::new();
            let mut defs = Vec::new();
            collect_inst_gp_uses_defs(&inst.kind, first_fp_reg, &mut uses, &mut defs)?;
            for reg in uses {
                mark_gp_use(
                    &mut block_use[block_index],
                    &block_def[block_index],
                    reg,
                    first_fp_reg,
                );
            }
            for reg in defs {
                mark_gp_def(&mut block_def[block_index], reg, first_fp_reg);
            }
        }

        let mut term_uses = Vec::new();
        collect_term_gp_uses(&block.terminator, first_fp_reg, &mut term_uses)?;
        for reg in term_uses {
            mark_gp_use(
                &mut block_use[block_index],
                &block_def[block_index],
                reg,
                first_fp_reg,
            );
        }
    }

    let mut live_in = vec![vec![false; gp_reg_count]; block_count];
    let mut live_out = vec![vec![false; gp_reg_count]; block_count];
    let mut changed = true;
    while changed {
        changed = false;
        for block in program.blocks.iter().rev() {
            let block_index = block.id.as_usize();
            let mut new_out = vec![false; gp_reg_count];
            for succ in term_successors(&block.terminator) {
                union_bitset(&mut new_out, &live_in[succ.as_usize()]);
            }

            let mut new_in = block_use[block_index].clone();
            let out_minus_defs = subtract_bitset(&new_out, &block_def[block_index]);
            union_bitset(&mut new_in, &out_minus_defs);

            if new_out != live_out[block_index] || new_in != live_in[block_index] {
                live_out[block_index] = new_out;
                live_in[block_index] = new_in;
                changed = true;
            }
        }
    }

    Ok(GpLiveness { live_in, live_out })
}

fn collect_inst_gp_uses_defs(
    inst: &MachineInstKind,
    first_fp_reg: u16,
    uses: &mut Vec<MachineReg>,
    defs: &mut Vec<MachineReg>,
) -> Result<(), WasmError> {
    match inst {
        MachineInstKind::Move { dst, src, .. } | MachineInstKind::Convert { dst, src, .. } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *src, first_fp_reg);
        }
        MachineInstKind::FloatConst { dst, .. } => {
            push_gp_def(defs, *dst, first_fp_reg);
        }
        MachineInstKind::Lea { dst, addr } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_use(uses, addr.base, first_fp_reg);
        }
        MachineInstKind::Load { dst, addr, .. } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_use(uses, addr.base, first_fp_reg);
        }
        MachineInstKind::Store { addr, src, .. } => {
            push_gp_use(uses, addr.base, first_fp_reg);
            push_gp_value_use(uses, *src, first_fp_reg);
        }
        MachineInstKind::IntUnary { dst, src, .. }
        | MachineInstKind::FloatUnary { dst, src, .. } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *src, first_fp_reg);
        }
        MachineInstKind::IntBinary { dst, lhs, rhs, .. }
        | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
        | MachineInstKind::FloatBinary { dst, lhs, rhs, .. }
        | MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *lhs, first_fp_reg);
            push_gp_value_use(uses, *rhs, first_fp_reg);
        }
        MachineInstKind::IntMulWide {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
            ..
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *lhs, first_fp_reg);
            push_gp_value_use(uses, *rhs, first_fp_reg);
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        }
        | MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *lhs_lo, first_fp_reg);
            push_gp_value_use(uses, *lhs_hi, first_fp_reg);
            push_gp_value_use(uses, *rhs_lo, first_fp_reg);
            push_gp_value_use(uses, *rhs_hi, first_fp_reg);
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            ..
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *src_lo, first_fp_reg);
            push_gp_value_use(uses, *src_hi, first_fp_reg);
        }
        MachineInstKind::Int64PairCompare {
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *lhs_lo, first_fp_reg);
            push_gp_value_use(uses, *lhs_hi, first_fp_reg);
            push_gp_value_use(uses, *rhs_lo, first_fp_reg);
            push_gp_value_use(uses, *rhs_hi, first_fp_reg);
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *lhs_lo, first_fp_reg);
            push_gp_value_use(uses, *lhs_hi, first_fp_reg);
            push_gp_value_use(uses, *rhs, first_fp_reg);
        }
        MachineInstKind::ConvertI64PairToFloat {
            dst,
            src_lo,
            src_hi,
            ..
        } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *src_lo, first_fp_reg);
            push_gp_value_use(uses, *src_hi, first_fp_reg);
        }
        MachineInstKind::ConvertFloatToI64Pair {
            dst_lo,
            dst_hi,
            src,
            ..
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *src, first_fp_reg);
        }
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            push_gp_def(defs, *dst_lo, first_fp_reg);
            push_gp_def(defs, *dst_hi, first_fp_reg);
            push_gp_value_use(uses, *src, first_fp_reg);
        }
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *src_lo, first_fp_reg);
            push_gp_value_use(uses, *src_hi, first_fp_reg);
        }
        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
            ..
        } => {
            push_gp_def(defs, *dst, first_fp_reg);
            push_gp_value_use(uses, *on_true, first_fp_reg);
            push_gp_value_use(uses, *on_false, first_fp_reg);
            push_gp_value_use(uses, *cond, first_fp_reg);
        }
        MachineInstKind::TrapIf { cond, .. } => {
            collect_branch_cond_gp_uses(cond, first_fp_reg, uses)?;
        }
        MachineInstKind::CallHelper(_) => {}
    }
    Ok(())
}

fn collect_term_gp_uses(
    term: &MachineTerminator,
    first_fp_reg: u16,
    uses: &mut Vec<MachineReg>,
) -> Result<(), WasmError> {
    match term {
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
        MachineTerminator::Jump(edge) => {
            collect_edge_gp_uses(edge, first_fp_reg, uses)?;
        }
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            collect_branch_cond_gp_uses(cond, first_fp_reg, uses)?;
            collect_edge_gp_uses(then_edge, first_fp_reg, uses)?;
            collect_edge_gp_uses(else_edge, first_fp_reg, uses)?;
        }
        MachineTerminator::JumpTable { index, entries } => {
            push_gp_value_use(uses, *index, first_fp_reg);
            for edge in entries {
                collect_edge_gp_uses(edge, first_fp_reg, uses)?;
            }
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            push_gp_use(uses, *callee_frame_base, first_fp_reg);
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            push_gp_value_use(uses, *callee_target, first_fp_reg);
            push_gp_use(uses, *callee_frame_base, first_fp_reg);
        }
    }
    Ok(())
}

fn collect_edge_gp_uses(
    edge: &MachineEdge,
    first_fp_reg: u16,
    uses: &mut Vec<MachineReg>,
) -> Result<(), WasmError> {
    for arg in &edge.args {
        push_gp_value_use(uses, *arg, first_fp_reg);
    }
    Ok(())
}

fn collect_branch_cond_gp_uses(
    cond: &MachineBranchCond,
    first_fp_reg: u16,
    uses: &mut Vec<MachineReg>,
) -> Result<(), WasmError> {
    match cond {
        MachineBranchCond::Value(value) => push_gp_value_use(uses, *value, first_fp_reg),
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            push_gp_value_use(uses, *lhs, first_fp_reg);
            push_gp_value_use(uses, *rhs, first_fp_reg);
        }
    }
    Ok(())
}

fn term_successors(term: &MachineTerminator) -> alloc::vec::IntoIter<MachineBlockId> {
    let mut succs = Vec::new();
    match term {
        MachineTerminator::Jump(edge) => succs.push(edge.target),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => {
            succs.push(then_edge.target);
            if else_edge.target != then_edge.target {
                succs.push(else_edge.target);
            }
        }
        MachineTerminator::JumpTable { entries, .. } => {
            for edge in entries {
                if !succs.contains(&edge.target) {
                    succs.push(edge.target);
                }
            }
        }
        MachineTerminator::CallDirect { continuation, .. }
        | MachineTerminator::CallIndirect { continuation, .. } => succs.push(*continuation),
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }
    succs.into_iter()
}

fn union_bitset(dst: &mut [bool], src: &[bool]) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst |= *src;
    }
}

fn subtract_bitset(lhs: &[bool], rhs: &[bool]) -> Vec<bool> {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| *lhs && !*rhs)
        .collect()
}

fn add_gp_interference(
    interference: &mut [Vec<bool>],
    reg: MachineReg,
    live: &[bool],
    first_fp_reg: u16,
    coalescing_partner: Option<MachineReg>,
) {
    if reg.0 >= first_fp_reg {
        return;
    }
    let reg_index = reg.0 as usize;
    let coalescing_partner = coalescing_partner.map(|partner| partner.0 as usize);
    for (other, is_live) in live.iter().enumerate() {
        if !*is_live || other == reg_index || Some(other) == coalescing_partner {
            continue;
        }
        interference[reg_index][other] = true;
        interference[other][reg_index] = true;
    }
}

fn add_gp_group_interference(
    interference: &mut [Vec<bool>],
    regs: &[MachineReg],
    first_fp_reg: u16,
) {
    for (index, &lhs) in regs.iter().enumerate() {
        if lhs.0 >= first_fp_reg {
            continue;
        }
        let lhs_index = lhs.0 as usize;
        for &rhs in &regs[index + 1..] {
            if rhs.0 >= first_fp_reg || lhs == rhs {
                continue;
            }
            let rhs_index = rhs.0 as usize;
            interference[lhs_index][rhs_index] = true;
            interference[rhs_index][lhs_index] = true;
        }
    }
}

fn gp_copy_coalescing_partner(
    inst: &MachineInstKind,
    reg: MachineReg,
    first_fp_reg: u16,
) -> Option<MachineReg> {
    match inst {
        MachineInstKind::Move {
            dst,
            src: MachineValue::Reg(src),
            ..
        } if *dst == reg && src.0 < first_fp_reg => Some(*src),
        _ => None,
    }
}

fn gp_interference_degree(interference: &[Vec<bool>], reg: MachineReg) -> usize {
    interference[reg.0 as usize]
        .iter()
        .filter(|interferes| **interferes)
        .count()
}

fn assign_compacted_gp_colors(
    interference: &[Vec<bool>],
    used_gp: &[bool],
    gp_map: &mut [Option<MachineReg>],
    free_gp_slots: &[MachineReg],
) -> bool {
    let mut color_order = Vec::new();
    for (index, used) in used_gp.iter().copied().enumerate() {
        if !used || gp_map[index].is_some() {
            continue;
        }
        color_order.push(MachineReg(index as u16));
    }
    color_order.sort_by(|lhs, rhs| {
        let lhs_degree = gp_interference_degree(interference, *lhs);
        let rhs_degree = gp_interference_degree(interference, *rhs);
        rhs_degree.cmp(&lhs_degree).then(lhs.0.cmp(&rhs.0))
    });

    let mut greedy_map = gp_map.to_vec();
    let greedy_ok = color_order.iter().copied().all(|reg| {
        let assigned = free_gp_slots
            .iter()
            .copied()
            .find(|slot| can_assign_gp_color(interference, &greedy_map, reg, *slot));
        let Some(slot) = assigned else {
            return false;
        };
        greedy_map[reg.0 as usize] = Some(slot);
        true
    });
    if greedy_ok {
        gp_map.copy_from_slice(&greedy_map);
        return true;
    }

    assign_compacted_gp_colors_exact(interference, used_gp, gp_map, free_gp_slots)
}

fn assign_compacted_gp_colors_exact(
    interference: &[Vec<bool>],
    used_gp: &[bool],
    gp_map: &mut [Option<MachineReg>],
    free_gp_slots: &[MachineReg],
) -> bool {
    let mut uncolored = Vec::new();
    for (index, used) in used_gp.iter().copied().enumerate() {
        if used && gp_map[index].is_none() {
            uncolored.push(index);
        }
    }
    if uncolored.is_empty() {
        return true;
    }
    color_backtrack(interference, gp_map, free_gp_slots, &uncolored)
}

fn color_backtrack(
    interference: &[Vec<bool>],
    gp_map: &mut [Option<MachineReg>],
    free_gp_slots: &[MachineReg],
    uncolored: &[usize],
) -> bool {
    let Some(next) = choose_next_uncolored_reg(interference, gp_map, free_gp_slots, uncolored)
    else {
        return true;
    };
    let reg = MachineReg(next as u16);
    let mut candidate_slots = candidate_gp_slots(interference, gp_map, free_gp_slots, reg);
    if candidate_slots.is_empty() {
        return false;
    }
    candidate_slots.sort_by_key(|slot| slot.0);
    for slot in candidate_slots {
        gp_map[next] = Some(slot);
        if color_backtrack(interference, gp_map, free_gp_slots, uncolored) {
            return true;
        }
        gp_map[next] = None;
    }
    false
}

fn choose_next_uncolored_reg(
    interference: &[Vec<bool>],
    gp_map: &[Option<MachineReg>],
    free_gp_slots: &[MachineReg],
    uncolored: &[usize],
) -> Option<usize> {
    uncolored
        .iter()
        .copied()
        .filter(|index| gp_map[*index].is_none())
        .max_by(|lhs, rhs| {
            let lhs_reg = MachineReg(*lhs as u16);
            let rhs_reg = MachineReg(*rhs as u16);
            let lhs_sat = gp_saturation_degree(interference, gp_map, lhs_reg);
            let rhs_sat = gp_saturation_degree(interference, gp_map, rhs_reg);
            lhs_sat
                .cmp(&rhs_sat)
                .then_with(|| {
                    let lhs_options =
                        candidate_gp_slots(interference, gp_map, free_gp_slots, lhs_reg).len();
                    let rhs_options =
                        candidate_gp_slots(interference, gp_map, free_gp_slots, rhs_reg).len();
                    rhs_options.cmp(&lhs_options)
                })
                .then_with(|| {
                    gp_interference_degree(interference, lhs_reg)
                        .cmp(&gp_interference_degree(interference, rhs_reg))
                })
                .then_with(|| rhs.cmp(lhs))
        })
}

fn gp_saturation_degree(
    interference: &[Vec<bool>],
    gp_map: &[Option<MachineReg>],
    reg: MachineReg,
) -> usize {
    let mut colors = Vec::new();
    for (other, interferes) in interference[reg.0 as usize].iter().copied().enumerate() {
        if !interferes {
            continue;
        }
        let Some(color) = gp_map[other] else {
            continue;
        };
        if !colors.contains(&color) {
            colors.push(color);
        }
    }
    colors.len()
}

fn candidate_gp_slots(
    interference: &[Vec<bool>],
    gp_map: &[Option<MachineReg>],
    free_gp_slots: &[MachineReg],
    reg: MachineReg,
) -> Vec<MachineReg> {
    free_gp_slots
        .iter()
        .copied()
        .filter(|slot| can_assign_gp_color(interference, gp_map, reg, *slot))
        .collect()
}

fn can_assign_gp_color(
    interference: &[Vec<bool>],
    gp_map: &[Option<MachineReg>],
    reg: MachineReg,
    slot: MachineReg,
) -> bool {
    interference[reg.0 as usize]
        .iter()
        .enumerate()
        .filter_map(|(other, interferes)| interferes.then_some(other))
        .all(|other| gp_map[other] != Some(slot))
}

fn max_live_gp_regs(
    interference: &[Vec<bool>],
    used_gp: &[bool],
    gp_map: &[Option<MachineReg>],
) -> usize {
    let reserved = gp_map.iter().filter(|slot| slot.is_some()).count();
    let mut max_degree = 0usize;
    for (reg, used) in used_gp.iter().enumerate() {
        if !*used || gp_map[reg].is_some() {
            continue;
        }
        max_degree = max_degree.max(
            interference[reg]
                .iter()
                .enumerate()
                .filter(|(other, interferes)| **interferes && used_gp[*other])
                .count()
                + 1,
        );
    }
    reserved + max_degree
}

#[cfg(test)]
pub(crate) fn debug_describe_gp_compaction(
    program: &MachineProgram,
    max_gp_regs: u16,
) -> Result<GpCompactionDebugInfo, WasmError> {
    let mut used_gp = vec![false; program.first_fp_reg as usize];
    collect_used_gp_regs(program, &mut used_gp)?;
    let mut gp_map = vec![None; program.first_fp_reg as usize];
    for reg in 0..max_gp_regs.min(program.first_fp_reg) {
        let reg = MachineReg(reg);
        gp_map[reg.0 as usize] = Some(reg);
    }
    let interference = build_gp_interference(program, program.first_fp_reg)?;
    let free_gp_slots = (MACHINE_FIXED_REG_COUNT..max_gp_regs)
        .map(MachineReg)
        .collect::<Vec<_>>();
    let used_gp_regs = used_gp
        .iter()
        .enumerate()
        .filter_map(|(reg, used)| used.then_some(MachineReg(reg as u16)))
        .collect::<Vec<_>>();
    let used_degrees = used_gp_regs
        .iter()
        .copied()
        .map(|reg| (reg, gp_interference_degree(&interference, reg)))
        .collect::<Vec<_>>();
    let peak_live_point = max_actual_live_gp_point(program, program.first_fp_reg)?;
    let peak_live_regs = peak_live_point.live_regs.len();
    let estimated_pressure = max_live_gp_regs(&interference, &used_gp, &gp_map);
    let extra_regs = collect_extra_gp_debug_info(
        program,
        &used_gp,
        &interference,
        &gp_map,
        &free_gp_slots,
        max_gp_regs,
    )?;

    Ok(GpCompactionDebugInfo {
        used_gp_regs,
        used_degrees,
        peak_live_regs,
        peak_live_point,
        estimated_pressure,
        extra_regs,
    })
}

#[cfg(test)]
pub(crate) fn debug_build_gp_interference(
    program: &MachineProgram,
) -> Result<Vec<Vec<bool>>, WasmError> {
    build_gp_interference(program, program.first_fp_reg)
}

#[cfg(test)]
fn max_actual_live_gp_point(
    program: &MachineProgram,
    first_fp_reg: u16,
) -> Result<GpPeakLivePoint, WasmError> {
    let liveness = build_gp_liveness(program, first_fp_reg)?;
    let mut peak = GpPeakLivePoint {
        block: MachineBlockId(0),
        stage: "unreachable".into(),
        live_regs: Vec::new(),
    };
    for block in &program.blocks {
        let block_index = block.id.as_usize();
        consider_peak_live(
            &mut peak,
            block.id,
            "block entry".into(),
            &liveness.live_in[block_index],
        );

        let mut live = liveness.live_out[block_index].clone();
        let mut term_uses = Vec::new();
        collect_term_gp_uses(&block.terminator, first_fp_reg, &mut term_uses)?;
        for reg in term_uses {
            mark_gp_live(&mut live, reg, first_fp_reg);
        }
        consider_peak_live(
            &mut peak,
            block.id,
            alloc::format!("before terminator {:?}", block.terminator),
            &live,
        );

        for (inst_index, inst) in block.ops.iter().enumerate().rev() {
            let mut uses = Vec::new();
            let mut defs = Vec::new();
            collect_inst_gp_uses_defs(&inst.kind, first_fp_reg, &mut uses, &mut defs)?;
            for reg in defs {
                clear_gp_live(&mut live, reg, first_fp_reg);
            }
            for reg in uses {
                mark_gp_live(&mut live, reg, first_fp_reg);
            }
            consider_peak_live(
                &mut peak,
                block.id,
                alloc::format!("before op{} {:?}", inst_index, inst.kind),
                &live,
            );
        }
    }
    Ok(peak)
}

#[cfg(test)]
fn consider_peak_live(
    peak: &mut GpPeakLivePoint,
    block: MachineBlockId,
    stage: String,
    live: &[bool],
) {
    let live_regs = live_regs_from_bitset(live);
    if live_regs.len() > peak.live_regs.len() {
        *peak = GpPeakLivePoint {
            block,
            stage,
            live_regs,
        };
    }
}

#[cfg(test)]
fn live_regs_from_bitset(live: &[bool]) -> Vec<MachineReg> {
    live.iter()
        .enumerate()
        .filter_map(|(reg, is_live)| is_live.then_some(MachineReg(reg as u16)))
        .collect()
}

#[cfg(test)]
fn collect_extra_gp_debug_info(
    program: &MachineProgram,
    used_gp: &[bool],
    interference: &[Vec<bool>],
    gp_map: &[Option<MachineReg>],
    free_gp_slots: &[MachineReg],
    max_gp_regs: u16,
) -> Result<Vec<GpExtraRegDebugInfo>, WasmError> {
    let mut extra_regs = Vec::new();
    for reg_index in max_gp_regs..program.first_fp_reg {
        if !used_gp[reg_index as usize] {
            continue;
        }
        extra_regs.push(GpExtraRegDebugInfo {
            reg: MachineReg(reg_index),
            candidate_slots: candidate_gp_slots(
                interference,
                gp_map,
                free_gp_slots,
                MachineReg(reg_index),
            ),
            pair_low_regs: Vec::new(),
            samples: Vec::new(),
        });
    }
    for block in &program.blocks {
        for param in &block.params {
            record_extra_gp_sample(
                &mut extra_regs,
                param.reg,
                None,
                alloc::format!("block{} param", block.id.0),
            );
        }
        for inst in &block.ops {
            collect_inst_extra_gp_debug(&mut extra_regs, &inst.kind);
        }
    }
    for info in &mut extra_regs {
        info.pair_low_regs.sort_by_key(|reg| reg.0);
        info.pair_low_regs.dedup();
    }
    Ok(extra_regs)
}

#[cfg(test)]
fn collect_inst_extra_gp_debug(extra_regs: &mut [GpExtraRegDebugInfo], inst: &MachineInstKind) {
    match inst {
        MachineInstKind::Move { dst, src, .. } => {
            record_extra_gp_sample(
                extra_regs,
                *dst,
                None,
                alloc::format!("move dst <- {:?}", src),
            );
            if let MachineValue::Reg(src) = src {
                record_extra_gp_sample(
                    extra_regs,
                    *src,
                    None,
                    alloc::format!("move src -> r{}", dst.0),
                );
            }
        }
        MachineInstKind::Load { dst, addr, .. } => {
            record_extra_gp_sample(
                extra_regs,
                *dst,
                None,
                alloc::format!("load dst from [r{}+{}]", addr.base.0, addr.offset),
            );
            record_extra_gp_sample(
                extra_regs,
                addr.base,
                None,
                alloc::format!("load base for r{}", dst.0),
            );
        }
        MachineInstKind::Store { addr, src, .. } => {
            record_extra_gp_sample(
                extra_regs,
                addr.base,
                None,
                alloc::format!("store base [r{}+{}]", addr.base.0, addr.offset),
            );
            if let MachineValue::Reg(src) = src {
                record_extra_gp_sample(
                    extra_regs,
                    *src,
                    None,
                    alloc::format!("store src -> [r{}+{}]", addr.base.0, addr.offset),
                );
            }
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            op,
        } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                alloc::format!("pair {op:?} dst_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *lhs_hi,
                *lhs_lo,
                alloc::format!("pair {op:?} lhs_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *rhs_hi,
                *rhs_lo,
                alloc::format!("pair {op:?} rhs_hi"),
            );
        }
        MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            sign,
            rem,
        } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                alloc::format!("pair divrem rem={rem} {sign:?} dst_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *lhs_hi,
                *lhs_lo,
                alloc::format!("pair divrem rem={rem} {sign:?} lhs_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *rhs_hi,
                *rhs_lo,
                alloc::format!("pair divrem rem={rem} {sign:?} rhs_hi"),
            );
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            op,
        } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                alloc::format!("pair {op:?} dst_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *src_hi,
                *src_lo,
                alloc::format!("pair {op:?} src_hi"),
            );
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            op,
            ..
        } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                alloc::format!("pair {op:?} dst_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *lhs_hi,
                *lhs_lo,
                alloc::format!("pair {op:?} lhs_hi"),
            );
        }
        MachineInstKind::Int64PairCompare {
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            kind,
            sign,
            ..
        } => {
            record_extra_gp_value_pair_sample(
                extra_regs,
                *lhs_hi,
                *lhs_lo,
                alloc::format!("pair cmp {kind:?}/{sign:?} lhs_hi"),
            );
            record_extra_gp_value_pair_sample(
                extra_regs,
                *rhs_hi,
                *rhs_lo,
                alloc::format!("pair cmp {kind:?}/{sign:?} rhs_hi"),
            );
        }
        MachineInstKind::ConvertI64PairToFloat {
            src_lo,
            src_hi,
            width,
            sign,
            ..
        } => {
            record_extra_gp_value_pair_sample(
                extra_regs,
                *src_hi,
                *src_lo,
                alloc::format!("convert i64->{width:?} {sign:?} src_hi"),
            );
        }
        MachineInstKind::ConvertFloatToI64Pair {
            dst_lo, dst_hi, op, ..
        } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                alloc::format!("convert {op:?} dst_hi"),
            );
        }
        MachineInstKind::ReinterpretF64ToI64Pair { dst_lo, dst_hi, .. } => {
            record_extra_gp_pair_sample(
                extra_regs,
                *dst_hi,
                *dst_lo,
                "reinterpret_f64 dst_hi".into(),
            );
        }
        MachineInstKind::ReinterpretI64PairToF64 { src_lo, src_hi, .. } => {
            record_extra_gp_value_pair_sample(
                extra_regs,
                *src_hi,
                *src_lo,
                "reinterpret_i64 src_hi".into(),
            );
        }
        _ => {}
    }
}

#[cfg(test)]
fn record_extra_gp_pair_sample(
    extra_regs: &mut [GpExtraRegDebugInfo],
    hi_reg: MachineReg,
    lo_reg: MachineReg,
    sample: String,
) {
    if let Some(info) = extra_regs.iter_mut().find(|info| info.reg == hi_reg) {
        if !info.pair_low_regs.contains(&lo_reg) {
            info.pair_low_regs.push(lo_reg);
        }
        record_sample(info, sample);
    }
}

#[cfg(test)]
fn record_extra_gp_value_pair_sample(
    extra_regs: &mut [GpExtraRegDebugInfo],
    hi_value: MachineValue,
    lo_value: MachineValue,
    sample: String,
) {
    let (MachineValue::Reg(hi_reg), MachineValue::Reg(lo_reg)) = (hi_value, lo_value) else {
        return;
    };
    record_extra_gp_pair_sample(extra_regs, hi_reg, lo_reg, sample);
}

#[cfg(test)]
fn record_extra_gp_sample(
    extra_regs: &mut [GpExtraRegDebugInfo],
    reg: MachineReg,
    lo_reg: Option<MachineReg>,
    sample: String,
) {
    if let Some(info) = extra_regs.iter_mut().find(|info| info.reg == reg) {
        if let Some(lo_reg) = lo_reg {
            if !info.pair_low_regs.contains(&lo_reg) {
                info.pair_low_regs.push(lo_reg);
            }
        }
        record_sample(info, sample);
    }
}

#[cfg(test)]
fn record_sample(info: &mut GpExtraRegDebugInfo, sample: String) {
    if info.samples.len() < 8 && !info.samples.contains(&sample) {
        info.samples.push(sample);
    }
}

fn push_gp_use(uses: &mut Vec<MachineReg>, reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg {
        uses.push(reg);
    }
}

fn push_gp_def(defs: &mut Vec<MachineReg>, reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg {
        defs.push(reg);
    }
}

fn push_gp_value_use(uses: &mut Vec<MachineReg>, value: MachineValue, first_fp_reg: u16) {
    if let MachineValue::Reg(reg) = value {
        push_gp_use(uses, reg, first_fp_reg);
    }
}

fn mark_gp_use(use_set: &mut [bool], def_set: &[bool], reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg && !def_set[reg.0 as usize] {
        use_set[reg.0 as usize] = true;
    }
}

fn mark_gp_def(def_set: &mut [bool], reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg {
        def_set[reg.0 as usize] = true;
    }
}

fn mark_gp_live(live: &mut [bool], reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg {
        live[reg.0 as usize] = true;
    }
}

fn clear_gp_live(live: &mut [bool], reg: MachineReg, first_fp_reg: u16) {
    if reg.0 < first_fp_reg {
        live[reg.0 as usize] = false;
    }
}

fn mark_gp_used(used: &mut [bool], reg: MachineReg) -> Result<(), WasmError> {
    if let Some(slot) = used.get_mut(reg.0 as usize) {
        *slot = true;
        return Ok(());
    }
    Ok(())
}

fn collect_value_gp_regs(value: MachineValue, used: &mut [bool]) -> Result<(), WasmError> {
    if let MachineValue::Reg(reg) = value {
        mark_gp_used(used, reg)?;
    }
    Ok(())
}

fn collect_branch_cond_gp_regs(
    cond: &MachineBranchCond,
    used: &mut [bool],
) -> Result<(), WasmError> {
    match cond {
        MachineBranchCond::Value(value) => collect_value_gp_regs(*value, used),
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            collect_value_gp_regs(*lhs, used)?;
            collect_value_gp_regs(*rhs, used)
        }
    }
}

fn collect_edge_gp_regs(edge: &MachineEdge, used: &mut [bool]) -> Result<(), WasmError> {
    for arg in &edge.args {
        collect_value_gp_regs(*arg, used)?;
    }
    Ok(())
}

fn collect_inst_gp_regs(inst: &MachineInstKind, used: &mut [bool]) -> Result<(), WasmError> {
    match inst {
        MachineInstKind::Move { dst, src, .. } | MachineInstKind::Convert { dst, src, .. } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*src, used)?;
        }
        MachineInstKind::FloatConst { .. } => {}
        MachineInstKind::Lea { dst, addr } => {
            mark_gp_used(used, *dst)?;
            mark_gp_used(used, addr.base)?;
        }
        MachineInstKind::Load { dst, addr, .. } => {
            mark_gp_used(used, *dst)?;
            mark_gp_used(used, addr.base)?;
        }
        MachineInstKind::Store { addr, src, .. } => {
            mark_gp_used(used, addr.base)?;
            collect_value_gp_regs(*src, used)?;
        }
        MachineInstKind::IntUnary { dst, src, .. }
        | MachineInstKind::FloatUnary { dst, src, .. } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*src, used)?;
        }
        MachineInstKind::IntBinary { dst, lhs, rhs, .. }
        | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
        | MachineInstKind::FloatBinary { dst, lhs, rhs, .. }
        | MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*lhs, used)?;
            collect_value_gp_regs(*rhs, used)?;
        }
        MachineInstKind::IntMulWide {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*lhs, used)?;
            collect_value_gp_regs(*rhs, used)?;
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*lhs_lo, used)?;
            collect_value_gp_regs(*lhs_hi, used)?;
            collect_value_gp_regs(*rhs_lo, used)?;
            collect_value_gp_regs(*rhs_hi, used)?;
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*src_lo, used)?;
            collect_value_gp_regs(*src_hi, used)?;
        }
        MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*lhs_lo, used)?;
            collect_value_gp_regs(*lhs_hi, used)?;
            collect_value_gp_regs(*rhs_lo, used)?;
            collect_value_gp_regs(*rhs_hi, used)?;
        }
        MachineInstKind::Int64PairCompare {
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*lhs_lo, used)?;
            collect_value_gp_regs(*lhs_hi, used)?;
            collect_value_gp_regs(*rhs_lo, used)?;
            collect_value_gp_regs(*rhs_hi, used)?;
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*lhs_lo, used)?;
            collect_value_gp_regs(*lhs_hi, used)?;
            collect_value_gp_regs(*rhs, used)?;
        }
        MachineInstKind::ConvertI64PairToFloat {
            dst,
            src_lo,
            src_hi,
            ..
        } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*src_lo, used)?;
            collect_value_gp_regs(*src_hi, used)?;
        }
        MachineInstKind::ConvertFloatToI64Pair {
            dst_lo,
            dst_hi,
            src,
            ..
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*src, used)?;
        }
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            mark_gp_used(used, *dst_lo)?;
            mark_gp_used(used, *dst_hi)?;
            collect_value_gp_regs(*src, used)?;
        }
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*src_lo, used)?;
            collect_value_gp_regs(*src_hi, used)?;
        }
        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
            ..
        } => {
            mark_gp_used(used, *dst)?;
            collect_value_gp_regs(*on_true, used)?;
            collect_value_gp_regs(*on_false, used)?;
            collect_value_gp_regs(*cond, used)?;
        }
        MachineInstKind::TrapIf { cond, .. } => collect_branch_cond_gp_regs(cond, used)?,
        MachineInstKind::CallHelper(_) => {}
    }
    Ok(())
}

fn collect_term_gp_regs(term: &MachineTerminator, used: &mut [bool]) -> Result<(), WasmError> {
    match term {
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
        MachineTerminator::Jump(edge) => collect_edge_gp_regs(edge, used)?,
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            collect_branch_cond_gp_regs(cond, used)?;
            collect_edge_gp_regs(then_edge, used)?;
            collect_edge_gp_regs(else_edge, used)?;
        }
        MachineTerminator::JumpTable { index, entries } => {
            collect_value_gp_regs(*index, used)?;
            for edge in entries {
                collect_edge_gp_regs(edge, used)?;
            }
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => mark_gp_used(used, *callee_frame_base)?,
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            collect_value_gp_regs(*callee_target, used)?;
            mark_gp_used(used, *callee_frame_base)?;
        }
    }
    Ok(())
}

fn remap_inst_regs(
    inst: &mut MachineInstKind,
    remap: &mut impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    match inst {
        MachineInstKind::Move { dst, src, .. } | MachineInstKind::Convert { dst, src, .. } => {
            *dst = remap(*dst)?;
            remap_value(src, remap)?;
        }
        MachineInstKind::FloatConst { dst, .. } => {
            *dst = remap(*dst)?;
        }
        MachineInstKind::Lea { dst, addr } => {
            *dst = remap(*dst)?;
            addr.base = remap(addr.base)?;
        }
        MachineInstKind::Load { dst, addr, .. } => {
            *dst = remap(*dst)?;
            addr.base = remap(addr.base)?;
        }
        MachineInstKind::Store { addr, src, .. } => {
            addr.base = remap(addr.base)?;
            remap_value(src, remap)?;
        }
        MachineInstKind::IntUnary { dst, src, .. }
        | MachineInstKind::FloatUnary { dst, src, .. } => {
            *dst = remap(*dst)?;
            remap_value(src, remap)?;
        }
        MachineInstKind::IntBinary { dst, lhs, rhs, .. }
        | MachineInstKind::IntCompare { dst, lhs, rhs, .. }
        | MachineInstKind::FloatBinary { dst, lhs, rhs, .. }
        | MachineInstKind::FloatCompare { dst, lhs, rhs, .. } => {
            *dst = remap(*dst)?;
            remap_value(lhs, remap)?;
            remap_value(rhs, remap)?;
        }
        MachineInstKind::IntMulWide {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(lhs, remap)?;
            remap_value(rhs, remap)?;
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(lhs_lo, remap)?;
            remap_value(lhs_hi, remap)?;
            remap_value(rhs_lo, remap)?;
            remap_value(rhs_hi, remap)?;
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(src_lo, remap)?;
            remap_value(src_hi, remap)?;
        }
        MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(lhs_lo, remap)?;
            remap_value(lhs_hi, remap)?;
            remap_value(rhs_lo, remap)?;
            remap_value(rhs_hi, remap)?;
        }
        MachineInstKind::Int64PairCompare {
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            *dst = remap(*dst)?;
            remap_value(lhs_lo, remap)?;
            remap_value(lhs_hi, remap)?;
            remap_value(rhs_lo, remap)?;
            remap_value(rhs_hi, remap)?;
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(lhs_lo, remap)?;
            remap_value(lhs_hi, remap)?;
            remap_value(rhs, remap)?;
        }
        MachineInstKind::Select {
            dst,
            on_true,
            on_false,
            cond,
            ..
        } => {
            *dst = remap(*dst)?;
            remap_value(on_true, remap)?;
            remap_value(on_false, remap)?;
            remap_value(cond, remap)?;
        }
        MachineInstKind::TrapIf { cond, .. } => {
            remap_branch_cond(cond, remap)?;
        }
        MachineInstKind::ConvertI64PairToFloat {
            dst,
            src_lo,
            src_hi,
            ..
        } => {
            *dst = remap(*dst)?;
            remap_value(src_lo, remap)?;
            remap_value(src_hi, remap)?;
        }
        MachineInstKind::ConvertFloatToI64Pair {
            dst_lo,
            dst_hi,
            src,
            ..
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(src, remap)?;
        }
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            *dst_lo = remap(*dst_lo)?;
            *dst_hi = remap(*dst_hi)?;
            remap_value(src, remap)?;
        }
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            *dst = remap(*dst)?;
            remap_value(src_lo, remap)?;
            remap_value(src_hi, remap)?;
        }
        MachineInstKind::CallHelper(_) => {}
    }
    Ok(())
}

fn remap_term_regs(
    term: &mut MachineTerminator,
    remap: &mut impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    match term {
        MachineTerminator::Jump(edge) => remap_edge(edge, remap)?,
        MachineTerminator::Branch {
            cond,
            then_edge,
            else_edge,
        } => {
            remap_branch_cond(cond, remap)?;
            remap_edge(then_edge, remap)?;
            remap_edge(else_edge, remap)?;
        }
        MachineTerminator::JumpTable { index, entries } => {
            remap_value(index, remap)?;
            for edge in entries {
                remap_edge(edge, remap)?;
            }
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            *callee_frame_base = remap(*callee_frame_base)?;
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            remap_value(callee_target, remap)?;
            *callee_frame_base = remap(*callee_frame_base)?;
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }
    Ok(())
}

fn remap_edge(
    edge: &mut MachineEdge,
    remap: &mut impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    for arg in &mut edge.args {
        remap_value(arg, remap)?;
    }
    Ok(())
}

fn remap_branch_cond(
    cond: &mut MachineBranchCond,
    remap: &mut impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    match cond {
        MachineBranchCond::Value(value) => remap_value(value, remap)?,
        MachineBranchCond::IntCompare { lhs, rhs, .. }
        | MachineBranchCond::FloatCompare { lhs, rhs, .. } => {
            remap_value(lhs, remap)?;
            remap_value(rhs, remap)?;
        }
    }
    Ok(())
}

fn remap_value(
    value: &mut MachineValue,
    remap: &mut impl FnMut(MachineReg) -> Result<MachineReg, WasmError>,
) -> Result<(), WasmError> {
    if let MachineValue::Reg(reg) = value {
        *reg = remap(*reg)?;
    }
    Ok(())
}

fn simulate_inst(
    state: &mut [MachineStorageFact],
    inst: &MachineInstKind,
    first_fp_reg: u16,
    _gp_word_mem_width: MachineMemWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match inst {
        MachineInstKind::Move { ty, dst, src } => {
            require_move_like_value_type(state, *src, *ty, first_fp_reg, "move source")?;
            write_reg_type(state, *dst, *ty, "move destination")?;
        }
        MachineInstKind::FloatConst { width, dst, .. } => {
            write_reg_type(
                state,
                *dst,
                float_storage_type(*width),
                "float const destination",
            )?;
        }
        MachineInstKind::Lea { dst, addr } => {
            require_reg_type(state, addr.base, MachineStorageType::GpWord, "address base")?;
            write_reg_type(state, *dst, MachineStorageType::GpWord, "lea destination")?;
        }
        MachineInstKind::Load { ty, dst, addr, .. } => {
            require_reg_type(state, addr.base, MachineStorageType::GpWord, "address base")?;
            write_reg_type(state, *dst, *ty, "load destination")?;
        }
        MachineInstKind::Store { ty, addr, src, .. } => {
            require_reg_type(state, addr.base, MachineStorageType::GpWord, "address base")?;
            require_move_like_value_type(state, *src, *ty, first_fp_reg, "store source")?;
        }
        MachineInstKind::IntUnary {
            width,
            op,
            dst,
            src,
            ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            require_value_type_if_reg(state, *src, operand_ty, "integer unary source")?;
            let result_ty = match op {
                MachineIntUnaryOp::Eqz => MachineStorageType::GpWord,
                _ => operand_ty,
            };
            write_reg_type(state, *dst, result_ty, "integer unary destination")?;
        }
        MachineInstKind::IntBinary {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            require_value_type_if_reg(state, *lhs, operand_ty, "integer binary lhs")?;
            require_value_type_if_reg(state, *rhs, operand_ty, "integer binary rhs")?;
            write_reg_type(state, *dst, operand_ty, "integer binary destination")?;
        }
        MachineInstKind::IntMulWide {
            dst_lo,
            dst_hi,
            lhs,
            rhs,
            ..
        } => {
            require_value_type_if_reg(state, *lhs, MachineStorageType::GpWord, "wide mul lhs")?;
            require_value_type_if_reg(state, *rhs, MachineStorageType::GpWord, "wide mul rhs")?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "wide mul low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "wide mul high destination",
            )?;
        }
        MachineInstKind::Int64PairBinary {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *lhs_lo,
                MachineStorageType::GpWord,
                "pair binary lhs low",
            )?;
            require_value_type_if_reg(
                state,
                *lhs_hi,
                MachineStorageType::GpWord,
                "pair binary lhs high",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_lo,
                MachineStorageType::GpWord,
                "pair binary rhs low",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_hi,
                MachineStorageType::GpWord,
                "pair binary rhs high",
            )?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "pair binary low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "pair binary high destination",
            )?;
        }
        MachineInstKind::Int64PairUnary {
            dst_lo,
            dst_hi,
            src_lo,
            src_hi,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *src_lo,
                MachineStorageType::GpWord,
                "pair unary low source",
            )?;
            require_value_type_if_reg(
                state,
                *src_hi,
                MachineStorageType::GpWord,
                "pair unary high source",
            )?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "pair unary low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "pair unary high destination",
            )?;
        }
        MachineInstKind::Int64PairDivRem {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *lhs_lo,
                MachineStorageType::GpWord,
                "pair div/rem lhs low",
            )?;
            require_value_type_if_reg(
                state,
                *lhs_hi,
                MachineStorageType::GpWord,
                "pair div/rem lhs high",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_lo,
                MachineStorageType::GpWord,
                "pair div/rem rhs low",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_hi,
                MachineStorageType::GpWord,
                "pair div/rem rhs high",
            )?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "pair div/rem low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "pair div/rem high destination",
            )?;
        }
        MachineInstKind::Int64PairShift {
            dst_lo,
            dst_hi,
            lhs_lo,
            lhs_hi,
            rhs,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *lhs_lo,
                MachineStorageType::GpWord,
                "pair shift lhs low",
            )?;
            require_value_type_if_reg(
                state,
                *lhs_hi,
                MachineStorageType::GpWord,
                "pair shift lhs high",
            )?;
            require_value_type_if_reg(state, *rhs, MachineStorageType::GpWord, "pair shift rhs")?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "pair shift low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "pair shift high destination",
            )?;
        }
        MachineInstKind::IntCompare {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = storage_type_for_int_width(*width, gp_word_int_width)?;
            require_value_type_if_reg(state, *lhs, operand_ty, "integer compare lhs")?;
            require_value_type_if_reg(state, *rhs, operand_ty, "integer compare rhs")?;
            write_reg_type(
                state,
                *dst,
                MachineStorageType::GpWord,
                "integer compare destination",
            )?;
        }
        MachineInstKind::Int64PairCompare {
            dst,
            lhs_lo,
            lhs_hi,
            rhs_lo,
            rhs_hi,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *lhs_lo,
                MachineStorageType::GpWord,
                "pair compare lhs low",
            )?;
            require_value_type_if_reg(
                state,
                *lhs_hi,
                MachineStorageType::GpWord,
                "pair compare lhs high",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_lo,
                MachineStorageType::GpWord,
                "pair compare rhs low",
            )?;
            require_value_type_if_reg(
                state,
                *rhs_hi,
                MachineStorageType::GpWord,
                "pair compare rhs high",
            )?;
            write_reg_type(
                state,
                *dst,
                MachineStorageType::GpWord,
                "pair compare destination",
            )?;
        }
        MachineInstKind::FloatUnary {
            width, dst, src, ..
        } => {
            let operand_ty = float_storage_type(*width);
            require_value_type_if_reg(state, *src, operand_ty, "float unary source")?;
            write_reg_type(state, *dst, operand_ty, "float unary destination")?;
        }
        MachineInstKind::FloatBinary {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = float_storage_type(*width);
            require_value_type_if_reg(state, *lhs, operand_ty, "float binary lhs")?;
            require_value_type_if_reg(state, *rhs, operand_ty, "float binary rhs")?;
            write_reg_type(state, *dst, operand_ty, "float binary destination")?;
        }
        MachineInstKind::FloatCompare {
            width,
            dst,
            lhs,
            rhs,
            ..
        } => {
            let operand_ty = float_storage_type(*width);
            require_value_type_if_reg(state, *lhs, operand_ty, "float compare lhs")?;
            require_value_type_if_reg(state, *rhs, operand_ty, "float compare rhs")?;
            write_reg_type(
                state,
                *dst,
                MachineStorageType::GpWord,
                "float compare destination",
            )?;
        }
        MachineInstKind::Convert { op, dst, src } => {
            let (src_ty, dst_ty) = storage_types_for_convert(*op);
            require_value_type_if_reg(state, *src, src_ty, "convert source")?;
            write_reg_type(state, *dst, dst_ty, "convert destination")?;
        }
        MachineInstKind::ConvertI64PairToFloat {
            width,
            dst,
            src_lo,
            src_hi,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *src_lo,
                MachineStorageType::GpWord,
                "i64-pair convert low source",
            )?;
            require_value_type_if_reg(
                state,
                *src_hi,
                MachineStorageType::GpWord,
                "i64-pair convert high source",
            )?;
            write_reg_type(
                state,
                *dst,
                float_storage_type(*width),
                "i64-pair convert destination",
            )?;
        }
        MachineInstKind::ConvertFloatToI64Pair {
            op,
            dst_lo,
            dst_hi,
            src,
        } => {
            let (src_ty, _) = storage_types_for_convert(*op);
            require_value_type_if_reg(state, *src, src_ty, "float-to-i64 convert source")?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "float-to-i64 convert low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "float-to-i64 convert high destination",
            )?;
        }
        MachineInstKind::ReinterpretF64ToI64Pair {
            dst_lo,
            dst_hi,
            src,
        } => {
            require_value_type_if_reg(
                state,
                *src,
                MachineStorageType::Fp64,
                "reinterpret f64->i64 source",
            )?;
            write_reg_type(
                state,
                *dst_lo,
                MachineStorageType::GpWord,
                "reinterpret f64->i64 low destination",
            )?;
            write_reg_type(
                state,
                *dst_hi,
                MachineStorageType::GpWord,
                "reinterpret f64->i64 high destination",
            )?;
        }
        MachineInstKind::ReinterpretI64PairToF64 {
            dst,
            src_lo,
            src_hi,
        } => {
            require_value_type_if_reg(
                state,
                *src_lo,
                MachineStorageType::GpWord,
                "reinterpret i64->f64 low source",
            )?;
            require_value_type_if_reg(
                state,
                *src_hi,
                MachineStorageType::GpWord,
                "reinterpret i64->f64 high source",
            )?;
            write_reg_type(
                state,
                *dst,
                MachineStorageType::Fp64,
                "reinterpret i64->f64 destination",
            )?;
        }
        MachineInstKind::Select {
            ty,
            dst,
            on_true,
            on_false,
            cond,
        } => {
            require_move_like_value_type(state, *on_true, *ty, first_fp_reg, "select on_true")?;
            require_move_like_value_type(state, *on_false, *ty, first_fp_reg, "select on_false")?;
            require_value_type_if_reg(
                state,
                *cond,
                MachineStorageType::GpWord,
                "select condition",
            )?;
            write_reg_type(state, *dst, *ty, "select destination")?;
        }
        MachineInstKind::TrapIf { cond, .. } => {
            require_branch_cond_types(state, *cond, gp_word_int_width)?;
        }
        MachineInstKind::CallHelper(_) => {}
    }

    Ok(())
}

fn simulate_term(
    state: &mut [MachineStorageFact],
    term: &MachineTerminator,
    first_fp_reg: u16,
    gp_word_mem_width: MachineMemWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match term {
        MachineTerminator::Jump(_) => {}
        MachineTerminator::Branch { cond, .. } => {
            require_branch_cond_types(state, *cond, gp_word_int_width)?;
        }
        MachineTerminator::JumpTable { index, .. } => {
            require_value_type_if_reg(
                state,
                *index,
                MachineStorageType::GpWord,
                "jump table index",
            )?;
        }
        MachineTerminator::CallDirect {
            callee_frame_base, ..
        } => {
            require_reg_type(
                state,
                *callee_frame_base,
                MachineStorageType::GpWord,
                "call frame base",
            )?;
        }
        MachineTerminator::CallIndirect {
            callee_target,
            callee_frame_base,
            ..
        } => {
            require_value_type_if_reg(
                state,
                *callee_target,
                storage_type_for_mem_width(gp_word_mem_width, false, gp_word_mem_width)?,
                "indirect callee target",
            )?;
            require_reg_type(
                state,
                *callee_frame_base,
                MachineStorageType::GpWord,
                "call frame base",
            )?;
        }
        MachineTerminator::Return | MachineTerminator::Trap { .. } => {}
    }

    let _ = first_fp_reg;
    Ok(())
}

fn require_block_params(
    state: &[MachineStorageFact],
    params: &[super::MachineBlockParam],
    context: &str,
) -> Result<(), WasmError> {
    for param in params {
        require_reg_type(
            state,
            param.reg,
            param.ty,
            &alloc::format!("{context} param r{}", param.reg.0),
        )?;
    }
    Ok(())
}

fn seed_block_params(
    state: &mut [MachineStorageFact],
    params: &[super::MachineBlockParam],
    context: &str,
) -> Result<(), WasmError> {
    for param in params {
        write_reg_type(
            state,
            param.reg,
            param.ty,
            &alloc::format!("{context} param r{}", param.reg.0),
        )?;
    }
    Ok(())
}

fn require_branch_cond_types(
    state: &[MachineStorageFact],
    cond: MachineBranchCond,
    gp_word_int_width: MachineIntWidth,
) -> Result<(), WasmError> {
    match cond {
        MachineBranchCond::Value(value) => {
            require_value_type_if_reg(state, value, MachineStorageType::GpWord, "branch value")?;
        }
        MachineBranchCond::IntCompare {
            width, lhs, rhs, ..
        } => {
            let operand_ty = storage_type_for_int_width(width, gp_word_int_width)?;
            require_value_type_if_reg(state, lhs, operand_ty, "branch integer lhs")?;
            require_value_type_if_reg(state, rhs, operand_ty, "branch integer rhs")?;
        }
        MachineBranchCond::FloatCompare {
            width, lhs, rhs, ..
        } => {
            let operand_ty = float_storage_type(width);
            require_value_type_if_reg(state, lhs, operand_ty, "branch float lhs")?;
            require_value_type_if_reg(state, rhs, operand_ty, "branch float rhs")?;
        }
    }

    Ok(())
}

fn require_move_like_value_type(
    state: &[MachineStorageFact],
    value: MachineValue,
    ty: MachineStorageType,
    first_fp_reg: u16,
    context: &str,
) -> Result<(), WasmError> {
    let MachineValue::Reg(reg) = value else {
        return Ok(());
    };
    let actual = require_known_reg_type(state, reg, context)?;
    if move_like_source_type_matches(actual, ty, reg.0 >= first_fp_reg) {
        return Ok(());
    }
    Err(WasmError::internal(alloc::format!(
        "machine register {} is {:?} at this 32-bit program point, but {} requires a source compatible with {:?}",
        reg.0,
        actual,
        context,
        ty,
    )))
}

fn require_value_type_if_reg(
    state: &[MachineStorageFact],
    value: MachineValue,
    ty: MachineStorageType,
    context: &str,
) -> Result<(), WasmError> {
    let MachineValue::Reg(reg) = value else {
        return Ok(());
    };
    require_reg_type(state, reg, ty, context)
}

fn require_reg_type(
    state: &[MachineStorageFact],
    reg: MachineReg,
    ty: MachineStorageType,
    context: &str,
) -> Result<(), WasmError> {
    let actual = require_known_reg_type(state, reg, context)?;
    if actual == ty {
        return Ok(());
    }
    Err(WasmError::internal(alloc::format!(
        "machine register {} is {:?} at this 32-bit program point, but {} requires {:?}",
        reg.0,
        actual,
        context,
        ty,
    )))
}

fn write_reg_type(
    state: &mut [MachineStorageFact],
    reg: MachineReg,
    ty: MachineStorageType,
    context: &str,
) -> Result<(), WasmError> {
    let slot = state.get_mut(reg.0 as usize).ok_or_else(|| {
        WasmError::internal(alloc::format!(
            "{context} references out-of-range machine register {}",
            reg.0,
        ))
    })?;
    *slot = MachineStorageFact::Known(ty);
    Ok(())
}

fn merge_block_state(
    slot: &mut Option<Vec<MachineStorageFact>>,
    incoming: Vec<MachineStorageFact>,
) -> Result<bool, WasmError> {
    match slot {
        None => {
            *slot = Some(incoming);
            Ok(true)
        }
        Some(existing) => {
            if existing.len() != incoming.len() {
                return Err(WasmError::internal(
                    "machine storage-flow state length mismatch across CFG merge".into(),
                ));
            }
            let mut changed = false;
            for (dst, src) in existing.iter_mut().zip(incoming.iter().copied()) {
                let merged = merge_storage_fact(*dst, src);
                if merged != *dst {
                    *dst = merged;
                    changed = true;
                }
            }
            Ok(changed)
        }
    }
}

fn merge_storage_fact(lhs: MachineStorageFact, rhs: MachineStorageFact) -> MachineStorageFact {
    match (lhs, rhs) {
        (MachineStorageFact::Known(a), MachineStorageFact::Known(b)) if a == b => {
            MachineStorageFact::Known(a)
        }
        (MachineStorageFact::Undefined, MachineStorageFact::Undefined) => {
            MachineStorageFact::Undefined
        }
        (MachineStorageFact::Conflict, _) | (_, MachineStorageFact::Conflict) => {
            MachineStorageFact::Conflict
        }
        (MachineStorageFact::Known(_), MachineStorageFact::Known(_))
        | (MachineStorageFact::Known(_), MachineStorageFact::Undefined)
        | (MachineStorageFact::Undefined, MachineStorageFact::Known(_)) => {
            MachineStorageFact::Conflict
        }
    }
}

fn float_storage_type(width: MachineFloatWidth) -> MachineStorageType {
    match width {
        MachineFloatWidth::F32 => MachineStorageType::Fp32,
        MachineFloatWidth::F64 => MachineStorageType::Fp64,
    }
}

fn require_known_reg_type(
    state: &[MachineStorageFact],
    reg: MachineReg,
    context: &str,
) -> Result<MachineStorageType, WasmError> {
    let fact = *state.get(reg.0 as usize).ok_or_else(|| {
        WasmError::internal(alloc::format!(
            "{context} references out-of-range machine register {}",
            reg.0,
        ))
    })?;

    match fact {
        MachineStorageFact::Known(existing) => Ok(existing),
        MachineStorageFact::Undefined => Err(WasmError::internal(alloc::format!(
            "machine register {} is undefined at this 32-bit program point, but {} requires a defined value",
            reg.0,
            context,
        ))),
        MachineStorageFact::Conflict => Err(WasmError::internal(alloc::format!(
            "machine register {} has conflicting incoming 32-bit storage types at this program point, but {} requires a defined value",
            reg.0,
            context,
        ))),
    }
}

fn move_like_source_type_matches(
    actual: MachineStorageType,
    ty: MachineStorageType,
    is_fp_reg: bool,
) -> bool {
    match (ty, is_fp_reg) {
        (MachineStorageType::Fp32, true) => actual == MachineStorageType::Fp32,
        (MachineStorageType::Fp64, true) => actual == MachineStorageType::Fp64,
        // Move-like GP transfers define zero-extend / low-word truncate across
        // the 32-bit legalizer boundary. Signed widening remains explicit via
        // MachineConvertOp.
        (MachineStorageType::GpWord, false) => {
            actual == MachineStorageType::GpWord || actual == MachineStorageType::GpI64
        }
        (MachineStorageType::GpI64, false) => {
            actual == MachineStorageType::GpI64 || actual == MachineStorageType::GpWord
        }
        (MachineStorageType::Fp32, false) => actual == MachineStorageType::GpWord,
        (MachineStorageType::Fp64, false) => actual == MachineStorageType::GpI64,
        (MachineStorageType::GpWord, true) => actual == MachineStorageType::Fp32,
        (MachineStorageType::GpI64, true) => actual == MachineStorageType::Fp64,
    }
}

fn storage_type_for_mem_width(
    width: MachineMemWidth,
    is_fp_reg: bool,
    gp_word_mem_width: MachineMemWidth,
) -> Result<MachineStorageType, WasmError> {
    if is_fp_reg {
        return match width {
            MachineMemWidth::U32 => Ok(MachineStorageType::Fp32),
            MachineMemWidth::U64 => Ok(MachineStorageType::Fp64),
            other => Err(WasmError::internal(alloc::format!(
                "unsupported FP memory width {:?} during 32-bit legalizer prep",
                other,
            ))),
        };
    }

    Ok(if width == gp_word_mem_width {
        MachineStorageType::GpWord
    } else {
        match width {
            MachineMemWidth::U8 | MachineMemWidth::U16 | MachineMemWidth::U32 => {
                MachineStorageType::GpWord
            }
            MachineMemWidth::U64 => MachineStorageType::GpI64,
        }
    })
}

fn storage_type_for_int_width(
    width: MachineIntWidth,
    gp_word_int_width: MachineIntWidth,
) -> Result<MachineStorageType, WasmError> {
    Ok(
        if width == gp_word_int_width || matches!(width, MachineIntWidth::I32) {
            MachineStorageType::GpWord
        } else {
            match width {
                MachineIntWidth::I64 => MachineStorageType::GpI64,
                MachineIntWidth::I32 => MachineStorageType::GpWord,
            }
        },
    )
}

fn storage_types_for_convert(op: MachineConvertOp) -> (MachineStorageType, MachineStorageType) {
    match op {
        MachineConvertOp::I32WrapI64 => (MachineStorageType::GpI64, MachineStorageType::GpWord),
        MachineConvertOp::I64ExtendI32S | MachineConvertOp::I64ExtendI32U => {
            (MachineStorageType::GpWord, MachineStorageType::GpI64)
        }
        MachineConvertOp::I32TruncF32S
        | MachineConvertOp::I32TruncF32U
        | MachineConvertOp::I32TruncSatF32S
        | MachineConvertOp::I32TruncSatF32U
        | MachineConvertOp::I32ReinterpretF32 => {
            (MachineStorageType::Fp32, MachineStorageType::GpWord)
        }
        MachineConvertOp::I32TruncF64S
        | MachineConvertOp::I32TruncF64U
        | MachineConvertOp::I32TruncSatF64S
        | MachineConvertOp::I32TruncSatF64U => {
            (MachineStorageType::Fp64, MachineStorageType::GpWord)
        }
        MachineConvertOp::I64TruncF32S
        | MachineConvertOp::I64TruncF32U
        | MachineConvertOp::I64TruncSatF32S
        | MachineConvertOp::I64TruncSatF32U => {
            (MachineStorageType::Fp32, MachineStorageType::GpI64)
        }
        MachineConvertOp::I64TruncF64S
        | MachineConvertOp::I64TruncF64U
        | MachineConvertOp::I64TruncSatF64S
        | MachineConvertOp::I64TruncSatF64U
        | MachineConvertOp::I64ReinterpretF64 => {
            (MachineStorageType::Fp64, MachineStorageType::GpI64)
        }
        MachineConvertOp::F32ConvertI32S
        | MachineConvertOp::F32ConvertI32U
        | MachineConvertOp::F32ReinterpretI32 => {
            (MachineStorageType::GpWord, MachineStorageType::Fp32)
        }
        MachineConvertOp::F32ConvertI64S | MachineConvertOp::F32ConvertI64U => {
            (MachineStorageType::GpI64, MachineStorageType::Fp32)
        }
        MachineConvertOp::F64ConvertI32S | MachineConvertOp::F64ConvertI32U => {
            (MachineStorageType::GpWord, MachineStorageType::Fp64)
        }
        MachineConvertOp::F64ConvertI64S
        | MachineConvertOp::F64ConvertI64U
        | MachineConvertOp::F64ReinterpretI64 => {
            (MachineStorageType::GpI64, MachineStorageType::Fp64)
        }
        MachineConvertOp::F32DemoteF64 => (MachineStorageType::Fp64, MachineStorageType::Fp32),
        MachineConvertOp::F64PromoteF32 => (MachineStorageType::Fp32, MachineStorageType::Fp64),
    }
}
