//! ABI register plan and named roles for the example backend.
//!
//! This file defines ONLY:
//! - which physical register plays which role
//! - mapping from MachineIR registers to physical registers
//!
//! It does NOT emit code. Prologue/epilogue live in `frame.rs`.

use crate::{
    error::WasmError,
    vm::machine::machine_ir::{
        MachineReg, MACHINE_CTX_REG, MACHINE_FIXED_REG_COUNT, MACHINE_FP_REG,
        MACHINE_MEM0_BASE_REG, MACHINE_MEM0_SIZE_REG,
    },
};

use super::reg::{fp, gp, FpReg, GpReg};

// ── Named role structs ───────────────────────────────────────────────────────
//
// Every register used in emission code is accessed through one of these
// role structs, never through raw hardware names like `gp::X19`.

/// Registers pinned to MachineIR fixed roles for the lifetime of a function.
#[derive(Clone, Copy, Debug)]
pub(super) struct FixedGpRoles {
    pub ctx: GpReg,
    pub frame: GpReg,
    pub mem0_base: GpReg,
    pub mem0_size: GpReg,
}

/// Registers used to receive arguments at function entry.
#[derive(Clone, Copy, Debug)]
pub(super) struct EntryGpRoles {
    pub ctx_arg: GpReg,
    pub frame_arg: GpReg,
}

/// Registers used to return values to the caller.
#[derive(Clone, Copy, Debug)]
pub(super) struct ReturnGpRoles {
    pub status: GpReg,
    pub value0: GpReg,
}

/// Dedicated temporaries with fixed, well-known assignments.
/// These are NOT in any scratch pool — they have specific protocol uses.
#[derive(Clone, Copy, Debug)]
pub(super) struct DedicatedTemps {
    /// Used by the parallel-move cycle-break protocol (GP).
    pub gp_cycle_break: GpReg,
    /// Used by the parallel-move cycle-break protocol (FP).
    pub fp_cycle_break: FpReg,
    /// Used to hold the branch target in call sequences.
    pub call_target: GpReg,
}

/// Registers used to pass arguments to runtime helper calls.
#[derive(Clone, Copy, Debug)]
pub(super) struct HelperArgRoles {
    pub arg0: GpReg,
    pub arg1: GpReg,
    pub arg2: GpReg,
}

// ── Register plan ────────────────────────────────────────────────────────────

pub(super) struct RegisterPlan;

impl RegisterPlan {
    pub(super) const FIXED: FixedGpRoles = FixedGpRoles {
        ctx: gp::X19,
        frame: gp::X20,
        mem0_base: gp::X21,
        mem0_size: gp::X22,
    };

    pub(super) const ENTRY: EntryGpRoles = EntryGpRoles {
        ctx_arg: gp::X0,
        frame_arg: gp::X1,
    };

    pub(super) const RETURN: ReturnGpRoles = ReturnGpRoles {
        status: gp::X0,
        value0: gp::X1,
    };

    /// Dedicated temps are separate from the scratch pool.
    /// They have fixed protocol-level uses and must never be pooled.
    pub(super) const TEMPS: DedicatedTemps = DedicatedTemps {
        gp_cycle_break: gp::X17,
        fp_cycle_break: fp::D2,
        call_target: gp::X16,
    };

    pub(super) const HELPER_ARGS: HelperArgRoles = HelperArgRoles {
        arg0: gp::X0,
        arg1: gp::X1,
        arg2: gp::X2,
    };

    /// General-purpose GP scratch registers. Allocated through `ScratchPool`.
    /// These must NOT overlap with dedicated temps, fixed roles, or dynamic regs.
    pub(super) const GP_SCRATCHES: [GpReg; 2] = [gp::X9, gp::X10];

    /// General-purpose FP scratch registers. Allocated through `ScratchPool`.
    pub(super) const FP_SCRATCHES: [FpReg; 2] = [fp::D0, fp::D1];

    /// Allocatable GP registers mapped from MachineReg dynamic indices.
    pub(super) const DYNAMIC_GP: [GpReg; 6] = [
        gp::X23, gp::X24, gp::X25, gp::X26, gp::X27, gp::X28,
    ];

    /// Allocatable FP registers mapped from MachineReg FP indices.
    pub(super) const DYNAMIC_FP: [FpReg; 4] = [fp::D16, fp::D17, fp::D18, fp::D19];

    /// All platform-callee-saved GP registers that the backend may clobber.
    /// This includes fixed roles (X19-X22) AND dynamic GP regs (X23-X28).
    pub(super) const CALLEE_SAVED_GP: [GpReg; 12] = [
        gp::X19, gp::X20, gp::X21, gp::X22,
        gp::X23, gp::X24, gp::X25, gp::X26, gp::X27, gp::X28,
        gp::X29, gp::X30,
    ];

    pub(super) const CALLEE_SAVED_FP: [FpReg; 8] = [
        fp::D8, fp::D9, fp::D10, fp::D11, fp::D12, fp::D13, fp::D14, fp::D15,
    ];

    pub(super) const STACK_ALIGNMENT_BYTES: u32 = 16;

    // ── Capacity queries ─────────────────────────────────────────────────

    #[inline]
    pub(super) const fn max_gp_mapped_regs() -> usize {
        MACHINE_FIXED_REG_COUNT as usize + Self::DYNAMIC_GP.len()
    }

    #[inline]
    pub(super) const fn max_fp_machine_regs() -> usize {
        Self::DYNAMIC_FP.len()
    }

    #[inline]
    pub(super) const fn max_total_machine_regs() -> usize {
        Self::max_gp_mapped_regs() + Self::max_fp_machine_regs()
    }

    // ── Mapping: MachineReg → physical register ──────────────────────────

    #[inline]
    pub(super) fn map_fixed_gp(reg: MachineReg) -> GpReg {
        match reg {
            MACHINE_CTX_REG => Self::FIXED.ctx,
            MACHINE_FP_REG => Self::FIXED.frame,
            MACHINE_MEM0_BASE_REG => Self::FIXED.mem0_base,
            MACHINE_MEM0_SIZE_REG => Self::FIXED.mem0_size,
            _ => unreachable!("not a fixed machine reg"),
        }
    }

    #[inline]
    pub(super) fn map_gp(reg: MachineReg) -> Result<GpReg, WasmError> {
        if reg.0 < MACHINE_FIXED_REG_COUNT {
            return Ok(Self::map_fixed_gp(reg));
        }
        Self::DYNAMIC_GP
            .get((reg.0 - MACHINE_FIXED_REG_COUNT) as usize)
            .copied()
            .ok_or_else(|| {
                WasmError::invalid(alloc::format!(
                    "example backend has no GP mapping for machine reg {}",
                    reg.0
                ))
            })
    }

    #[inline]
    pub(super) fn map_fp(index: usize) -> Option<FpReg> {
        Self::DYNAMIC_FP.get(index).copied()
    }
}
