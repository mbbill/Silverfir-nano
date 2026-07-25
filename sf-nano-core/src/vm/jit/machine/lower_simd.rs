use crate::{
    error::WasmError,
    vm::{
        jit::machine::machine_ir::{
            MachineInst, MachineInstKind, MachineLoadExtension, MachineReg, MachineRegOwner,
            MachineStorageType, MachineValue,
        },
        jit::middle::frame::FrameSlot,
    },
};

use super::lower_context::BlockLowerContext;

impl<'a> BlockLowerContext<'a> {
    #[cfg(sf_has_simd)]
    pub(super) fn ensure_v128_raw_abi(&self) -> Result<(), WasmError> {
        if self.gp_reg_width() != 8 {
            return Err(WasmError::internal(
                "SIMD lowering requires a 64-bit raw-handle ABI".into(),
            ));
        }
        Ok(())
    }

    #[cfg(sf_has_simd)]
    pub(super) fn emit_v128_from_raw_reg(
        &mut self,
        dst: MachineReg,
        raw: MachineReg,
        owner: MachineRegOwner,
    ) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::V128FromRaw {
                owner,
                dst,
                raw: MachineValue::Reg(raw),
            },
        });
    }

    #[cfg(sf_has_simd)]
    pub(super) fn emit_v128_to_raw_reg(&mut self, raw: MachineReg, src: MachineReg) {
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::V128ToRaw {
                dst: raw,
                src: MachineValue::Reg(src),
            },
        });
    }

    #[cfg(sf_has_simd)]
    pub(super) fn emit_load_frame_v128(
        &mut self,
        slot: FrameSlot,
        dst: MachineReg,
        owner: MachineRegOwner,
    ) -> Result<(), WasmError> {
        self.ensure_v128_raw_abi()?;
        let raw = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Load {
                owner: MachineRegOwner::LinearValue,
                ty: MachineStorageType::GpWord,
                dst: raw,
                addr: self.frame_addr(slot)?,
                width: self.gp_word_mem_width(),
                extension: MachineLoadExtension::None,
            },
        });
        self.emit_v128_from_raw_reg(dst, raw, owner);
        Ok(())
    }

    #[cfg(sf_has_simd)]
    pub(super) fn emit_store_frame_v128(
        &mut self,
        slot: FrameSlot,
        src: MachineReg,
    ) -> Result<(), WasmError> {
        self.ensure_v128_raw_abi()?;
        let raw = self.borrow_free_gp_dynamic_regs(1)?[0];
        self.emit_v128_to_raw_reg(raw, src);
        self.emit_machine_inst(MachineInst {
            kind: MachineInstKind::Store {
                ty: MachineStorageType::GpWord,
                addr: self.frame_addr(slot)?,
                width: self.gp_word_mem_width(),
                src: MachineValue::Reg(raw),
            },
        });
        Ok(())
    }
}
