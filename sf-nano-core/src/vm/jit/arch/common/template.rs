//! Backend-facing template JIT emission interface.
//!
//! The template JIT streams Wasm directly into backend emission primitives. It
//! does not build a MachineIR function; these methods are the narrow backend
//! contract used by the streaming template frontend.

use crate::{
    error::WasmError,
    vm::jit::machine::machine_ir::{
        MachineAddr, MachineBranchCond, MachineCompareKind, MachineConvertOp, MachineInst,
        MachineInstKind, MachineIntBinaryOp, MachineIntUnaryOp, MachineIntWidth,
        MachineLoadExtension, MachineMemWidth, MachineReg, MachineRegOwner, MachineSign,
        MachineStorageType, MachineTerminator, MachineTrapKind, MachineValue,
    },
};

use super::backend::ArchBackend;

pub(crate) const TEMPLATE_PATCH_CHAIN_END: usize = usize::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemplateBranchSense {
    IfTrue,
    IfFalse,
}

pub(crate) fn encode_template_chain_next(next: usize) -> Result<u32, WasmError> {
    if next == TEMPLATE_PATCH_CHAIN_END {
        return Ok(u32::MAX);
    }
    u32::try_from(next)
        .map_err(|_| WasmError::internal("template jit branch chain offset overflow"))
}

pub(crate) fn decode_template_chain_next(encoded: u32) -> usize {
    if encoded == u32::MAX {
        TEMPLATE_PATCH_CHAIN_END
    } else {
        encoded as usize
    }
}

#[allow(dead_code)]
pub(crate) fn template_i32_delta(from: usize, to: usize) -> Result<i32, WasmError> {
    let delta = to as i64 - from as i64;
    i32::try_from(delta).map_err(|_| WasmError::internal("template jit branch out of range"))
}

pub(crate) trait TemplateBackend<'a>: ArchBackend<'a> {
    const TEMPLATE_JUMP_BYTES: usize;

    /// Emit code that skips over the following `skip_bytes` bytes unless
    /// `cond` evaluates with `jump_when` sense. The skipped bytes are the
    /// unconditional template jump emitted immediately after this hook.
    fn emit_template_skip_unless(
        &mut self,
        cond: &MachineBranchCond,
        jump_when: TemplateBranchSense,
        skip_bytes: usize,
    ) -> Result<(), WasmError>;

    /// Emit an unresolved unconditional template jump and encode `next` in
    /// its placeholder bytes so unresolved sites form an intrusive chain.
    fn emit_template_jump_placeholder(&mut self, next: usize) -> Result<usize, WasmError>;

    fn read_template_jump_next(&self, site: usize) -> Result<usize, WasmError>;
    fn patch_template_jump(&mut self, site: usize, target: usize) -> Result<(), WasmError>;
    fn emit_template_jump_to_offset(&mut self, target: usize) -> Result<(), WasmError>;

    fn emit_template_return(&mut self) -> Result<(), WasmError> {
        self.lower_terminator(&MachineTerminator::Return, None)
    }

    fn emit_template_trap(&mut self, kind: MachineTrapKind) -> Result<(), WasmError> {
        self.lower_trap(kind);
        Ok(())
    }

    fn emit_template_load(
        &mut self,
        owner: MachineRegOwner,
        ty: MachineStorageType,
        dst: MachineReg,
        addr: MachineAddr,
        width: MachineMemWidth,
        extension: MachineLoadExtension,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::Load {
            owner,
            ty,
            dst,
            addr,
            width,
            extension,
        })
    }

    fn emit_template_store(
        &mut self,
        ty: MachineStorageType,
        addr: MachineAddr,
        width: MachineMemWidth,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::Store {
            ty,
            addr,
            width,
            src,
        })
    }

    fn emit_template_move(
        &mut self,
        owner: MachineRegOwner,
        ty: MachineStorageType,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::Move {
            owner,
            ty,
            dst,
            src,
        })
    }

    fn emit_template_convert(
        &mut self,
        op: MachineConvertOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::Convert { op, dst, src })
    }

    fn emit_template_int_binary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntBinaryOp,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::IntBinary {
            width,
            op,
            dst,
            lhs,
            rhs,
        })
    }

    fn emit_template_int_unary(
        &mut self,
        width: MachineIntWidth,
        op: MachineIntUnaryOp,
        dst: MachineReg,
        src: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::IntUnary {
            width,
            op,
            dst,
            src,
        })
    }

    fn emit_template_int_compare(
        &mut self,
        width: MachineIntWidth,
        kind: MachineCompareKind,
        sign: MachineSign,
        dst: MachineReg,
        lhs: MachineValue,
        rhs: MachineValue,
    ) -> Result<(), WasmError> {
        self.emit_template_inst(MachineInstKind::IntCompare {
            width,
            kind,
            sign,
            dst,
            lhs,
            rhs,
        })
    }

    fn emit_template_inst(&mut self, kind: MachineInstKind) -> Result<(), WasmError> {
        self.lower_inst(&MachineInst { kind })
    }
}
