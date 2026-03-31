use alloc::{collections::BTreeMap, vec::Vec};

use crate::vm::{
    machine::machine_ir::{
        MachineConstData, MachineConstId, MachineExternBinding, MachineExternId,
        MachineHelperSymbol,
    },
    runtime::helper_meta::{
        encode_record, CallExternalMeta, CallIndirectExternalMeta,
    },
};

pub(super) struct SidecarBuilder {
    consts: Vec<MachineConstData>,
    externs: Vec<MachineExternBinding>,
    const_ids: BTreeMap<(u32, Vec<u8>), MachineConstId>,
    extern_ids: BTreeMap<MachineHelperSymbol, MachineExternId>,
}

impl SidecarBuilder {
    pub(super) fn new() -> Self {
        Self {
            consts: Vec::new(),
            externs: Vec::new(),
            const_ids: BTreeMap::new(),
            extern_ids: BTreeMap::new(),
        }
    }

    pub(super) fn finish(self) -> (Vec<MachineConstData>, Vec<MachineExternBinding>) {
        (self.consts, self.externs)
    }

    pub(super) fn extern_target(&mut self, symbol: MachineHelperSymbol) -> MachineExternId {
        if let Some(existing) = self.extern_ids.get(&symbol).copied() {
            return existing;
        }
        let id = MachineExternId(self.externs.len() as u32);
        self.externs.push(MachineExternBinding { id, symbol });
        self.extern_ids.insert(symbol, id);
        id
    }

    pub(super) fn call_external_meta(&mut self, meta: CallExternalMeta) -> MachineConstId {
        self.push_encoded(meta)
    }

    pub(super) fn call_indirect_external_meta(
        &mut self,
        meta: CallIndirectExternalMeta,
    ) -> MachineConstId {
        self.push_encoded(meta)
    }

    fn push_encoded<T: Copy>(&mut self, meta: T) -> MachineConstId {
        let (align, bytes) = encode_record(&meta);
        let key = (align, bytes);
        if let Some(existing) = self.const_ids.get(&key).copied() {
            return existing;
        }
        let id = MachineConstId(self.consts.len() as u32);
        self.consts.push(MachineConstData {
            id,
            align,
            bytes: key.1.clone(),
        });
        self.const_ids.insert(key, id);
        id
    }
}
