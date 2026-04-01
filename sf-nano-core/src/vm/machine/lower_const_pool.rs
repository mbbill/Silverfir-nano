use alloc::{collections::BTreeMap, vec::Vec};
use core::mem::{align_of, size_of};

use crate::vm::{
    machine::machine_ir::{
        MachineConstData, MachineConstId,
    },
    runtime::external::ExternalCallMeta,
};

pub(super) struct ConstPoolBuilder {
    consts: Vec<MachineConstData>,
    const_ids: BTreeMap<(u32, Vec<u8>), MachineConstId>,
}

impl ConstPoolBuilder {
    pub(super) fn new() -> Self {
        Self {
            consts: Vec::new(),
            const_ids: BTreeMap::new(),
        }
    }

    pub(super) fn finish(self) -> Vec<MachineConstData> {
        self.consts
    }

    pub(super) fn call_external_meta(&mut self, meta: ExternalCallMeta) -> MachineConstId {
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

#[inline]
fn encode_record<T: Copy>(record: &T) -> (u32, Vec<u8>) {
    let bytes =
        unsafe { core::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    (align_of::<T>() as u32, bytes.to_vec())
}
