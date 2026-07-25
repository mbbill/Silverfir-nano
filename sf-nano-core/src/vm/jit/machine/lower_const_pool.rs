use crate::collections;
use tracked_alloc::collections::BTreeMap;

use core::mem::{align_of, size_of};

use crate::vm::{
    jit::machine::machine_ir::{MachineConstData, MachineConstId},
    jit::runtime::runtime_call::RuntimeCallMeta,
};

pub(crate) struct ConstPoolBuilder {
    consts: collections::Vec<MachineConstData>,
    const_ids: BTreeMap<(u32, collections::Vec<u8>), MachineConstId>,
}

impl ConstPoolBuilder {
    pub(crate) fn new() -> Self {
        Self {
            consts: collections::Vec::new(),
            const_ids: BTreeMap::new(),
        }
    }

    pub(crate) fn finish(self) -> collections::Vec<MachineConstData> {
        self.consts
    }

    pub(crate) fn records(&self) -> &[MachineConstData] {
        &self.consts
    }

    pub(super) fn call_runtime_meta(&mut self, meta: RuntimeCallMeta) -> MachineConstId {
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
fn encode_record<T: Copy>(record: &T) -> (u32, collections::Vec<u8>) {
    let bytes =
        unsafe { core::slice::from_raw_parts((record as *const T).cast::<u8>(), size_of::<T>()) };
    (align_of::<T>() as u32, bytes.to_vec().into())
}
