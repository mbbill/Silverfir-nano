//! Shared debug dump layout helpers.

use alloc::{format, string::String};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpRoot {
    pub root: String,
    pub module: String,
}

impl DumpRoot {
    #[inline]
    pub fn function_dir(&self, func_idx: u32) -> String {
        format!("{}/{}/func_{:04}", self.root, self.module, func_idx)
    }

    #[inline]
    pub fn lowered_ir_path(&self, func_idx: u32) -> String {
        format!("{}/lowered_ir.txt", self.function_dir(func_idx))
    }

    #[inline]
    pub fn native_dir(&self, func_idx: u32) -> String {
        format!("{}/native", self.function_dir(func_idx))
    }

    #[inline]
    pub fn fast_path(&self, func_idx: u32) -> String {
        format!("{}/fast_resolved.txt", self.function_dir(func_idx))
    }
}
