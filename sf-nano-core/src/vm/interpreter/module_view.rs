//! Parsed-module queries used by the interpreter's retained module view.
//! The JIT consumes these definitions when building its own runtime entities.
use crate::module::entities::{ElementInit, Function, FunctionDef, Tag, TagDef};
use crate::FunctionType;
use tracked_alloc::rc::Rc;

impl Function {
    pub(crate) fn def(&self) -> &FunctionDef {
        &self.def
    }
}

impl Function {
    #[inline]
    pub(crate) fn func_type_rc(&self) -> Rc<FunctionType> {
        match &self.def {
            FunctionDef::Local(spec) => spec.func_type_rc(),
            FunctionDef::Import { func_type, .. } => func_type.clone(),
        }
    }
}

impl Tag {
    pub(crate) fn type_index(&self) -> u32 {
        match self.def() {
            TagDef::Local(spec) => spec.type_index(),
            TagDef::Import { type_index, .. } => *type_index,
        }
    }
}

impl ElementInit {
    pub(crate) fn len(&self) -> usize {
        match self {
            ElementInit::FunctionIndexes(vec) => vec.len(),
            ElementInit::InitExprs { exprs, .. } => exprs.len(),
        }
    }
}
