//! LIR-native leaf-op vocabulary.
//!
//! This mirrors the reusable Wasm leaf-op set, but it is owned by LIR rather
//! than borrowing `CoreOpKind` directly.

use crate::vm::wasm::core_op::{CoreOpKind, for_each_core_op};

macro_rules! define_lir_leaf_ops {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum LirLeafOp {
            $( $name $( { $($field : $ty),* } )?, )*
        }

        impl From<CoreOpKind> for LirLeafOp {
            fn from(kind: CoreOpKind) -> Self {
                match kind {
                    $( CoreOpKind::$name $( { $($field),* } )? => Self::$name $( { $($field),* } )?, )*
                }
            }
        }
    };
}

for_each_core_op!(define_lir_leaf_ops);
