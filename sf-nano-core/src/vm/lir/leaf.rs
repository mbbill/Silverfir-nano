//! LIR-native leaf-op vocabulary.
//!
//! This mirrors the reusable Wasm leaf-op set, but it is owned by LIR rather
//! than borrowing `PrimitiveOpKind` directly.

use crate::vm::wasm::primitive_op::{PrimitiveOpKind, for_each_primitive_op};

macro_rules! define_lir_leaf_ops {
    ($(
        $name:ident $( { $($field:ident : $ty:ty),* $(,)? } )? => ($pops:expr, $pushes:expr),
    )* ) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub enum LirLeafOp {
            $( $name $( { $($field : $ty),* } )?, )*
        }

        impl From<PrimitiveOpKind> for LirLeafOp {
            fn from(kind: PrimitiveOpKind) -> Self {
                match kind {
                    $( PrimitiveOpKind::$name $( { $($field),* } )? => Self::$name $( { $($field),* } )?, )*
                }
            }
        }
    };
}

for_each_primitive_op!(define_lir_leaf_ops);
