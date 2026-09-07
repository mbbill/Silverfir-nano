//! Internal fixtures for deliberately malformed linking metadata.
use super::*;

impl Import {
    /// As above, with the source-context type index, so a cross-module
    /// rec-group identity check has both halves it needs.
    pub(crate) fn func_typed_with_context_and_index<F>(
        module: &str,
        name: &str,
        f: F,
        func_type: FunctionType,
        type_index: u32,
        type_ctx: TypeContext,
    ) -> Self
    where
        F: for<'a, 'b, 'c, 'd> Fn(
                &'a mut Caller<'b>,
                &'c [Value],
                &'d mut [Value],
            ) -> Result<(), WasmError>
            + 'static,
    {
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Func(ImportedFunction::Host {
                callback: HostCallback::from_host(f),
                func_type: Some(func_type),
                type_index,
                type_ctx: Some(type_ctx),
            }),
        }
    }

    pub(crate) fn global_with_state(module: &str, name: &str, state: ImportedGlobalState) -> Self {
        let mutable = state.global.mutable;
        Import {
            source: None,
            module: module.to_string(),
            name: name.to_string(),
            value: ImportValue::Global(ImportedGlobal::State(state), mutable),
        }
    }
}

use crate::module::Module;
use crate::vm::instance::Instance;
use crate::{Config, Engine, Tier};
#[test]
fn typed_host_import_keeps_recursive_group_identity() {
    let source = wat::parse_str(
        r#"(module
        (rec (type $a (func)) (type $b (func)))
        (func (type $a)))"#,
    )
    .unwrap();
    let target = wat::parse_str(
        r#"(module
        (rec (type $a (func)) (type $b (func)))
        (import "host" "call" (func (type $b))))"#,
    )
    .unwrap();
    let source = Module::new("source", &source).unwrap();
    for &tier in Tier::ALL {
        let import = Import::func_typed_with_context_and_index(
            "host",
            "call",
            |_, _, _| Ok(()),
            source.functions()[0].func_type().clone(),
            0,
            source.types().clone(),
        );
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        assert!(
            matches!(Instance::new(&engine, &target, &[import]), Err(e) if e.is_unlinkable()),
            "{tier:?}: equal signatures at different recursive-group positions must not link"
        );
        let matching = Import::func_typed_with_context_and_index(
            "host",
            "call",
            |_, _, _| Ok(()),
            source.functions()[0].func_type().clone(),
            1,
            source.types().clone(),
        );
        assert!(
            Instance::new(&engine, &target, &[matching]).is_ok(),
            "{tier:?}: matching recursive-group positions must link"
        );
    }
}
