//! An external consumer must see the same allocation types with memprof on/off.
use sf_nano_core::value_type::ValueType;
use sf_nano_core::Module;
use sf_nano_core::{Config, Engine, FunctionType, Instance, RuntimeWorld, Tier, Value};

fn answer_wasm() -> Vec<u8> {
    wat::parse_str("(module (func (export \"answer\") (result i32) i32.const 42))")
        .expect("valid module")
}

#[test]
fn embedding_calls_return_standard_vectors_for_every_engine() {
    let wasm = answer_wasm();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).expect("engine");
        let mut instance = Instance::new(&engine, &wasm, &[]).expect("instance");
        let named: Vec<Value> = instance.invoke("answer", &[]).expect("named call");
        let indexed: Vec<Value> = instance
            .invoke_function_index(0, &[])
            .expect("indexed call");
        assert_eq!(named, vec![Value::I32(42)]);
        assert_eq!(indexed, named);

        let mut world = RuntimeWorld::new();
        let id = world
            .instantiate(&engine, Module::new("answer", &wasm).expect("module"), &[])
            .expect("world instance");
        let owned: Vec<Value> = world.invoke(id, "answer", &[]).expect("world call");
        let borrowed: Vec<Value> = world
            .handle()
            .invoke(id, 0, &[])
            .expect("world handle call");
        assert_eq!(owned, named);
        assert_eq!(borrowed, named);
    }
}

#[test]
fn function_types_accept_standard_vectors() {
    let ty = FunctionType::new(vec![ValueType::I32, ValueType::I64], vec![ValueType::I32]);
    assert_eq!(ty.params(), &[ValueType::I32, ValueType::I64]);
    assert_eq!(ty.results(), &[ValueType::I32]);
}

#[test]
fn function_handles_reject_other_instances() {
    let wasm = answer_wasm();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).expect("engine");
        let mut first = Instance::new(&engine, &wasm, &[]).expect("first");
        let mut second = Instance::new(&engine, &wasm, &[]).expect("second");
        let func = first.get_func("answer").expect("export");
        let mut results = [Value::I32(0)];
        assert!(second.call(&func, &[], &mut results).is_err());
        assert_eq!(results, [Value::I32(0)]);
        first.call(&func, &[], &mut results).expect("owner call");
        assert_eq!(results, [Value::I32(42)]);
    }
}

#[test]
fn world_handles_reject_ids_from_another_world() {
    let wasm = answer_wasm();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).expect("engine");
        let mut first = RuntimeWorld::new();
        let mut second = RuntimeWorld::new();
        let a = first
            .instantiate(&engine, Module::new("a", &wasm).unwrap(), &[])
            .unwrap();
        let b = second
            .instantiate(&engine, Module::new("b", &wasm).unwrap(), &[])
            .unwrap();
        assert!(second.handle().invoke(a, 0, &[]).is_err());
        assert!(second.invoke(a, "answer", &[]).is_err());
        assert!(second.free(a).is_err());
        assert_eq!(
            second.invoke(b, "answer", &[]).unwrap(),
            vec![Value::I32(42)]
        );
    }
}

#[test]
fn function_handles_do_not_follow_reused_instance_slots() {
    let wasm = answer_wasm();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).expect("engine");
        let mut world = RuntimeWorld::new();
        let old = world
            .instantiate(&engine, Module::new("old", &wasm).unwrap(), &[])
            .unwrap();
        let func = world.instance(old).unwrap().get_func("answer").unwrap();
        world.free(old).unwrap();
        let new = world
            .instantiate(&engine, Module::new("new", &wasm).unwrap(), &[])
            .unwrap();
        assert_eq!(old.index(), new.index());
        assert_ne!(old.generation(), new.generation());
        let mut results = [Value::I32(0)];
        assert!(world
            .instance_mut(new)
            .unwrap()
            .call(&func, &[], &mut results)
            .is_err());
        assert_eq!(results, [Value::I32(0)]);
    }
}

#[test]
fn opaque_errors_preserve_host_traps_and_exception_payloads() {
    use sf_nano_core::{Caller, Import, WasmError};
    let host_wasm = wat::parse_str(
        r#"(module
        (import "host" "fail" (func $fail))
        (func (export "run") call $fail))"#,
    )
    .unwrap();
    let throw_wasm = wat::parse_str(
        r#"(module
        (tag $e (export "exception") (param i32))
        (func (export "run") i32.const 42 throw $e))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let import = Import::func("host", "fail", |_, _, _| {
            Err(WasmError::trap("host failure"))
        });
        assert_eq!((import.module(), import.name()), ("host", "fail"));
        let mut host = Instance::new(&engine, &host_wasm, &[import]).unwrap();
        let error = host.invoke("run", &[]).unwrap_err();
        let standard: &dyn std::error::Error = &error;
        assert!(standard.to_string().contains("host failure"));
        assert!(error.is_trap());
        assert_eq!(error.message(), "host failure");
        assert_eq!(error.class(), "trap");
        assert!(error.exception().is_none());

        let mut throwing = Instance::new(&engine, &throw_wasm, &[]).unwrap();
        let error = throwing.invoke("run", &[]).unwrap_err();
        assert!(error.is_exception());
        assert_eq!(error.exception_tag(), throwing.tag_identity("exception"));
        assert_eq!(
            throwing.exception_fields(error.exception().unwrap()),
            Some(vec![Value::I32(42)])
        );

        // Host throws use the public Caller helper, without exposing the VM's
        // inbound exception transport representation.
        let host_throw = Caller::throw(error.exception_tag().unwrap(), vec![Value::I32(7)]);
        assert_eq!(host_throw.class(), "host_throw");
    }
    let exit = WasmError::exit_with_code(17);
    assert_eq!(exit.exit_code(), Some(17));
}

#[test]
fn limit_construction_uses_public_errors_and_immutable_width() {
    use sf_nano_core::{Limits, WasmError, WASM_PAGE_SIZE};
    assert_eq!(WASM_PAGE_SIZE, 65536);
    let error: WasmError = Limits::new(2, Some(1)).unwrap_err();
    assert_eq!(error.class(), "invalid");
    let narrow = Limits::new(1, Some(3)).unwrap();
    let wide = Limits::new_64(1, None).unwrap();
    assert_eq!(
        (narrow.min(), narrow.max(), narrow.is_64()),
        (1, Some(3), false)
    );
    assert!(wide.is_64());
}

#[cfg(feature = "interp")]
#[test]
fn interpreter_diagnostics_are_owned_display_data() {
    let engine = Engine::new(Config::new().tier(Tier::Interp)).unwrap();
    let wasm = wat::parse_str(
        "(module (func (export \"run\") (param i32) (result i32) local.get 0 i32.const 1 i32.add))",
    )
    .unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    assert_eq!(
        instance.invoke("run", &[Value::I32(1)]).unwrap(),
        [Value::I32(2)]
    );
    let stats = instance.interpreter_stats().unwrap();
    drop(instance);
    assert!(stats.engine_code_bytes > 0);
    assert_eq!(stats.dispatches.is_some(), cfg!(feature = "interp-count"));
    for ((first, second), count) in &stats.bigrams {
        assert!(!first.is_empty() && !second.is_empty() && *count > 0);
    }
    for (op, count) in &stats.slow_exits {
        assert!(!op.is_empty() && *count > 0);
    }
}

#[cfg(feature = "jit")]
#[test]
fn dropping_peer_instances_preserves_live_jit_traps() {
    let engine = Engine::new(Config::new().tier(Tier::Jit)).unwrap();
    let wasm = wat::parse_str("(module (memory 1) (func (export \"load\") (param i32) (result i32) local.get 0 i32.load))").unwrap();
    let mut survivor = Instance::new(&engine, &wasm, &[]).unwrap();
    assert_eq!(survivor.function_has_native_code("load"), Some(true));
    assert_eq!(survivor.function_has_native_code("missing"), None);
    for _ in 0..8 {
        let mut peer = Instance::new(&engine, &wasm, &[]).unwrap();
        assert!(peer
            .invoke("load", &[Value::I32(65536)])
            .unwrap_err()
            .is_trap());
        drop(peer);
        assert_eq!(
            survivor.invoke("load", &[Value::I32(0)]).unwrap(),
            [Value::I32(0)]
        );
        assert!(survivor
            .invoke("load", &[Value::I32(65536)])
            .unwrap_err()
            .is_trap());
    }
}

#[test]
fn tag_aliases_preserve_the_original_signature() {
    use sf_nano_core::Import;
    let original_wasm = wat::parse_str(
        "(module (import \"h\" \"tag\" (tag (param i32))) (export \"tag\" (tag 0)))",
    )
    .unwrap();
    let mismatched_wasm =
        wat::parse_str("(module (import \"other\" \"tag\" (tag (param i64))))").unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let (original, identity) = Import::tag_typed_with_handle(
            "h",
            "tag",
            FunctionType::new(vec![ValueType::I32], vec![]),
        );
        let instance =
            Instance::new(&engine, &original_wasm, core::slice::from_ref(&original)).unwrap();
        assert_eq!(instance.tag_identity("tag"), Some(identity));
        let alias = original.alias("other", "tag");
        assert!(
            Instance::new(&engine, &mismatched_wasm, &[alias]).is_err(),
            "{tier:?}: tag identity accepted under another signature"
        );
    }
}

#[test]
fn concrete_tag_parameters_require_a_preserved_type_context() {
    use sf_nano_core::{
        value_type::{HeapType, RefType},
        Import,
    };
    let provider = wat::parse_str(
        "(module (type $f (func (param i32))) (tag (export \"tag\") (param (ref null $f))))",
    )
    .unwrap();
    let good = wat::parse_str("(module (type (func (result f64))) (type $f (func (param i32))) (import \"other\" \"tag\" (tag (param (ref null $f)))))").unwrap();
    let bad = wat::parse_str("(module (type $f (func (param i64))) (import \"other\" \"tag\" (tag (param (ref null $f)))))").unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let host = Import::tag_typed(
            "other",
            "tag",
            FunctionType::new(
                vec![ValueType::Ref(RefType::new(true, HeapType::Concrete(0)))],
                vec![],
            ),
        );
        let error = match Instance::new(&engine, &bad, &[host]) {
            Ok(_) => panic!("context-free concrete tag must be rejected"),
            Err(error) => error,
        };
        assert!(error.is_unlinkable());
        let mut world = RuntimeWorld::new();
        let id = world
            .instantiate(&engine, Module::new("provider", &provider).unwrap(), &[])
            .unwrap();
        let export = world
            .instance(id)
            .unwrap()
            .get_export("tag")
            .unwrap()
            .unwrap();
        let imported = Import::new("other", "tag", export);
        world
            .instantiate(
                &engine,
                Module::new("good", &good).unwrap(),
                core::slice::from_ref(&imported),
            )
            .unwrap();
        assert!(world
            .instantiate(&engine, Module::new("bad", &bad).unwrap(), &[imported])
            .is_err());
    }
}
