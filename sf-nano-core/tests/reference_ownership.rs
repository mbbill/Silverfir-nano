use sf_nano_core::value_type::RefType;
use sf_nano_core::{Config, Engine, Import, Instance, Module, RefValue, RuntimeWorld, Tier, Value};

const PROVIDER: &str = r#"(module (func (export "f") (result i32) i32.const 7))"#;
const CONSUMER: &str = r#"(module
  (type $f (func (result i32)))
  (table 1 funcref)
  (global $calls (export "calls") (mut i32) (i32.const 0))
  (func (export "call") (param funcref) (result i32)
    global.get $calls i32.const 1 i32.add global.set $calls
    i32.const 0 local.get 0 table.set
    i32.const 0 call_indirect (type $f)))"#;

fn module(wat: &str) -> Module {
    Module::new("reference test", &wat::parse_str(wat).unwrap()).unwrap()
}

#[test]
fn function_references_reject_other_worlds_and_freed_owners() {
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut first = RuntimeWorld::new();
        let mut second = RuntimeWorld::new();
        let a = first.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        let b = second.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        let reference = first.instance(a).unwrap().get_func("f").unwrap().to_value();
        let foreign = second
            .instance(b)
            .unwrap()
            .get_func("f")
            .unwrap()
            .to_value();
        assert_ne!(reference, foreign);
        let consumer = first.instantiate(&engine, module(CONSUMER), &[]).unwrap();
        assert!(first.invoke(consumer, "call", &[foreign]).is_err());
        assert!(first.handle().invoke(consumer, 0, &[foreign]).is_err());
        assert_eq!(
            first
                .instance(consumer)
                .unwrap()
                .get_global("calls")
                .unwrap(),
            Some(Value::I32(0))
        );
        assert_eq!(
            first.invoke(consumer, "call", &[reference]).unwrap(),
            [Value::I32(7)]
        );
        first.free(a).unwrap();
        let replacement = first.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        assert_eq!(replacement.index(), a.index());
        assert_ne!(replacement.generation(), a.generation());
        assert!(first.invoke(consumer, "call", &[reference]).is_err());
        assert_eq!(
            first
                .instance(consumer)
                .unwrap()
                .get_global("calls")
                .unwrap(),
            Some(Value::I32(1))
        );
    }
}

#[test]
fn callbacks_preserve_references_and_reject_foreign_results() {
    let wasm = r#"(module
      (func $host (import "host" "identity") (param funcref) (result funcref))
      (func (export "call") (param funcref) (result funcref) local.get 0 call $host))"#;
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut first = RuntimeWorld::new();
        let mut second = RuntimeWorld::new();
        let provider = first.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        let other = second.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        let reference = first
            .instance(provider)
            .unwrap()
            .get_func("f")
            .unwrap()
            .to_value();
        let foreign = second
            .instance(other)
            .unwrap()
            .get_func("f")
            .unwrap()
            .to_value();
        for returned in [reference, foreign] {
            let host = Import::func("host", "identity", move |_, args, results| {
                assert_eq!(RefValue::from(args[0]), RefValue::from(reference));
                results[0] = returned;
                Ok(())
            });
            let consumer = first.instantiate(&engine, module(wasm), &[host]).unwrap();
            let result = first.invoke(consumer, "call", &[reference]);
            if returned == reference {
                assert_eq!(
                    RefValue::from(result.unwrap()[0]),
                    RefValue::from(reference)
                );
            } else {
                assert!(result.unwrap_err().is_trap());
            }
        }
    }
}

#[test]
fn global_imports_check_world_lifetime_and_dynamic_function_type() {
    let reader = r#"(module
      (global (import "host" "ref") funcref)
      (export "ref" (global 0)))"#;
    let typed_reader = r#"(module
      (type $different (func (param i64)))
      (global (import "host" "ref") (ref null $different)))"#;
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut first = RuntimeWorld::new();
        let mut second = RuntimeWorld::new();
        let provider = first.instantiate(&engine, module(PROVIDER), &[]).unwrap();
        let reference = first
            .instance(provider)
            .unwrap()
            .get_func("f")
            .unwrap()
            .to_value();
        let import = Import::global("host", "ref", reference, false);
        let consumer = first
            .instantiate(&engine, module(reader), &[import.clone()])
            .unwrap();
        let observed = first
            .instance(consumer)
            .unwrap()
            .get_global("ref")
            .unwrap()
            .unwrap();
        assert_eq!(RefValue::from(observed), RefValue::from(reference));
        assert!(second
            .instantiate(&engine, module(reader), &[import.clone()])
            .is_err());
        let forged_annotation = Value::Ref(reference.into(), RefType::nullable_concrete(0));
        assert!(first
            .instantiate(
                &engine,
                module(typed_reader),
                &[Import::global("host", "ref", forged_annotation, false),]
            )
            .is_err());
        first.free(provider).unwrap();
        assert!(first
            .instantiate(&engine, module(reader), &[import])
            .is_err());
    }
}

#[test]
fn start_callbacks_can_round_trip_the_initializing_instances_reference() {
    let wasm = r#"(module
      (func $identity (import "host" "identity") (param funcref) (result funcref))
      (func $f (export "f") (result i32) i32.const 7)
      (elem declare func $f)
      (global $saved (export "saved") (mut funcref) (ref.null func))
      (func $start ref.func $f call $identity global.set $saved)
      (start $start))"#;
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let import = Import::func("host", "identity", |_, args, results| {
            results[0] = args[0];
            Ok(())
        });
        let instance = Instance::from_module(&engine, module(wasm), &[import]).unwrap();
        let saved = RefValue::from(instance.get_global("saved").unwrap().unwrap());
        let function = RefValue::from(instance.get_func("f").unwrap().to_value());
        assert_eq!(saved, function);
    }
}

#[test]
fn exception_handles_and_propagation_are_world_bound() {
    let throwing = r#"(module (tag $e (param i32))
      (func (export "throw") i32.const 7 throw $e))"#;
    let catching = r#"(module (func $f (import "host" "throw"))
      (func (export "run") call $f))"#;
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut first = Instance::from_module(&engine, module(throwing), &[]).unwrap();
        let mut second = Instance::from_module(&engine, module(throwing), &[]).unwrap();
        let error = first.invoke("throw", &[]).unwrap_err();
        let other = second.invoke("throw", &[]).unwrap_err();
        assert_ne!(error.exception(), other.exception());
        assert_eq!(
            first.exception_fields(error.exception().unwrap()).unwrap(),
            [Value::I32(7)]
        );
        assert!(second
            .exception_fields(error.exception().unwrap())
            .is_none());
        let host = Import::func("host", "throw", move |_, _, _| Err(error.clone()));
        let mut caller = Instance::from_module(&engine, module(catching), &[host]).unwrap();
        assert!(caller.invoke("run", &[]).unwrap_err().is_trap());
    }
}

#[test]
fn portable_host_labels_and_nulls_do_not_claim_engine_ownership() {
    let wasm =
        r#"(module (func (export "identity") (param externref) (result externref) local.get 0))"#;
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        for reference in [RefValue::null(), RefValue::externref(13)] {
            let value = Value::Ref(reference, RefType::externref());
            for _ in 0..2 {
                let mut instance = Instance::from_module(&engine, module(wasm), &[]).unwrap();
                assert_eq!(instance.invoke("identity", &[value]).unwrap(), [value]);
            }
        }
        assert_eq!(RefValue::externref(13).host_id(), Some(13));
        assert_eq!(RefValue::null().host_id(), None);
    }
}

#[cfg(feature = "jit")]
#[test]
fn externalized_gc_references_still_check_world_and_owner_lifetime() {
    let provider = r#"(module (type $s (struct (field i32)))
      (func (export "make") (result externref) i32.const 7 struct.new $s extern.convert_any))"#;
    let consumer =
        r#"(module (func (export "identity") (param externref) (result externref) local.get 0))"#;
    let engine = Engine::new(Config::new().tier(Tier::Jit)).unwrap();
    let mut first = RuntimeWorld::new();
    let mut second = RuntimeWorld::new();
    let owner = first.instantiate(&engine, module(provider), &[]).unwrap();
    let target = first.instantiate(&engine, module(consumer), &[]).unwrap();
    let foreign = second.instantiate(&engine, module(consumer), &[]).unwrap();
    let reference = first.invoke(owner, "make", &[]).unwrap()[0];
    assert_eq!(RefValue::from(reference).host_id(), None);
    assert_eq!(
        first.invoke(target, "identity", &[reference]).unwrap(),
        [reference]
    );
    assert!(second.invoke(foreign, "identity", &[reference]).is_err());
    first.free(owner).unwrap();
    assert!(first.invoke(target, "identity", &[reference]).is_err());
}
