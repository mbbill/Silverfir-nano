//! Host values must be checked before their bits enter a typed Wasm frame.
use sf_nano_core::value_type::{RefType, ValueType};
use sf_nano_core::{Caller, FunctionType, RefValue};
use sf_nano_core::{Config, Engine, Import, Instance, Module, RuntimeWorld, Tier, Value};
use std::{cell::Cell, rc::Rc};

#[test]
fn narrow_and_wide_host_signatures_preserve_values_and_reject_bad_inputs() {
    // Eight is the interpreter's existing host-result limit. Five and eight
    // exercise heap conversion; four and below exercise inline conversion.
    for count in [0, 1, 4, 5, 8] {
        let types = " i32".repeat(count);
        let signature = if count == 0 {
            String::new()
        } else {
            format!("(param{types}) (result{types})")
        };
        let forward = (0..count)
            .map(|i| format!("local.get {i} "))
            .collect::<String>();
        let wasm = wat::parse_str(format!(
            "(module (import \"host\" \"echo\" (func $echo {signature}))
             (func (export \"run\") {signature} {forward} call $echo))"
        ))
        .unwrap();
        for &tier in Tier::ALL {
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            let calls = Rc::new(Cell::new(0));
            let counter = calls.clone();
            let imports = [Import::func("host", "echo", move |_, args, results| {
                counter.set(counter.get() + 1);
                results.copy_from_slice(args);
                Ok(())
            })];
            let mut instance = Instance::new(&engine, &wasm, &imports).unwrap();
            let mut args = (0..count)
                .map(|i| Value::I32(i as i32 + 10))
                .collect::<Vec<_>>();
            let func = instance.get_func("run").unwrap();
            let mut results = vec![Value::Unknown; count];
            instance.call(&func, &args, &mut results).unwrap();
            assert_eq!(results, args);
            assert_eq!(instance.invoke("run", &args).unwrap(), args);
            assert_eq!(calls.get(), 2);
            if let Some(last) = args.last_mut() {
                *last = Value::I64(99);
                let before = results.clone();
                assert!(instance.call(&func, &args, &mut results).is_err());
                assert_eq!(results, before);
                assert_eq!(calls.get(), 2);
            }
        }
    }
}

#[test]
fn wrong_numeric_arguments_are_rejected_before_guest_side_effects() {
    let wasm = wat::parse_str(
        r#"(module
          (global $calls (export "calls") (mut i32) (i32.const 0))
          (func (export "run") (param i32) (result i32)
            global.get $calls i32.const 1 i32.add global.set $calls
            local.get 0))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let func = instance.get_func("run").unwrap();
        for wrong in [
            Value::I64(7),
            Value::F32(7.0),
            Value::F64(7.0),
            Value::Unknown,
        ] {
            assert!(instance.invoke("run", &[wrong]).is_err(), "{tier:?}: named");
            assert!(
                instance.invoke_function_index(0, &[wrong]).is_err(),
                "{tier:?}: index"
            );
            let mut results = [Value::I32(99)];
            assert!(
                instance.call(&func, &[wrong], &mut results).is_err(),
                "{tier:?}: resolved"
            );
            assert_eq!(results, [Value::I32(99)]);
        }
        assert_eq!(instance.get_global("calls").unwrap(), Some(Value::I32(0)));
        assert_eq!(
            instance.invoke("run", &[Value::I32(7)]).unwrap(),
            [Value::I32(7)]
        );

        let mut world = RuntimeWorld::new();
        let id = world
            .instantiate(&engine, Module::new("run", &wasm).unwrap(), &[])
            .unwrap();
        assert!(world.invoke(id, "run", &[Value::I64(7)]).is_err());
        assert!(world.handle().invoke(id, 0, &[Value::I64(7)]).is_err());
        assert_eq!(
            world.instance(id).unwrap().get_global("calls").unwrap(),
            Some(Value::I32(0))
        );
    }
}

#[test]
fn imported_host_functions_reject_wrong_arguments_and_results() {
    let wasm = wat::parse_str(
        r#"(module
          (func $host (import "host" "run") (param i32) (result i32))
          (export "direct" (func $host))
          (global $calls (export "calls") (mut i32) (i32.const 0))
          (func (export "guest") (param i32) (result i32)
            local.get 0 call $host
            global.get $calls i32.const 1 i32.add global.set $calls))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        for wrong in [Some(Value::I64(7)), Some(Value::Unknown), None] {
            let calls = Rc::new(Cell::new(0));
            let observed = calls.clone();
            let import = Import::func("host", "run", move |_, _, results| {
                observed.set(observed.get() + 1);
                if let Some(wrong) = wrong {
                    results[0] = wrong;
                }
                Ok(())
            });
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            let mut instance = Instance::new(&engine, &wasm, &[import]).unwrap();
            assert!(
                instance.invoke("direct", &[Value::I64(7)]).is_err(),
                "{tier:?}: host args"
            );
            assert_eq!(calls.get(), 0);
            assert!(
                instance.invoke("direct", &[Value::I32(7)]).is_err(),
                "{tier:?}: host results {wrong:?}"
            );
            assert!(
                instance.invoke("guest", &[Value::I32(7)]).is_err(),
                "{tier:?}: guest continuation {wrong:?}"
            );
            assert_eq!(calls.get(), 2);
            assert_eq!(instance.get_global("calls").unwrap(), Some(Value::I32(0)));
        }
    }
}

#[test]
fn reference_arguments_check_kind_and_nullability() {
    let wasm = wat::parse_str(
        r#"(module
          (func (export "nullable") (param funcref) (result i32) i32.const 7)
          (func (export "nonnull") (param (ref func)) (result i32) i32.const 7))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        for wrong in [
            Value::I32(0),
            Value::Ref(RefValue::externref(0), RefType::funcref()),
            Value::Ref(RefValue::hostref(0), RefType::funcref()),
            Value::Ref(RefValue::null(), RefType::externref()),
        ] {
            assert!(
                instance.invoke("nullable", &[wrong]).is_err(),
                "{tier:?}: {wrong:?}"
            );
        }
        let null = Value::Ref(RefValue::null(), RefType::nullfuncref());
        assert_eq!(
            instance.invoke("nullable", &[null]).unwrap(),
            [Value::I32(7)]
        );
        assert!(
            instance.invoke("nonnull", &[null]).is_err(),
            "{tier:?}: nonnull"
        );
    }
}

#[test]
fn host_reference_results_check_kind_before_guest_continues() {
    let wasm = wat::parse_str(
        r#"(module
          (func $host (import "host" "ref") (result funcref))
          (export "direct" (func $host))
          (global $calls (export "calls") (mut i32) (i32.const 0))
          (func (export "guest") (result i32)
            call $host drop
            i32.const 1 global.set $calls i32.const 7))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        for wrong in [
            Value::I32(0),
            Value::Ref(RefValue::hostref(0), RefType::funcref()),
            Value::Ref(RefValue::null(), RefType::externref()),
        ] {
            let import = Import::func("host", "ref", move |_, _, results| {
                results[0] = wrong;
                Ok(())
            });
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            let mut instance = Instance::new(&engine, &wasm, &[import]).unwrap();
            for name in ["direct", "guest"] {
                let error = instance.invoke(name, &[]).expect_err("mistyped result");
                assert!(error.is_trap(), "{tier:?}: {name}: {error}");
            }
            assert_eq!(instance.get_global("calls").unwrap(), Some(Value::I32(0)));
        }
    }
}

#[test]
fn host_exception_reference_payloads_check_the_dynamic_kind() {
    let wasm = wat::parse_str(
        r#"(module
          (tag $tag (import "host" "tag") (param funcref))
          (func $throw (import "host" "throw"))
          (func (export "uncaught") call $throw)
          (func (export "guest") (result i32)
            (block $caught (result funcref)
              (try_table (catch $tag $caught)
                call $throw)
              ref.null func)
            drop i32.const 7))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        for valid in [false, true] {
            let (tag_import, tag) = Import::tag_typed_with_handle(
                "host",
                "tag",
                FunctionType::new(vec![ValueType::Ref(RefType::funcref())], vec![]),
            );
            let payload = if valid {
                Value::Ref(RefValue::null(), RefType::nullfuncref())
            } else {
                Value::Ref(RefValue::hostref(0), RefType::funcref())
            };
            let callback = Import::func("host", "throw", move |_, _, _| {
                Err(Caller::throw(tag, vec![payload]))
            });
            let engine = Engine::new(Config::new().tier(tier)).unwrap();
            let mut instance = Instance::new(&engine, &wasm, &[tag_import, callback]).unwrap();
            if valid {
                let error = instance
                    .invoke("uncaught", &[])
                    .expect_err("valid host throw");
                assert_eq!(error.exception_tag(), Some(tag));
                let fields = instance
                    .exception_fields(error.exception().unwrap())
                    .unwrap();
                assert!(matches!(&fields[..], [Value::Ref(handle, _)] if handle.is_null()));
            } else {
                let error = instance
                    .invoke("guest", &[])
                    .expect_err("wrong reference kind must not be catchable");
                assert!(error.is_trap(), "{tier:?}: {error}");
                assert_eq!(error.message(), "host threw mistyped exception");
            }
        }
    }
}

#[test]
fn start_callbacks_are_checked_before_world_registration() {
    let direct =
        wat::parse_str(r#"(module (func $start (import "host" "start")) (start $start))"#).unwrap();
    let guest = wat::parse_str(
        r#"(module
          (func $get (import "host" "get") (result i32 funcref))
          (global $number (export "number") (mut i32) (i32.const 0))
          (global $ref (export "ref") (mut funcref) (ref.null func))
          (func $start call $get global.set $ref global.set $number)
          (start $start))"#,
    )
    .unwrap();
    for &tier in Tier::ALL {
        let engine = Engine::new(Config::new().tier(tier)).unwrap();
        let called = Rc::new(Cell::new(false));
        let observed = called.clone();
        let start = Import::func("host", "start", move |_, _, _| {
            observed.set(true);
            Ok(())
        });
        Instance::new(&engine, &direct, &[start]).unwrap();
        assert!(called.get());
        for valid in [false, true] {
            let get = Import::func("host", "get", move |_, _, results| {
                results[0] = if valid { Value::I32(7) } else { Value::I64(7) };
                results[1] = Value::Ref(RefValue::null(), RefType::nullfuncref());
                Ok(())
            });
            let result = Instance::new(&engine, &guest, &[get]);
            if valid {
                let instance = result.unwrap();
                assert_eq!(instance.get_global("number").unwrap(), Some(Value::I32(7)));
                assert!(
                    matches!(instance.get_global("ref").unwrap(), Some(Value::Ref(handle, _)) if handle.is_null())
                );
            } else {
                let error = result.err().expect("mistyped start callback result");
                assert!(error.is_trap(), "{tier:?}: {error}");
                assert_eq!(error.message(), "host returned mistyped value");
            }
        }
    }
}
