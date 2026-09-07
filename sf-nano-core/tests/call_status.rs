//! Calls preserve success and failure independently of the returned value.
use sf_nano_core::{Caller, Config, Engine, Import, Instance, Value, WasmError};

#[test]
fn nested_direct_indirect_and_tail_calls_propagate_errors_before_continuations() {
    let mut source = String::from(
        r#"(module
        (type $t (func (param i32) (result i32)))
        (import "env" "callback" (func $callback (param i32) (result i32)))
        (memory 1)
        (table 2 funcref)
        (elem (i32.const 0) $work $other)
        (func $other (param i64) (result i64) (local.get 0))
        (func $work (type $t) (param $mode i32) (result i32)
            (if (i32.eqz (local.get $mode)) (then (return (i32.const 37))))
            (if (i32.eq (local.get $mode) (i32.const 1)) (then unreachable))
            (if (i32.eq (local.get $mode) (i32.const 2))
                (then (return (i32.div_s (i32.const 37) (i32.const 0)))))
            (if (i32.eq (local.get $mode) (i32.const 3))
                (then (return (i32.load (i32.const 65536)))))
            (call $callback (local.get $mode)))
        (func $tail (type $t) (param i32) (result i32)
            (return_call $work (local.get 0)))
        (func $nested (param $depth i32) (param $mode i32) (result i32)
            (if (i32.eqz (local.get $depth))
                (then (return (call $work (local.get $mode)))))
            (i32.add (i32.const 1)
                (call $nested (i32.sub (local.get $depth) (i32.const 1)) (local.get $mode))))
        (func (export "marker") (result i32) (i32.load (i32.const 0)))"#,
    );
    for (name, call) in [
        ("direct", "(call $work (local.get $mode))"),
        (
            "indirect",
            "(call_indirect (type $t) (local.get $mode) (local.get $index))",
        ),
        ("tail", "(call $tail (local.get $mode))"),
        ("nested", "(call $nested (i32.const 32) (local.get $mode))"),
    ] {
        source.push_str(&format!(
            r#"(func (export "{name}") (param $mode i32) (param $index i32) (result i32)
                (local $result i32)
                (i32.store (i32.const 0) (i32.const 0))
                (local.set $result {call})
                (i32.store (i32.const 0) (i32.const 99))
                (local.get $result))"#
        ));
    }
    source.push(')');
    let wasm = wat::parse_str(source).unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let imports = [Import::func(
        "env",
        "callback",
        |_: &mut Caller, args: &[Value], results: &mut [Value]| {
            if args[0] == Value::I32(4) {
                return Err(WasmError::trap("requested host failure"));
            }
            results[0] = Value::I32(-11);
            Ok(())
        },
    )];
    let mut instance = Instance::new(&engine, &wasm, &imports).unwrap();
    for name in ["direct", "indirect", "tail", "nested"] {
        for mode in [0, 1, 0, 2, 0, 3, 0, 4, 5, 0] {
            let result = instance.invoke(name, &[Value::I32(mode), Value::I32(0)]);
            let success = mode == 0 || mode == 5;
            if success {
                let value = if mode == 0 { 37 } else { -11 };
                let value = value + if name == "nested" { 32 } else { 0 };
                assert_eq!(result.unwrap().as_slice(), &[Value::I32(value)]);
            } else {
                assert!(result.is_err(), "{name}, mode={mode}");
            }
            assert_eq!(
                instance.invoke("marker", &[]).unwrap().as_slice(),
                &[Value::I32(if success { 99 } else { 0 })],
                "{name}, mode={mode}"
            );
        }
    }
    for index in [1, 2, -1] {
        assert!(instance
            .invoke("indirect", &[Value::I32(0), Value::I32(index)])
            .is_err());
        assert_eq!(
            instance.invoke("marker", &[]).unwrap().as_slice(),
            &[Value::I32(0)]
        );
    }
}
