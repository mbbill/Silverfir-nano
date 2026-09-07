//! Small-call expansion must preserve stack operands, call effects and traps.
use sf_nano_core::{Caller, Config, Engine, Import, Instance, Value, WasmError};

fn instance(source: &str) -> Instance {
    let engine = Engine::new(Config::new()).unwrap();
    Instance::new(&engine, &wat::parse_str(source).unwrap(), &[]).unwrap()
}

#[test]
fn small_calls_preserve_argument_order_and_reset_locals_in_loops() {
    let mut instance = instance(
        r#"(module
        (global $next (mut i32) (i32.const 0))
        (func $next (result i32)
            (global.set $next (i32.add (global.get $next) (i32.const 1)))
            (global.get $next))
        (func $pair (param $a i32) (param $b i32) (result i32)
            (i32.add (i32.mul (local.get $a) (i32.const 100)) (local.get $b)))
        (func $local (param $value i32) (result i32) (local $fresh i32)
            (local.get $fresh)
            (local.set $fresh (local.get $value)))
        (func (export "run") (param $n i32) (result i32)
            (local $i i32) (local $sum i32)
            (global.set $next (i32.const 0))
            (block $done (loop $again
                (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
                (local.set $sum (i32.add (local.get $sum)
                    (call $pair (call $next) (call $next))))
                (local.set $sum (i32.add (local.get $sum)
                    (call $local (i32.const 999))))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $again)))
            (local.get $sum)))"#,
    );
    for n in [0, 1, 2, 3, 20, 99, 2] {
        let expected = 101 * n * n + n;
        assert_eq!(
            instance.invoke("run", &[Value::I32(n)]).unwrap().as_slice(),
            &[Value::I32(expected)],
            "n={n}"
        );
    }
}

#[test]
fn mixed_multiple_results_keep_lower_stack_and_branch_table_targets() {
    let mut instance = instance(
        r#"(module
        (func $pair (param i64) (param f64) (result i64 f64) (local f64)
            (local.get 0) (f64.add (local.get 1) (local.get 2)))
        (func (export "run") (param $choice i32) (param $a i64) (param $b f64)
            (result i64 f64) (local $x i64) (local $y f64)
            (block $outer (block $middle (block $inner
                (br_table $inner $middle $outer (local.get $choice)))
                (local.set $a (i64.add (local.get $a) (i64.const 7)))))
            (i64.const 55)
            (if (result i64 f64) (i32.eqz (local.get $choice))
                (then (call $pair (local.get $a) (local.get $b)))
                (else (call $pair (i64.sub (local.get $a) (i64.const 3))
                    (f64.neg (local.get $b)))))
            (local.set $y) (local.set $x)
            (i64.add (local.get $x)) (local.get $y)))"#,
    );
    for choice in [0, 1, 2, -1, 100] {
        for a in [0i64, -1, i64::MIN, i64::MAX, 0x1234_5678_9abc_def0] {
            let result = instance
                .invoke("run", &[Value::I32(choice), Value::I64(a), Value::F64(3.5)])
                .unwrap();
            let x = a
                .wrapping_add(if choice == 0 { 7 } else { -3 })
                .wrapping_add(55);
            assert_eq!(
                result.as_slice(),
                &[
                    Value::I64(x),
                    Value::F64(if choice == 0 { 3.5 } else { -3.5 })
                ]
            );
        }
    }
}

#[test]
fn wrapper_calls_keep_memory_and_host_failure_before_continuation() {
    let source = r#"(module
        (import "env" "host" (func $host (param i32) (result i32)))
        (memory 1)
        (func $host_wrapper (param i32) (result i32)
            (i32.add (call $host (local.get 0)) (i32.const 3)))
        (func $memory_wrapper (param i32) (result i32)
            (i32.load (local.get 0)))
        (func $division_wrapper (param i32) (result i32)
            (i32.div_s (i32.const -2147483648) (local.get 0)))
        (func (export "run") (param $kind i32) (param $arg i32) (result i32)
            (local $value i32)
            (i32.store (i32.const 0) (i32.const 17))
            (local.set $value
                (if (result i32) (i32.eqz (local.get $kind))
                    (then (call $host_wrapper (local.get $arg)))
                    (else (if (result i32) (i32.eq (local.get $kind) (i32.const 1))
                        (then (call $memory_wrapper (local.get $arg)))
                        (else (call $division_wrapper (local.get $arg)))))))
            (i32.store (i32.const 0) (i32.const 99))
            (local.get $value))
        (func (export "marker") (result i32) (i32.load (i32.const 0))))"#;
    let engine = Engine::new(Config::new()).unwrap();
    let imports = [Import::func(
        "env",
        "host",
        |_: &mut Caller, args: &[Value], results: &mut [Value]| {
            if args[0] == Value::I32(-1) {
                return Err(WasmError::trap("host failure"));
            }
            results[0] = args[0];
            Ok(())
        },
    )];
    let mut instance = Instance::new(&engine, &wat::parse_str(source).unwrap(), &imports).unwrap();
    for (kind, arg, expected) in [
        (0, 8, Some(11)),
        (0, -1, None),
        (1, 65536, None),
        (1, 0, Some(17)),
        (2, 0, None),
        (2, -1, None),
        (2, 2, Some(-1073741824)),
        (0, 8, Some(11)),
    ] {
        let result = instance.invoke("run", &[Value::I32(kind), Value::I32(arg)]);
        if let Some(expected) = expected {
            assert_eq!(result.unwrap().as_slice(), &[Value::I32(expected)]);
        } else {
            assert!(result.is_err(), "kind={kind}, arg={arg}");
        }
        assert_eq!(
            instance.invoke("marker", &[]).unwrap().as_slice(),
            &[Value::I32(if expected.is_some() { 99 } else { 17 })]
        );
    }
}
