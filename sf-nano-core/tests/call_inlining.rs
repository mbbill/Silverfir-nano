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

#[test]
fn structured_callees_preserve_return_floor_loop_and_function_branches() {
    let mut instance = instance(
        r#"(module
        (func $early (param i32) (result i32)
            (i64.const 123456789)
            (if (i32.eqz (local.get 0)) (then (return (i32.const 41))))
            drop (i32.const 33))
        (func $sum (param $n i32) (result i32) (local $sum i32)
            (block $done (loop $again
                (br_if $done (i32.eqz (local.get $n)))
                (local.set $sum (i32.add (local.get $sum) (local.get $n)))
                (local.set $n (i32.sub (local.get $n) (i32.const 1)))
                (br $again)))
            (local.get $sum))
        (func $switch (param i32) (result i64)
            (block (result i64)
                (block (result i64)
                    (i64.const 17) (local.get 0) (br_table 0 1 2))
                (i64.const 3) i64.add)
            (i64.const 7) i64.add)
        (func (export "run") (param $n i32) (param $choice i32) (param $older i64)
            (result i64)
            (local.get $older)
            (call $early (local.get $n)) i64.extend_i32_s i64.add
            (call $sum (local.get $n)) i64.extend_i32_s i64.add
            (call $switch (local.get $choice)) i64.add))"#,
    );
    for n in [0i32, 1, 2, 7, 19] {
        for choice in [0, 1, 2, -1, 9] {
            for older in [i64::MIN, -1, 0x1234_5678_9abc_def0, i64::MAX] {
                let expected = older
                    .wrapping_add(if n == 0 { 41 } else { 33 })
                    .wrapping_add(i64::from(n * (n + 1) / 2))
                    .wrapping_add(match choice {
                        0 => 27,
                        1 => 24,
                        _ => 17,
                    });
                assert_eq!(
                    instance
                        .invoke(
                            "run",
                            &[Value::I32(n), Value::I32(choice), Value::I64(older)]
                        )
                        .unwrap()
                        .as_slice(),
                    &[Value::I64(expected)]
                );
            }
        }
    }
}

#[test]
fn bounded_recursive_expansion_preserves_both_call_results() {
    let mut instance = instance(
        r#"(module
        (func $tree (export "run") (param $n i64) (result i64)
            (if (i64.le_s (local.get $n) (i64.const 1))
                (then (return (i64.add (local.get $n) (i64.const 2)))))
            (i64.add
                (i64.mul (call $tree (i64.sub (local.get $n) (i64.const 1))) (i64.const 3))
                (call $tree (i64.sub (local.get $n) (i64.const 2))))))"#,
    );
    for n in [-99, -3, -1, 0, 1, 2, 3, 8, 12, 17] {
        let expected = if n <= 1 {
            n + 2
        } else {
            let (mut a, mut b) = (2i64, 3i64);
            for _ in 2..=n {
                (a, b) = (b, b.wrapping_mul(3).wrapping_add(a));
            }
            b
        };
        assert_eq!(
            instance.invoke("run", &[Value::I64(n)]).unwrap().as_slice(),
            &[Value::I64(expected)]
        );
    }
    assert!(instance.invoke("run", &[Value::I64(1_000_000)]).is_err());
    assert_eq!(
        instance.invoke("run", &[Value::I64(2)]).unwrap().as_slice(),
        &[Value::I64(11)]
    );
}

#[test]
fn inlined_loop_parameters_keep_their_backedge_values() {
    let mut instance = instance(
        r#"(module
        (func $loop (param $n i32) (param $value i64) (result i64)
            (local.get $value) (local.get $n)
            (loop (param i64 i32) (result i64)
                (local.set $n)
                (i64.const 5) i64.add
                (local.get $n) (i32.const 1) i32.sub (local.tee $n)
                (local.get $n) (br_if 0)
                drop))
        (func (export "run") (param i32) (param i64) (param i64) (result i64)
            (local.get 2)
            (call $loop (local.get 0) (local.get 1)) i64.add))"#,
    );
    for n in [1i32, 2, 3, 7, 31] {
        for value in [i64::MIN, -1, 0, i64::MAX] {
            let older = 0x1234_5678_9abc_def0i64;
            assert_eq!(
                instance
                    .invoke(
                        "run",
                        &[Value::I32(n), Value::I64(value), Value::I64(older)]
                    )
                    .unwrap()
                    .as_slice(),
                &[Value::I64(
                    older.wrapping_add(value).wrapping_add(i64::from(n) * 5)
                )]
            );
        }
    }
}
