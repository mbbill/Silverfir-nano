//! Exercise non-destructive arithmetic, wrapping, and branches on its result.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn sums_preserve_inputs_and_branch_on_wrapped_result() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let wasm = wat::parse_str(format!(
            r#"(module
                (func (export "sum") (param $a {ty}) (param $b {ty})
                    (result {ty} i32 {ty} {ty}) (local $s {ty})
                    (local.set $s ({ty}.add (local.get $a) (local.get $b)))
                    (local.get $s)
                    (block $zero (result i32)
                        (br_if $zero (i32.const 17) ({ty}.eqz (local.get $s)))
                        drop
                        (i32.const 29))
                    (local.get $a) (local.get $b))
                (func (export "add_min") (param $a {ty}) (result {ty} {ty})
                    ({ty}.add (local.get $a) ({ty}.const -2147483648))
                    (local.get $a))
                (func (export "sub_min") (param $a {ty}) (result {ty} {ty})
                    ({ty}.sub (local.get $a) ({ty}.const -2147483648))
                    (local.get $a))
                (func (export "sub_one") (param $a {ty}) (result {ty} {ty})
                    ({ty}.sub (local.get $a) ({ty}.const 1))
                    (local.get $a)))"#
        ))
        .unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |x: i64| {
            if width == 32 {
                Value::I32(x as i32)
            } else {
                Value::I64(x)
            }
        };
        let inputs = [
            0,
            1,
            -1,
            i32::MAX as i64,
            i32::MIN as i64,
            i64::MAX,
            i64::MIN,
        ];
        for a in inputs {
            for b in inputs {
                let sum = a.wrapping_add(b);
                let is_zero = if width == 32 {
                    sum as i32 == 0
                } else {
                    sum == 0
                };
                let result = instance.invoke("sum", &[value(a), value(b)]).unwrap();
                assert_eq!(
                    result.as_slice(),
                    &[
                        value(sum),
                        Value::I32(if is_zero { 17 } else { 29 }),
                        value(a),
                        value(b)
                    ],
                    "{ty}: {a} + {b}"
                );
            }
            for (name, expected) in [
                ("add_min", a.wrapping_add(i32::MIN as i64)),
                ("sub_min", a.wrapping_sub(i32::MIN as i64)),
                ("sub_one", a.wrapping_sub(1)),
            ] {
                let result = instance.invoke(name, &[value(a)]).unwrap();
                assert_eq!(
                    result.as_slice(),
                    &[value(expected), value(a)],
                    "{ty}: {name}({a})"
                );
            }
        }
    }
}

#[test]
fn sum_of_wrapped_i64_inputs_discards_high_halves() {
    let wasm = wat::parse_str(
        r#"(module
            (func (export "run") (param $a i64) (param $b i64) (result i64 i64 i64)
                (i64.extend_i32_u (i32.add
                    (i32.wrap_i64 (local.get $a))
                    (i32.wrap_i64 (local.get $b))))
                (local.get $a) (local.get $b)))"#,
    )
    .unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    for (a, b) in [(0x1234_5678_ffff_ffff, 1), (i64::MIN, i64::MAX), (-1, -1)] {
        let result = instance
            .invoke("run", &[Value::I64(a), Value::I64(b)])
            .unwrap();
        assert_eq!(
            result.as_slice(),
            &[
                Value::I64((a as u32).wrapping_add(b as u32) as i64),
                Value::I64(a),
                Value::I64(b),
            ]
        );
    }
}

#[test]
fn arithmetic_conditions_survive_stores_and_use_the_requested_width() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut module = String::from("(module (memory 1)");
        for op in ["add", "sub", "and", "or", "xor"] {
            for compare in ["eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u"] {
                for zero_store in [false, true] {
                    let name = format!("{op}_{compare}_{zero_store}");
                    let stored = if zero_store {
                        format!("({ty}.const 0)")
                    } else {
                        "(local.get $c)".to_string()
                    };
                    module.push_str(&format!(
                        r#"(func (export "{name}") (param $a {ty}) (param $b {ty})
                            (result i32 {ty}) (local $c {ty})
                            (local.set $c ({ty}.{op} (local.get $a) (local.get $b)))
                            ({ty}.store (i32.const 0) {stored})
                            (if ({ty}.{compare} (local.get $c) ({ty}.const 0))
                                (then (i32.store (i32.const 16) (i32.const 17)))
                                (else (i32.store (i32.const 16) (i32.const 29))))
                            (i32.load (i32.const 16)) ({ty}.load (i32.const 0)))"#
                    ));
                }
            }
        }
        module.push(')');
        let wasm = wat::parse_str(module).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |x: i64| {
            if width == 32 {
                Value::I32(x as i32)
            } else {
                Value::I64(x)
            }
        };
        let inputs = [
            0,
            1,
            -1,
            i32::MIN as i64,
            i32::MAX as i64,
            i64::MIN,
            i64::MAX,
        ];
        for a in inputs {
            for b in inputs {
                for (op, result) in [
                    ("add", a.wrapping_add(b)),
                    ("sub", a.wrapping_sub(b)),
                    ("and", a & b),
                    ("or", a | b),
                    ("xor", a ^ b),
                ] {
                    let result = if width == 32 {
                        result as i32 as i64
                    } else {
                        result
                    };
                    for (compare, condition) in [
                        ("eq", result == 0),
                        ("ne", result != 0),
                        ("lt_s", result < 0),
                        ("lt_u", false),
                        ("gt_s", result > 0),
                        ("gt_u", result != 0),
                    ] {
                        for zero_store in [false, true] {
                            let name = format!("{op}_{compare}_{zero_store}");
                            let actual = instance.invoke(&name, &[value(a), value(b)]).unwrap();
                            assert_eq!(
                                actual.as_slice(),
                                &[
                                    Value::I32(if condition { 17 } else { 29 }),
                                    value(if zero_store { 0 } else { result }),
                                ],
                                "{ty} {name}({a}, {b})"
                            );
                        }
                    }
                }
            }
        }
    }
}
