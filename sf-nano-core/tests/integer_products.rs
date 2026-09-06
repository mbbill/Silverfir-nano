//! Constant products must preserve input values and the requested word width.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn constant_products_preserve_live_inputs_and_zero_conditions() {
    let constants = [
        i64::MIN,
        -2_147_483_649,
        -2_147_483_648,
        -129,
        -128,
        -1,
        0,
        1,
        3,
        127,
        128,
        255,
        2_147_483_647,
        2_147_483_648,
        4_294_967_295,
        4_294_967_296,
        0x1234_5678_9abc_def0,
        i64::MAX,
    ];
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut module = String::from("(module");
        for (index, constant) in constants.iter().copied().enumerate() {
            let constant = if width == 32 {
                constant as i32 as i64
            } else {
                constant
            };
            for left in [false, true] {
                let input = "(local.get $a)";
                let literal = format!("({ty}.const {constant})");
                let operands = if left {
                    format!("{literal} {input}")
                } else {
                    format!("{input} {literal}")
                };
                module.push_str(&format!(
                    r#"
                    (func (export "live_{index}_{left}") (param $a {ty})
                        (result {ty} i32 {ty}) (local $p {ty})
                        (local.set $p ({ty}.mul {operands}))
                        (local.get $p) ({ty}.eqz (local.get $p)) (local.get $a))
                    (func (export "branch_{index}_{left}") (param $a {ty}) (result i32)
                        (if (result i32) ({ty}.eqz ({ty}.mul {operands}))
                            (then (i32.const 17)) (else (i32.const 29))))"#
                ));
            }
        }
        module.push(')');
        let wasm = wat::parse_str(module).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |n: i64| {
            if width == 32 {
                Value::I32(n as i32)
            } else {
                Value::I64(n)
            }
        };
        for raw in [
            0,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            1 << 32,
            -1 << 32,
            0x1234_5678_9abc_def0,
        ] {
            for (index, constant) in constants.iter().copied().enumerate() {
                let product = raw.wrapping_mul(constant);
                let zero = if width == 32 {
                    product as i32 == 0
                } else {
                    product == 0
                };
                for left in [false, true] {
                    let actual = instance
                        .invoke(&format!("live_{index}_{left}"), &[value(raw)])
                        .unwrap();
                    assert_eq!(
                        actual.as_slice(),
                        &[value(product), Value::I32(i32::from(zero)), value(raw)],
                        "i{width}: {raw} * {constant}, constant on left={left}"
                    );
                    let actual = instance
                        .invoke(&format!("branch_{index}_{left}"), &[value(raw)])
                        .unwrap();
                    assert_eq!(actual.as_slice(), &[Value::I32(if zero { 17 } else { 29 })]);
                }
            }
        }
    }
}
