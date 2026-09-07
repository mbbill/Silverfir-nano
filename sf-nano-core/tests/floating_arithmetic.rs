//! Scalar arithmetic must preserve live inputs, signed zero and subnormals
//! when the backend reuses either operand's register for the result.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn scalar_arithmetic_preserves_live_operands_and_ieee_results() {
    for width in [32, 64] {
        let ty = format!("f{width}");
        let bits_ty = format!("i{width}");
        let mut module = String::from("(module");
        for op in ["add", "sub", "mul", "div"] {
            for live in ["a", "b"] {
                module.push_str(&format!(
                    r#"(func (export "{op}_{live}") (param $a {ty}) (param $b {ty})
                        (result {bits_ty} {bits_ty})
                        ({bits_ty}.reinterpret_{ty} ({ty}.{op} (local.get $a) (local.get $b)))
                        ({bits_ty}.reinterpret_{ty} (local.get ${live})))"#
                ));
            }
        }
        module.push(')');
        let wasm = wat::parse_str(module).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let ordinary = [0.0, -0.0, 1.5, -3.25, f64::INFINITY, f64::NEG_INFINITY];
        let mut inputs: Vec<u64> = ordinary
            .into_iter()
            .map(|x| {
                if width == 32 {
                    (x as f32).to_bits() as u64
                } else {
                    x.to_bits()
                }
            })
            .collect();
        inputs.extend(if width == 32 {
            [1, 0x8000_0001, 0x7fc0_1234, 0x7f80_1234]
        } else {
            [
                1,
                0x8000_0000_0000_0001,
                0x7ff8_0000_0000_1234,
                0x7ff0_0000_0000_1234,
            ]
        });
        let value = |x: u64| {
            if width == 32 {
                Value::F32(f32::from_bits(x as u32))
            } else {
                Value::F64(f64::from_bits(x))
            }
        };
        let bits_value = |x: u64| {
            if width == 32 {
                Value::I32(x as i32)
            } else {
                Value::I64(x as i64)
            }
        };
        for &a_bits in &inputs {
            for &b_bits in &inputs {
                for op in ["add", "sub", "mul", "div"] {
                    let expected = if width == 32 {
                        let (a, b) = (f32::from_bits(a_bits as u32), f32::from_bits(b_bits as u32));
                        (match op {
                            "add" => a + b,
                            "sub" => a - b,
                            "mul" => a * b,
                            "div" => a / b,
                            _ => unreachable!(),
                        }) as f64
                    } else {
                        let (a, b) = (f64::from_bits(a_bits), f64::from_bits(b_bits));
                        match op {
                            "add" => a + b,
                            "sub" => a - b,
                            "mul" => a * b,
                            "div" => a / b,
                            _ => unreachable!(),
                        }
                    };
                    for live in ["a", "b"] {
                        let result = instance
                            .invoke(&format!("{op}_{live}"), &[value(a_bits), value(b_bits)])
                            .unwrap();
                        if expected.is_nan() {
                            match result[0] {
                                Value::I32(x) => assert!(
                                    f32::from_bits(x as u32).is_nan() && x & 0x0040_0000 != 0
                                ),
                                Value::I64(x) => assert!(
                                    f64::from_bits(x as u64).is_nan()
                                        && x & 0x0008_0000_0000_0000 != 0
                                ),
                                _ => panic!("unexpected result type"),
                            }
                        } else {
                            let expected_bits = if width == 32 {
                                (expected as f32).to_bits() as u64
                            } else {
                                expected.to_bits()
                            };
                            assert_eq!(
                                result[0],
                                bits_value(expected_bits),
                                "{ty}.{op}({a_bits:x}, {b_bits:x})"
                            );
                        }
                        assert_eq!(
                            result[1],
                            bits_value(if live == "a" { a_bits } else { b_bits })
                        );
                    }
                }
            }
        }
    }
}
