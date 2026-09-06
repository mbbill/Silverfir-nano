//! Scalar shifts preserve modulo-width counts and still-live input values.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn shifted_binary_operands_preserve_live_and_overlapping_inputs() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut wat = String::from("(module");
        let mut cases = Vec::new();
        for op in ["add", "sub", "and", "or", "xor"] {
            for shift in ["shl", "shr_u", "shr_s", "rotr"] {
                for amount in [0, 1, 7, 31, 32, 63, 64] {
                    for live in [false, true] {
                        let name = format!("{op}_{shift}_{amount}_{live}");
                        let extra_types = if live {
                            format!(" {ty} {ty}")
                        } else {
                            String::new()
                        };
                        let extra_values = if live {
                            "(local.get 0) (local.get 1)"
                        } else {
                            ""
                        };
                        wat.push_str(&format!(
                            r#"(func (export "{name}") (param {ty} {ty}) (result {ty}{extra_types})
                                ({ty}.{op} (local.get 0)
                                    ({ty}.{shift} (local.get 1) ({ty}.const {amount})))
                                {extra_values})"#
                        ));
                        cases.push((name, op, shift, amount, live));
                    }
                }
            }
        }
        wat.push(')');
        let wasm = wat::parse_str(&wat).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |raw| {
            if width == 32 {
                Value::I32(raw as i32)
            } else {
                Value::I64(raw as i64)
            }
        };
        let width_mask = if width == 32 {
            u64::from(u32::MAX)
        } else {
            u64::MAX
        };
        for (left, right) in [
            (0u64, 0u64),
            (1, 1),
            (u64::MAX, u64::MAX),
            (0x1234_5678_abcd_0000, 0xfedc_ba98_0123_ffff),
            (1 << 31, 1 << 63),
            (u64::MAX, 0),
            (0, u64::MAX),
        ] {
            for (name, op, shift, amount, live) in &cases {
                let right_word = right & width_mask;
                let count = amount & (width - 1);
                let shifted = match *shift {
                    "shl" => right_word.wrapping_shl(count),
                    "shr_u" => right_word.wrapping_shr(count),
                    "shr_s" if width == 32 => (right as i32).wrapping_shr(count) as u32 as u64,
                    "shr_s" => (right as i64).wrapping_shr(count) as u64,
                    "rotr" if width == 32 => u64::from((right as u32).rotate_right(count)),
                    "rotr" => right.rotate_right(count),
                    _ => unreachable!(),
                } & width_mask;
                let result = match *op {
                    "add" => left.wrapping_add(shifted),
                    "sub" => left.wrapping_sub(shifted),
                    "and" => left & shifted,
                    "or" => left | shifted,
                    "xor" => left ^ shifted,
                    _ => unreachable!(),
                };
                let mut expected = vec![value(result)];
                if *live {
                    expected.extend([value(left), value(right)]);
                }
                assert_eq!(
                    instance.invoke(name, &[value(left), value(right)]).unwrap(),
                    expected,
                    "{ty} {name}({left:x}, {right:x})"
                );
            }
        }
    }
}

#[test]
fn shifts_and_rotates_match_scalar_widths_with_live_inputs() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut wat = String::from("(module");
        for op in ["shl", "shr_u", "shr_s"] {
            wat.push_str(&format!(
                r#"
                (func (export "{op}") (param {ty} {ty}) (result {ty} {ty} {ty})
                    ({ty}.{op} (local.get 0) (local.get 1))
                    (local.get 0) (local.get 1))"#
            ));
        }
        for op in ["rotl", "rotr"] {
            for amount in [0, 1, 7, 31, 32, 33, 63, 64, 65, 255] {
                wat.push_str(&format!(
                    r#"
                    (func (export "{op}{amount}") (param {ty}) (result {ty} {ty})
                        ({ty}.{op} (local.get 0) ({ty}.const {amount})) (local.get 0))"#
                ));
            }
        }
        wat.push(')');
        let wasm = wat::parse_str(&wat).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |raw| {
            if width == 32 {
                Value::I32(raw as i32)
            } else {
                Value::I64(raw as i64)
            }
        };
        for raw in [
            0u64,
            1,
            u64::MAX,
            1 << 31,
            1 << 32,
            1 << 63,
            0x1234_5678_9abc_def0,
        ] {
            for amount in [
                0u64,
                1,
                7,
                31,
                32,
                33,
                63,
                64,
                65,
                255,
                0x8123_4567_89ab_cde0,
            ] {
                let n = amount as u32;
                for op in ["shl", "shr_u", "shr_s"] {
                    let expected = if width == 64 {
                        match op {
                            "shl" => raw.wrapping_shl(n),
                            "shr_u" => raw.wrapping_shr(n),
                            "shr_s" => (raw as i64).wrapping_shr(n) as u64,
                            _ => unreachable!(),
                        }
                    } else {
                        let raw = raw as u32;
                        u64::from(match op {
                            "shl" => raw.wrapping_shl(n),
                            "shr_u" => raw.wrapping_shr(n),
                            "shr_s" => (raw as i32).wrapping_shr(n) as u32,
                            _ => unreachable!(),
                        })
                    };
                    let result = instance.invoke(op, &[value(raw), value(amount)]).unwrap();
                    assert_eq!(
                        result,
                        vec![value(expected), value(raw), value(amount)],
                        "{ty}.{op}({raw:x}, {amount:x})"
                    );
                }
            }
            for amount in [0, 1, 7, 31, 32, 33, 63, 64, 65, 255] {
                for op in ["rotl", "rotr"] {
                    let expected = if width == 64 {
                        if op == "rotl" {
                            raw.rotate_left(amount)
                        } else {
                            raw.rotate_right(amount)
                        }
                    } else {
                        let raw = raw as u32;
                        u64::from(if op == "rotl" {
                            raw.rotate_left(amount)
                        } else {
                            raw.rotate_right(amount)
                        })
                    };
                    let result = instance
                        .invoke(&format!("{op}{amount}"), &[value(raw)])
                        .unwrap();
                    assert_eq!(
                        result,
                        vec![value(expected), value(raw)],
                        "{ty}.{op}({raw:x}, {amount})"
                    );
                }
            }
        }
    }
}
