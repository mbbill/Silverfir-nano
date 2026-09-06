//! Scalar shifts preserve modulo-width counts and still-live input values.
use sf_nano_core::{Config, Engine, Instance, Value};

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
