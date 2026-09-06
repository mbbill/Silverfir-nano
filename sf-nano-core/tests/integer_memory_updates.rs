//! Non-atomic integer updates preserve width, wrapping and escaping values.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn immediate_memory_updates_match_wrapping_arithmetic() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let constants = [
            0,
            1,
            127,
            128,
            0x8000_0000,
            0xffff_ff7f,
            u64::MAX,
            i32::MIN as i64 as u64,
            i32::MAX as u64,
            i64::MAX as u64,
        ];
        let ops = ["add", "sub", "and", "or", "xor"];
        let mut wat = format!(
            r#"(module (memory 1)
            (func (export "set") (param i32 {ty})
                ({ty}.store (local.get 0) (local.get 1)))
            (func (export "get") (param i32) (result {ty})
                ({ty}.load (local.get 0)))"#
        );
        for (op_id, op) in ops.iter().enumerate() {
            for (constant_id, &raw) in constants.iter().enumerate() {
                let constant = if width == 32 {
                    u64::from(raw as u32)
                } else {
                    raw
                };
                for live in [false, true] {
                    let result = if live {
                        format!("(result {ty})")
                    } else {
                        String::new()
                    };
                    let returned = if live { "(local.get $value)" } else { "" };
                    wat.push_str(&format!(r#"
                        (func (export "f{op_id}_{constant_id}_{live}") (param $addr i32)
                            {result} (local $value {ty})
                            (local.set $value ({ty}.{op} ({ty}.load (local.get $addr)) ({ty}.const {constant})))
                            ({ty}.store (local.get $addr) (local.get $value))
                            {returned})"#));
                }
            }
        }
        wat.push(')');
        let wasm = wat::parse_str(wat).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let value = |raw: u64| {
            if width == 32 {
                Value::I32(raw as i32)
            } else {
                Value::I64(raw as i64)
            }
        };
        for address in [0, 1, 7, 128, 65536 - width / 8] {
            for raw in [
                0,
                1,
                u64::MAX,
                0x8000_0000,
                0x8000_0000_0000_0000,
                0x57c1_d08a_994e_b361,
            ] {
                for (op_id, op) in ops.iter().enumerate() {
                    for (constant_id, &constant) in constants.iter().enumerate() {
                        let expected = value(match *op {
                            "add" => raw.wrapping_add(constant),
                            "sub" => raw.wrapping_sub(constant),
                            "and" => raw & constant,
                            "or" => raw | constant,
                            "xor" => raw ^ constant,
                            _ => unreachable!(),
                        });
                        for live in [false, true] {
                            let neighbor = width == 32 && address <= 65528;
                            if neighbor {
                                instance
                                    .invoke(
                                        "set",
                                        &[Value::I32(address + 4), Value::I32(0x7123_4567)],
                                    )
                                    .unwrap();
                            }
                            instance
                                .invoke("set", &[Value::I32(address), value(raw)])
                                .unwrap();
                            let name = format!("f{op_id}_{constant_id}_{live}");
                            let returned = instance.invoke(&name, &[Value::I32(address)]).unwrap();
                            if live {
                                assert_eq!(returned.as_slice(), &[expected]);
                            } else {
                                assert!(returned.is_empty());
                            }
                            assert_eq!(
                                instance
                                    .invoke("get", &[Value::I32(address)])
                                    .unwrap()
                                    .as_slice(),
                                &[expected]
                            );
                            if neighbor {
                                assert_eq!(
                                    instance
                                        .invoke("get", &[Value::I32(address + 4)])
                                        .unwrap()
                                        .as_slice(),
                                    &[Value::I32(0x7123_4567)]
                                );
                            }
                        }
                    }
                }
            }
        }
        // Both the load and the writeback would cross the memory boundary.
        for address in [65536 - width / 8 + 1, 65536, -1] {
            assert!(instance
                .invoke("f0_1_false", &[Value::I32(address)])
                .is_err());
        }
    }
}
