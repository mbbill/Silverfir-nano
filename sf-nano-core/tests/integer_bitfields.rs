//! Bitfield extraction must respect scalar width and preserve live input lanes.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn unsigned_bitfields_match_shift_and_mask_at_width_boundaries() {
    for width in [32, 64] {
        let fields: &[(u32, u32)] = if width == 32 {
            &[(0, 8), (1, 1), (2, 4), (5, 7), (7, 16), (1, 31), (31, 1)]
        } else {
            &[(0, 32), (1, 32), (17, 40), (1, 63), (63, 1)]
        };
        let ty = format!("i{width}");
        let mut module = String::from("(module");
        for (index, &(lsb, bits)) in fields.iter().enumerate() {
            let mask = (1u64 << bits) - 1;
            for live in [false, true] {
                let suffix = if live { "_live" } else { "" };
                let second_result = if live { ty.as_str() } else { "" };
                let second_value = if live { "(local.get 0)" } else { "" };
                module.push_str(&format!(
                    r#"(func (export "f{index}{suffix}") (param {ty}) (result {ty} {second_result})
                        ({ty}.and ({ty}.shr_u (local.get 0) ({ty}.const {lsb})) ({ty}.const {mask}))
                        {second_value})"#
                ));
            }
        }
        module.push(')');
        let wasm = wat::parse_str(module).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        let mut inputs = vec![
            0,
            1,
            u64::MAX,
            1 << 31,
            1 << 32,
            1 << 63,
            0xaaaa_aaaa_5555_5555,
        ];
        let mut random = 0xf08d_371b_24e9_c6a5u64;
        for _ in 0..500 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            inputs.push(random);
        }
        let value = |x| {
            if width == 32 {
                Value::I32(x as i32)
            } else {
                Value::I64(x as i64)
            }
        };
        for raw in inputs {
            let input = if width == 32 {
                u64::from(raw as u32)
            } else {
                raw
            };
            for (index, &(lsb, bits)) in fields.iter().enumerate() {
                let expected = value((input >> lsb) & ((1u64 << bits) - 1));
                for live in [false, true] {
                    let name = format!("f{index}{}", if live { "_live" } else { "" });
                    let output = instance.invoke(&name, &[value(input)]).unwrap();
                    assert_eq!(
                        output[0], expected,
                        "{ty} field {lsb}:{bits}, input {input:x}"
                    );
                    if live {
                        assert_eq!(output[1], value(input));
                    }
                }
            }
        }
    }
}
