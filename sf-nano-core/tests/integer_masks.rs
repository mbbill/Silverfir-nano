//! Truncation preserves live inputs and supplies fresh conditions after an ALU op.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn masks_preserve_widths_inputs_and_zero_tests() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut wat = String::from("(module");
        for mask in [0xff, 0xffff, 0x7fff, 0x100ff] {
            wat.push_str(&format!(
                r#"
                (func (export "mask{mask}") (param {ty}) (result {ty} {ty} i32)
                    (local $next {ty}) (local $masked {ty})
                    (local.set $next ({ty}.add (local.get 0) ({ty}.const 1)))
                    (local.set $masked ({ty}.and (local.get $next) ({ty}.const {mask})))
                    (local.get $masked) (local.get 0)
                    (if (result i32) ({ty}.eqz (local.get $masked))
                        (then (i32.const 7)) (else (i32.const 11))))
                (func (export "wrap{mask}") (param i64) (result i64)
                    (i64.extend_i32_u
                        (i32.and (i32.wrap_i64 (local.get 0)) (i32.const {mask}))))"#
            ));
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
            0xfe,
            0xff,
            0x100,
            0xfffe,
            0xffff,
            0x10000,
            0xffff_ffff,
            u64::MAX,
            0xdead_beef_8765_4321,
        ] {
            for mask in [0xff, 0xffff, 0x7fff, 0x100ff] {
                let masked = raw.wrapping_add(1) & mask;
                assert_eq!(
                    instance
                        .invoke(&format!("mask{mask}"), &[value(raw)])
                        .unwrap(),
                    vec![
                        value(masked),
                        value(raw),
                        Value::I32(if masked == 0 { 7 } else { 11 })
                    ],
                    "{ty} raw={raw:x} mask={mask:x}"
                );
                assert_eq!(
                    instance
                        .invoke(&format!("wrap{mask}"), &[Value::I64(raw as i64)])
                        .unwrap(),
                    vec![Value::I64((raw & mask) as i64)]
                );
            }
        }
    }
}
