//! Constant divisions must return Wasm results or traps, never exhaust codegen temps.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn constant_integer_division_and_remainder_preserve_results_and_traps() {
    for width in [32, 64] {
        let ty = format!("i{width}");
        let minimum = if width == 32 {
            i32::MIN as i64
        } else {
            i64::MIN
        };
        let maximum = if width == 32 {
            i32::MAX as i64
        } else {
            i64::MAX
        };
        let mut source = String::from("(module");
        let mut cases = Vec::new();
        for op in ["div_s", "div_u", "rem_s", "rem_u"] {
            for lhs in [minimum, -17, 0, 37, maximum] {
                for rhs in [0, -1, 1, 7] {
                    let name = format!("case{}", cases.len());
                    source.push_str(&format!(
                        r#"(func (export "{name}") (param i64) (result i64 {ty})
                            (local.get 0) ({ty}.{op} ({ty}.const {lhs}) ({ty}.const {rhs})))"#
                    ));
                    cases.push((name, op, lhs, rhs));
                }
            }
        }
        source.push(')');
        let wasm = wat::parse_str(source).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        for (name, op, lhs, rhs) in cases {
            let result = instance.invoke(&name, &[Value::I64(0x7654_3210_ffff_eeee)]);
            if rhs == 0 || (op == "div_s" && lhs == minimum && rhs == -1) {
                assert!(result.is_err(), "{ty}.{op}: {lhs}, {rhs}");
                continue;
            }
            let unsigned = |x| {
                if width == 32 {
                    x as u32 as u64
                } else {
                    x as u64
                }
            };
            let expected = match op {
                "div_s" => lhs / rhs,
                "rem_s" => lhs.wrapping_rem(rhs),
                "div_u" => (unsigned(lhs) / unsigned(rhs)) as i64,
                "rem_u" => (unsigned(lhs) % unsigned(rhs)) as i64,
                _ => unreachable!(),
            };
            let expected = if width == 32 {
                Value::I32(expected as i32)
            } else {
                Value::I64(expected)
            };
            assert_eq!(
                result.unwrap().as_slice(),
                &[Value::I64(0x7654_3210_ffff_eeee), expected],
                "{ty}.{op}: {lhs}, {rhs}"
            );
        }
    }
}
