//! Early returns must preserve branch width, result lanes, and the caller frame.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn conditional_integer_returns_match_signed_and_unsigned_boundaries() {
    let inputs = [
        0u64,
        1,
        36,
        37,
        38,
        u32::MAX as u64,
        1 << 31,
        1 << 32,
        1 << 63,
        u64::MAX,
    ];
    let conditions = [
        "eq", "ne", "lt_s", "lt_u", "le_s", "le_u", "gt_s", "gt_u", "ge_s", "ge_u",
    ];
    for width in [32, 64] {
        let ty = format!("i{width}");
        let mut wat = String::from("(module");
        for cond in conditions {
            for flip in [false, true] {
                let fast = "(local.get 2)";
                let slow = format!("({ty}.xor (local.get 2) ({ty}.const 37))");
                let (yes, no) = if flip {
                    (slow.as_str(), fast)
                } else {
                    (fast, slow.as_str())
                };
                wat.push_str(&format!(
                    r#"
                    (func (export "{cond}_{flip}") (param {ty} {ty} {ty}) (result {ty})
                        (if ({ty}.{cond} (local.get 0) (local.get 1))
                            (then (return {yes})))
                        {no})"#
                ));
            }
        }
        wat.push(')');
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wat::parse_str(&wat).unwrap(), &[]).unwrap();
        let narrow = |x: u64| if width == 32 { x as u32 as u64 } else { x };
        let signed = |x: u64| {
            if width == 32 {
                x as i32 as i64
            } else {
                x as i64
            }
        };
        let value = |x: u64| {
            if width == 32 {
                Value::I32(x as i32)
            } else {
                Value::I64(x as i64)
            }
        };
        for lhs in inputs.map(narrow) {
            for rhs in inputs.map(narrow) {
                let payload = lhs.wrapping_mul(71).wrapping_add(rhs).wrapping_add(0x1234);
                for cond in conditions {
                    let matches = match cond {
                        "eq" => lhs == rhs,
                        "ne" => lhs != rhs,
                        "lt_s" => signed(lhs) < signed(rhs),
                        "lt_u" => lhs < rhs,
                        "le_s" => signed(lhs) <= signed(rhs),
                        "le_u" => lhs <= rhs,
                        "gt_s" => signed(lhs) > signed(rhs),
                        "gt_u" => lhs > rhs,
                        "ge_s" => signed(lhs) >= signed(rhs),
                        "ge_u" => lhs >= rhs,
                        _ => unreachable!(),
                    };
                    for flip in [false, true] {
                        let expected = if matches != flip {
                            payload
                        } else {
                            payload ^ 37
                        };
                        let name = format!("{cond}_{flip}");
                        let output = instance
                            .invoke(&name, &[value(lhs), value(rhs), value(payload)])
                            .unwrap();
                        assert_eq!(
                            output.as_slice(),
                            &[value(expected)],
                            "{ty} {name}, {lhs:x}, {rhs:x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn recursive_returns_restore_frames_and_preserve_nonfirst_result_parameters() {
    let wasm = wat::parse_str(
        r#"(module
        (func $sum (export "sum") (param $n i64) (param $seed i64) (result i64)
            (if (i64.le_s (local.get $n) (i64.const 0))
                (then (return (local.get $seed))))
            (i64.add (local.get $n)
                (call $sum (i64.sub (local.get $n) (i64.const 1)) (local.get $seed)))))"#,
    )
    .unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    for n in [-37i64, 0, 1, 2, 31, 64] {
        for seed in [0i64, 1, -1, i64::MIN, i64::MAX] {
            let sum = if n > 0 { n * (n + 1) / 2 } else { 0 };
            let output = instance
                .invoke("sum", &[Value::I64(n), Value::I64(seed)])
                .unwrap();
            assert_eq!(output.as_slice(), &[Value::I64(seed.wrapping_add(sum))]);
        }
    }
}

#[test]
fn constant_early_returns_preserve_every_result_bit() {
    for width in [32, 64] {
        for raw in [0u64, 1, u64::MAX, 1 << 63, 0xabc0_1234_5678] {
            let bits = if width == 32 { raw as u32 as u64 } else { raw };
            let ty = format!("i{width}");
            let wasm = wat::parse_str(format!(
                r#"(module
                (memory 1)
                (func (export "guard") (param i64 i64) (result {ty})
                    (if (i64.lt_s (local.get 0) (local.get 1))
                        (then (return ({ty}.const {bits}))))
                    ({ty}.load (i32.const -1))))"#
            ))
            .unwrap();
            let engine = Engine::new(Config::new()).unwrap();
            let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
            let expected = if width == 32 {
                Value::I32(bits as i32)
            } else {
                Value::I64(bits as i64)
            };
            assert_eq!(
                instance
                    .invoke("guard", &[Value::I64(7), Value::I64(8)])
                    .unwrap()
                    .as_slice(),
                &[expected]
            );
            assert!(instance
                .invoke("guard", &[Value::I64(8), Value::I64(8)])
                .is_err());
        }
    }
}

#[test]
fn returning_arm_skips_guest_memory_but_the_other_arm_still_traps() {
    let wasm = wat::parse_str(
        r#"(module
        (memory 1)
        (func (export "guard") (param $x i64) (param $bound i64) (result i64)
            (if (i64.lt_u (local.get $x) (local.get $bound))
                (then (return (local.get $x))))
            (i64.load (i32.const -1))))"#,
    )
    .unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    for _ in 0..3 {
        assert_eq!(
            instance
                .invoke("guard", &[Value::I64(5), Value::I64(6)])
                .unwrap()
                .as_slice(),
            &[Value::I64(5)]
        );
        assert!(instance
            .invoke("guard", &[Value::I64(6), Value::I64(6)])
            .is_err());
        assert_eq!(
            instance
                .invoke("guard", &[Value::I64(7), Value::I64(-1)])
                .unwrap()
                .as_slice(),
            &[Value::I64(7)]
        );
    }
}
