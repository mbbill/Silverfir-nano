//! Sparse table dispatch must retain i32 indices and the complete default.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn repeated_table_edges_preserve_indices_with_stale_high_halves() {
    for targets in [
        vec![1, 0, 1],
        vec![0, 1, 0, 1],
        vec![0, 1, 1, 1, 1],
        vec![1, 0, 1, 2],
        vec![1, 1, 0, 0, 1],
    ] {
        let labels = targets
            .iter()
            .map(|target| match target {
                1 => "$one",
                2 => "$two",
                _ => "$default",
            })
            .collect::<Vec<_>>()
            .join(" ");
        let wasm = wat::parse_str(format!(
            r#"(module (func (export "run") (param $index i64) (result i32)
                (block $exit (result i32)
                    (block $default
                        (block $two
                            (block $one
                                (br_table {labels} $default (i32.wrap_i64 (local.get $index))))
                            (br $exit (i32.const 111)))
                        (br $exit (i32.const 222)))
                    (i32.const 333))))"#
        ))
        .unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        for index in (0..20u32).chain([u32::MAX, 0x8000_0000, 0x10000]) {
            let expected = match targets.get(index as usize) {
                Some(1) => 111,
                Some(2) => 222,
                _ => 333,
            };
            for upper in [0, 0x1234_5678_0000_0000u64, 0xffff_ffff_0000_0000] {
                let result = instance
                    .invoke("run", &[Value::I64((upper | u64::from(index)) as i64)])
                    .unwrap();
                assert_eq!(
                    result.as_slice(),
                    &[Value::I32(expected)],
                    "{targets:?} index={index} upper={upper:x}"
                );
            }
        }
    }
}

#[test]
fn dense_tables_preserve_bounded_loops_and_unknown_initial_indices() {
    for masked in [false, true] {
        let initial = if masked {
            "(i32.and (i32.wrap_i64 (local.get $input)) (i32.const 7))"
        } else {
            "(i32.wrap_i64 (local.get $input))"
        };
        let mut dispatch =
            "(br_table $c0 $c1 $c2 $c3 $c4 $c5 $c6 $c7 $default (local.get $state))".to_string();
        for index in 0..8 {
            dispatch = format!(
                "(block $c{index} {dispatch}) (br $join (i32.const {}))",
                17 + index * 13
            );
        }
        let wasm = wat::parse_str(format!(r#"(module
            (func (export "run") (param $input i64) (param $rounds i32) (result i32)
                (local $state i32) (local $sum i32)
                (local.set $state {initial})
                (loop $again
                    (local.set $sum (i32.add (local.get $sum)
                        (block $join (result i32)
                            (block $default {dispatch}) (i32.const 333))))
                    (local.set $state (i32.and (i32.add (local.get $state) (i32.const 3)) (i32.const 7)))
                    (br_if $again (local.tee $rounds (i32.sub (local.get $rounds) (i32.const 1)))))
                (local.get $sum)))"#)).unwrap();
        let engine = Engine::new(Config::new()).unwrap();
        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
        for input in (0..20u32).chain([u32::MAX, 0x8000_0000, 0x10000]) {
            for rounds in [1, 2, 7, 19] {
                let mut state = if masked { input & 7 } else { input };
                let mut expected = 0;
                for _ in 0..rounds {
                    expected += if state < 8 {
                        17 + state as i32 * 13
                    } else {
                        333
                    };
                    state = state.wrapping_add(3) & 7;
                }
                for high in [0, 0xffff_ffff_0000_0000u64] {
                    let result = instance
                        .invoke(
                            "run",
                            &[
                                Value::I64((high | u64::from(input)) as i64),
                                Value::I32(rounds),
                            ],
                        )
                        .unwrap();
                    assert_eq!(
                        result.as_slice(),
                        &[Value::I32(expected)],
                        "masked={masked} input={input} rounds={rounds} high={high:x}"
                    );
                }
            }
        }
    }
}
