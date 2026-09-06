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
