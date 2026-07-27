//! Regression tests for native bulk-memory helper calls.

use sf_nano_core::{Config, Engine, Instance, Tier, Value};

fn compile(wat_src: &str) -> Vec<u8> {
    wat::parse_str(wat_src).expect("wat parse failed")
}

#[test]
fn bulk_memory_preserves_c_argument_lanes() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "fill")
            (result i32)
            (local $l0 i32) (local $l1 i32) (local $l2 i32)
            (local $l3 i32) (local $l4 i32) (local $l5 i32)
            (local $l6 i32) (local $l7 i32) (local $l8 i32)
            (local.set $l0 (i32.const 1))
            (local.set $l1 (i32.const 2))
            (local.set $l2 (i32.const 3))
            (local.set $l3 (i32.const 4))
            (local.set $l4 (i32.const 5))
            (local.set $l5 (i32.const 6))
            (local.set $l6 (i32.const 7))
            (local.set $l7 (i32.const 8))
            (local.set $l8 (i32.const 9))
            (memory.fill (i32.const 128) (i32.const 37) (i32.const 32))
            (i32.add
              (i32.load8_u (i32.const 128))
              (i32.add (local.get $l0)
                (i32.add (local.get $l1)
                  (i32.add (local.get $l2)
                    (i32.add (local.get $l3)
                      (i32.add (local.get $l4)
                        (i32.add (local.get $l5)
                          (i32.add (local.get $l6)
                            (i32.add (local.get $l7) (local.get $l8)))))))))))
          (func (export "copy")
            (result i32)
            (memory.fill (i32.const 256) (i32.const 42) (i32.const 32))
            (memory.copy (i32.const 512) (i32.const 256) (i32.const 32))
            (i32.load8_u (i32.const 512))))
        "#,
    );
    let mut instance = Instance::new(&interp_engine(), &wasm, &[]).expect("instantiation failed");
    let fill = instance.invoke("fill", &[]).expect("memory.fill failed");
    assert_eq!(fill.as_slice(), &[Value::I32(82)]);

    let copy = instance.invoke("copy", &[]).expect("memory.copy failed");
    assert_eq!(copy.as_slice(), &[Value::I32(42)]);
}

#[test]
fn bulk_memory_bounds_trap_unwinds_helper_frame() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "fill") (param $dest i32) (param $len i32)
            (memory.fill (local.get $dest) (i32.const 0) (local.get $len))))
        "#,
    );
    let mut instance = Instance::new(&interp_engine(), &wasm, &[]).expect("instantiation failed");
    let err = instance
        .invoke("fill", &[Value::I32(65_520), Value::I32(32)])
        .expect_err("out-of-bounds fill should trap");
    assert!(err.message().contains("out of bounds"), "{err:?}");
}

#[test]
fn interpreter_overlapping_copy_matches_memmove_across_block_boundaries() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "copy") (param $dst i32) (param $src i32) (param $len i32)
            (memory.copy
              (local.get $dst)
              (local.get $src)
              (local.get $len))))
        "#,
    );
    let mut cases = vec![
        (1, 63),
        (15, 64),
        (31, 65),
        (63, 127),
        (1, 128),
        (31, 129),
        (1, 255),
        (127, 5_000),
        (4_095, 5_000),
    ];
    // Exercise every destination-end alignment through the pipelined
    // backward-copy path.
    cases.extend((1..=32).map(|distance| (distance, 5_000)));
    for (distance, len) in cases {
        let engine = Engine::new(Config::new().tier(Tier::Interp))
            .expect("interpreter engine configuration failed");
        let mut instance = Instance::new(&engine, &wasm, &[]).expect("instantiation failed");
        let src = 64usize;
        let dst = src + distance;
        let initial: Vec<u8> = (0..16_384).map(|i| (i as u8).wrapping_mul(37)).collect();
        instance.memory_mut().expect("memory")[..initial.len()].copy_from_slice(&initial);

        let mut expected = initial.clone();
        expected.copy_within(src..src + len, dst);
        instance
            .invoke(
                "copy",
                &[
                    Value::I32(dst as i32),
                    Value::I32(src as i32),
                    Value::I32(len as i32),
                ],
            )
            .expect("overlapping memory.copy failed");
        assert_eq!(
            &instance.memory().expect("memory")[..expected.len()],
            expected.as_slice(),
            "distance={distance}, len={len}",
        );
    }
}

#[test]
fn adjacent_fill_copy_fusion_matches_sequential_effects_and_traps() {
    let wasm = compile(
        r#"
        (module
          (memory 1)
          (func (export "run")
            (param $fill-dst i32) (param $value i32) (param $fill-len i32)
            (param $copy-dst i32) (param $copy-src i32) (param $copy-len i32)
            (memory.fill
              (local.get $fill-dst) (local.get $value) (local.get $fill-len))
            (memory.copy
              (local.get $copy-dst) (local.get $copy-src) (local.get $copy-len))))
        "#,
    );

    // Covers right/left overlap, disjoint destination fallback, a source
    // not made uniform by the fill, zero length, a copy trap after its fill
    // commits, and a fill trap that must leave memory untouched.
    let cases = [
        (100u32, 0x12u32, 128u32, 132u32, 100u32, 128u32),
        (100, 0x34, 128, 50, 100, 100),
        (100, 0x56, 32, 300, 100, 32),
        (100, 0x78, 32, 300, 90, 32),
        (65_536, 0x9a, 0, 65_536, 65_536, 0),
        (100, 0xbc, 32, 65_530, 100, 16),
        (100, 0xde, 32, 300, 65_530, 16),
        (65_530, 0xf0, 16, 0, 0, 1),
    ];
    for (fill_dst, value, fill_len, copy_dst, copy_src, copy_len) in cases {
        let mut instance =
            Instance::new(&interp_engine(), &wasm, &[]).expect("instantiation failed");
        let initial: Vec<u8> = (0..65_536)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(11))
            .collect();
        instance
            .memory_mut()
            .expect("memory")
            .copy_from_slice(&initial);

        let mut expected = initial;
        let expected_result = reference_fill_copy_pair(
            &mut expected,
            fill_dst,
            value,
            fill_len,
            copy_dst,
            copy_src,
            copy_len,
        );
        let actual = instance.invoke(
            "run",
            &[
                Value::I32(fill_dst as i32),
                Value::I32(value as i32),
                Value::I32(fill_len as i32),
                Value::I32(copy_dst as i32),
                Value::I32(copy_src as i32),
                Value::I32(copy_len as i32),
            ],
        );
        match expected_result {
            Ok(()) => assert!(actual.expect("pair should complete").is_empty()),
            Err(()) => {
                let error = actual.expect_err("pair should trap");
                assert!(
                    error.message().contains("out of bounds"),
                    "fill=({fill_dst}, {value}, {fill_len}), \
                     copy=({copy_dst}, {copy_src}, {copy_len}): {error:?}",
                );
            }
        }
        assert_eq!(
            instance.memory().expect("memory"),
            expected.as_slice(),
            "fill=({fill_dst}, {value}, {fill_len}), \
             copy=({copy_dst}, {copy_src}, {copy_len})",
        );
    }
}

fn reference_fill_copy_pair(
    memory: &mut [u8],
    fill_dst: u32,
    value: u32,
    fill_len: u32,
    copy_dst: u32,
    copy_src: u32,
    copy_len: u32,
) -> Result<(), ()> {
    let (fill_dst, fill_len) = (fill_dst as u64, fill_len as u64);
    let fill_end = fill_dst + fill_len;
    if fill_end > memory.len() as u64 {
        return Err(());
    }
    memory[fill_dst as usize..fill_end as usize].fill(value as u8);

    let (copy_dst, copy_src, copy_len) = (copy_dst as u64, copy_src as u64, copy_len as u64);
    let copy_end = copy_dst + copy_len;
    let source_end = copy_src + copy_len;
    if copy_end > memory.len() as u64 || source_end > memory.len() as u64 {
        return Err(());
    }
    memory.copy_within(copy_src as usize..source_end as usize, copy_dst as usize);
    Ok(())
}

fn interp_engine() -> sf_nano_core::Engine {
    Engine::new(Config::new().tier(Tier::Interp)).expect("interpreter engine configuration failed")
}
