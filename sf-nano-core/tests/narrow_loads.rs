//! Narrow address reads preserve sign extension, trap order and instance reuse.
use sf_nano_core::{Config, Engine, Instance, Value};

#[test]
fn narrow_read_loops_match_memory_and_preserve_the_last_write_before_trapping() {
    let mut wat = String::from("(module (memory (export \"memory\") 1)");
    for (name, op) in [
        ("u8", "i64.load8_u"),
        ("s8", "i64.load8_s"),
        ("u16", "i64.load16_u"),
        ("s16", "i64.load16_s"),
    ] {
        wat.push_str(&format!(
            r#"(func (export "{name}") (param $p i32) (param $n i32) (result i64)
                (local $sum i64)
                (block $done (loop $next
                    (br_if $done (i32.eqz (local.get $n)))
                    (local.set $sum (i64.add (local.get $sum) ({op} offset=1 (local.get $p))))
                    (i64.store (i32.const 0) (local.get $sum))
                    (local.set $p (i32.add (local.get $p) (i32.const 1)))
                    (local.set $n (i32.sub (local.get $n) (i32.const 1)))
                    (br $next)))
                (local.get $sum))"#
        ));
    }
    wat.push(')');
    let wasm = wat::parse_str(&wat).unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    let initial: Vec<u8> = (0..65536).map(|n| ((n * 71) ^ (n >> 8)) as u8).collect();
    for (name, width, signed) in [
        ("u8", 1, false),
        ("s8", 1, true),
        ("u16", 2, false),
        ("s16", 2, true),
    ] {
        for (start, count) in [
            (32u32, 127u32),
            (65532, 5),
            (0xffff_ffff, 1),
            (0x8000_0000, 1),
            (65534, 1),
            (32, 1),
            (65535, 0),
        ] {
            instance.memory_mut().unwrap().copy_from_slice(&initial);
            let mut expected = initial.clone();
            let mut sum = 0i64;
            let mut success = true;
            for index in 0..count {
                let address = u64::from(start.wrapping_add(index)) + 1;
                let bytes = usize::try_from(address).ok().and_then(|address| {
                    address
                        .checked_add(width)
                        .and_then(|end| expected.get(address..end))
                });
                let Some(bytes) = bytes else {
                    success = false;
                    break;
                };
                let value = match (width, signed) {
                    (1, false) => i64::from(bytes[0]),
                    (1, true) => i64::from(bytes[0] as i8),
                    (2, false) => i64::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                    (2, true) => i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
                    _ => unreachable!(),
                };
                sum = sum.wrapping_add(value);
                expected[..8].copy_from_slice(&sum.to_le_bytes());
            }
            let actual =
                instance.invoke(name, &[Value::I32(start as i32), Value::I32(count as i32)]);
            assert_eq!(
                actual.is_ok(),
                success,
                "{name}({start}, {count}): {actual:?}"
            );
            if success {
                assert_eq!(
                    actual.unwrap(),
                    vec![Value::I64(sum)],
                    "{name}({start}, {count})"
                );
            }
            assert_eq!(
                instance.memory().unwrap(),
                expected.as_slice(),
                "{name}({start}, {count})"
            );
        }
    }
}

#[test]
fn narrow_equality_search_preserves_high_bits_branch_direction_and_traps() {
    let mut wat = String::from("(module (memory (export \"memory\") 1)");
    for (width, op, mask) in [(1, "i32.load8_u", 255), (2, "i32.load16_u", 65535)] {
        for xor in [false, true] {
            for ne in [false, true] {
                let loaded = format!("({op} (local.get $p))");
                let masked = format!("(i32.and (local.get $needle) (i32.const {mask}))");
                let comparison = if xor {
                    let difference = format!("(i32.xor {loaded} {masked})");
                    format!(
                        "(i32.{} {difference} (i32.const 0))",
                        if ne { "ne" } else { "eq" }
                    )
                } else {
                    format!("(i32.{} {loaded} {masked})", if ne { "ne" } else { "eq" })
                };
                wat.push_str(&format!(
                    r#"
                    (func (export "search{width}{xor}{ne}")
                        (param $p i32) (param $n i32) (param $needle i32) (result i32)
                        (block $done (loop $next
                            (br_if $done (i32.eqz (local.get $n)))
                            (i32.store (i32.const 0) (local.get $p))
                            (br_if $done {comparison})
                            (local.set $p (i32.add (local.get $p) (i32.const 1)))
                            (local.set $n (i32.sub (local.get $n) (i32.const 1)))
                            (br $next)))
                        (local.get $p))"#
                ));
            }
        }
    }
    wat.push(')');
    let wasm = wat::parse_str(&wat).unwrap();
    let engine = Engine::new(Config::new()).unwrap();
    let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
    let initial: Vec<u8> = (0..65536).map(|n| ((n * 73) ^ (n >> 7)) as u8).collect();
    for width in [1usize, 2] {
        let mask = if width == 1 { 255 } else { 65535 };
        for xor in [false, true] {
            for ne in [false, true] {
                for needle in [0u32, 255, 65535, 0xabcd_12ff, 0x1234_8000, 0xffff_ffff] {
                    for (start, count) in [
                        (32u32, 256u32),
                        (65533, 5),
                        (65535, 1),
                        (0xffff_ffff, 1),
                        (32, 1),
                        (0xffff_ffff, 0),
                    ] {
                        let mut expected = initial.clone();
                        let mut p = start;
                        let mut success = true;
                        for _ in 0..count {
                            expected[..4].copy_from_slice(&p.to_le_bytes());
                            let value = usize::try_from(p).ok().and_then(|p| {
                                p.checked_add(width).and_then(|end| expected.get(p..end))
                            });
                            let Some(value) = value else {
                                success = false;
                                break;
                            };
                            let loaded = if width == 1 {
                                u32::from(value[0])
                            } else {
                                u32::from(u16::from_le_bytes([value[0], value[1]]))
                            };
                            if (loaded == needle & mask) != ne {
                                break;
                            }
                            p = p.wrapping_add(1);
                        }
                        instance.memory_mut().unwrap().copy_from_slice(&initial);
                        let name = format!("search{width}{xor}{ne}");
                        let actual = instance.invoke(
                            &name,
                            &[
                                Value::I32(start as i32),
                                Value::I32(count as i32),
                                Value::I32(needle as i32),
                            ],
                        );
                        assert_eq!(
                            actual.is_ok(),
                            success,
                            "{name}({start}, {count}, {needle}): {actual:?}"
                        );
                        if success {
                            assert_eq!(actual.unwrap(), vec![Value::I32(p as i32)]);
                        }
                        assert_eq!(instance.memory().unwrap(), expected.as_slice());
                    }
                }
            }
        }
    }
}
