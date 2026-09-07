//! Bulk operations interpret each index with its declared Wasm width.
use sf_nano_core::{Config, Engine, Instance, Tier, Value};

fn check_bulk_indices(table: bool, compiler_budget: u32) {
    let mut failures = Vec::new();
    for &tier in Tier::ALL {
        for dst_width in [32, 64] {
            for src_width in [32, 64] {
                let marker = |width| if width == 64 { "i64 " } else { "" };
                let operand = |slot, width| {
                    if width == 64 {
                        format!("(local.get {slot})")
                    } else {
                        format!("(i32.wrap_i64 (local.get {slot}))")
                    }
                };
                let length_width = dst_width.min(src_width);
                let dst = operand(0, dst_width);
                let src = operand(1, src_width);
                let copy_len = operand(2, length_width);
                let fill_len = operand(2, dst_width);
                let init_src = operand(1, 32);
                let init_len = operand(2, 32);
                let (resources, fill, copy, init, read, capacity) = if table {
                    (
                        format!(
                            "(func $f) (table {}2 funcref) (table {}2 funcref)
                             (elem (table 0) (i{dst_width}.const 0) func $f $f)
                             (elem (table 1) (i{src_width}.const 0) func $f $f)
                             (elem $data func $f $f)",
                            marker(dst_width),
                            marker(src_width)
                        ),
                        format!("(table.fill 0 {dst} (ref.null func) {fill_len})"),
                        format!("(table.copy 0 1 {dst} {src} {copy_len})"),
                        format!("(table.init 0 $data {dst} {init_src} {init_len})"),
                        format!("(ref.is_null (table.get 0 (i{dst_width}.const 0)))"),
                        2_u64,
                    )
                } else {
                    (
                        format!(
                            "(memory {}1) (memory {}1)
                             (data (memory 0) (i{dst_width}.const 0) \"**\")
                             (data $data \"++\")",
                            marker(dst_width),
                            marker(src_width)
                        ),
                        format!("(memory.fill 0 {dst} (i32.const 7) {fill_len})"),
                        format!("(memory.copy 0 1 {dst} {src} {copy_len})"),
                        format!("(memory.init 0 $data {dst} {init_src} {init_len})"),
                        format!("(i32.load8_u (i{dst_width}.const 0))"),
                        65_536,
                    )
                };
                let mut operations = vec!["fill", "copy", "init"];
                let copy_self = if dst_width == src_width {
                    operations.push("copy_self");
                    let op = if table { "table.copy" } else { "memory.copy" };
                    format!("(func (export \"copy_self\") (param i64 i64 i64) ({op} 0 0 {dst} {src} {copy_len}))")
                } else {
                    String::new()
                };
                let wasm = wat::parse_str(format!(
                    "(module {resources} {copy_self}
                     (func (export \"fill\") (param i64 i64 i64) {fill})
                     (func (export \"copy\") (param i64 i64 i64) {copy})
                     (func (export \"init\") (param i64 i64 i64) {init})
                     (func (export \"read\") (result i32) {read}))"
                ))
                .unwrap();
                let engine = Engine::new(
                    Config::new()
                        .tier(tier)
                        .compiler_ram_budget_bytes(compiler_budget),
                )
                .unwrap();
                // Passing i64 through wrap_i64 also exercises producers whose
                // unused high bits must not influence a 32-bit index.
                for operation in operations {
                    for args in [
                        [0, 0, 1],
                        [0, 0, 0],
                        [capacity as i64, 0, 0],
                        [1_i64 << 32, 0, 1],
                        [0, 1_i64 << 32, 1],
                        [1_i64 << 32, 0, 0],
                        [0, 1_i64 << 32, 0],
                        [0, 0, 1_i64 << 32],
                        [-1, 0, 1],
                        [0, -1, 1],
                        [0, 0, -1],
                    ] {
                        let index = |raw: i64, width| {
                            if width == 64 {
                                raw as u64
                            } else {
                                raw as u32 as u64
                            }
                        };
                        let d = index(args[0], dst_width);
                        let s = index(args[1], if operation == "init" { 32 } else { src_width });
                        let n = index(
                            args[2],
                            match operation {
                                "fill" => dst_width,
                                "copy" | "copy_self" => length_width,
                                _ => 32,
                            },
                        );
                        let fits = |offset: u64, limit| {
                            offset.checked_add(n).is_some_and(|end| end <= limit)
                        };
                        let expected_trap = !fits(d, capacity)
                            || (operation != "fill"
                                && !fits(s, if operation == "init" { 2 } else { capacity }));
                        let mut instance = Instance::new(&engine, &wasm, &[]).unwrap();
                        let before = instance.invoke("read", &[]).unwrap();
                        let result = instance.invoke(operation, &args.map(Value::I64));
                        let case = format!("tier={tier:?}, budget={compiler_budget}, table={table}, widths={dst_width}/{src_width}, op={operation}, args={args:?}");
                        if result.is_err() != expected_trap {
                            failures.push(format!(
                                "{case}: expected trap={expected_trap}, got {result:?}"
                            ));
                        }
                        if expected_trap && instance.invoke("read", &[]).unwrap() != before {
                            failures.push(format!(
                                "{case}: trapping operation changed the destination"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn bulk_memory_indices_keep_their_declared_widths() {
    // Exercise both finite-budget streaming and unrestricted compilation.
    for budget in [65_536, u32::MAX] {
        check_bulk_indices(false, budget);
    }
}

#[test]
fn bulk_table_indices_keep_their_declared_widths() {
    for budget in [65_536, u32::MAX] {
        check_bulk_indices(true, budget);
    }
}
