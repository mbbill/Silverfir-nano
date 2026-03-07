//! Hot local analysis for the register cache.
//!
//! Walks raw Wasm bytecode and counts local variable accesses weighted by
//! loop nesting depth. The top N locals by weight are cached in l0, l1, ...

use alloc::vec::Vec;
use crate::utils::leb128;
use super::stack::HOT_LOCAL_COUNT;

/// Read an unsigned LEB128 u32 from `code` at position `i`, returning (value, new_position).
/// Silently returns 0 on malformed input (best-effort for analysis).
#[inline]
fn read_u32(code: &[u8], i: usize) -> (u32, usize) {
    match leb128::read_leb128_u32(&code[i..]) {
        Ok((val, len)) => (val, i + len),
        Err(_) => (0, i + 1),
    }
}

/// Read a signed LEB128 i32 from `code` at position `i`, returning (value, new_position).
#[inline]
fn read_i32(code: &[u8], i: usize) -> (i32, usize) {
    match leb128::read_leb128_i32(&code[i..]) {
        Ok((val, len)) => (val, i + len),
        Err(_) => (0, i + 1),
    }
}

/// Read a signed LEB128 i64 from `code` at position `i`, returning (value, new_position).
#[inline]
fn read_i64(code: &[u8], i: usize) -> (i64, usize) {
    match leb128::read_leb128_i64(&code[i..]) {
        Ok((val, len)) => (val, i + len),
        Err(_) => (0, i + 1),
    }
}

/// Find the top-N hottest local variables in a function body.
///
/// Returns an array where each element is the index of a local with the
/// highest loop-depth-weighted access count, or None if unavailable.
/// Element 0 is cached in l0, element 1 in l1, etc.
///
/// `code` is the raw Wasm function body bytecode (starting after the locals section).
/// `frame_size` is params_count + locals_count.
pub fn local_weights(code: &[u8], frame_size: usize) -> Vec<u64> {
    if frame_size == 0 {
        return Vec::new();
    }

    let mut weights: Vec<u64> = alloc::vec![0u64; frame_size];
    let mut loop_depth: u32 = 0;
    let mut control_stack: Vec<bool> = Vec::new();
    let mut i = 0;

    while i < code.len() {
        let opcode = code[i];
        i += 1;

        match opcode {
            // block, if: enter block (not loop)
            0x02 | 0x04 => {
                i = skip_block_type(code, i);
                control_stack.push(false);
            }
            // loop: increment depth
            0x03 => {
                loop_depth = loop_depth.saturating_add(1);
                i = skip_block_type(code, i);
                control_stack.push(true);
            }
            // else: stay within the same if-frame, no loop-depth change
            0x05 => {}
            // end: pop one control frame and decrement loop depth only for loops
            0x0B => {
                if control_stack.pop().unwrap_or(false) {
                    loop_depth = loop_depth.saturating_sub(1);
                }
            }
            // local.get, local.set, local.tee (0x20, 0x21, 0x22)
            0x20 | 0x21 | 0x22 => {
                let (idx, new_i) = read_u32(code, i);
                i = new_i;
                if (idx as usize) < frame_size {
                    let w = loop_weight(loop_depth);
                    weights[idx as usize] = weights[idx as usize].saturating_add(w);
                }
            }
            // Instructions with a single LEB128 immediate
            // br, br_if, call, global.get/set, table.get/set, ref.func, memory.size/grow
            0x0C | 0x0D | 0x10 | 0x23 | 0x24 | 0x25 | 0x26 | 0xD2
            | 0x3F | 0x40 => {
                let (_, new_i) = read_u32(code, i);
                i = new_i;
            }
            // call_indirect: type_idx + table_idx
            0x11 => {
                let (_, new_i) = read_u32(code, i);
                let (_, new_i2) = read_u32(code, new_i);
                i = new_i2;
            }
            // br_table
            0x0E => {
                let (count, new_i) = read_u32(code, i);
                i = new_i;
                for _ in 0..=count {
                    let (_, new_i2) = read_u32(code, i);
                    i = new_i2;
                }
            }
            // Memory load/store instructions (0x28-0x3E): alignment + offset
            0x28..=0x3E => {
                let (_, new_i) = read_u32(code, i); // align
                let (_, new_i2) = read_u32(code, new_i); // offset
                i = new_i2;
            }
            // i32.const
            0x41 => {
                let (_, new_i) = read_i32(code, i);
                i = new_i;
            }
            // i64.const
            0x42 => {
                let (_, new_i) = read_i64(code, i);
                i = new_i;
            }
            // f32.const
            0x43 => {
                i += 4;
            }
            // f64.const
            0x44 => {
                i += 8;
            }
            // select_t (0x1C): has a type count + types
            0x1C => {
                let (count, new_i) = read_u32(code, i);
                i = new_i;
                for _ in 0..count {
                    let (_, new_i2) = read_u32(code, i);
                    i = new_i2;
                }
            }
            // FC prefix (0xFC): bulk memory, saturating truncation
            0xFC => {
                let (sub_opcode, new_i) = read_u32(code, i);
                i = new_i;
                match sub_opcode {
                    // memory.init: data_idx + mem_idx
                    8 => {
                        let (_, new_i2) = read_u32(code, i);
                        i = new_i2;
                        i += 1; // mem_idx (always 0x00)
                    }
                    // data.drop: data_idx
                    9 => {
                        let (_, new_i2) = read_u32(code, i);
                        i = new_i2;
                    }
                    // memory.copy: src_mem + dst_mem
                    10 => {
                        i += 2; // two memory indices
                    }
                    // memory.fill: mem_idx
                    11 => {
                        i += 1;
                    }
                    // table.init: elem_idx + table_idx
                    12 => {
                        let (_, new_i2) = read_u32(code, i);
                        let (_, new_i3) = read_u32(code, new_i2);
                        i = new_i3;
                    }
                    // elem.drop: elem_idx
                    13 => {
                        let (_, new_i2) = read_u32(code, i);
                        i = new_i2;
                    }
                    // table.copy: dst_table + src_table
                    14 => {
                        let (_, new_i2) = read_u32(code, i);
                        let (_, new_i3) = read_u32(code, new_i2);
                        i = new_i3;
                    }
                    // table.grow, table.size, table.fill: table_idx
                    15 | 16 | 17 => {
                        let (_, new_i2) = read_u32(code, i);
                        i = new_i2;
                    }
                    // Saturating truncation (0-7): no immediates
                    _ => {}
                }
            }
            // FB prefix (GC ops) - skip for now
            0xFB => {
                let (_, new_i) = read_u32(code, i);
                i = new_i;
                // Most FB ops have additional immediates but we don't need
                // precise parsing for hot local analysis - worst case we
                // misparse a few bytes but still get correct local weights.
            }
            // Everything else: no immediates (single-byte ops)
            _ => {}
        }
    }

    weights
}

pub fn find_hot_locals(code: &[u8], frame_size: usize) -> [Option<u32>; HOT_LOCAL_COUNT] {
    if frame_size == 0 {
        return [None; HOT_LOCAL_COUNT];
    }

    let weights = local_weights(code, frame_size);

    // Find top-N by weight
    // Each element is (local_index, weight), sorted descending by weight.
    let mut best: [(u32, u64); HOT_LOCAL_COUNT] = [(0, 0); HOT_LOCAL_COUNT];
    let mut found = 0usize;

    for (idx, &w) in weights.iter().enumerate() {
        if w == 0 {
            continue;
        }
        // Insert into sorted best array
        let mut insert_at = found.min(HOT_LOCAL_COUNT);
        for i in 0..found.min(HOT_LOCAL_COUNT) {
            if w > best[i].1 {
                insert_at = i;
                break;
            }
        }
        if insert_at < HOT_LOCAL_COUNT {
            // Shift down
            let mut j = (found.min(HOT_LOCAL_COUNT)).min(HOT_LOCAL_COUNT - 1);
            while j > insert_at {
                best[j] = best[j - 1];
                j -= 1;
            }
            best[insert_at] = (idx as u32, w);
            found += 1;
        }
    }

    let mut result = [None; HOT_LOCAL_COUNT];
    for i in 0..found.min(HOT_LOCAL_COUNT) {
        if best[i].1 > 0 {
            result[i] = Some(best[i].0);
        }
    }
    result
}

/// Compute effective indices after sequential swaps.
///
/// Each init_lN swaps fp[N]↔fp[KN]. If a later swap's target was moved by
/// an earlier swap, we adjust it. For example, if init_l0 swaps fp[0]↔fp[K0],
/// then K1_eff must account for the fact that the value originally at fp[K1]
/// may now be at a different position.
///
/// `raw` contains the original (pre-swap) hot local indices from `find_hot_locals`.
/// Returns effective indices suitable for passing to StackTracker and init_lN.
pub fn compute_effective_indices(
    raw: &[Option<u32>; HOT_LOCAL_COUNT],
    frame_size: usize,
) -> [Option<u32>; HOT_LOCAL_COUNT] {
    let mut eff = [None; HOT_LOCAL_COUNT];
    for slot in 0..HOT_LOCAL_COUNT {
        if frame_size <= slot {
            break; // not enough locals for this register
        }
        let Some(k) = raw[slot] else {
            // No hot local found for this slot; earlier slots disable later ones
            break;
        };
        // Apply all previous swaps to find where k ended up
        let mut k_eff = k;
        for prev in 0..slot {
            if let Some(kp) = eff[prev] {
                let prev_slot = prev as u32;
                if k_eff == prev_slot {
                    k_eff = kp;
                } else if k_eff == kp {
                    k_eff = prev_slot;
                }
            }
        }
        eff[slot] = Some(k_eff);
    }
    eff
}

/// Compute loop weight: 10^min(depth, 6)
#[inline]
fn loop_weight(depth: u32) -> u64 {
    const WEIGHTS: [u64; 7] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000];
    WEIGHTS[depth.min(6) as usize]
}

/// Skip a block type (single byte 0x40, or valtype, or s33 type index)
#[inline]
fn skip_block_type(code: &[u8], i: usize) -> usize {
    if i >= code.len() {
        return i;
    }
    let b = code[i];
    if b == 0x40 || (b >= 0x6C && b <= 0x7F) {
        // Empty block type or single value type
        i + 1
    } else {
        // s33 type index (LEB128)
        let (_, new_i) = read_i32(code, i);
        new_i
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use super::{compute_effective_indices, find_hot_locals, local_weights};

    #[test]
    fn test_local_weights_keep_loop_depth_through_nested_block_end() {
        let code = [
            0x03, 0x40,       // loop
            0x02, 0x40,       // block
            0x20, 0x00,       // local.get 0
            0x0B,             // end block
            0x20, 0x01,       // local.get 1
            0x0B,             // end loop
            0x20, 0x02,       // local.get 2
        ];

        let weights = local_weights(&code, 3);
        assert_eq!(weights, vec![10, 10, 1]);
        assert_eq!(find_hot_locals(&code, 3), [Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_local_weights_keep_loop_depth_across_if_else() {
        let code = [
            0x03, 0x40,       // loop
            0x04, 0x40,       // if
            0x20, 0x00,       // local.get 0
            0x05,             // else
            0x20, 0x01,       // local.get 1
            0x0B,             // end if
            0x20, 0x02,       // local.get 2
            0x0B,             // end loop
        ];

        let weights = local_weights(&code, 3);
        assert_eq!(weights, vec![10, 10, 10]);
    }

    #[test]
    fn test_compute_effective_indices_still_applies_swaps_in_order() {
        let raw = [Some(4), Some(5), Some(6)];
        let eff = compute_effective_indices(&raw, 8);
        assert_eq!(eff, [Some(4), Some(5), Some(6)]);

        let raw = [Some(1), Some(0), Some(2)];
        let eff = compute_effective_indices(&raw, 8);
        assert_eq!(eff, [Some(1), Some(1), Some(2)]);
    }
}
