//! Compact membership tests for tables with few distinct non-default edges.

use crate::collections;
use crate::vm::jit::machine::machine_ir::MachineEdge;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexSet {
    Range { first: u32, last: u32 },
    Mask { mask: u32, expected: u32 },
}

impl IndexSet {
    fn max_code_bytes(self) -> usize {
        // Conservative sizes including each near conditional branch, using
        // extended GP registers and imm32 encodings. No edge stubs are counted:
        // both the table and direct form share their parallel-move semantics.
        match self {
            Self::Range { first, last } if first == last || first == 0 => 13,
            Self::Range { .. } => 20,
            Self::Mask { expected: 0, .. } => 13,
            Self::Mask { .. } => 23,
        }
    }
}

pub(super) struct DirectCase {
    pub set: IndexSet,
    pub entry_index: usize,
}

pub(super) struct DirectTable {
    pub cases: collections::Vec<DirectCase>,
    pub default_index: usize,
}

struct Group {
    first: u32,
    last: u32,
    count: usize,
    varying_bits: u32,
}

/// Each group must be a contiguous range or every combination of its varying
/// bits. The latter admits a single masked equality test without a bounds
/// check: all high index bits remain in the mask, so out-of-range values fail.
/// Group equality includes edge arguments, not just destination block ids.
pub(super) fn plan_direct_table(entries: &[MachineEdge]) -> Option<DirectTable> {
    if entries.len() <= 1 || entries.len() > 4097 {
        return None;
    }
    let default_index = entries.len() - 1;
    let mut groups: collections::Vec<Group> = collections::Vec::new();
    for (index, edge) in entries[..default_index].iter().enumerate() {
        if *edge == entries[default_index] {
            continue;
        }
        if let Some(group) = groups
            .iter_mut()
            .find(|g| entries[g.first as usize] == *edge)
        {
            group.last = index as u32;
            group.count += 1;
            group.varying_bits |= group.first ^ index as u32;
        } else {
            if groups.len() == 2 {
                return None;
            }
            groups.push(Group {
                first: index as u32,
                last: index as u32,
                count: 1,
                varying_bits: 0,
            });
        }
    }
    let mut cases = collections::Vec::new();
    let mut bytes = 5; // final near jump to the default edge
    for group in groups {
        let set = if group.count == (group.last - group.first + 1) as usize {
            IndexSet::Range {
                first: group.first,
                last: group.last,
            }
        } else if group.count == 1usize << group.varying_bits.count_ones() {
            let mask = !group.varying_bits;
            IndexSet::Mask {
                mask,
                expected: group.first & mask,
            }
        } else {
            return None;
        };
        bytes += set.max_code_bytes();
        cases.push(DirectCase {
            set,
            entry_index: group.first as usize,
        });
    }
    // Dense dispatch needs at least 23 bytes for its bound constant, CMP,
    // CMOV, movabs and indirect jump, before preparing the index. Require a
    // substantial size reduction before trading one indirect branch for Jccs.
    let dense_min = 23 + 8 * entries.len();
    if bytes * 2 > dense_min || dense_min - bytes < 16 {
        return None;
    }
    Some(DirectTable {
        cases,
        default_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{MachineBlockId, MachineValue};

    fn edge(target: u32) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: collections::Vec::new(),
        }
    }

    fn matches(set: IndexSet, index: u32) -> bool {
        match set {
            IndexSet::Range { first, last } => index.wrapping_sub(first) <= last - first,
            IndexSet::Mask { mask, expected } => index & mask == expected,
        }
    }

    #[test]
    fn compresses_complete_bit_sets_and_ranges() {
        let entries = [edge(1), edge(0), edge(1), edge(0)];
        let plan = plan_direct_table(&entries).unwrap();
        assert_eq!(
            plan.cases[0].set,
            IndexSet::Mask {
                mask: !2,
                expected: 0
            }
        );
        let entries = [edge(0), edge(1), edge(1), edge(1), edge(1), edge(0)];
        let plan = plan_direct_table(&entries).unwrap();
        assert_eq!(plan.cases[0].set, IndexSet::Range { first: 1, last: 4 });
    }

    #[test]
    fn every_compressed_small_table_preserves_all_edges_and_default() {
        let mut compressed = 0;
        for length in 2..=8 {
            for mut pattern in 0usize..3usize.pow(length as u32) {
                let mut entries = collections::Vec::new();
                for _ in 0..length {
                    // Same block with different args must remain distinct.
                    entries.push(MachineEdge {
                        target: MachineBlockId(1),
                        args: collections::vec![MachineValue::Imm64((pattern % 3) as u64)],
                    });
                    pattern /= 3;
                }
                let Some(plan) = plan_direct_table(&entries) else {
                    continue;
                };
                compressed += 1;
                for index in (0..32).chain([u32::MAX, 0x8000_0000, 0x10000]) {
                    let found = plan
                        .cases
                        .iter()
                        .find(|case| matches(case.set, index))
                        .map_or(plan.default_index, |case| case.entry_index);
                    let expected = (index as usize).min(entries.len() - 1);
                    assert_eq!(entries[found], entries[expected]);
                }
            }
        }
        assert!(compressed > 100);
    }

    #[test]
    fn leaves_large_and_many_destination_tables_dense() {
        assert!(plan_direct_table(&collections::vec![edge(0); 4098]).is_none());
        assert!(plan_direct_table(&(0..8).map(edge).collect::<collections::Vec<_>>()).is_none());
    }
}
