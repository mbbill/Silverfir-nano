//! One list per block, stored back to back.
//!
//! Graph helpers (predecessors, successors, dominator-tree children) used to
//! keep a `Vec` per block, which costs one allocation for every block with a
//! nonempty list, plus regrowth. `BlockLists` keeps every list in one buffer
//! indexed by per-block offsets, so a whole function's lists take two.

use core::ops::Index;

use crate::collections;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLists {
    /// `starts[b]..starts[b + 1]` is block `b`'s range of `items`.
    starts: collections::Vec<u32>,
    items: collections::Vec<usize>,
}

impl BlockLists {
    /// Block `b`'s list holds the `item` of every `(b, item)` pair, in pair
    /// order. Every block index must be below `block_count`.
    pub(crate) fn from_pairs(block_count: usize, pairs: &[(usize, usize)]) -> Self {
        let mut starts = collections::vec![0u32; block_count + 1];
        for &(block, _) in pairs {
            starts[block] += 1;
        }
        // Running sums: `starts[b]` becomes the end of block `b`'s range.
        let mut end = 0;
        for start in &mut starts {
            end += *start;
            *start = end;
        }
        // Placing the pairs last to first moves each `starts[b]` from the
        // end of its range back to its start, and keeps pair order.
        let mut items = collections::vec![0usize; pairs.len()];
        for &(block, item) in pairs.iter().rev() {
            starts[block] -= 1;
            items[starts[block] as usize] = item;
        }
        Self { starts, items }
    }

    /// Number of blocks.
    pub(crate) fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    pub(crate) fn get(&self, block: usize) -> Option<&[usize]> {
        (block < self.len()).then(|| &self[block])
    }

    /// Every block's list, in block order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &[usize]> + '_ {
        (0..self.len()).map(|block| &self[block])
    }
}

impl Index<usize> for BlockLists {
    type Output = [usize];

    fn index(&self, block: usize) -> &[usize] {
        &self.items[self.starts[block] as usize..self.starts[block + 1] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::BlockLists;

    #[test]
    fn groups_pairs_by_block_in_pair_order() {
        let lists = BlockLists::from_pairs(4, &[(2, 7), (0, 1), (2, 3), (0, 5), (2, 7)]);
        assert_eq!(lists.len(), 4);
        assert_eq!(&lists[0], &[1, 5]);
        assert_eq!(&lists[1], &[] as &[usize]);
        assert_eq!(&lists[2], &[7, 3, 7]);
        assert_eq!(&lists[3], &[] as &[usize]);
        assert_eq!(lists.get(4), None);
        assert_eq!(lists.iter().map(<[usize]>::len).sum::<usize>(), 5);
    }

    #[test]
    fn empty_lists() {
        let lists = BlockLists::from_pairs(0, &[]);
        assert_eq!(lists.len(), 0);
        assert_eq!(lists.iter().count(), 0);
        let lists = BlockLists::from_pairs(3, &[]);
        assert!(lists.iter().all(<[usize]>::is_empty));
    }
}
