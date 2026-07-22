//! Cell identity.
//!
//! A cell is the middle-end's multi-use value slot: named, written and read
//! any number of times, with its lifetime managed by the residency planner —
//! in contrast to the anonymous single-use linear SSA values. Wasm locals are
//! one origin of cells (today the only one); cell `i` corresponds 1:1 to wasm
//! local `i`. A cell's frame home is a separate property published in
//! `SsaProgram::cell_homes`, keyed by this identity — `CellId` itself is not
//! a frame address.

/// Identity of a cell. Distinct from `FrameSlot` (frame geometry): plan rows,
/// resident sets, and cache ops are keyed by `CellId`; frame addressing goes
/// through the cell's home slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub(crate) struct CellId(pub u16);

/// Small ordered set used for the resident cache.
///
/// Resident rows are bounded by the machine register budget, so a contiguous
/// sorted vector is cheaper to scan and mutate than an allocating tree while
/// preserving deterministic highest-cell eviction order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CellSet {
    cells: crate::collections::Vec<CellId>,
}

impl CellSet {
    #[inline]
    pub(crate) fn contains(&self, cell: &CellId) -> bool {
        self.cells.binary_search(cell).is_ok()
    }

    pub(crate) fn insert(&mut self, cell: CellId) -> bool {
        match self.cells.binary_search(&cell) {
            Ok(_) => false,
            Err(index) => {
                self.cells.insert(index, cell);
                true
            }
        }
    }

    pub(crate) fn remove(&mut self, cell: &CellId) -> bool {
        match self.cells.binary_search(cell) {
            Ok(index) => {
                self.cells.remove(index);
                true
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.cells.clear();
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    #[inline]
    pub(crate) fn iter(&self) -> core::slice::Iter<'_, CellId> {
        self.cells.iter()
    }

    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&CellId) -> bool) {
        self.cells.retain(|cell| keep(cell));
    }

    pub(crate) fn replace_from_iter(&mut self, iter: impl IntoIterator<Item = CellId>) {
        self.cells.clear();
        self.cells.extend(iter);
        self.cells.sort_unstable();
        self.cells.dedup();
    }
}

impl core::iter::FromIterator<CellId> for CellSet {
    fn from_iter<T: IntoIterator<Item = CellId>>(iter: T) -> Self {
        let mut set = Self::default();
        set.replace_from_iter(iter);
        set
    }
}

impl IntoIterator for CellSet {
    type Item = CellId;
    type IntoIter = <crate::collections::Vec<CellId> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.cells.into_iter()
    }
}

impl<'a> IntoIterator for &'a CellSet {
    type Item = &'a CellId;
    type IntoIter = core::slice::Iter<'a, CellId>;

    fn into_iter(self) -> Self::IntoIter {
        self.cells.iter()
    }
}
