//! Find cycle entries for code alignment without depending on text layout.
//!
//! A branch to an earlier text address can be an ordinary acyclic join.
//! Only a DFS edge back to an active ancestor proves re-entry into a cycle.
//! This is an alignment hint, not a dominance proof for moving instructions.

use crate::{
    collections,
    vm::jit::machine::machine_ir::{MachineBlock, MachineBlockId, MachineTerminator},
};

/// One iterative DFS, with O(blocks) scratch and O(blocks + edges) work.
/// The caller has validated dense block IDs and all successor targets.
pub(super) fn loop_headers(
    blocks: &[MachineBlock],
    entry: MachineBlockId,
) -> collections::Vec<bool> {
    let mut headers = collections::vec![false; blocks.len()];
    let mut color = collections::vec![0u8; blocks.len()];
    let mut stack = collections::Vec::new();
    // The emitter also lays out unreachable blocks; handle their cycles too.
    for start in core::iter::once(entry.as_usize()).chain(0..blocks.len()) {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        stack.push((start, 0));
        while let Some((block, next)) = stack.last_mut() {
            let Some(target) = successor(&blocks[*block].terminator, *next) else {
                color[*block] = 2;
                stack.pop();
                continue;
            };
            *next += 1;
            let target = target.as_usize();
            match color[target] {
                0 => {
                    color[target] = 1;
                    stack.push((target, 0));
                }
                1 => headers[target] = true,
                _ => {}
            }
        }
    }
    headers
}

fn successor(term: &MachineTerminator, index: usize) -> Option<MachineBlockId> {
    match term {
        MachineTerminator::Jump(edge) => (index == 0).then_some(edge.target),
        MachineTerminator::Branch {
            then_edge,
            else_edge,
            ..
        } => match index {
            0 => Some(then_edge.target),
            1 => Some(else_edge.target),
            _ => None,
        },
        MachineTerminator::JumpTable { entries, .. } => entries.get(index).map(|edge| edge.target),
        MachineTerminator::Call { success, .. } => (index == 0).then_some(success.target),
        MachineTerminator::TailCall { .. }
        | MachineTerminator::Return
        | MachineTerminator::ReturnScalar { .. }
        | MachineTerminator::Trap { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::jit::machine::machine_ir::{MachineEdge, MachineValue};

    fn edge(target: u32) -> MachineEdge {
        MachineEdge {
            target: MachineBlockId(target),
            args: collections::Vec::new(),
        }
    }

    fn blocks(targets: &[&[u32]]) -> collections::Vec<MachineBlock> {
        targets
            .iter()
            .enumerate()
            .map(|(id, targets)| MachineBlock {
                id: MachineBlockId(id as u32),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: match *targets {
                    [] => MachineTerminator::Return,
                    [target] => MachineTerminator::Jump(edge(*target)),
                    many => MachineTerminator::JumpTable {
                        index: MachineValue::Imm64(0),
                        entries: many.iter().map(|target| edge(*target)).collect(),
                    },
                },
            })
            .collect()
    }

    #[test]
    fn acyclic_backward_joins_are_not_loop_entries() {
        let blocks = blocks(&[&[], &[0], &[1, 3], &[0]]);
        assert_eq!(loop_headers(&blocks, MachineBlockId(2)), [false; 4]);
    }

    #[test]
    fn nested_and_unreachable_cycles_keep_their_entries() {
        let blocks = blocks(&[&[1], &[2, 5], &[3, 4], &[2], &[1], &[], &[6], &[8], &[7]]);
        assert_eq!(
            loop_headers(&blocks, MachineBlockId(0)),
            [false, true, true, false, false, false, true, true, false]
        );
    }

    #[test]
    fn deep_cycles_do_not_recurse_on_the_host_stack() {
        let mut blocks = blocks(&[&[]]);
        for id in 1..20_000 {
            blocks.push(MachineBlock {
                id: MachineBlockId(id),
                params: collections::Vec::new(),
                ops: collections::Vec::new(),
                terminator: MachineTerminator::Jump(edge(id - 1)),
            });
        }
        blocks[0].terminator = MachineTerminator::Jump(edge(19_999));
        let headers = loop_headers(&blocks, MachineBlockId(0));
        assert!(headers[0]);
        assert_eq!(headers.into_iter().filter(|header| *header).count(), 1);
    }
}
