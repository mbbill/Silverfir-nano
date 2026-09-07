//! Bounded expansion of small local callees before frame planning.
//!
//! Arguments are consumed once into fresh locals in reverse stack order. Every
//! non-parameter local is initialized at each expansion, including loop sites.
//! Callee calls remain ordinary calls: this pass never recursively expands its
//! own output. Caller control targets and dynamic result-type rows are relocated
//! together; no frame, register, or machine-call ABI exists at this stage.

use crate::{collections, value_type::ValueType};
use tracked_alloc::collections::BTreeMap;

use super::{
    common::SemanticTarget,
    primitive_op::{stack_effect, PrimitiveOpKind},
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

const MAX_CALLEE_OPS: usize = 32;
const MAX_ADDED_OPS: usize = 256;
const MAX_ADDED_LOCALS: u16 = 64;

struct InlineBody {
    program: SemanticProgram,
    /// Empty for the straight-line fast path. Otherwise, every early return
    /// becomes a branch to the copied function End, with this operand drop.
    return_drops: collections::Vec<u32>,
}

/// The resolver must return an unexpanded body from the caller's own module.
pub(crate) fn inline_small_calls(
    caller: &mut SemanticProgram,
    call_scratch_slots: u16,
    mut resolve: impl FnMut(u32) -> Option<SemanticProgram>,
) {
    if caller.local_types.len() != usize::from(caller.local_count)
        || !caller
            .ops
            .iter()
            .any(|op| matches!(op.kind, SemanticOpKind::CallDirect { .. }))
    {
        return;
    }

    let original_locals = caller.local_count;
    let original_stack = caller.max_stack_height;
    let mut added_ops = 0;
    let mut bodies = BTreeMap::new();
    let mut expansions = collections::Vec::new();
    for (index, op) in caller.ops.iter().enumerate() {
        let SemanticOpKind::CallDirect {
            callee,
            params,
            results,
        } = op.kind
        else {
            continue;
        };
        let body = bodies.entry(callee).or_insert_with(|| {
            let program = resolve(callee)?;
            if !within_body_limits(&program) {
                return None;
            }
            let return_drops = if eligible_straight_line_body(&program) {
                // Expanding a straight-line wrapper leaves all inner calls
                // while growing the caller's live locals and frame. Reserve
                // non-leaf expansion for bodies that expose control flow.
                if program.ops.iter().any(|op| {
                    matches!(
                        op.kind,
                        SemanticOpKind::CallDirect { .. }
                            | SemanticOpKind::CallIndirect { .. }
                            | SemanticOpKind::CallRef { .. }
                    )
                }) {
                    return None;
                }
                collections::Vec::new()
            } else {
                structured_return_drops(&program)?
            };
            Some(InlineBody {
                program,
                return_drops,
            })
        });
        let Some(body) = body else { continue };
        let body = &body.program;
        if body.params != params || body.results != results {
            continue;
        }
        let extra_ops = body.ops.len() + usize::from(body.local_count) * 2;
        let Some(local_count) = caller.local_count.checked_add(body.local_count) else {
            continue;
        };
        let Some(stack_height) = original_stack.checked_add(body.max_stack_height) else {
            continue;
        };
        let stack_height = caller.max_stack_height.max(stack_height);
        if added_ops + extra_ops > MAX_ADDED_OPS
            || local_count - original_locals > MAX_ADDED_LOCALS
            || local_count
                .checked_add(call_scratch_slots)
                .and_then(|n| n.checked_add(stack_height))
                .is_none()
        {
            continue;
        }
        expansions.push((index, callee, caller.local_count));
        caller.local_count = local_count;
        caller.max_stack_height = stack_height;
        caller.local_types.extend_from_slice(&body.local_types);
        added_ops += extra_ops;
    }
    if expansions.is_empty() {
        return;
    }

    let old_ops = core::mem::take(&mut caller.ops);
    let old_types = core::mem::take(&mut caller.op_result_types);
    let mut offsets = collections::vec![0; old_ops.len() + 1];
    let mut caller_ops = collections::Vec::with_capacity(old_ops.len());
    caller.ops.reserve(old_ops.len() + added_ops);
    let mut expansion = expansions.iter().peekable();
    for (index, op) in old_ops.into_iter().enumerate() {
        offsets[index] = caller.ops.len();
        if let Some(&&(site, callee, local_base)) = expansion.peek() {
            if site == index {
                let body = bodies[&callee].as_ref().expect("validated inline body");
                append_body(caller, body, local_base);
                expansion.next();
                continue;
            }
        }
        caller_ops.push(caller.ops.len());
        caller.ops.push(op);
        if let Some(types) = old_types.get(&index) {
            caller
                .op_result_types
                .insert(caller.ops.len() - 1, types.clone());
        }
    }
    *offsets.last_mut().expect("one sentinel offset") = caller.ops.len();
    for index in caller_ops {
        relocate_targets(&mut caller.ops[index].kind, &offsets);
    }
}

fn within_body_limits(body: &SemanticProgram) -> bool {
    body.ops.len() <= MAX_CALLEE_OPS
        && body.local_count <= 8
        && body.local_types.len() == usize::from(body.local_count)
        && body.result_types.len() == usize::from(body.results)
        && body.local_types.iter().chain(&body.result_types).all(|ty| {
            matches!(
                ty,
                ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
            )
        })
}

fn eligible_straight_line_body(body: &SemanticProgram) -> bool {
    let mut height = 0usize;
    let mut returned = false;
    let mut ended = false;
    for op in &body.ops {
        // The decoder keeps the function-closing End marker. An explicit
        // return can precede it; an implicit return follows it. No nested
        // control or executable operation after either marker is accepted.
        match op.kind {
            SemanticOpKind::End if !ended => {
                ended = true;
                continue;
            }
            SemanticOpKind::ReturnVoid if !returned && body.results == 0 && height == 0 => {
                returned = true;
                continue;
            }
            SemanticOpKind::ReturnOne if !returned && body.results == 1 && height == 1 => {
                returned = true;
                continue;
            }
            SemanticOpKind::Return { arity }
                if !returned && arity == body.results && height == usize::from(arity) =>
            {
                returned = true;
                continue;
            }
            _ if returned || ended => return false,
            _ => {}
        }
        let (pops, pushes) = match &op.kind {
            SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => return false,
            SemanticOpKind::Primitive(op) => stack_effect(op),
            SemanticOpKind::LocalGet { .. } => (0, 1),
            SemanticOpKind::LocalSet { .. } => (1, 0),
            SemanticOpKind::LocalTee { .. } => (1, 1),
            SemanticOpKind::CallDirect {
                params, results, ..
            } => (usize::from(*params), usize::from(*results)),
            SemanticOpKind::CallIndirect {
                params, results, ..
            }
            | SemanticOpKind::CallRef {
                params, results, ..
            } => (usize::from(*params) + 1, usize::from(*results)),
            _ => return false,
        };
        let Some(rest) = height.checked_sub(pops) else {
            return false;
        };
        height = rest + pushes;
    }
    returned && ended
}

/// Track operand heights through a small structured body. Heights are relative
/// to the callee's operand-stack floor; the caller's older operands therefore
/// never enter an early return's drop count. Unreachable paths are polymorphic,
/// while Else/End restore the declared structured signature, just as decode
/// does. EH and tail transfers retain their ordinary call boundary for now.
fn structured_return_drops(body: &SemanticProgram) -> Option<collections::Vec<u32>> {
    let end_index = body.ops.len().checked_sub(2)?;
    if !matches!(body.ops[end_index].kind, SemanticOpKind::End)
        || !body.ops.iter().any(|op| {
            matches!(
                op.kind,
                SemanticOpKind::Block { .. }
                    | SemanticOpKind::Loop { .. }
                    | SemanticOpKind::If { .. }
            )
        })
    {
        return None;
    }
    #[derive(Clone, Copy)]
    struct Frame {
        base: usize,
        params: u16,
        results: u16,
        has_else: bool,
    }
    let mut frames = collections::vec![Frame {
        base: 0,
        params: 0,
        results: body.results,
        has_else: false
    }];
    let mut drops = collections::vec![0; body.ops.len()];
    let mut height = 0usize;
    let mut unreachable = false;
    for (index, op) in body.ops.iter().enumerate() {
        if index > end_index
            && !matches!(
                op.kind,
                SemanticOpKind::ReturnVoid
                    | SemanticOpKind::ReturnOne
                    | SemanticOpKind::Return { .. }
            )
        {
            return None;
        }
        let mut effect = None;
        match &op.kind {
            SemanticOpKind::If {
                params, results, ..
            }
            | SemanticOpKind::Block { params, results }
            | SemanticOpKind::Loop { params, results } => {
                if matches!(op.kind, SemanticOpKind::If { .. }) && !unreachable {
                    height = height.checked_sub(1)?;
                }
                let base = if unreachable {
                    height.saturating_sub(usize::from(*params))
                } else {
                    height.checked_sub(usize::from(*params))?
                };
                frames.push(Frame {
                    base,
                    params: *params,
                    results: *results,
                    has_else: matches!(op.kind, SemanticOpKind::If { .. }),
                });
            }
            SemanticOpKind::Else { .. } => {
                let frame = frames.last_mut()?;
                if !frame.has_else {
                    return None;
                }
                frame.has_else = false;
                height = frame.base + usize::from(frame.params);
                unreachable = false;
            }
            SemanticOpKind::End => {
                let frame = frames.pop()?;
                if frames.is_empty() != (index == end_index) {
                    return None;
                }
                height = frame.base + usize::from(frame.results);
                unreachable = false;
            }
            SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. } => {
                let arity = match op.kind {
                    SemanticOpKind::ReturnVoid => 0,
                    SemanticOpKind::ReturnOne => 1,
                    SemanticOpKind::Return { arity } => arity,
                    _ => unreachable!(),
                };
                if arity != body.results {
                    return None;
                }
                if !unreachable {
                    drops[index] = u32::try_from(height.checked_sub(usize::from(arity))?).ok()?;
                }
                unreachable = true;
            }
            SemanticOpKind::Br { .. } => unreachable = true,
            SemanticOpKind::BrIf { .. } => effect = Some((1, 0)),
            SemanticOpKind::BrTable { .. } => {
                if !unreachable {
                    height = height.checked_sub(1)?;
                }
                unreachable = true;
            }
            SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable) => unreachable = true,
            SemanticOpKind::Primitive(op) => effect = Some(stack_effect(op)),
            SemanticOpKind::LocalGet { .. } => effect = Some((0, 1)),
            SemanticOpKind::LocalSet { .. } => effect = Some((1, 0)),
            SemanticOpKind::LocalTee { .. } => effect = Some((1, 1)),
            SemanticOpKind::CallDirect {
                params, results, ..
            } => effect = Some((usize::from(*params), usize::from(*results))),
            SemanticOpKind::CallIndirect {
                params, results, ..
            }
            | SemanticOpKind::CallRef {
                params, results, ..
            } => effect = Some((usize::from(*params) + 1, usize::from(*results))),
            _ => return None,
        }
        if !unreachable {
            if let Some((pops, pushes)) = effect {
                height = height.checked_sub(pops)? + pushes;
            }
            if height > usize::from(body.max_stack_height) {
                return None;
            }
        }
    }
    frames.is_empty().then_some(drops)
}

fn append_body(caller: &mut SemanticProgram, inline: &InlineBody, local_base: u16) {
    let body = &inline.program;
    for param in (0..body.params).rev() {
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::LocalSet {
                idx: local_base + param,
            },
        });
    }
    for local in body.params..body.local_count {
        let zero = match body.local_types[usize::from(local)] {
            ValueType::I32 => PrimitiveOpKind::I32Const { value: 0 },
            ValueType::I64 => PrimitiveOpKind::I64Const { value: 0 },
            ValueType::F32 => PrimitiveOpKind::F32Const { value: 0 },
            ValueType::F64 => PrimitiveOpKind::F64Const { value: 0 },
            _ => unreachable!("eligible inline local type"),
        };
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::Primitive(zero),
        });
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::LocalSet {
                idx: local_base + local,
            },
        });
    }
    let structured = !inline.return_drops.is_empty();
    if structured {
        caller
            .op_result_types
            .insert(caller.ops.len(), body.result_types.clone());
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::Block {
                params: 0,
                results: body.results,
            },
        });
    }
    let body_base = caller.ops.len();
    for (index, op) in body.ops.iter().enumerate() {
        if structured && index + 1 == body.ops.len() {
            break; // The copied function End leaves results on the caller stack.
        }
        let mut op = op.clone();
        match &mut op.kind {
            SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. }
                if structured =>
            {
                op.kind = SemanticOpKind::Br {
                    stack_drop: inline.return_drops[index],
                    arity: body.results,
                    target: SemanticTarget::new(body.ops.len() - 2),
                };
            }
            SemanticOpKind::End if structured => {}
            SemanticOpKind::End
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. } => continue,
            SemanticOpKind::LocalGet { idx }
            | SemanticOpKind::LocalSet { idx }
            | SemanticOpKind::LocalTee { idx } => *idx += local_base,
            _ => {}
        }
        if structured {
            map_targets(&mut op.kind, |index| body_base + index);
        }
        if let Some(types) = body.op_result_types.get(&index) {
            caller
                .op_result_types
                .insert(caller.ops.len(), types.clone());
        }
        caller.ops.push(op);
    }
}

fn relocate_targets(op: &mut SemanticOpKind, offsets: &[usize]) {
    map_targets(op, |index| offsets[index]);
}

fn map_targets(op: &mut SemanticOpKind, map: impl Fn(usize) -> usize) {
    let relocate = |target: &mut SemanticTarget| {
        *target = SemanticTarget::new(map(target.index().as_usize()));
    };
    match op {
        SemanticOpKind::If { else_target, .. } => relocate(else_target),
        SemanticOpKind::Else { end_target } => relocate(end_target),
        SemanticOpKind::Br { target, .. }
        | SemanticOpKind::BrIf { target, .. }
        | SemanticOpKind::BrOnNull { target, .. }
        | SemanticOpKind::BrOnNonNull { target, .. }
        | SemanticOpKind::BrOnCast { target, .. }
        | SemanticOpKind::BrOnCastFail { target, .. } => relocate(target),
        SemanticOpKind::BrTable { entries } => {
            for entry in entries {
                relocate(&mut entry.target);
            }
        }
        SemanticOpKind::TryTable { catches, .. } => {
            for catch in catches {
                relocate(&mut catch.target);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recursive_body() -> SemanticProgram {
        SemanticProgram {
            params: 1,
            results: 1,
            local_count: 1,
            max_stack_height: 2,
            local_types: collections::vec![ValueType::I64],
            result_types: collections::vec![ValueType::I64],
            ops: [
                SemanticOpKind::LocalGet { idx: 0 },
                SemanticOpKind::Primitive(PrimitiveOpKind::I64Const { value: 0 }),
                SemanticOpKind::Primitive(PrimitiveOpKind::I64GtS),
                SemanticOpKind::If {
                    params: 0,
                    results: 1,
                    else_target: SemanticTarget::new(7),
                },
                SemanticOpKind::LocalGet { idx: 0 },
                SemanticOpKind::CallDirect {
                    callee: 0,
                    params: 1,
                    results: 1,
                },
                SemanticOpKind::Else {
                    end_target: SemanticTarget::new(8),
                },
                SemanticOpKind::LocalGet { idx: 0 },
                SemanticOpKind::End,
                SemanticOpKind::End,
                SemanticOpKind::ReturnOne,
            ]
            .into_iter()
            .map(|kind| SemanticOp { kind })
            .collect(),
            op_result_types: [
                (3, collections::vec![ValueType::I64]),
                (5, collections::vec![ValueType::I64]),
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn expansion_is_single_level_and_keeps_nested_call_types() {
        let body = recursive_body();
        let mut caller = body.clone();
        let mut resolutions = 0;
        inline_small_calls(&mut caller, 3, |_| {
            resolutions += 1;
            Some(body.clone())
        });
        assert_eq!(resolutions, 1);
        assert_eq!(caller.local_count, 2);
        assert_eq!(
            caller
                .ops
                .iter()
                .filter(|op| matches!(op.kind, SemanticOpKind::CallDirect { .. }))
                .count(),
            1
        );
        let call = caller
            .ops
            .iter()
            .position(|op| matches!(op.kind, SemanticOpKind::CallDirect { .. }))
            .unwrap();
        assert_eq!(
            caller.op_result_types[&call],
            collections::vec![ValueType::I64]
        );
        caller.validate().unwrap();
    }

    #[test]
    fn expansion_refuses_frame_overflow_and_unsupported_control() {
        let body = recursive_body();
        let mut caller = body.clone();
        caller.max_stack_height = u16::MAX - 4;
        let before = caller.clone();
        inline_small_calls(&mut caller, 3, |_| Some(body.clone()));
        assert_eq!(caller, before);

        for kind in [
            SemanticOpKind::Loop {
                params: 0,
                results: 0,
            },
            SemanticOpKind::ReturnVoid,
            SemanticOpKind::ReturnCallDirect {
                callee: 0,
                params: 1,
                results: 1,
            },
        ] {
            let mut invalid = body.clone();
            invalid.ops.insert(0, SemanticOp { kind });
            let mut caller = body.clone();
            inline_small_calls(&mut caller, 3, |_| Some(invalid.clone()));
            assert_eq!(caller, body);
        }
    }

    #[test]
    fn straight_line_nonleaf_wrapper_is_not_expanded() {
        let mut wrapper = recursive_body();
        wrapper.ops = [
            SemanticOpKind::LocalGet { idx: 0 },
            SemanticOpKind::CallDirect {
                callee: 0,
                params: 1,
                results: 1,
            },
            SemanticOpKind::End,
            SemanticOpKind::ReturnOne,
        ]
        .into_iter()
        .map(|kind| SemanticOp { kind })
        .collect();
        wrapper.op_result_types = [(1, collections::vec![ValueType::I64])]
            .into_iter()
            .collect();
        let mut caller = wrapper.clone();
        inline_small_calls(&mut caller, 3, |_| Some(wrapper.clone()));
        assert_eq!(caller, wrapper);
    }
}
