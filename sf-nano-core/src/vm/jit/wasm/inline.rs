//! Bounded expansion of small local callees before frame planning.
//!
//! Arguments are consumed once into fresh locals in reverse stack order. Every
//! non-parameter local is initialized at each expansion, including loop sites.
//! Callee calls remain ordinary calls: this pass never recursively expands its
//! own output. Caller control targets and dynamic result-type rows are relocated
//! together; no frame, register, or machine-call ABI exists at this stage.

use crate::{collections, value_type::ValueType};

use super::{
    common::SemanticTarget,
    primitive_op::{stack_effect, PrimitiveOpKind},
    semantic_ir::{SemanticOp, SemanticOpKind, SemanticProgram},
};

const MAX_CALLEE_OPS: usize = 32;
// Bound the whole expanded caller, not just each callee. These limits keep
// the semantic body, canonical frame and Algorithm4's regions × locals
// tables small together. The bytecode prefilter in build.rs also bounds
// variable-sized operands before any candidate body is decoded.
const MAX_FUNCTION_OPS: usize = 128;
const MAX_FUNCTION_LOCALS: u16 = 8;
const MAX_FRAME_SLOTS: u16 = 32;
const MAX_REGION_LOCALS: usize = 32;
const MAX_CANDIDATE_BODIES: usize = 8;

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
    if caller.ops.len() > MAX_FUNCTION_OPS
        || caller.local_count > MAX_FUNCTION_LOCALS
        || caller.local_types.len() != usize::from(caller.local_count)
        || !caller
            .ops
            .iter()
            .any(|op| matches!(op.kind, SemanticOpKind::CallDirect { .. }))
    {
        return;
    }

    let mut region_count = 1 + loop_count(caller);
    if !within_caller_resources(
        caller.local_count,
        caller.max_stack_height,
        call_scratch_slots,
        region_count,
    ) {
        return;
    }
    let original_stack = caller.max_stack_height;
    let mut added_ops = 0;
    // Rejected candidates need only a key and a negative result, not a tree
    // node containing a full InlineBody-sized entry for every possible key.
    let mut keys = [0; MAX_CANDIDATE_BODIES];
    let mut body_ids = [None::<u8>; MAX_CANDIDATE_BODIES];
    let mut cached = 0;
    let mut bodies = collections::Vec::new();
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
        // Negative candidates consume cache space too. A caller with many
        // distinct calls must not decode and retain an unbounded callee set.
        let body_id = if let Some(slot) = keys[..cached].iter().position(|&key| key == callee) {
            body_ids[slot]
        } else {
            if cached == MAX_CANDIDATE_BODIES {
                continue;
            }
            let body_id = resolve(callee).and_then(inline_body).map(|body| {
                let id = bodies.len() as u8;
                bodies.push(body);
                id
            });
            keys[cached] = callee;
            body_ids[cached] = body_id;
            cached += 1;
            body_id
        };
        let Some(body_id) = body_id else { continue };
        let body = &bodies[usize::from(body_id)].program;
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
        let next_regions = region_count + loop_count(body);
        if caller.ops.len() + added_ops + extra_ops > MAX_FUNCTION_OPS
            || !within_caller_resources(local_count, stack_height, call_scratch_slots, next_regions)
        {
            continue;
        }
        expansions.push((index, body_id, caller.local_count));
        caller.local_count = local_count;
        caller.max_stack_height = stack_height;
        caller.local_types.extend_from_slice(&body.local_types);
        added_ops += extra_ops;
        region_count = next_regions;
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
        if let Some(&&(site, body_id, local_base)) = expansion.peek() {
            if site == index {
                let body = &bodies[usize::from(body_id)];
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

fn inline_body(program: SemanticProgram) -> Option<InlineBody> {
    if !within_body_limits(&program) {
        return None;
    }
    let return_drops = if eligible_straight_line_body(&program) {
        // Expanding a straight-line wrapper leaves all inner calls while
        // growing live locals and frame. Reserve non-leaf expansion for
        // bodies that expose control flow.
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
}

fn loop_count(body: &SemanticProgram) -> usize {
    body.ops
        .iter()
        .filter(|op| matches!(op.kind, SemanticOpKind::Loop { .. }))
        .count()
}

fn within_caller_resources(locals: u16, stack: u16, scratch: u16, regions: usize) -> bool {
    locals <= MAX_FUNCTION_LOCALS
        && regions * usize::from(locals.max(1)) <= MAX_REGION_LOCALS
        && locals
            .checked_add(scratch)
            .and_then(|n| n.checked_add(stack))
            .is_some_and(|slots| slots <= MAX_FRAME_SLOTS)
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

    fn prepend(body: &mut SemanticProgram, prefix: collections::Vec<SemanticOp>) {
        let offset = prefix.len();
        for op in &mut body.ops {
            map_targets(&mut op.kind, |target| target + offset);
        }
        body.op_result_types = core::mem::take(&mut body.op_result_types)
            .into_iter()
            .map(|(index, types)| (index + offset, types))
            .collect();
        body.ops.splice(0..0, prefix);
    }

    fn prepend_loops(body: &mut SemanticProgram, count: usize) {
        let mut prefix = collections::Vec::new();
        for _ in 0..count {
            prefix.push(SemanticOp {
                kind: SemanticOpKind::Loop {
                    params: 0,
                    results: 0,
                },
            });
            prefix.push(SemanticOp {
                kind: SemanticOpKind::End,
            });
        }
        prepend(body, prefix);
    }

    #[test]
    fn large_callers_are_rejected_before_resolving_any_callee() {
        let body = recursive_body();
        let mut many_locals = body.clone();
        many_locals.local_count = 9;
        many_locals.local_types.resize(9, ValueType::I64);
        let mut many_ops = body.clone();
        prepend(
            &mut many_ops,
            collections::vec![SemanticOp { kind: SemanticOpKind::Primitive(PrimitiveOpKind::Nop) }; 128],
        );
        let mut large_frame = body.clone();
        large_frame.max_stack_height = 32;
        let mut many_regions = body.clone();
        prepend_loops(&mut many_regions, 32);
        for mut caller in [many_locals, many_ops, large_frame, many_regions] {
            caller.validate().unwrap();
            let before = caller.clone();
            inline_small_calls(&mut caller, 3, |_| {
                panic!("over-budget caller decoded a callee")
            });
            assert_eq!(caller, before);
        }
    }

    #[test]
    fn separately_small_functions_cannot_exceed_the_combined_resource_limit() {
        let caller = recursive_body();
        let mut callee = caller.clone();
        callee.local_count = 8;
        callee.local_types.resize(8, ValueType::I64);
        let mut local_limited = caller.clone();
        inline_small_calls(&mut local_limited, 3, |_| Some(callee.clone()));
        assert_eq!(local_limited, caller);

        let mut regional = caller;
        regional.local_count = 3;
        regional.local_types.resize(3, ValueType::I64);
        prepend_loops(&mut regional, 3);
        regional.validate().unwrap();
        let mut region_limited = regional.clone();
        let mut resolutions = 0;
        inline_small_calls(&mut region_limited, 3, |_| {
            resolutions += 1;
            Some(regional.clone())
        });
        // Each input has 4 regions × 3 locals; expansion would have 7 × 6.
        assert_eq!(resolutions, 1);
        assert_eq!(region_limited, regional);
    }

    #[test]
    fn negative_candidate_cache_has_a_decode_bound() {
        let mut caller = recursive_body();
        caller.ops.clear();
        caller.op_result_types.clear();
        caller.results = 0;
        caller.result_types.clear();
        for callee in (0..12).chain(0..2) {
            caller.ops.push(SemanticOp {
                kind: SemanticOpKind::LocalGet { idx: 0 },
            });
            caller
                .op_result_types
                .insert(caller.ops.len(), collections::vec![ValueType::I64]);
            caller.ops.push(SemanticOp {
                kind: SemanticOpKind::CallDirect {
                    callee,
                    params: 1,
                    results: 1,
                },
            });
            caller.ops.push(SemanticOp {
                kind: SemanticOpKind::Primitive(PrimitiveOpKind::Drop),
            });
        }
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::End,
        });
        caller.ops.push(SemanticOp {
            kind: SemanticOpKind::ReturnVoid,
        });
        caller.validate().unwrap();
        let before = caller.clone();
        let mut decoded = collections::Vec::new();
        inline_small_calls(&mut caller, 3, |callee| {
            decoded.push(callee);
            None
        });
        assert_eq!(decoded, (0..8).collect::<collections::Vec<_>>());
        assert_eq!(caller, before);
    }
}
