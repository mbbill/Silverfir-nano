//! Bounded expansion of small, straight-line local callees before frame planning.
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
            resolve(callee).and_then(|body| eligible_body(&body).then_some(body))
        });
        let Some(body) = body else { continue };
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

fn eligible_body(body: &SemanticProgram) -> bool {
    if body.ops.len() > MAX_CALLEE_OPS
        || body.local_count > 8
        || body.local_types.len() != usize::from(body.local_count)
        || body.result_types.len() != usize::from(body.results)
        || body.local_types.iter().chain(&body.result_types).any(|ty| {
            !matches!(
                ty,
                ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
            )
        })
    {
        return false;
    }
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

fn append_body(caller: &mut SemanticProgram, body: &SemanticProgram, local_base: u16) {
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
    for (index, op) in body.ops.iter().enumerate() {
        let mut op = op.clone();
        match &mut op.kind {
            SemanticOpKind::End
            | SemanticOpKind::ReturnVoid
            | SemanticOpKind::ReturnOne
            | SemanticOpKind::Return { .. } => continue,
            SemanticOpKind::LocalGet { idx }
            | SemanticOpKind::LocalSet { idx }
            | SemanticOpKind::LocalTee { idx } => *idx += local_base,
            _ => {}
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
    let relocate = |target: &mut SemanticTarget| {
        *target = SemanticTarget::new(offsets[target.index().as_usize()]);
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
            max_stack_height: 1,
            local_types: collections::vec![ValueType::I64],
            result_types: collections::vec![ValueType::I64],
            ops: [
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
            .collect(),
            op_result_types: [(1, collections::vec![ValueType::I64])]
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
    fn expansion_refuses_frame_overflow_and_nonstraight_line_bodies() {
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
            SemanticOpKind::Primitive(PrimitiveOpKind::Unreachable),
        ] {
            let mut invalid = body.clone();
            invalid.ops.insert(0, SemanticOp { kind });
            let mut caller = body.clone();
            inline_small_calls(&mut caller, 3, |_| Some(invalid.clone()));
            assert_eq!(caller, body);
        }
    }
}
