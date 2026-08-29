//! Static function/loop tier planning over the validated baseline artifact.
//!
//! This module selects metadata only. It neither enters the executor nor
//! changes the production interpreter's representation choice.

use super::baseline_artifact::{BaselineArtifact, BaselineFunction, LoopRegion};
use crate::collections::{vec, Vec};
use crate::error::WasmError;
use crate::module::entities::Function;
use crate::module::Module;
use crate::op_decoder::raw_cursor::{RawDecodeError, RawImmediate, RawOp, RawOpCursor};
use crate::opcodes::{Opcode, OpcodeFC, WasmOpcode};
use crate::value_type::ValueType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FunctionPlanKind {
    Raw,
    Hybrid,
    FullFold,
    Import,
}

pub(crate) fn select_function_plans(
    module: &Module,
    artifact: &BaselineArtifact,
) -> Result<Vec<FunctionPlanKind>, WasmError> {
    if artifact.functions.len() != module.functions().len() {
        return Err(WasmError::invalid("baseline plan function count mismatch"));
    }
    let function_count = module.functions().len();
    let local: Vec<bool> = module
        .functions()
        .iter()
        .map(|function| function.spec().is_some())
        .collect();
    let mut graph = vec![Vec::new(); function_count];
    let mut opaque = vec![false; function_count];
    // The shared artifact assembler suppresses dead non-structural ops, so
    // these are exactly the reachable direct-call edges. Dynamic calls stay
    // opaque: type compatibility is not a points-to proof, even for a table
    // that is private today.
    for edge in &artifact.direct_calls {
        let caller = edge.caller as usize;
        let callee = edge.callee as usize;
        if caller >= function_count || callee >= function_count {
            return Err(WasmError::invalid(
                "baseline plan direct call index overflow",
            ));
        }
        if local[callee] {
            graph[caller].push(callee);
        } else {
            opaque[caller] = true;
        }
    }
    for site in &artifact.indirect_calls {
        let caller = site.function as usize;
        if caller >= function_count {
            return Err(WasmError::invalid(
                "baseline plan indirect caller index overflow",
            ));
        }
        opaque[caller] = true;
    }
    for edges in &mut graph {
        edges.sort_unstable();
        edges.dedup();
    }

    let components = strongly_connected_components(&graph, &local);
    let mut component_of = vec![usize::MAX; function_count];
    for (component_index, component) in components.iter().enumerate() {
        for &member in component {
            component_of[member] = component_index;
        }
    }
    let mut recursive = vec![false; components.len()];
    let mut tail_scc = vec![false; components.len()];
    for (component_index, component) in components.iter().enumerate() {
        recursive[component_index] = component.len() > 1
            || component
                .first()
                .is_some_and(|&only| graph[only].binary_search(&only).is_ok());
        // Inspect the original (not deduplicated) edges: one non-tail edge
        // invalidates the proof even when the same pair also has a tail edge.
        tail_scc[component_index] = recursive[component_index]
            && component.iter().all(|&member| !opaque[member])
            && artifact
                .direct_calls
                .iter()
                .filter(|edge| {
                    component_of[edge.caller as usize] == component_index
                        && component_of[edge.callee as usize] == component_index
                })
                .all(|edge| edge.tail);
    }

    let mut full = vec![false; function_count];
    // Only callees reached from a loop-hot direct call seed the hot closure.
    // Cold unsupported/EH/opaque callers fold themselves without heating all
    // of their otherwise raw callees.
    let mut direct_closure = vec![false; function_count];
    for (index, function) in module.functions().iter().enumerate() {
        let Some(spec) = function.spec() else {
            continue;
        };
        let artifact_function = artifact.functions[index]
            .as_ref()
            .ok_or_else(|| WasmError::invalid("baseline plan local artifact missing"))?;
        full[index] |= function_frame_is_unsupported(function);
        full[index] |= has_uncovered_raw_opcode(module, spec.code(), artifact, artifact_function)?;
        full[index] |= eh_crosses_region_boundary(artifact, artifact_function);
        let component = component_of[index];
        if recursive[component] {
            full[index] = true;
        }
        if !recursive[component]
            && linear_tail_wrapper_target(spec.code())?.is_some_and(|callee| {
                let callee = callee as usize;
                callee < function_count && local[callee] && tail_scc[component_of[callee]]
            })
        {
            full[index] = true;
        }
    }

    for edge in artifact
        .direct_calls
        .iter()
        .filter(|edge| edge.loop_depth != 0)
    {
        let callee = edge.callee as usize;
        if callee < function_count && local[callee] {
            full[callee] = true;
            direct_closure[callee] = true;
        }
    }
    for (index, &is_opaque) in opaque.iter().enumerate() {
        if is_opaque && local[index] {
            full[index] = true;
        }
    }

    let mut queue: Vec<usize> = direct_closure
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| selected.then_some(index))
        .collect();
    let mut cursor = 0usize;
    while let Some(&caller) = queue.get(cursor) {
        cursor += 1;
        for &callee in &graph[caller] {
            if !direct_closure[callee] {
                direct_closure[callee] = true;
                full[callee] = true;
                queue.push(callee);
            }
        }
    }

    Ok(module
        .functions()
        .iter()
        .enumerate()
        .map(|(index, function)| {
            if function.spec().is_none() {
                FunctionPlanKind::Import
            } else if full[index] {
                FunctionPlanKind::FullFold
            } else if artifact.functions[index]
                .as_ref()
                .is_some_and(|function| !function.loop_regions.is_empty())
            {
                FunctionPlanKind::Hybrid
            } else {
                FunctionPlanKind::Raw
            }
        })
        .collect())
}

fn has_uncovered_raw_opcode(
    module: &Module,
    code: &[u8],
    artifact: &BaselineArtifact,
    function: &BaselineFunction,
) -> Result<bool, WasmError> {
    let regions = &artifact.loop_regions[function.loop_regions.clone()];
    let mut cursor = RawOpCursor::new(code);
    loop {
        let raw = match cursor.next() {
            Ok(Some(raw)) => raw,
            Ok(None) => return Ok(false),
            Err(RawDecodeError::Unsupported { .. }) => return Ok(true),
            Err(RawDecodeError::Decode(error)) => return Err(error),
            Err(RawDecodeError::InvalidPc { .. }) => {
                return Err(WasmError::invalid("baseline plan raw pc overflow"))
            }
        };
        let supported = if inside_any_region(regions, raw.start as u32) {
            raw_op_supported_inside_folded_region(raw.wasm_op)
        } else {
            raw_op_supported_outside_folded_region(module, &raw)
        };
        if supported {
            continue;
        }
        return Ok(true);
    }
}

fn raw_op_supported_inside_folded_region(opcode: WasmOpcode) -> bool {
    !matches!(
        opcode,
        WasmOpcode::OP(
            Opcode::RETURN_CALL | Opcode::RETURN_CALL_INDIRECT | Opcode::RETURN_CALL_REF
        ) | WasmOpcode::FB(_)
            | WasmOpcode::FD(_)
    )
}

fn function_frame_is_unsupported(function: &Function) -> bool {
    let Some(spec) = function.spec() else {
        return false;
    };
    function
        .func_type()
        .params()
        .iter()
        .chain(function.func_type().results())
        .chain(spec.locals())
        .any(|value_type| !is_baseline_scalar(*value_type))
}

fn is_baseline_scalar(value_type: ValueType) -> bool {
    matches!(
        value_type,
        ValueType::I32 | ValueType::I64 | ValueType::F32 | ValueType::F64
    )
}

fn raw_op_supported_outside_folded_region(module: &Module, raw: &RawOp<'_>) -> bool {
    match raw.wasm_op {
        WasmOpcode::FC(opcode) => matches!(
            opcode,
            OpcodeFC::I32_TRUNC_SAT_F32_S
                | OpcodeFC::I32_TRUNC_SAT_F32_U
                | OpcodeFC::I32_TRUNC_SAT_F64_S
                | OpcodeFC::I32_TRUNC_SAT_F64_U
                | OpcodeFC::I64_TRUNC_SAT_F32_S
                | OpcodeFC::I64_TRUNC_SAT_F32_U
                | OpcodeFC::I64_TRUNC_SAT_F64_S
                | OpcodeFC::I64_TRUNC_SAT_F64_U
        ),
        WasmOpcode::OP(Opcode::CALL | Opcode::RETURN_CALL) => {
            // A local direct tail transfer is a planner-supported raw
            // boundary. Keeping it distinct lets SCC/wrapper proof decide
            // whether heavy folding is required; no production executor is
            // selected by this test-only plan.
            let RawImmediate::FunctionIndex(index) = raw.imm else {
                return false;
            };
            module
                .functions()
                .get(index as usize)
                .is_some_and(|function| function.spec().is_some())
        }
        WasmOpcode::OP(Opcode::GLOBAL_GET | Opcode::GLOBAL_SET) => {
            let RawImmediate::GlobalIndex(index) = raw.imm else {
                return false;
            };
            module
                .globals()
                .get(index as usize)
                .is_some_and(|global| global.value_type() == ValueType::I32)
        }
        WasmOpcode::OP(opcode) => {
            let byte = opcode as u8;
            matches!(
                opcode,
                Opcode::UNREACHABLE
                    | Opcode::NOP
                    | Opcode::BLOCK
                    | Opcode::LOOP
                    | Opcode::IF
                    | Opcode::ELSE
                    | Opcode::END
                    | Opcode::BR
                    | Opcode::BR_IF
                    | Opcode::BR_TABLE
                    | Opcode::RETURN
                    | Opcode::DROP
                    | Opcode::SELECT
                    | Opcode::LOCAL_GET
                    | Opcode::LOCAL_SET
                    | Opcode::LOCAL_TEE
                    | Opcode::MEMORY_SIZE
                    | Opcode::MEMORY_GROW
                    | Opcode::I32_CONST
                    | Opcode::I64_CONST
                    | Opcode::F32_CONST
                    | Opcode::F64_CONST
            ) || (Opcode::I32_LOAD as u8..=Opcode::I64_STORE32 as u8).contains(&byte)
                || (Opcode::I32_EQZ as u8..=Opcode::I64_EXTEND32_S as u8).contains(&byte)
        }
        WasmOpcode::FB(_) | WasmOpcode::FD(_) => false,
    }
}

fn inside_any_region(regions: &[LoopRegion], pc: u32) -> bool {
    regions
        .iter()
        .any(|region| pc >= region.body_start_pc && pc < region.exit_pc)
}

fn eh_crosses_region_boundary(artifact: &BaselineArtifact, function: &BaselineFunction) -> bool {
    let regions = &artifact.loop_regions[function.loop_regions.clone()];
    if regions.iter().any(|region| region.eh_depth != 0) {
        return true;
    }
    for table in &artifact.try_tables[function.try_tables.clone()] {
        let catches =
            table.catches_start as usize..table.catches_start as usize + table.catches_len as usize;
        for catch in artifact.catches.get(catches).into_iter().flatten() {
            for region in regions {
                let source_inside = inside_region(region, table.source_pc);
                let target_inside = inside_region(region, catch.target_pc);
                if source_inside != target_inside {
                    return true;
                }
            }
        }
    }
    false
}

fn inside_region(region: &LoopRegion, pc: u32) -> bool {
    pc >= region.body_start_pc && pc < region.exit_pc
}

fn linear_tail_wrapper_target(code: &[u8]) -> Result<Option<u32>, WasmError> {
    let mut cursor = RawOpCursor::new(code);
    while let Some(raw) = match cursor.next() {
        Ok(raw) => raw,
        Err(RawDecodeError::Unsupported { .. }) => return Ok(None),
        Err(RawDecodeError::Decode(error)) => return Err(error),
        Err(RawDecodeError::InvalidPc { .. }) => {
            return Err(WasmError::invalid("baseline tail-wrapper pc overflow"))
        }
    } {
        match (raw.wasm_op, raw.imm) {
            (WasmOpcode::OP(Opcode::RETURN_CALL), RawImmediate::FunctionIndex(callee)) => {
                let Some(end) = cursor.next().map_err(|error| match error {
                    RawDecodeError::Decode(error) => error,
                    RawDecodeError::Unsupported { .. } => {
                        WasmError::invalid("baseline tail-wrapper unsupported final opcode")
                    }
                    RawDecodeError::InvalidPc { .. } => {
                        WasmError::invalid("baseline tail-wrapper final pc overflow")
                    }
                })?
                else {
                    return Ok(None);
                };
                return Ok((end.wasm_op == WasmOpcode::OP(Opcode::END)
                    && cursor.remaining().is_empty())
                .then_some(callee));
            }
            (
                WasmOpcode::OP(
                    Opcode::NOP
                    | Opcode::LOCAL_GET
                    | Opcode::I32_CONST
                    | Opcode::I64_CONST
                    | Opcode::F32_CONST
                    | Opcode::F64_CONST,
                ),
                _,
            ) => {}
            _ => return Ok(None),
        }
    }
    Ok(None)
}

fn strongly_connected_components(graph: &[Vec<usize>], local: &[bool]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.len()];
    let mut order = Vec::new();
    for start in 0..graph.len() {
        if !local[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next_edge)) = stack.last_mut() {
            if *next_edge < graph[*node].len() {
                let target = graph[*node][*next_edge];
                *next_edge += 1;
                if !visited[target] {
                    visited[target] = true;
                    stack.push((target, 0));
                }
            } else {
                order.push(stack.pop().expect("SCC stack").0);
            }
        }
    }
    let mut reverse = vec![Vec::new(); graph.len()];
    for (caller, callees) in graph.iter().enumerate() {
        for &callee in callees {
            reverse[callee].push(caller);
        }
    }
    let mut assigned = vec![false; graph.len()];
    let mut components = Vec::new();
    for &start in order.iter().rev() {
        if assigned[start] {
            continue;
        }
        assigned[start] = true;
        let mut members = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            members.push(node);
            for &caller in &reverse[node] {
                if !assigned[caller] {
                    assigned[caller] = true;
                    stack.push(caller);
                }
            }
        }
        members.sort_unstable();
        components.push(members);
    }
    components.sort_by_key(|members| members[0]);
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::interpreter::baseline_artifact::{
        artifact_test_guard, BaselineArtifact, LoopBoundaryTypes,
    };
    use crate::vm::interpreter::baseline_raw_artifact::build_baseline_artifact_raw;
    use std::vec::Vec as StdVec;

    fn fixture(wat: &str) -> (Module, BaselineArtifact) {
        let _guard = artifact_test_guard();
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new("baseline-function-plan", &wasm).expect("module");
        let artifact = build_baseline_artifact_raw(&module).expect("raw artifact");
        (module, artifact)
    }

    fn plans(wat: &str) -> StdVec<FunctionPlanKind> {
        let (module, artifact) = fixture(wat);
        select_function_plans(&module, &artifact).expect("function plans")
    }

    #[test]
    fn loop_regions_record_outer_reachable_multivalue_boundary() {
        let (module, artifact) = fixture(
            r#"(module
                (type $pair (func (param i32 i32) (result i32 i32)))
                (func (result i32)
                    i32.const 10
                    i32.const 20
                    block (type $pair)
                        i32.const 0
                        br_if 0
                        loop (type $pair)
                            loop (type $pair)
                                br 2
                            end
                        end
                    end
                    i32.add))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        let regions = &artifact.loop_regions[function.loop_regions.clone()];
        assert_eq!(regions.len(), 1, "nested loop must merge into outer region");
        let region = regions[0];

        let mut cursor = RawOpCursor::new(module.functions()[0].spec().unwrap().code());
        let mut raw = StdVec::new();
        while let Some(opcode) = cursor.next().expect("raw opcode") {
            raw.push(opcode);
        }
        assert_eq!(region.entry_pc, raw[5].start as u32);
        assert_eq!(region.body_start_pc, raw[5].end as u32);
        assert_eq!(region.matching_end_pc, raw[9].start as u32);
        assert_eq!(region.exit_pc, raw[9].end as u32);
        assert_eq!(region.relative_stp, 1);
        assert_eq!(region.operand_height, 2);
        assert_eq!(region.control_depth, 3);
        assert_eq!(region.eh_depth, 0);
        assert_eq!(region.boundary_types, LoopBoundaryTypes::Unavailable);
    }

    #[test]
    fn loop_regions_keep_live_siblings_in_order_and_omit_dead_loops() {
        let (_module, artifact) = fixture(
            r#"(module
                (func
                    loop nop end
                    loop nop end
                    unreachable
                    loop nop end))"#,
        );
        let function = artifact.functions[0].as_ref().expect("function");
        let regions = &artifact.loop_regions[function.loop_regions.clone()];
        assert_eq!(regions.len(), 2);
        assert!(regions[0].entry_pc < regions[0].matching_end_pc);
        assert!(regions[0].exit_pc <= regions[1].entry_pc);
        assert!(regions[1].entry_pc < regions[1].matching_end_pc);
    }

    #[test]
    fn enclosing_eh_depth_is_recorded_and_forces_full_fold() {
        let wat = r#"(module
            (func
                block $handler
                    try_table (catch_all $handler)
                        loop nop end
                    end
                end))"#;
        let (module, artifact) = fixture(wat);
        let function = artifact.functions[0].as_ref().expect("function");
        let region = artifact.loop_regions[function.loop_regions.start];
        assert_eq!(region.control_depth, 4);
        assert_eq!(region.eh_depth, 1);
        assert_eq!(
            select_function_plans(&module, &artifact).expect("plans"),
            [FunctionPlanKind::FullFold]
        );
    }

    #[test]
    fn loop_direct_targets_and_transitive_callees_fold_but_exports_do_not_seed() {
        assert_eq!(
            plans(
                r#"(module
                    (func $leaf)
                    (func $middle call $leaf)
                    (func $hot (export "hot")
                        block $done
                            loop
                                call $middle
                                br $done
                            end
                        end)
                    (func (export "cold")))"#,
            ),
            [
                FunctionPlanKind::FullFold,
                FunctionPlanKind::FullFold,
                FunctionPlanKind::Hybrid,
                FunctionPlanKind::Raw,
            ]
        );
    }

    #[test]
    fn tail_scc_and_only_its_linear_entry_wrapper_fold() {
        assert_eq!(
            plans(
                r#"(module
                    (type $i (func (param i32) (result i32)))
                    (func $worker (type $i) (param i32) (result i32)
                        local.get 0
                        i32.eqz
                        if (result i32)
                            i32.const 0
                        else
                            local.get 0
                            i32.const 1
                            i32.sub
                            return_call $worker
                        end)
                    (func $entry (type $i) (param i32) (result i32)
                        local.get 0
                        return_call $worker)
                    (func $leaf (type $i) (param i32) (result i32)
                        local.get 0)
                    (func $not_entry (type $i) (param i32) (result i32)
                        local.get 0
                        return_call $leaf))"#,
            ),
            [
                FunctionPlanKind::FullFold,
                FunctionPlanKind::FullFold,
                FunctionPlanKind::Raw,
                FunctionPlanKind::Raw,
            ]
        );
    }

    #[test]
    fn a_non_tail_internal_edge_blocks_tail_entry_proof() {
        assert_eq!(
            plans(
                r#"(module
                    (func $a
                        i32.const 0
                        if
                            call $b
                        else
                            return_call $b
                        end)
                    (func $b return_call $a)
                    (func $entry return_call $a))"#,
            ),
            [
                FunctionPlanKind::FullFold,
                FunctionPlanKind::FullFold,
                FunctionPlanKind::Raw,
            ]
        );
    }

    #[test]
    fn unreachable_direct_edges_do_not_create_an_scc() {
        let wat = r#"(module
            (func $a unreachable call $b)
            (func $b call $a))"#;
        let (module, artifact) = fixture(wat);
        assert_eq!(artifact.direct_calls.len(), 1);
        assert_eq!(artifact.direct_calls[0].caller, 1);
        assert_eq!(artifact.direct_calls[0].callee, 0);
        assert_eq!(
            select_function_plans(&module, &artifact).expect("plans"),
            [FunctionPlanKind::Raw, FunctionPlanKind::Raw]
        );
    }

    #[test]
    fn imported_direct_calls_are_opaque_not_graph_edges() {
        assert_eq!(
            plans(
                r#"(module
                    (import "host" "call" (func $host))
                    (func call $host))"#,
            ),
            [FunctionPlanKind::Import, FunctionPlanKind::FullFold]
        );
    }

    #[test]
    fn cold_full_fold_blockers_do_not_seed_the_hot_direct_closure() {
        assert_eq!(
            plans(
                r#"(module
                    (table 1 funcref)
                    (func $cold_leaf)
                    (func (param i32)
                        local.get 0
                        table.get 0
                        drop
                        call $cold_leaf))"#,
            ),
            [FunctionPlanKind::Raw, FunctionPlanKind::FullFold]
        );
        assert_eq!(
            plans(
                r#"(module
                    (import "host" "call" (func $host))
                    (func $cold_leaf)
                    (func
                        call $host
                        call $cold_leaf))"#,
            ),
            [
                FunctionPlanKind::Import,
                FunctionPlanKind::Raw,
                FunctionPlanKind::FullFold,
            ]
        );
    }

    #[test]
    fn closed_and_open_indirect_tables_remain_opaque_in_v1() {
        for table in ["(table 1 funcref)", "(table (export \"open\") 1 funcref)"] {
            let wat = std::format!(
                r#"(module
                    (type $target (func))
                    {table}
                    (func $candidate (type $target))
                    (func $same_type (type $target))
                    (func (param i32)
                        block $done
                            loop
                                i32.const 0
                                call_indirect (type $target)
                                br $done
                            end
                        end))"#,
            );
            assert_eq!(
                plans(&wat),
                [
                    FunctionPlanKind::Raw,
                    FunctionPlanKind::Raw,
                    FunctionPlanKind::FullFold,
                ],
                "table declaration: {table}"
            );
        }
    }

    #[test]
    fn return_call_ref_and_eh_crossing_are_full_fold_blockers() {
        assert_eq!(
            plans(
                r#"(module
                    (type $f (func))
                    (func $target (type $f))
                    (elem declare func $target)
                    (func (type $f) ref.func $target return_call_ref $f)
                    (func
                        block $handler
                            loop
                                try_table (catch_all $handler) nop end
                            end
                        end))"#,
            ),
            [
                FunctionPlanKind::Raw,
                FunctionPlanKind::FullFold,
                FunctionPlanKind::FullFold,
            ]
        );
    }

    #[test]
    fn unsupported_ops_outside_regions_fold_while_folded_loop_ops_stay_hybrid() {
        assert_eq!(
            plans(
                r#"(module
                    (table 1 funcref)
                    (func (param i32)
                        local.get 0
                        table.get 0
                        drop)
                    (func (param i32)
                        loop
                            local.get 0
                            table.get 0
                            drop
                        end))"#,
            ),
            [FunctionPlanKind::FullFold, FunctionPlanKind::Hybrid]
        );
    }

    #[test]
    fn checked_in_corpus_plan_counts_are_stable() {
        let _guard = artifact_test_guard();
        for (name, wasm, expected) in [
            (
                "fib-min",
                &include_bytes!("../../../../benchmarks/wasi/fib/fib_min.wasm")[..],
                [1, 0, 1, 0],
            ),
            (
                "sha256",
                &include_bytes!("../../../../benchmarks/wasi/sha256/sha256.wasm")[..],
                [25, 9, 39, 9],
            ),
            (
                "coremark",
                &include_bytes!("../../../../benchmarks/wasi/coremark/coremark.wasm")[..],
                [23, 9, 60, 8],
            ),
        ] {
            let module = Module::new(name, wasm).expect("module");
            let artifact = build_baseline_artifact_raw(&module).expect("raw artifact");
            let mut counts = [0usize; 4];
            for plan in select_function_plans(&module, &artifact).expect("plans") {
                let index = match plan {
                    FunctionPlanKind::Raw => 0,
                    FunctionPlanKind::Hybrid => 1,
                    FunctionPlanKind::FullFold => 2,
                    FunctionPlanKind::Import => 3,
                };
                counts[index] += 1;
            }
            assert_eq!(counts, expected, "{name} plan counts: {counts:?}");
        }
    }
}
