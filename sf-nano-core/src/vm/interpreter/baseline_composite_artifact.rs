//! Single-decode validation plus baseline-artifact construction.
//!
//! Every function body is decoded by
//! the generic decoder once: full type validation consumes each borrowed op
//! first, then an independent height/reachability tracker commits the matching
//! artifact event. Function parts remain staged until the validator's later
//! module-wide phases succeed, so failed validation never publishes an
//! artifact or increments the artifact build census.

use super::baseline_artifact::{BaselineArtifact, BaselineFunctionBuilder, BaselineFunctionParts};
use super::baseline_raw_artifact::{DecodedLayoutError, RawFunctionScanner};
use crate::collections::Vec;
use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::module::validator::{FunctionValidator, Validator};
use crate::module::Module;
use crate::op_decoder::{DecodedOp, Decoder, Immediate, OpStream, OpcodeHandler};
use crate::opcodes::{Opcode, WasmOpcode};

struct ValidatingArtifactHandler<'a, 't> {
    module: &'a Module,
    validator: FunctionValidator<'a>,
    builder: Option<BaselineFunctionBuilder>,
    layout: RawFunctionScanner,
    parts: Option<BaselineFunctionParts>,
    boundary_types: &'t mut Vec<crate::value_type::ValueType>,
    boundary_start: usize,
    syntactic_direct_tail_callees: Vec<u32>,
    has_syntactic_dynamic_tail: bool,
}

impl<'a, 't> ValidatingArtifactHandler<'a, 't> {
    fn new(
        module: &'a Module,
        function_index: usize,
        function: &'a FunctionSpec,
        declared_functions: &'a [bool],
        boundary_types: &'t mut Vec<crate::value_type::ValueType>,
    ) -> Result<Self, WasmError> {
        let raw_end = function
            .code_offset()
            .checked_add(function.code().len())
            .ok_or_else(|| WasmError::invalid("composite artifact code range overflow"))?;
        let boundary_start = boundary_types.len();
        Ok(Self {
            module,
            validator: FunctionValidator::new(module, function, declared_functions)?,
            builder: Some(BaselineFunctionBuilder::new(
                function_index,
                function.code_offset()..raw_end,
                function.func_type().results().len(),
            )?),
            layout: RawFunctionScanner::new(function.func_type().results().len()),
            parts: None,
            boundary_types,
            boundary_start,
            syntactic_direct_tail_callees: Vec::new(),
            has_syntactic_dynamic_tail: false,
        })
    }

    fn append_operand_types(&mut self) -> core::ops::Range<usize> {
        let start = self.boundary_types.len();
        self.boundary_types
            .extend_from_slice(self.validator.operand_types());
        start..self.boundary_types.len()
    }

    fn validate_and_plan(&mut self, decoded: &DecodedOp) -> Result<(), WasmError> {
        self.validator.validate_decoded(decoded)?;
        match decoded.wasm_op {
            WasmOpcode::OP(Opcode::RETURN_CALL) => {
                let Immediate::FunctionIndex(callee) = &decoded.imm else {
                    return Err(WasmError::invalid("return_call immediate mismatch"));
                };
                if !self.syntactic_direct_tail_callees.contains(callee) {
                    self.syntactic_direct_tail_callees.push(*callee);
                }
            }
            WasmOpcode::OP(Opcode::RETURN_CALL_INDIRECT | Opcode::RETURN_CALL_REF) => {
                self.has_syntactic_dynamic_tail = true;
            }
            _ => {}
        }
        let Some(builder) = self.builder.as_ref() else {
            return Ok(());
        };
        let event = builder.plan(
            self.module,
            decoded,
            self.layout.height(),
            self.layout.dead(),
        )?;
        let opens_outer_loop = BaselineFunctionBuilder::event_opens_outer_loop(&event);
        let closes_outer_loop = builder.event_closes_outer_loop(&event);
        let entry_types = opens_outer_loop.then(|| self.append_operand_types());
        match self.layout.apply_decoded(self.module, decoded) {
            Ok(()) => {}
            Err(DecodedLayoutError::Ineligible) => {
                self.boundary_types.truncate(self.boundary_start);
                self.builder = None;
                return Ok(());
            }
            Err(DecodedLayoutError::Wasm(error)) => return Err(error),
        }
        let exit_reachable = closes_outer_loop && !self.layout.dead();
        let fallthrough_types = exit_reachable.then(|| self.append_operand_types());
        let builder = self.builder.as_mut().expect("eligible builder exists");
        if closes_outer_loop {
            builder.set_closing_loop_fallthrough_types(fallthrough_types, exit_reachable)?;
        }
        builder.commit(event, self.layout.height())?;
        if let Some(entry_types) = entry_types {
            builder.set_open_loop_entry_types(entry_types)?;
        }
        Ok(())
    }

    fn into_output(self) -> (Option<BaselineFunctionParts>, Vec<u32>, bool) {
        (
            self.parts,
            self.syntactic_direct_tail_callees,
            self.has_syntactic_dynamic_tail,
        )
    }
}

impl OpcodeHandler for ValidatingArtifactHandler<'_, '_> {
    fn on_decode_begin(&mut self) -> Result<(), WasmError> {
        Ok(())
    }

    fn on_stream<'x, 'y, 'z>(
        &mut self,
        stream: &mut OpStream<'x, 'y, 'z>,
    ) -> Result<(), WasmError> {
        while let Some(decoded) = stream.next()? {
            self.validate_and_plan(decoded)?;
        }
        Ok(())
    }

    fn on_decode_end(&mut self) -> Result<(), WasmError> {
        self.validator.finish()?;
        if let Some(builder) = self.builder.take() {
            self.layout.ensure_finished()?;
            self.parts = Some(builder.finish()?);
        }
        Ok(())
    }
}

pub(super) fn build_baseline_artifact_composite(
    module: &Module,
) -> Result<BaselineArtifact, WasmError> {
    // This arena is private staging: `artifact_build_count` and the caller do
    // not observe it unless every function and the module suffix validate.
    let mut artifact = BaselineArtifact::new_unpublished(module.functions().len());
    let mut boundary_types = Vec::new();
    Validator::new(module).validate_with_function_driver(
        |module, function_index, function, declared_functions| {
            let mut handler = ValidatingArtifactHandler::new(
                module,
                function_index,
                function,
                declared_functions,
                &mut boundary_types,
            )?;
            let mut decoder = Decoder::new(function.code());
            decoder.add_handler(&mut handler);
            decoder.decode_function()?;
            let (parts, direct_tail_callees, has_dynamic_tail) = handler.into_output();
            for callee in direct_tail_callees {
                artifact.record_syntactic_direct_tail(callee);
            }
            if has_dynamic_tail {
                artifact.record_syntactic_dynamic_tail();
            }
            if let Some(parts) = parts {
                artifact.publish_function(function_index, parts)?;
            }
            Ok(())
        },
    )?;
    artifact.boundary_types = boundary_types;
    Ok(artifact.into_published())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::validator::Validator;
    use crate::value_type::ValueType;
    use crate::vm::interpreter::baseline_artifact::{
        artifact_build_count, artifact_test_guard, BaselineFunctionEligibility, LoopBoundaryTypes,
    };
    use crate::vm::interpreter::baseline_raw_artifact::build_baseline_artifact_raw;
    use std::string::{String, ToString};

    fn module(wat: &str) -> Module {
        let wasm = wat::parse_str(wat).expect("wat");
        Module::new("composite-artifact", &wasm).expect("module")
    }

    fn without_boundary_types(mut artifact: BaselineArtifact) -> BaselineArtifact {
        artifact.boundary_types.clear();
        for region in &mut artifact.loop_regions {
            region.boundary_types =
                crate::vm::interpreter::baseline_artifact::LoopBoundaryTypes::Unavailable;
            region.escaping_types_available = false;
        }
        artifact
    }

    fn assert_matches_two_pass(wat: &str) {
        let module = module(wat);
        let two_pass = build_baseline_artifact_raw(&module).expect("validated raw artifact");
        let composite = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert_eq!(
            without_boundary_types(composite),
            without_boundary_types(two_pass)
        );
    }

    fn validator_error(module: &Module) -> String {
        Validator::new(module)
            .validate()
            .expect_err("fixture must fail validation")
            .to_string()
    }

    fn region<'a>(
        artifact: &'a BaselineArtifact,
        function_index: usize,
    ) -> &'a crate::vm::interpreter::baseline_artifact::LoopRegion {
        let function = artifact.functions[function_index]
            .as_ref()
            .expect("eligible function");
        assert_eq!(function.loop_regions.len(), 1);
        &artifact.loop_regions[function.loop_regions.start]
    }

    fn available_ranges(
        region: &crate::vm::interpreter::baseline_artifact::LoopRegion,
    ) -> (
        &core::ops::Range<usize>,
        Option<&core::ops::Range<usize>>,
        bool,
    ) {
        let LoopBoundaryTypes::Available {
            entry_range,
            fallthrough_range,
            exit_reachable,
        } = &region.boundary_types
        else {
            panic!("loop boundary types unavailable");
        };
        (entry_range, fallthrough_range.as_ref(), *exit_reachable)
    }

    #[test]
    fn composite_matches_two_pass_structural_boundaries() {
        let _guard = artifact_test_guard();
        for wat in [
            r#"(module
                (func
                    i32.const 99
                    unreachable
                    if
                        nop
                    else
                        nop
                    end))"#,
            r#"(module
                (type $pair (func (param i32 i32) (result i32 i32)))
                (func (param i32 i32 i32) (result i32)
                    block $exit (result i32)
                        local.get 0
                        local.get 1
                        local.get 2
                        br_table $exit $exit $exit
                    end)
                (func (param i32 i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    local.get 2
                    select (result i32)))"#,
            r#"(module
                (tag $e (param i32))
                (func (result i32)
                    block $handler (result i32)
                        try_table (result i32) (catch $e $handler)
                            i32.const 2
                        end
                    end))"#,
        ] {
            assert_matches_two_pass(wat);
        }
    }

    #[test]
    fn composite_matches_two_pass_checked_in_coremark() {
        let _guard = artifact_test_guard();
        let module = Module::new(
            "composite-coremark",
            include_bytes!("../../../../benchmarks/wasi/coremark/coremark.wasm"),
        )
        .expect("module");
        let two_pass = build_baseline_artifact_raw(&module).expect("validated raw artifact");
        let composite = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert_eq!(
            without_boundary_types(composite),
            without_boundary_types(two_pass)
        );
    }

    #[test]
    fn typed_loop_entry_and_fallthrough_use_one_flat_arena() {
        let _guard = artifact_test_guard();
        let module = module(
            r#"(module
                (func (result i32)
                    i32.const 7
                    i64.const 8
                    f32.const 1
                    loop (param i64 f32) (result i32)
                        drop
                        drop
                        i32.const 9
                    end
                    i32.add)
                (func
                    f64.const 0
                    i32.const 2
                    loop (param i32) (result i64)
                        drop
                        i64.const 3
                    end
                    drop
                    drop))"#,
        );
        let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
        let first = region(&artifact, 0);
        let (first_entry, first_fallthrough, first_reachable) = available_ranges(first);
        let first_fallthrough = first_fallthrough.expect("first fallthrough");
        assert_eq!(
            &artifact.boundary_types[first_entry.clone()],
            &[ValueType::I32, ValueType::I64, ValueType::F32]
        );
        assert_eq!(
            &artifact.boundary_types[first_fallthrough.clone()],
            &[ValueType::I32, ValueType::I32]
        );
        assert!(first_reachable);
        assert_eq!(first.operand_height, 3);

        let second = region(&artifact, 1);
        let (second_entry, second_fallthrough, second_reachable) = available_ranges(second);
        let second_fallthrough = second_fallthrough.expect("second fallthrough");
        assert_eq!(first_entry.start, 0);
        assert_eq!(first_entry.end, first_fallthrough.start);
        assert_eq!(first_fallthrough.end, second_entry.start);
        assert_eq!(second_entry.end, second_fallthrough.start);
        assert_eq!(second_fallthrough.end, artifact.boundary_types.len());
        assert_eq!(
            &artifact.boundary_types[second_entry.clone()],
            &[ValueType::F64, ValueType::I32]
        );
        assert_eq!(
            &artifact.boundary_types[second_fallthrough.clone()],
            &[ValueType::F64, ValueType::I64]
        );
        assert!(second_reachable);
        assert!(!first.escaping_types_available);
        assert!(!first.execution_switching_available());
    }

    #[test]
    fn nested_dead_infinite_branch_and_eh_loop_boundaries_stay_conservative() {
        let _guard = artifact_test_guard();
        let module = module(
            r#"(module
                (func loop loop nop end end)
                (func unreachable loop nop end)
                (func loop br 0 end)
                (func block $out loop br $out end end)
                (func
                    block $handler
                        try_table (catch_all $handler)
                            loop nop end
                        end
                    end))"#,
        );
        let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert_eq!(
            artifact.functions[0]
                .as_ref()
                .expect("nested function")
                .loop_regions
                .len(),
            1
        );
        assert!(artifact.functions[1]
            .as_ref()
            .expect("dead function")
            .loop_regions
            .is_empty());
        for index in [2, 3] {
            let region = region(&artifact, index);
            let (entry, fallthrough, reachable) = available_ranges(region);
            assert!(entry.is_empty());
            assert!(fallthrough.is_none());
            assert!(!reachable);
            assert!(!region.escaping_types_available);
            assert!(!region.execution_switching_available());
        }
        let eh = region(&artifact, 4);
        let (_, fallthrough, reachable) = available_ranges(eh);
        assert!(fallthrough.is_some());
        assert!(reachable);
        assert_eq!(eh.eh_depth, 1);
        assert!(!eh.escaping_types_available);
        assert!(!eh.execution_switching_available());
    }

    #[test]
    fn valid_unsupported_eh_and_gc_functions_require_full_fold() {
        let _guard = artifact_test_guard();
        for wat in [
            r#"(module
                (tag $e)
                (func nop)
                (func throw $e)
                (func (param exnref) local.get 0 throw_ref))"#,
            r#"(module
                (type $s (struct))
                (func nop)
                (func (result (ref $s)) struct.new $s))"#,
        ] {
            let module = module(wat);
            Validator::new(&module).validate().expect("valid module");
            let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
            assert!(matches!(
                artifact.function_eligibility(&module, 0),
                Some(BaselineFunctionEligibility::Baseline(_))
            ));
            for index in 1..module.functions().len() {
                assert_eq!(
                    artifact.function_eligibility(&module, index),
                    Some(BaselineFunctionEligibility::FullFold)
                );
            }
        }
    }

    #[test]
    fn syntactic_tail_census_survives_ineligible_eh_and_gc_prefixes() {
        let _guard = artifact_test_guard();
        for wat in [
            r#"(module
                (tag $e)
                (func $target)
                (func
                    throw $e
                    return_call $target))"#,
            r#"(module
                (type $s (struct))
                (func $target)
                (func
                    struct.new $s drop
                    return_call $target))"#,
        ] {
            let module = module(wat);
            let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
            assert_eq!(artifact.syntactic_direct_tail_callees, [0]);
            assert!(!artifact.has_syntactic_dynamic_tail);
            assert!(artifact.functions[1].is_none(), "prefix must be ineligible");
        }

        let module = module(
            r#"(module
                (type $s (struct))
                (type $f (func))
                (table 1 funcref)
                (func $target (type $f))
                (elem (i32.const 0) func $target)
                (func
                    struct.new $s drop
                    i32.const 0 return_call_indirect (type $f))
                (func
                    struct.new $s drop
                    ref.func $target return_call_ref $f))"#,
        );
        let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert!(artifact.has_syntactic_dynamic_tail);
        assert!(artifact.functions[1].is_none());
        assert!(artifact.functions[2].is_none());
    }

    #[cfg(sf_has_simd)]
    #[test]
    fn syntactic_tail_census_survives_an_ineligible_simd_prefix() {
        let _guard = artifact_test_guard();
        let module = module(
            r#"(module
                (func $target)
                (func
                    v128.const i32x4 0 0 0 0 drop
                    return_call $target))"#,
        );
        let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert_eq!(artifact.syntactic_direct_tail_callees, [0]);
        assert!(artifact.functions[1].is_none());
    }

    #[cfg(sf_has_simd)]
    #[test]
    fn valid_simd_function_requires_full_fold() {
        let _guard = artifact_test_guard();
        let module = module(
            r#"(module
                (func nop)
                (func (result v128) v128.const i32x4 0 0 0 0))"#,
        );
        Validator::new(&module)
            .validate()
            .expect("valid SIMD module");
        let artifact = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert!(matches!(
            artifact.function_eligibility(&module, 0),
            Some(BaselineFunctionEligibility::Baseline(_))
        ));
        assert_eq!(
            artifact.function_eligibility(&module, 1),
            Some(BaselineFunctionEligibility::FullFold)
        );
    }

    #[test]
    fn invalid_body_and_suffix_never_publish_and_keep_error_order() {
        let _guard = artifact_test_guard();
        for wat in [
            "(module (func local.get 0 drop))",
            "(module (global i32 (f64.const 0)) (func nop))",
            "(module (global i32 (f64.const 0)) (func local.get 0 drop))",
        ] {
            let module = module(wat);
            let expected = validator_error(&module);
            let before = artifact_build_count();
            let actual = build_baseline_artifact_composite(&module)
                .expect_err("composite validation must fail")
                .to_string();
            assert_eq!(actual, expected, "error order changed for {wat}");
            assert_eq!(artifact_build_count(), before, "failed build published");
        }
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn composite_allocation_census_does_not_exceed_two_pass() {
        let _guard = artifact_test_guard();
        let module = module(
            r#"(module
                (func (param i32 i32 i32) (result i32)
                    block $exit (result i32)
                        local.get 0
                        local.get 1
                        local.get 2
                        br_table $exit $exit $exit $exit $exit
                    end)
                (func (param i32 i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    local.get 2
                    select (result i32)))"#,
        );
        let (two_pass, two_census) =
            crate::test_alloc::measure(|| build_baseline_artifact_raw(&module));
        two_pass.expect("two-pass artifact");
        let (composite, composite_census) =
            crate::test_alloc::measure(|| build_baseline_artifact_composite(&module));
        composite.expect("composite artifact");
        assert!(composite_census.allocations <= two_census.allocations);
        assert!(composite_census.reallocations <= two_census.reallocations);
        assert!(composite_census.allocated_bytes <= two_census.allocated_bytes);
        assert!(composite_census.reallocated_bytes <= two_census.reallocated_bytes);
    }

    #[cfg(not(feature = "memprof"))]
    #[test]
    fn many_loop_boundaries_grow_flat_arenas_not_per_loop_vectors() {
        let _guard = artifact_test_guard();
        let one = module(
            r#"(module
                (type $loop (func (param i32) (result i32)))
                (func i32.const 0 loop (type $loop) end drop))"#,
        );
        let many = module(
            r#"(module
                (type $loop (func (param i32) (result i32)))
                (func
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop
                    i32.const 0 loop (type $loop) end drop))"#,
        );
        let (one_validation, one_validation_census) =
            crate::test_alloc::measure(|| Validator::new(&one).validate());
        one_validation.expect("one-loop validation");
        let (many_validation, many_validation_census) =
            crate::test_alloc::measure(|| Validator::new(&many).validate());
        many_validation.expect("many-loop validation");
        let (one, one_census) =
            crate::test_alloc::measure(|| build_baseline_artifact_composite(&one));
        one.expect("one-loop artifact");
        let (many, many_census) =
            crate::test_alloc::measure(|| build_baseline_artifact_composite(&many));
        let many = many.expect("many-loop artifact");
        assert_eq!(many.loop_regions.len(), 8);
        assert_eq!(many.boundary_types.len(), 16);
        let one_boundary_allocations = one_census
            .allocations
            .saturating_sub(one_validation_census.allocations);
        let many_boundary_allocations = many_census
            .allocations
            .saturating_sub(many_validation_census.allocations);
        let one_boundary_reallocations = one_census
            .reallocations
            .saturating_sub(one_validation_census.reallocations);
        let many_boundary_reallocations = many_census
            .reallocations
            .saturating_sub(many_validation_census.reallocations);
        assert!(many_boundary_allocations <= one_boundary_allocations + 1);
        assert!(many_boundary_reallocations <= one_boundary_reallocations + 8);
    }
}
