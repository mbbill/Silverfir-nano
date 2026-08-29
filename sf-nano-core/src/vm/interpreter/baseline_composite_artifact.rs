//! Single-decode validation plus baseline-artifact construction.
//!
//! This is a test-only architecture probe. Every function body is decoded by
//! the generic decoder once: full type validation consumes each borrowed op
//! first, then an independent height/reachability tracker commits the matching
//! artifact event. Function parts remain staged until the validator's later
//! module-wide phases succeed, so failed validation never publishes an
//! artifact or increments the artifact build census.

use super::baseline_artifact::{BaselineArtifact, BaselineFunctionBuilder, BaselineFunctionParts};
use super::baseline_raw_artifact::{DecodedLayoutError, RawFunctionScanner};
use crate::error::WasmError;
use crate::module::entities::FunctionSpec;
use crate::module::validator::{FunctionValidator, Validator};
use crate::module::Module;
use crate::op_decoder::{DecodedOp, Decoder, OpStream, OpcodeHandler};

struct ValidatingArtifactHandler<'a> {
    module: &'a Module,
    validator: FunctionValidator<'a>,
    builder: Option<BaselineFunctionBuilder>,
    layout: RawFunctionScanner,
    parts: Option<BaselineFunctionParts>,
}

impl<'a> ValidatingArtifactHandler<'a> {
    fn new(
        module: &'a Module,
        function_index: usize,
        function: &'a FunctionSpec,
        declared_functions: &'a [bool],
    ) -> Result<Self, WasmError> {
        let raw_end = function
            .code_offset()
            .checked_add(function.code().len())
            .ok_or_else(|| WasmError::invalid("composite artifact code range overflow"))?;
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
        })
    }

    fn validate_and_plan(&mut self, decoded: &DecodedOp) -> Result<(), WasmError> {
        self.validator.validate_decoded(decoded)?;
        let Some(builder) = self.builder.as_ref() else {
            return Ok(());
        };
        let event = builder.plan(
            self.module,
            decoded,
            self.layout.height(),
            self.layout.dead(),
        )?;
        match self.layout.apply_decoded(self.module, decoded) {
            Ok(()) => {}
            Err(DecodedLayoutError::Ineligible) => {
                self.builder = None;
                return Ok(());
            }
            Err(DecodedLayoutError::Wasm(error)) => return Err(error),
        }
        self.builder
            .as_mut()
            .expect("eligible builder exists")
            .commit(event, self.layout.height())
    }

    fn into_parts(self) -> Option<BaselineFunctionParts> {
        self.parts
    }
}

impl OpcodeHandler for ValidatingArtifactHandler<'_> {
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
    Validator::new(module).validate_with_function_driver(
        |module, function_index, function, declared_functions| {
            let mut handler = ValidatingArtifactHandler::new(
                module,
                function_index,
                function,
                declared_functions,
            )?;
            let mut decoder = Decoder::new(function.code());
            decoder.add_handler(&mut handler);
            decoder.decode_function()?;
            if let Some(parts) = handler.into_parts() {
                artifact.publish_function(function_index, parts)?;
            }
            Ok(())
        },
    )?;
    Ok(artifact.into_published())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::validator::Validator;
    use crate::vm::interpreter::baseline_artifact::{
        artifact_build_count, artifact_test_guard, BaselineFunctionEligibility,
    };
    use crate::vm::interpreter::baseline_raw_artifact::build_baseline_artifact_raw;
    use std::string::{String, ToString};

    fn module(wat: &str) -> Module {
        let wasm = wat::parse_str(wat).expect("wat");
        Module::new("composite-artifact", &wasm).expect("module")
    }

    fn assert_matches_two_pass(wat: &str) {
        let module = module(wat);
        let two_pass = build_baseline_artifact_raw(&module).expect("validated raw artifact");
        let composite = build_baseline_artifact_composite(&module).expect("composite artifact");
        assert_eq!(composite, two_pass);
    }

    fn validator_error(module: &Module) -> String {
        Validator::new(module)
            .validate()
            .expect_err("fixture must fail validation")
            .to_string()
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
        assert_eq!(composite, two_pass);
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
}
