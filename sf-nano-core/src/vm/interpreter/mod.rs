//! Interpreter v2: the folded stack machine.
//!
//! Design of record: `mcts_mem/silverfir/interpreter/`. Quantitative
//! basis: `tools/foldsim` v4 over the `benchmarks/wasi` corpus, plus the
//! measured record in the same subtree.
//!
//! Architecture split: the two engines differ only in how code is RUN.
//! Shared are the module layer (parsing, decode, the value-type model),
//! validation, the entity model (`vm::entities` — a memory here is the
//! same `MemInst` the JIT exports, which is what lets one instance import
//! another's), imports (`vm::imports`), values, config, and the WASI host.
//!
//! Not shared is code generation: this module never touches `middle/`,
//! `machine/`, or `arch/`, and the JIT's `JitInstance` and `jit/runtime/` layer
//! stay the JIT's, so interpreter work can never break JIT builds.
//!
//! Pipeline:
//! - `predecode` folds wasm bytecode into fixed 32-byte instruction cells.
//!   Routing opcodes (`local.get/set/tee`, consts) fold into the operand
//!   and destination fields of semantic instructions and never dispatch.
//! - `layout` defines the handler variant space — which operand residency
//!   classes exist and where each combination's handler lives.
//! - `engine` links a predecoded function into dispatch cells pointing at
//!   handlers that were generated at BUILD time (`interp_gen/`, driven by
//!   `build.rs`) and live in this binary's `.text`. No executable memory
//!   is allocated or mapped at run time.
//! - `exec` drives the chain and provides its slow path: one shared
//!   single-instruction executor covering every op without a native
//!   handler, plus host calls, traps with messages, and the activation
//!   boundary.

#[cfg(any(test, sf_module_validator))]
mod baseline_artifact;
#[cfg(sf_module_validator)]
mod baseline_composite_artifact;
#[cfg(any(test, sf_module_validator))]
mod baseline_exec;
#[cfg(sf_module_validator)]
mod baseline_function_plan;
#[cfg(sf_module_validator)]
mod baseline_raw_artifact;
mod engine;
mod exec;
mod fmath;
mod instr;
// The variant layout describes the generated handler set. The build script
// compiles this same file independently, via `#[path]`, so the generator and
// the linker agree on the space by construction.
mod layout;
mod predecode;

#[cfg(sf_module_validator)]
/// Owned output of the interpreter's single-decode validation pass.
///
/// Construction still predecodes and links every local function. The
/// validator-enabled interpreter may route preflighted whole functions Raw;
/// this correctness rollout does not yet remove eager startup work.
pub(crate) struct ValidatedBaselinePlan {
    artifact: baseline_artifact::BaselineArtifact,
    function_plan: crate::collections::Vec<baseline_function_plan::FunctionPlanKind>,
}

#[cfg(sf_module_validator)]
impl ValidatedBaselinePlan {
    pub(crate) fn validate(module: &crate::module::Module) -> Result<Self, crate::WasmError> {
        let artifact = baseline_composite_artifact::build_baseline_artifact_composite(module)?;
        let function_plan = baseline_function_plan::select_function_plans(module, &artifact)?;
        Ok(Self {
            artifact,
            function_plan,
        })
    }

    fn into_parts_for(
        self,
        module: &crate::module::Module,
    ) -> Result<
        (
            baseline_artifact::BaselineArtifact,
            crate::collections::Vec<baseline_function_plan::FunctionPlanKind>,
        ),
        crate::WasmError,
    > {
        use baseline_function_plan::FunctionPlanKind::{FullFold, Hybrid, Import, Raw};

        let function_count = module.functions().len();
        if self.artifact.functions.len() != function_count
            || self.function_plan.len() != function_count
        {
            return Err(crate::WasmError::invalid(
                "interpreter baseline metadata function count mismatch",
            ));
        }
        for ((function, artifact), plan) in module
            .functions()
            .iter()
            .zip(&self.artifact.functions)
            .zip(&self.function_plan)
        {
            let classification_matches = match (function.is_import(), artifact.is_some(), *plan) {
                (true, false, Import) => true,
                (false, false, FullFold) => true,
                (false, true, Raw | Hybrid | FullFold) => true,
                _ => false,
            };
            if !classification_matches {
                return Err(crate::WasmError::invalid(
                    "interpreter baseline metadata function classification mismatch",
                ));
            }
        }
        Ok((self.artifact, self.function_plan))
    }

    #[cfg(test)]
    pub(crate) fn force_all_full_fold_for_test(mut self, module: &crate::module::Module) -> Self {
        for (plan, function) in self.function_plan.iter_mut().zip(module.functions()) {
            *plan = if function.is_import() {
                baseline_function_plan::FunctionPlanKind::Import
            } else {
                baseline_function_plan::FunctionPlanKind::FullFold
            };
        }
        self
    }

    #[cfg(test)]
    pub(crate) fn plan_label_for_test(&self, function: usize) -> Option<&'static str> {
        self.function_plan.get(function).map(|kind| match kind {
            baseline_function_plan::FunctionPlanKind::Raw => "raw",
            baseline_function_plan::FunctionPlanKind::Hybrid => "hybrid",
            baseline_function_plan::FunctionPlanKind::FullFold => "full-fold",
            baseline_function_plan::FunctionPlanKind::Import => "import",
        })
    }
}

#[cfg(all(test, sf_module_validator))]
pub(crate) fn baseline_artifact_build_count_for_test() -> usize {
    baseline_artifact::artifact_build_count()
}

#[cfg(all(test, sf_module_validator))]
pub(crate) fn baseline_artifact_guard_for_test() -> std::sync::MutexGuard<'static, ()> {
    baseline_artifact::artifact_test_guard()
}

// Interpreter slots are always backed by `u64`, but references inside them
// use the target GP wire width consumed by the generated dispatch chain.
const SLOT_GP_UNIT_BYTES: u8 = core::mem::size_of::<usize>() as u8;

const EXTERNAL_FUNCREF_HOST_REQUIRED: &str =
    "interp: external function reference calls require a FuncRefHost hook";

// `InterpInstance` is the engine's public face. The predecoded
// representation behind it -- instructions, the opcode enum, operand
// flags -- stays inside the engine: it is how a function is stored, not
// an interface anything outside builds against.
pub use exec::{FuncRefHost, InterpInstance};
// The boundary converts host values through the same shared slot encoding
// that the executor imports from `vm::value`.
pub(crate) use exec::{
    raw_to_value_for_interp, value_to_raw_for_interp, InterpInstanceAccess, InterpInstanceLease,
};
