//! Backend assembly for the fast interpreter instruction format.
//!
//! Pipeline:
//!   Wasm bytecode
//!     → semantic decode
//!     → backend-lowered IR
//!     → selected backend
//!     → Vec<ResolvedInst>
//!     → finalizer
//!     → Box<[Instruction]>
//!
//! Modules:
//! - `backend`: ResolvedInst type, base 1:1 resolution
//! - `ir_resolve`: Handler/operand resolution for IR ops
//! - `finalizer_ir`: Compact, patch, and build final instructions
//!
//! The backend-neutral compile frontend now lives in `vm::compile`.

pub mod backend;
mod finalizer_ir;
#[cfg(feature = "ir-dump")]
mod ir_dump;
pub mod ir_resolve;

use crate::{
    error::WasmError,
    module::entities::FunctionSpec,
    vm::{
        backend::{BackendKind, BackendMode, active_backend, backend_mode},
        compile::{self, CompileContext, CompilePlan, StackTracker},
        entities::ModuleInst,
        interp::fast::instruction::Instruction,
        store::Store,
    },
};

use alloc::{rc::Rc, vec::Vec};
use crate::module::type_defs::FunctionType;

#[cfg(feature = "fusion")]
fn resolve_fusion_backend(ir_ops: &[compile::lowered_ir::IrOp]) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    Ok(super::fusion::resolve::resolve_fusion(ir_ops))
}

#[cfg(not(feature = "fusion"))]
fn resolve_fusion_backend(_ir_ops: &[compile::lowered_ir::IrOp]) -> Result<Vec<backend::ResolvedInst>, &'static str> {
    Err("fusion backend not compiled in")
}

/// Build fast code via the shared compile frontend + backend-specific resolution.
///
/// The compile frontend decodes Wasm into semantic IR and lowers it into the
/// current backend-lowered IR. Backends then resolve that IR into the fast
/// interpreter's `Instruction` stream.
pub fn build_for_function(
    function: &FunctionSpec,
    types: Option<&[Rc<FunctionType>]>,
    store: &Store,
    module: &ModuleInst,
    func_idx: u32,
) -> Result<*mut Instruction, WasmError> {
    let code = function.code();
    let func_type = function.func_type();
    let params_count = func_type.params().len();
    let results_count = func_type.results().len();
    let locals_count = function.locals().len();

    let selected_backend = match backend_mode() {
        BackendMode::Base => BackendKind::Base,
        BackendMode::Fusion => BackendKind::Fusion,
        BackendMode::Native => BackendKind::Native,
        BackendMode::Auto => active_backend().unwrap_or(BackendKind::Base),
    };
    let compile_config = selected_backend.compile_config();

    let frame_size = params_count + locals_count;
    let compile_plan = CompilePlan::for_config(code, frame_size, compile_config);
    let raw_hot_locals = compile_plan.hot_locals().raw();
    let hot_local_plan = compile_plan.hot_locals();
    let hot_locals = hot_local_plan.effective();
    let ctx = CompileContext::new(types, store, module, results_count);

    let mut stack = StackTracker::new(
        compile_plan.config(),
        params_count,
        locals_count,
        results_count,
    );

    // Compile pipeline: lower to backend-lowered IR
    let ir_ops = compile::ir_lower::lower_to_lowered_ir(code, &ctx, &mut stack, hot_local_plan)?;

    #[cfg(feature = "ir-dump")]
    ir_dump::dump_ir(func_idx, code, frame_size, &ir_ops, raw_hot_locals, hot_locals);

    let hot_mask = hot_local_plan.hot_mask();

    // Backend: resolve IR to handlers.
    let resolved = match backend_mode() {
        BackendMode::Base => backend::resolve_base(&ir_ops),
        BackendMode::Fusion => resolve_fusion_backend(&ir_ops)
            .map_err(|err| WasmError::invalid(alloc::format!("requested backend fusion is unavailable: {}", err)))?,
        BackendMode::Native => crate::vm::native::resolve_backend(&ir_ops, module, hot_mask, func_idx)
            .map_err(|err| WasmError::invalid(alloc::format!("requested backend native is unavailable: {}", err)))?,
        BackendMode::Auto => match active_backend().unwrap_or(BackendKind::Base) {
            BackendKind::Native => match crate::vm::native::resolve_backend(&ir_ops, module, hot_mask, func_idx) {
                Ok(resolved) => resolved,
                Err(_) => match resolve_fusion_backend(&ir_ops) {
                    Ok(resolved) => resolved,
                    Err(_) => backend::resolve_base(&ir_ops),
                },
            },
            BackendKind::Fusion => match resolve_fusion_backend(&ir_ops) {
                Ok(resolved) => resolved,
                Err(_) => backend::resolve_base(&ir_ops),
            },
            BackendKind::Base => backend::resolve_base(&ir_ops),
        },
    };

    #[cfg(feature = "ir-dump")]
    ir_dump::dump_resolved(func_idx, &resolved, &ir_ops);

    // Finalize: Vec<ResolvedInst> → Box<[Instruction]>
    let code_box = finalizer_ir::finalize(resolved, &mut stack);

    // Store in function spec
    use crate::vm::interp::fast::fast_code::create_fast_code;
    let (fast_code, fast_cache) = create_fast_code(code_box, params_count, locals_count, results_count);
    let entry = fast_cache.entry();
    function.set_fast_code(fast_code, fast_cache);
    Ok(entry)
}

#[cfg(all(test, feature = "micro-jit"))]
mod tests {
    use super::*;
    use alloc::{
        string::{String, ToString},
        vec,
        vec::Vec,
    };
    use crate::vm::compile::{self, FAST_COMPILE_CONFIG, ir};
    use crate::vm::interp::fast::{
        context::Context,
        handlers::{self, run_trampoline, NextHandler},
        instruction::Instruction,
    };
    use crate::vm::native::{CodeBuffer, resolve_native};

    #[derive(Clone, Copy)]
    enum BackendUnderTest {
        Base,
        Jit,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct ExecOutcome {
        frame: Vec<u64>,
        memory: Vec<u8>,
        trap: Option<String>,
    }

    fn depth_variant_for(depth: u16) -> u8 {
        if depth == 0 {
            1
        } else {
            ((((depth - 1) as usize) % FAST_COMPILE_CONFIG.tos_register_count) + 1) as u8
        }
    }

    fn post_push_variant(pre_height: u16) -> u8 {
        depth_variant_for(pre_height + 1)
    }

    fn current_variant(pre_height: u16) -> u8 {
        depth_variant_for(pre_height)
    }

    fn make_op(kind: compile::lowered_ir::IrOpKind, pre_height: u16, variant: u8) -> compile::lowered_ir::IrOp {
        compile::lowered_ir::IrOp {
            kind,
            variant,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    fn run_backend(
        backend_impl: BackendUnderTest,
        ir_ops: &[ir::IrOp],
        locals_count: usize,
        frame_words: usize,
        observe_frame_words: usize,
        initial_frame: &[u64],
        initial_memory: &[u8],
    ) -> ExecOutcome {
        run_backend_with_hot(
            backend_impl,
            ir_ops,
            locals_count,
            frame_words,
            observe_frame_words,
            initial_frame,
            initial_memory,
            [false; 3],
            [0; 3],
        )
    }

    fn run_backend_with_hot(
        backend_impl: BackendUnderTest,
        ir_ops: &[ir::IrOp],
        locals_count: usize,
        frame_words: usize,
        observe_frame_words: usize,
        initial_frame: &[u64],
        initial_memory: &[u8],
        hot_mask: [bool; 3],
        hot_values: [u64; 3],
    ) -> ExecOutcome {
        let mut stack = StackTracker::new(FAST_COMPILE_CONFIG, 0, locals_count, 0);

        let mut maybe_jit_buf = None;
        let resolved = match backend_impl {
            BackendUnderTest::Base => backend::resolve_base(ir_ops),
            BackendUnderTest::Jit => {
                let mut buf = CodeBuffer::new().expect("mmap failed");
                let resolved = resolve_native(ir_ops, &mut buf, hot_mask);
                maybe_jit_buf = Some(buf);
                resolved
            }
        };

        let mut code = super::finalizer_ir::finalize(resolved, &mut stack);
        let mut frame = vec![0u64; frame_words.max(initial_frame.len()).max(observe_frame_words)];
        frame[..initial_frame.len()].copy_from_slice(initial_frame);

        let mut memory = initial_memory.to_vec();
        let mem_base = if memory.is_empty() {
            core::ptr::null_mut()
        } else {
            memory.as_mut_ptr()
        };

        let stack_end = unsafe { frame.as_mut_ptr().add(frame.len()) };
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            stack_end,
            mem_base,
            memory.len() as u64,
        );
        ctx.hot.term_inst = handlers::term();

        let entry = code.as_mut_ptr();
        unsafe {
            let nh: NextHandler = core::mem::transmute((*entry.add(1)).handler);
            run_trampoline(
                &mut ctx,
                entry,
                frame.as_mut_ptr(),
                hot_values[0],
                hot_values[1],
                hot_values[2],
                0,
                0,
                0,
                0,
                nh,
            );
        }

        let trap = if let Some(ref error) = ctx.error {
            Some(error.message())
        } else if !ctx.hot.trap_message.is_null() {
            Some(unsafe {
                core::ffi::CStr::from_ptr(ctx.hot.trap_message)
                    .to_str()
                    .unwrap_or("trap")
                    .to_string()
            })
        } else {
            None
        };

        drop(maybe_jit_buf);

        ExecOutcome {
            frame: frame[..observe_frame_words].to_vec(),
            memory,
            trap,
        }
    }

    fn assert_base_equals_jit(
        ir_ops: &[ir::IrOp],
        locals_count: usize,
        frame_words: usize,
        observe_frame_words: usize,
        initial_frame: &[u64],
        initial_memory: &[u8],
    ) {
        assert_base_equals_jit_with_hot(
            ir_ops,
            locals_count,
            frame_words,
            observe_frame_words,
            initial_frame,
            initial_memory,
            [false; 3],
            [0; 3],
        );
    }

    fn assert_base_equals_jit_with_hot(
        ir_ops: &[ir::IrOp],
        locals_count: usize,
        frame_words: usize,
        observe_frame_words: usize,
        initial_frame: &[u64],
        initial_memory: &[u8],
        hot_mask: [bool; 3],
        hot_values: [u64; 3],
    ) {
        let base = run_backend_with_hot(
            BackendUnderTest::Base,
            ir_ops,
            locals_count,
            frame_words,
            observe_frame_words,
            initial_frame,
            initial_memory,
            hot_mask,
            hot_values,
        );
        let jit = run_backend_with_hot(
            BackendUnderTest::Jit,
            ir_ops,
            locals_count,
            frame_words,
            observe_frame_words,
            initial_frame,
            initial_memory,
            hot_mask,
            hot_values,
        );
        assert_eq!(base, jit);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_spill_fill_roundtrip() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 1 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 2 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::I32Const { value: 3 }, 2, post_push_variant(2)),
            make_op(ir::IrOpKind::I32Const { value: 4 }, 3, post_push_variant(3)),
            make_op(ir::IrOpKind::Spill { slot: 3, count: 1 }, 4, 1),
            make_op(ir::IrOpKind::I32Const { value: 5 }, 4, post_push_variant(4)),
            make_op(ir::IrOpKind::I32Add, 5, current_variant(5)),
            make_op(ir::IrOpKind::I32Add, 4, current_variant(4)),
            make_op(ir::IrOpKind::I32Add, 3, current_variant(3)),
            make_op(ir::IrOpKind::Fill { slot: 3, count: 1 }, 2, 1),
            make_op(ir::IrOpKind::I32Add, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_when_branch_targets_candidate_interior() {
        // This is the shape that would have caught the old bug: a branch targets an
        // op inside an otherwise JIT-able window. If JIT ever swallows that target
        // into an internal-only group interior again, base and JIT will diverge here.
        let mut br = make_op(ir::IrOpKind::BrIfSimple, 2, current_variant(2));
        br.has_target = true;
        br.alt_target = Some(ir::OpIndex::from(5));

        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 42 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 1 }, 1, post_push_variant(1)),
            br,
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 1 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::Drop, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 2, 8, 2, &[11, 7], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_on_memory_oob_trap() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 4 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 0xDEAD_BEEF }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::I32Store { offset: 0, memidx: 0 }, 2, current_variant(2)),
        ];

        assert_base_equals_jit(&ops, 0, 8, 0, &[], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_if_terminator() {
        for cond in [0u64, 1u64] {
            let mut if_op = make_op(ir::IrOpKind::If, 1, current_variant(1));
            if_op.has_target = true;
            if_op.alt_target = Some(ir::OpIndex::from(4));

            let ops = vec![
                make_op(ir::IrOpKind::I32Const { value: cond as u32 }, 0, post_push_variant(0)),
                if_op,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_f64_compare_br_if_suffix() {
        for (lhs, rhs) in [(3.0f64, 2.0f64), (1.0f64, 2.0f64)] {
            let mut br = make_op(ir::IrOpKind::BrIfSimple, 1, current_variant(1));
            br.has_target = true;
            br.alt_target = Some(ir::OpIndex::from(6));

            let ops = vec![
                make_op(ir::IrOpKind::F64Const { value: lhs.to_bits() }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::F64Const { value: rhs.to_bits() }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::F64Gt, 2, current_variant(2)),
                br,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_f64_zero_compare_br_if_suffix() {
        for lhs in [-1.0f64, 1.0f64] {
            let mut br = make_op(ir::IrOpKind::BrIfSimple, 1, current_variant(1));
            br.has_target = true;
            br.alt_target = Some(ir::OpIndex::from(6));

            let ops = vec![
                make_op(ir::IrOpKind::F64Const { value: lhs.to_bits() }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::F64Const { value: 0 }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::F64Gt, 2, current_variant(2)),
                br,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_bool_xor_eqz_if_suffix() {
        for (lhs, rhs) in [(3.0f64, 2.0f64), (1.0f64, 2.0f64)] {
            let mut if_op = make_op(ir::IrOpKind::If, 1, current_variant(1));
            if_op.has_target = true;
            if_op.alt_target = Some(ir::OpIndex::from(9));

            let ops = vec![
                make_op(ir::IrOpKind::F64Const { value: lhs.to_bits() }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::F64Const { value: rhs.to_bits() }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::F64Gt, 2, current_variant(2)),
                make_op(ir::IrOpKind::I32Const { value: 1 }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::I32Xor, 2, current_variant(2)),
                make_op(ir::IrOpKind::I32Eqz, 1, current_variant(1)),
                if_op,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_f64_zero_compare_if_suffix() {
        for lhs in [-1.0f64, 1.0f64] {
            let mut if_op = make_op(ir::IrOpKind::If, 1, current_variant(1));
            if_op.has_target = true;
            if_op.alt_target = Some(ir::OpIndex::from(9));

            let ops = vec![
                make_op(ir::IrOpKind::F64Const { value: lhs.to_bits() }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::F64Const { value: 0 }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::F64Gt, 2, current_variant(2)),
                make_op(ir::IrOpKind::I32Const { value: 1 }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::I32Xor, 2, current_variant(2)),
                make_op(ir::IrOpKind::I32Eqz, 1, current_variant(1)),
                if_op,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_bool_xor_br_if_suffix() {
        for (lhs, rhs) in [(3u32, 3u32), (3u32, 4u32)] {
            let mut br = make_op(ir::IrOpKind::BrIfSimple, 1, current_variant(1));
            br.has_target = true;
            br.alt_target = Some(ir::OpIndex::from(8));

            let ops = vec![
                make_op(ir::IrOpKind::I32Const { value: lhs }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::I32Const { value: rhs }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::I32Eq, 2, current_variant(2)),
                make_op(ir::IrOpKind::I32Const { value: 1 }, 1, post_push_variant(1)),
                make_op(ir::IrOpKind::I32Xor, 2, current_variant(2)),
                br,
                make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
                make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
                make_op(ir::IrOpKind::Nop, 0, 0),
            ];

            assert_base_equals_jit(&ops, 1, 8, 1, &[99], &[]);
        }
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_br_terminator_no_fixup() {
        let mut br = make_op(
            ir::IrOpKind::Br {
                stack_drop: 1,
                arity: 0,
                height: 1,
                operand_base_offset: 0,
            },
            1,
            0,
        );
        br.has_target = true;
        br.alt_target = Some(ir::OpIndex::from(5));

        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::I32Const { value: 22 }, 0, post_push_variant(0)),
            br,
            make_op(ir::IrOpKind::I32Const { value: 99 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 0 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 1 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 2, 8, 2, &[0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_br_terminator_arity_no_fixup() {
        let mut br = make_op(
            ir::IrOpKind::Br {
                stack_drop: 0,
                arity: 1,
                height: 1,
                operand_base_offset: 0,
            },
            1,
            0,
        );
        br.has_target = true;
        br.alt_target = Some(ir::OpIndex::from(4));

        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 22 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::Spill { slot: 0, count: 1 }, 1, current_variant(1)),
            br,
            make_op(ir::IrOpKind::I32Const { value: 99 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::Fill { slot: 0, count: 1 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 1 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 2, 8, 2, &[0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_return_one_terminator() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 42 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::Spill { slot: 3, count: 1 }, 1, current_variant(1)),
            make_op(
                ir::IrOpKind::ReturnOne {
                    frame_size: 0,
                    operand_base_offset: 24,
                    height: 1,
                },
                1,
                0,
            ),
        ];

        assert_base_equals_jit(&ops, 0, 8, 1, &[0, 0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_return_void_terminator() {
        let ops = vec![make_op(
            ir::IrOpKind::ReturnVoid { frame_size: 0 },
            0,
            0,
        )];

        assert_base_equals_jit(&ops, 0, 8, 3, &[0, 0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_hot_local_alias_chain() {
        let hot = 3.0f64.to_bits();
        let ops = vec![
            make_op(ir::IrOpKind::LocalGetHot { reg: 0 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalGetHot { reg: 0 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::F64Mul, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalTeeHot { reg: 1 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalGetHot { reg: 1 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::F64Add, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 3 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit_with_hot(&ops, 4, 8, 4, &[0, 0, 0, 0], &[], [false, true, false], [hot, 0, 0]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_hot_and_frame_float_mix() {
        let ops = vec![
            make_op(ir::IrOpKind::LocalGetFrame { idx: 3 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalGetHot { reg: 1 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 4 }, 2, post_push_variant(2)),
            make_op(ir::IrOpKind::F64Sub, 3, current_variant(3)),
            make_op(ir::IrOpKind::F64Add, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 5 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit_with_hot(
            &ops,
            6,
            8,
            6,
            &[0, 0, 0, 5.0f64.to_bits(), 2.0f64.to_bits(), 0],
            &[],
            [false, true, false],
            [0, 3.0f64.to_bits(), 0],
        );
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_float_convert_chain() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 9 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::F64ConvertI32S, 1, current_variant(1)),
            make_op(ir::IrOpKind::F32DemoteF64, 1, current_variant(1)),
            make_op(ir::IrOpKind::F64PromoteF32, 1, current_variant(1)),
            make_op(ir::IrOpKind::F64Sqrt, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 1, 8, 1, &[0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_f64_load_store_chain() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 8 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 0 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::F64Load { offset: 0, memidx: 0 }, 2, current_variant(2)),
            make_op(
                ir::IrOpKind::F64Const {
                    value: 2.0f64.to_bits(),
                },
                2,
                post_push_variant(2),
            ),
            make_op(ir::IrOpKind::F64Mul, 3, current_variant(3)),
            make_op(ir::IrOpKind::F64Store { offset: 0, memidx: 0 }, 2, current_variant(2)),
            make_op(ir::IrOpKind::I32Const { value: 8 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::F64Load { offset: 0, memidx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 0 }, 1, current_variant(1)),
        ];

        let mut memory = vec![0u8; 16];
        memory[0..8].copy_from_slice(&1.5f64.to_bits().to_le_bytes());
        assert_base_equals_jit(&ops, 1, 8, 1, &[0], &memory);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_f64_frame_tee_reload_chain() {
        let ops = vec![
            make_op(
                ir::IrOpKind::F64Const {
                    value: 1.5f64.to_bits(),
                },
                0,
                post_push_variant(0),
            ),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::Drop, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 0 }, 0, post_push_variant(0)),
            make_op(
                ir::IrOpKind::F64Const {
                    value: 2.0f64.to_bits(),
                },
                1,
                post_push_variant(1),
            ),
            make_op(ir::IrOpKind::F64Mul, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 1 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 2, 8, 2, &[0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_i32_frame_tee_reload_chain() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::Drop, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 0 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 7 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::I32Add, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 1 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 2, 8, 2, &[0, 0], &[]);
    }

    #[test]
    fn test_base_vs_jit_equivalent_with_two_frame_aliases_and_eviction() {
        let ops = vec![
            make_op(ir::IrOpKind::I32Const { value: 11 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 0 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::Drop, 1, current_variant(1)),
            make_op(ir::IrOpKind::I32Const { value: 7 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 1 }, 1, current_variant(1)),
            make_op(ir::IrOpKind::Drop, 1, current_variant(1)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 0 }, 0, post_push_variant(0)),
            make_op(ir::IrOpKind::I32Const { value: 3 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::LocalTeeFrame { idx: 2 }, 2, current_variant(2)),
            make_op(ir::IrOpKind::Drop, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalGetFrame { idx: 1 }, 1, post_push_variant(1)),
            make_op(ir::IrOpKind::I32Add, 2, current_variant(2)),
            make_op(ir::IrOpKind::LocalSetFrame { idx: 3 }, 1, current_variant(1)),
        ];

        assert_base_equals_jit(&ops, 4, 8, 4, &[0, 0, 0, 0], &[]);
    }
}
