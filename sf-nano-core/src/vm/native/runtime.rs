//! Native runtime entry point.

use alloc::string::ToString;
use alloc::vec;
use core::arch::global_asm;

use crate::error::WasmError;
use crate::vm::entities::{FunctionInst, MemInst, ModuleInst};
use crate::vm::interp::stack::InterpreterStack;
use crate::vm::store::Store;
use crate::vm::value::Value;

use super::context::Context;
use super::instruction::NativeEntry;

const MAX_SLOTS: usize = crate::constants::MAX_STACK_SIZE / core::mem::size_of::<u64>();

#[allow(improper_ctypes)]
unsafe extern "C" {
    fn native_run_entry(
        ctx: *mut Context,
        entry: NativeEntry,
        fp: *mut u64,
        l0: u64,
        l1: u64,
        l2: u64,
        t0: u64,
        t1: u64,
        t2: u64,
        t3: u64,
    );

    fn native_term();
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use alloc::rc::Rc;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;
    use crate::vm::backend::BackendKind;
    use crate::vm::compile::StackTracker;
    use crate::module::entities::FunctionSpec;
    use crate::module::type_context::TypeContext;
    use crate::module::type_defs::FunctionType;
    use crate::utils::limits::Limits;
    use crate::value_type::ValueType;
    use crate::vm::lowered::{IrOp, IrOpKind, OpIndex};
    use crate::vm::native::instruction::{NativeEntry, NativeInst};
    use crate::vm::native::{CodeBuffer, code::create_native_code, finalizer, resolve_native};
    use crate::vm::entities::{FunctionInst, TableInst};
    use crate::vm::store::Store;
    use crate::vm::value::RefHandle;
    #[test]
    fn test_native_run_entry_can_enter_terminal_instruction() {
        let mut stack = [0u64; 16];
        let stack_end = stack.as_mut_ptr().wrapping_add(stack.len());
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        let mut insts = [
            NativeInst::new_entry_only(term_entry()),
            NativeInst::new_entry_only(term_entry()),
        ];

        unsafe {
            native_run_entry(
                &mut ctx,
                insts[0].entry,
                stack.as_mut_ptr(),
                0, 0, 0, 0, 0, 0, 0,
            );
        }
    }

    fn make_op(kind: IrOpKind, pre_height: u16) -> IrOp {
        IrOp {
            kind,
            variant: 0,
            pre_height,
            fallthrough: None,
            alt_target: None,
            has_target: false,
        }
    }

    fn compile_into_spec(
        module: &crate::vm::entities::ModuleInst,
        spec: &FunctionSpec,
        ops: Vec<IrOp>,
        params: usize,
        locals: usize,
        results: usize,
    ) {
        let mut buf = module.native_code_buffer().expect("native code buffer");
        let resolved = resolve_native(&ops, &mut buf, [false; 3]).expect("native resolve");
        let mut stack = StackTracker::new(BackendKind::Native.compile_config(), params, locals, results);
        let (code, metadata) = finalizer::finalize(resolved, &mut stack, &mut buf, "test", 0);
        drop(buf);
        let (native_code, native_cache) = create_native_code(code, metadata, params, locals, results);
        spec.set_native_code(native_code, native_cache);
    }

    #[test]
    fn test_native_br_if_block_first_shape() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let operand_base_offset = 8u32;
        let frame_size = 1u16;

        let mut ops = vec![
            make_op(IrOpKind::Block, 0),
            make_op(IrOpKind::LocalGetFrame { idx: 0 }, 0),
            make_op(
                IrOpKind::BrIf {
                    stack_drop: 0,
                    arity: 0,
                    height: 1,
                    operand_base_offset,
                },
                1,
            ),
            make_op(IrOpKind::I32Const { value: 2 }, 0),
            make_op(
                IrOpKind::ReturnOne {
                    frame_size,
                    operand_base_offset,
                    height: 1,
                },
                1,
            ),
            make_op(IrOpKind::End, 0),
            make_op(IrOpKind::I32Const { value: 3 }, 0),
            make_op(
                IrOpKind::ReturnOne {
                    frame_size,
                    operand_base_offset,
                    height: 1,
                },
                1,
            ),
        ];
        ops[2].has_target = true;
        ops[2].alt_target = Some(OpIndex::from(5));

        let resolved = resolve_native(&ops, &mut buf, [false; 3]).expect("native resolve");
        let mut stack = StackTracker::new(BackendKind::Native.compile_config(), 1, 0, 1);
        let (mut code, _meta) = finalizer::finalize(resolved, &mut stack, &mut buf, "test", 0);

        for &(param, expected) in &[(0u64, 2u64), (1u64, 3u64)] {
            let mut frame = [0u64; 8];
            frame[0] = param;
            let stack_end = unsafe { frame.as_mut_ptr().add(frame.len()) };
            let mut ctx = Context::new(
                core::ptr::null_mut(),
                core::ptr::null(),
                stack_end,
                core::ptr::null_mut(),
                0,
            );
            ctx.hot.term_entry = term_entry();

            unsafe {
                native_run_entry(
                    &mut ctx,
                    code[0].entry,
                    frame.as_mut_ptr(),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                );
            }

            assert_eq!(frame[0], expected, "param {}", param);
            assert!(ctx.error.is_none(), "unexpected native error for param {}: {:?}", param, ctx.error);
        }
    }

    #[test]
    fn test_native_return_many_preserves_result_order() {
        let mut buf = CodeBuffer::new().expect("mmap failed");
        let operand_base_offset = 24u32;
        let frame_size = 0u16;

        let ops = vec![
            make_op(IrOpKind::I32Const { value: 4 }, 0),
            make_op(IrOpKind::I32Const { value: 5 }, 1),
            make_op(
                IrOpKind::Return {
                    arity: 2,
                    frame_size,
                    operand_base_offset,
                    height: 2,
                },
                2,
            ),
        ];

        let resolved = resolve_native(&ops, &mut buf, [false; 3]).expect("native resolve");
        let mut stack = StackTracker::new(BackendKind::Native.compile_config(), 0, 0, 2);
        let (mut code, _meta) = finalizer::finalize(resolved, &mut stack, &mut buf, "test", 0);

        let mut frame = [0u64; 8];
        let stack_end = unsafe { frame.as_mut_ptr().add(frame.len()) };
        let mut ctx = Context::new(
            core::ptr::null_mut(),
            core::ptr::null(),
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(
                &mut ctx,
                code[0].entry,
                frame.as_mut_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            );
        }

        assert_eq!(frame[0], 4);
        assert_eq!(frame[1], 5);
        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
    }

    #[test]
    fn test_native_internal_call_returns_single_i32() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });

        let callee_spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            callee_spec,
            vec![
                make_op(IrOpKind::I32Const { value: 0x132 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let callee_ptr = &module.functions[0] as *const FunctionInst as u64;
        let caller_spec = match &module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            caller_spec,
            vec![
                make_op(
                    IrOpKind::CallInternal {
                        callee: callee_ptr,
                        delta: crate::vm::compile::SlotRef::operand_relative(0),
                    },
                    0,
                ),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let caller_code = caller_spec.get_native_code().expect("caller native code");
        assert_eq!(caller_code.helper_metadata().len(), 1);
        assert_eq!(caller_code.helper_metadata()[0].data1, 3);
        assert_eq!(
            caller_code.helper_metadata()[0].next_entry,
            caller_code.code()[1].entry as usize as u64
        );
        assert_eq!(caller_code.code()[1].tos_slots & 0xffff, 3);

        let mut raw_stack = [0u64; 16];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let func_ptr = store.function(1) as *const FunctionInst;
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();
        let entry = match unsafe { &*func_ptr } {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 0, 0, 0, 0, 0, 0, 0);
        }

        assert_ne!(raw_stack[4], 0, "callee saved_fp slot should hold caller fp");
        assert_eq!(ctx.hot.call_depth, 0, "entry-frame return should branch to term before decrement");
        assert_eq!(raw_stack[3], 0x132, "caller operand slot should receive callee result");
        assert_eq!(raw_stack[0], 0x132, "top-level return should copy result to slot 0");
    }

    #[test]
    fn test_native_internal_call_runs_init_locals_before_frame_access() {
        let callee_sig = Rc::new(FunctionType::new(vec![ValueType::I32], vec![ValueType::I32]));
        let caller_sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![callee_sig.clone(), caller_sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(callee_sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(caller_sig, 0),
            type_index: 1,
        });

        match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec.set_locals(vec![
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
            ]),
            _ => unreachable!(),
        }

        let callee_spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            callee_spec,
            vec![
                make_op(IrOpKind::InitLocals { k0: 6, k1: 1, k2: 2 }, 0),
                make_op(IrOpKind::LocalGetFrame { idx: 6 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 7,
                        operand_base_offset: 80,
                        height: 1,
                    },
                    1,
                ),
            ],
            1,
            6,
            1,
        );

        let callee_ptr = &module.functions[0] as *const FunctionInst as u64;
        let caller_spec = match &module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            caller_spec,
            vec![
                make_op(IrOpKind::I32Const { value: 0x1234 }, 0),
                make_op(
                    IrOpKind::CallInternal {
                        callee: callee_ptr,
                        delta: crate::vm::compile::SlotRef::operand_relative(0),
                    },
                    1,
                ),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 32];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let func_ptr = store.function(1) as *const FunctionInst;
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();
        let entry = match unsafe { &*func_ptr } {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 0, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 0x1234, "callee should observe param0 after InitLocals swap");
    }

    #[test]
    fn test_native_top_level_entry_runs_init_locals_before_frame_access() {
        let sig = Rc::new(FunctionType::new(vec![ValueType::I32], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });

        match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec.set_locals(vec![
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
            ]),
            _ => unreachable!(),
        }

        let spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            spec,
            vec![
                make_op(IrOpKind::InitLocals { k0: 6, k1: 1, k2: 2 }, 0),
                make_op(IrOpKind::LocalGetFrame { idx: 6 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 7,
                        operand_base_offset: 80,
                        height: 1,
                    },
                    1,
                ),
            ],
            1,
            6,
            1,
        );

        let mut raw_stack = [0u64; 32];
        raw_stack[0] = 0x1234;
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();
        let entry = match store.function(0) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), raw_stack[0], 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 0x1234, "top-level native entry should run InitLocals swap");
    }

    #[test]
    fn test_native_internal_call_with_mixed_six_args_preserves_param0_after_init_locals() {
        let callee_sig = Rc::new(FunctionType::new(
            vec![
                ValueType::I32,
                ValueType::F64,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
                ValueType::I32,
            ],
            vec![ValueType::I32],
        ));
        let caller_sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![callee_sig.clone(), caller_sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(callee_sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(caller_sig, 0),
            type_index: 1,
        });

        match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec.set_locals(vec![ValueType::I32, ValueType::I32]),
            _ => unreachable!(),
        }

        let callee_spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            callee_spec,
            vec![
                make_op(IrOpKind::InitLocals { k0: 6, k1: 7, k2: 3 }, 0),
                make_op(IrOpKind::LocalGetFrame { idx: 6 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 8,
                        operand_base_offset: 88,
                        height: 1,
                    },
                    1,
                ),
            ],
            6,
            2,
            1,
        );

        let callee_ptr = &module.functions[0] as *const FunctionInst as u64;
        let caller_spec = match &module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            caller_spec,
            vec![
                make_op(IrOpKind::I32Const { value: 0xb78 }, 0),
                make_op(IrOpKind::F64Const { value: 1.25f64.to_bits() }, 1),
                make_op(IrOpKind::I32Const { value: 0 }, 2),
                make_op(IrOpKind::I32Const { value: u32::MAX }, 3),
                IrOp {
                    kind: IrOpKind::Spill { slot: 3, count: 1 },
                    variant: 1,
                    pre_height: 4,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::I32Const { value: 0 }, 4),
                IrOp {
                    kind: IrOpKind::Spill { slot: 4, count: 1 },
                    variant: 2,
                    pre_height: 5,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::I32Const { value: 0x66 }, 5),
                make_op(
                    IrOpKind::CallInternal {
                        callee: callee_ptr,
                        delta: crate::vm::compile::SlotRef::operand_relative(0),
                    },
                    6,
                ),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 48];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let func_ptr = store.function(1) as *const FunctionInst;
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();
        let entry = match unsafe { &*func_ptr } {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 0, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 0xb78, "mixed-arg internal call should preserve param0 after InitLocals");
    }

    #[test]
    fn test_native_select_with_hot_local_alias_preserves_lower_stack_value() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });

        let spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            spec,
            vec![
                make_op(IrOpKind::I32Const { value: 5 }, 0),
                make_op(IrOpKind::LocalGetHot { reg: 0 }, 1),
                make_op(IrOpKind::I32Const { value: 20 }, 2),
                make_op(IrOpKind::I32Const { value: 1 }, 3),
                make_op(IrOpKind::Select, 4),
                make_op(IrOpKind::I32Add, 2),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 16];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let entry = match store.function(0) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 11, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 16);
    }

    #[test]
    fn test_native_call_internal_spill_materializes_hot_local_alias() {
        let callee_sig = Rc::new(FunctionType::new(
            vec![ValueType::I32, ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
        ));
        let caller_sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![callee_sig.clone(), caller_sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(callee_sig, 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(caller_sig, 0),
            type_index: 1,
        });

        let callee_spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            callee_spec,
            vec![
                make_op(IrOpKind::LocalGetFrame { idx: 0 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 3,
                        operand_base_offset: 48,
                        height: 1,
                    },
                    1,
                ),
            ],
            3,
            0,
            1,
        );

        let callee_ptr = &module.functions[0] as *const FunctionInst as u64;
        let caller_spec = match &module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            caller_spec,
            vec![
                make_op(IrOpKind::LocalGetHot { reg: 0 }, 0),
                IrOp {
                    kind: IrOpKind::Spill { slot: 3, count: 1 },
                    variant: 1,
                    pre_height: 1,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::I32Const { value: 42 }, 1),
                IrOp {
                    kind: IrOpKind::Spill { slot: 4, count: 1 },
                    variant: 2,
                    pre_height: 2,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::I32Const { value: 7 }, 2),
                IrOp {
                    kind: IrOpKind::Spill { slot: 5, count: 1 },
                    variant: 3,
                    pre_height: 3,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(
                    IrOpKind::CallInternal {
                        callee: callee_ptr,
                        delta: crate::vm::compile::SlotRef::operand_relative(0),
                    },
                    3,
                ),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 24];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let entry = match store.function(1) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 11, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 11);
    }

    #[test]
    fn test_native_call_internal_preserves_oldest_spilled_arg_across_select_shape() {
        let callee_sig = Rc::new(FunctionType::new(
            vec![ValueType::I32, ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
        ));
        let caller_sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![callee_sig.clone(), caller_sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(callee_sig, 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(caller_sig, 0),
            type_index: 1,
        });

        let callee_spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            callee_spec,
            vec![
                make_op(IrOpKind::LocalGetFrame { idx: 0 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 3,
                        operand_base_offset: 48,
                        height: 1,
                    },
                    1,
                ),
            ],
            3,
            0,
            1,
        );

        let callee_ptr = &module.functions[0] as *const FunctionInst as u64;
        let caller_spec = match &module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            caller_spec,
            vec![
                make_op(IrOpKind::LocalGetFrame { idx: 6 }, 0),
                make_op(IrOpKind::LocalGetHot { reg: 0 }, 1),
                make_op(IrOpKind::I32Const { value: 9 }, 2),
                make_op(IrOpKind::LocalGetFrame { idx: 10 }, 3),
                IrOp {
                    kind: IrOpKind::Spill { slot: 3, count: 1 },
                    variant: 1,
                    pre_height: 4,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::LocalGetFrame { idx: 10 }, 4),
                IrOp {
                    kind: IrOpKind::Spill { slot: 4, count: 1 },
                    variant: 2,
                    pre_height: 5,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(IrOpKind::I32Const { value: 9 }, 5),
                make_op(IrOpKind::I32GeS, 6),
                make_op(IrOpKind::Select, 5),
                IrOp {
                    kind: IrOpKind::Spill { slot: 5, count: 1 },
                    variant: 3,
                    pre_height: 3,
                    fallthrough: None,
                    alt_target: None,
                    has_target: false,
                },
                make_op(
                    IrOpKind::CallInternal {
                        callee: callee_ptr,
                        delta: crate::vm::compile::SlotRef::operand_relative(0),
                    },
                    3,
                ),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 40];
        raw_stack[6] = 0xb78;
        raw_stack[10] = 6;
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let entry = match store.function(1) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 9, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(
            raw_stack[0],
            0xb78,
            "frame6={:#x} frame10={} arg_slots=[{:#x},{:#x},{:#x}]",
            raw_stack[6],
            raw_stack[10],
            raw_stack[3],
            raw_stack[4],
            raw_stack[5],
        );
    }

    #[test]
    fn test_native_integer_binop_does_not_clobber_hot_local_alias() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });

        let spec = match &module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        compile_into_spec(
            &module,
            spec,
            vec![
                make_op(IrOpKind::LocalGetHot { reg: 0 }, 0),
                make_op(IrOpKind::I32Const { value: 1 }, 1),
                make_op(IrOpKind::I32Add, 2),
                make_op(IrOpKind::Drop, 1),
                make_op(IrOpKind::LocalGetHot { reg: 0 }, 0),
                make_op(
                    IrOpKind::ReturnOne {
                        frame_size: 0,
                        operand_base_offset: 24,
                        height: 1,
                    },
                    1,
                ),
            ],
            0,
            0,
            1,
        );

        let mut raw_stack = [0u64; 16];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let entry = match store.function(0) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(&mut ctx, entry, raw_stack.as_mut_ptr(), 11, 0, 0, 0, 0, 0, 0);
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 11);
    }

    #[test]
    fn test_native_compiler_path_simple_const_return() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig.clone(), 0),
            type_index: 0,
        });

        let spec = match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        spec.set_code(crate::module::entities::Bytecode::from(&[0x41, 0xb2, 0x02, 0x0b][..]));

        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        let spec = match store.function(0) {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };

        crate::vm::native::compiler::build_for_function(
            spec,
            Some(&store.module().types),
            &store,
            store.module(),
            0,
        )
        .expect("native build");
        let entry = spec.native_cache().entry();

        let mut raw_stack = [0u64; 16];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(
                &mut ctx,
                entry,
                raw_stack.as_mut_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            );
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 306);
    }

    #[test]
    fn test_native_compiler_path_call_indirect() {
        let sig = Rc::new(FunctionType::new(vec![], vec![ValueType::I32]));
        let types = TypeContext::new(vec![sig.clone()]);
        let mut module = crate::vm::entities::ModuleInst::new("test".to_string(), types);
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig.clone(), 0),
            type_index: 0,
        });
        module.functions.push(FunctionInst::Local {
            spec: FunctionSpec::new(sig, 0),
            type_index: 0,
        });
        module.tables.push(TableInst::new(Limits::new(1, Some(1)).unwrap(), ValueType::funcref()));
        module.tables[0].elements[0] = RefHandle::new(0);

        let callee_spec = match &mut module.functions[0] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        callee_spec.set_code(crate::module::entities::Bytecode::from(&[0x41, 0xb2, 0x02, 0x0b][..]));

        let caller_spec = match &mut module.functions[1] {
            FunctionInst::Local { spec, .. } => spec,
            _ => unreachable!(),
        };
        caller_spec.set_code(crate::module::entities::Bytecode::from(&[
            0x41, 0x00, // i32.const 0
            0x11, 0x00, 0x00, // call_indirect (type 0) (table 0)
            0x0b, // end
        ][..]));

        let mut store = Store::new(module);
        let module_ptr = store.module() as *const crate::vm::entities::ModuleInst;
        for func_idx in 0..2 {
            let spec = match store.function(func_idx) {
                FunctionInst::Local { spec, .. } => spec,
                _ => unreachable!(),
            };
            crate::vm::native::compiler::build_for_function(
                spec,
                Some(&store.module().types),
                &store,
                store.module(),
                func_idx as u32,
            )
            .expect("native build");
        }

        let entry: NativeEntry = match store.function(1) {
            FunctionInst::Local { spec, .. } => spec.native_cache().entry(),
            _ => unreachable!(),
        };

        let mut raw_stack = [0u64; 16];
        let stack_end = unsafe { raw_stack.as_mut_ptr().add(raw_stack.len()) };
        let mut ctx = Context::new(
            &mut store as *mut Store,
            module_ptr,
            stack_end,
            core::ptr::null_mut(),
            0,
        );
        ctx.hot.term_entry = term_entry();

        unsafe {
            native_run_entry(
                &mut ctx,
                entry,
                raw_stack.as_mut_ptr(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            );
        }

        assert!(ctx.error.is_none(), "unexpected native error: {:?}", ctx.error);
        assert_eq!(raw_stack[0], 306);
    }
}

#[cfg(target_arch = "aarch64")]
global_asm!(
    r#"
    .text
    .p2align 2
    .global _native_run_entry
_native_run_entry:
    sub sp, sp, #96
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    str x30, [sp, #80]
    mov x19, x0
    mov x20, xzr
    mov x21, x2
    mov x22, x3
    mov x23, x4
    mov x24, x5
    mov x25, x6
    mov x26, x7
    ldr x27, [sp, #96]
    ldr x28, [sp, #104]
    mov x16, x1
    br x16

    .p2align 2
    .global _native_term
_native_term:
    ldr x30, [sp, #80]
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    add sp, sp, #96
    ret
"#
);

pub fn term_entry() -> NativeEntry {
    native_term
}

pub fn eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    args: &[Value],
) -> Result<InterpreterStack, WasmError> {
    let FunctionInst::Local { spec, .. } = func_inst else {
        return Err(WasmError::invalid(
            "native runtime only supports local functions".into(),
        ));
    };

    let ft = spec.func_type();
    let params_len = ft.params().len();
    if args.len() != params_len {
        return Err(WasmError::invalid(alloc::format!(
            "invalid argument count: got {}, expected {}",
            args.len(),
            params_len
        )));
    }

    let mut stack = vec![0u64; MAX_SLOTS];
    let stack_base = stack.as_mut_ptr();
    let stack_end = unsafe { stack_base.add(MAX_SLOTS) };

    unsafe {
        for (i, a) in args.iter().enumerate() {
            core::ptr::write(stack_base.add(i), a.to_raw());
        }
    }

    internal_eval(func_inst, store, stack_base, stack_end, args.len())?;

    let results_len = ft.results().len();
    let mut out = InterpreterStack::with_exact_capacity(results_len);
    unsafe {
        for i in 0..results_len {
            out.push(core::ptr::read(stack_base.add(i)));
        }
    }
    Ok(out)
}

pub fn internal_eval(
    func_inst: &FunctionInst,
    store: &mut Store,
    stack_base: *mut u64,
    stack_end: *mut u64,
    sp_offset: usize,
) -> Result<(), WasmError> {
    let spec = match func_inst {
        FunctionInst::Local { spec, .. } => spec,
        FunctionInst::External { .. } => {
            return Err(WasmError::internal("external functions should not reach native runtime".into()))
        }
    };

    if !spec.has_native_code() {
        crate::vm::native::precompile::precompile_module(store)?;
    }
    if !spec.has_native_code() {
        return Err(WasmError::invalid("native backend unavailable for function".into()));
    }

    let ft = spec.func_type();
    let params_len = ft.params().len();
    if sp_offset < params_len {
        return Err(WasmError::internal("invalid stack size".into()));
    }
    let fp_index = sp_offset - params_len;
    let fp = unsafe { stack_base.add(fp_index) };

    let locals_len = spec.locals().len();
    if locals_len > 0 {
        unsafe { core::ptr::write_bytes(fp.add(params_len), 0, locals_len) };
    }

    let (heap_base, heap_size) = if !store.module().memories.is_empty() {
        let m = &store.module().memories[0];
        (m.data.as_ptr() as *mut u8, m.data.len())
    } else {
        (core::ptr::null_mut(), 0usize)
    };

    let module_ptr = store.module() as *const ModuleInst;
    let store_ptr = store as *mut Store;
    let mut ctx = Context::new(store_ptr, module_ptr, stack_end, heap_base, heap_size as u64);

    let frame_size = params_len + locals_len;
    unsafe {
        *fp.add(frame_size) = 0;
        *fp.add(frame_size + 1) = 0;
        *fp.add(frame_size + 2) = 0;
    }

    let entry = spec.native_cache().entry();
    ctx.hot.term_entry = term_entry();

    unsafe {
        native_run_entry(&mut ctx, entry, fp, 0, 0, 0, 0, 0, 0, 0);
    }

    if !ctx.hot.trap_message.is_null() && ctx.error.is_none() {
        let msg = unsafe {
            core::ffi::CStr::from_ptr(ctx.hot.trap_message)
                .to_str()
                .unwrap_or("trap")
        };
        ctx.error = Some(WasmError::trap(msg.to_string()));
    }

    if let Some(error) = ctx.error {
        return Err(error);
    }

    Ok(())
}
