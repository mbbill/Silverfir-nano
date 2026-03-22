#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, rc::Rc, string::String, vec};

    use super::super::compile::{compile_module, IndexedMemFusion};
    use super::super::compile_fusion::{indexed_mem_fusion, int_binary_imm_inst};
    use crate::{
        module::{type_context::TypeContext, type_defs::FunctionType},
        vm::{
            arch::arm64::{enc, reg::Arm64Reg},
            backend::BackendConfig,
            entities::ModuleInst,
            machine::machine_ir::{
                MachineAddr, MachineBlock, MachineBlockId, MachineCallLinkLayout,
                MachineConstData, MachineConstId, MachineConvertOp, MachineEdge,
                MachineExternBinding, MachineExternId, MachineFloatUnaryOp, MachineFloatWidth,
                MachineFrameRegion, MachineFuncId, MachineFunction, MachineFunctionRuntime,
                MachineHelperCall, MachineHelperSymbol, MachineInst, MachineInstKind,
                MachineIntBinaryOp, MachineIntWidth, MachineLoadExtension, MachineMemWidth,
                MachineModule, MachineProgram, MachineReg, MachineRuntimeContract,
                MachineStorageType, MachineTerminator, MachineValue, MACHINE_FIXED_REG_COUNT,
                MACHINE_FP_REG,
            },
            runtime::{code::CompiledNativeModule, context::NativeContext},
            store::Store,
        },
    };

    #[test]
    fn selects_small_wrapping_i32_add_as_sub_immediate() {
        assert_eq!(
            int_binary_imm_inst(
                MachineIntWidth::I32,
                MachineIntBinaryOp::Add,
                Arm64Reg::X9,
                MachineValue::Reg(MachineReg(4)),
                MachineValue::Imm64(u32::MAX as u64),
            )
            .expect("immediate selection should succeed"),
            Some(enc::sub_imm_32(Arm64Reg::X9, Arm64Reg::X23, 1))
        );
    }

    #[test]
    fn selects_constant_shift_immediate() {
        assert_eq!(
            int_binary_imm_inst(
                MachineIntWidth::I32,
                MachineIntBinaryOp::ShrU,
                Arm64Reg::X9,
                MachineValue::Reg(MachineReg(4)),
                MachineValue::Imm64(8),
            )
            .expect("shift-immediate selection should succeed"),
            Some(enc::lsr_imm_32(Arm64Reg::X9, Arm64Reg::X23, 8))
        );
    }

    #[test]
    fn selects_power_of_two_mul_as_shift_immediate() {
        assert_eq!(
            int_binary_imm_inst(
                MachineIntWidth::I64,
                MachineIntBinaryOp::Mul,
                Arm64Reg::X9,
                MachineValue::Reg(MachineReg(4)),
                MachineValue::Imm64(8),
            )
            .expect("mul-immediate selection should succeed"),
            Some(enc::lsl_imm_64(Arm64Reg::X9, Arm64Reg::X23, 3))
        );
    }

    #[test]
    fn selects_logical_and_immediate() {
        assert_eq!(
            int_binary_imm_inst(
                MachineIntWidth::I32,
                MachineIntBinaryOp::And,
                Arm64Reg::X9,
                MachineValue::Reg(MachineReg(4)),
                MachineValue::Imm64(15),
            )
            .expect("logical-immediate selection should succeed"),
            enc::and_imm_32(Arm64Reg::X9, Arm64Reg::X23, 15)
        );
    }

    #[test]
    fn selects_xor_all_ones_as_mvn() {
        assert_eq!(
            int_binary_imm_inst(
                MachineIntWidth::I32,
                MachineIntBinaryOp::Xor,
                Arm64Reg::X9,
                MachineValue::Reg(MachineReg(4)),
                MachineValue::Imm64(u32::MAX as u64),
            )
            .expect("xor-all-ones selection should succeed"),
            Some(enc::mvn_32(Arm64Reg::X9, Arm64Reg::X23))
        );
    }

    #[test]
    fn fuses_single_use_add_into_indexed_load() {
        let block = MachineBlock {
            id: MachineBlockId(0),
            params: vec![],
            ops: vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Load {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(6),
                        addr: MachineAddr {
                            base: MachineReg(6),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        extension: MachineLoadExtension::None,
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        };

        assert_eq!(
            indexed_mem_fusion(&block, 0),
            Some(IndexedMemFusion::Load {
                dst: MachineReg(6),
                base: MachineReg(2),
                index: MachineReg(5),
                width: MachineMemWidth::U64,
                extension: MachineLoadExtension::None,
                scaled: false,
                uxtw: false,
            })
        );
    }

    #[test]
    fn does_not_fuse_store_that_writes_computed_address_value() {
        let block = MachineBlock {
            id: MachineBlockId(0),
            params: vec![],
            ops: vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(6),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        };

        assert_eq!(indexed_mem_fusion(&block, 0), None);
    }

    #[test]
    fn does_not_fuse_when_computed_address_value_is_used_later() {
        let block = MachineBlock {
            id: MachineBlockId(0),
            params: vec![],
            ops: vec![
                MachineInst {
                    kind: MachineInstKind::IntBinary {
                        width: MachineIntWidth::I64,
                        op: MachineIntBinaryOp::Add,
                        dst: MachineReg(6),
                        lhs: MachineValue::Reg(MachineReg(2)),
                        rhs: MachineValue::Reg(MachineReg(5)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Store {
                        ty: MachineStorageType::GpWord,
                        addr: MachineAddr {
                            base: MachineReg(6),
                            offset: 0,
                        },
                        width: MachineMemWidth::U64,
                        src: MachineValue::Reg(MachineReg(7)),
                    },
                },
                MachineInst {
                    kind: MachineInstKind::Move {
                        ty: MachineStorageType::GpWord,
                        dst: MachineReg(8),
                        src: MachineValue::Reg(MachineReg(6)),
                    },
                },
            ],
            terminator: MachineTerminator::Return,
        };

        assert_eq!(indexed_mem_fusion(&block, 0), None);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn compiles_helper_with_live_fp_transient() {
        let function = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 7,
                reg_count: 8,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: MachineReg(4),
                                src: MachineValue::Imm64(0x3f800000),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::FloatUnary {
                                width: MachineFloatWidth::F32,
                                op: MachineFloatUnaryOp::Abs,
                                dst: MachineReg(7),
                                src: MachineValue::Reg(MachineReg(4)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::CallHelper(MachineHelperCall {
                                target: MachineExternId(0),
                                metadata: MachineConstId(0),
                            }),
                        },
                        MachineInst {
                            kind: MachineInstKind::FloatUnary {
                                width: MachineFloatWidth::F32,
                                op: MachineFloatUnaryOp::Neg,
                                dst: MachineReg(7),
                                src: MachineValue::Reg(MachineReg(7)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::Fp32,
                                addr: MachineAddr {
                                    base: MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: MachineMemWidth::U32,
                                src: MachineValue::Reg(MachineReg(7)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![function],
                consts: vec![MachineConstData {
                    id: MachineConstId(0),
                    align: 1,
                    bytes: vec![0],
                }],
                externs: vec![MachineExternBinding {
                    id: MachineExternId(0),
                    symbol: MachineHelperSymbol::MemoryGrow,
                }],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        compile_module(&module, &compiled)
            .expect("arm64 compile should preserve live FP widths across helpers");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn float_to_float_converts_do_not_bounce_through_gp_scratch() {
        let function = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 7,
                reg_count: 10,
                fp_transient_count: 3,
                fp_reg_init_widths: vec![],
                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::FloatConst {
                                width: MachineFloatWidth::F64,
                                dst: MachineReg(7),
                                bits: 1.5f64.to_bits(),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Convert {
                                op: MachineConvertOp::F32DemoteF64,
                                dst: MachineReg(8),
                                src: MachineValue::Reg(MachineReg(7)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Convert {
                                op: MachineConvertOp::F64PromoteF32,
                                dst: MachineReg(9),
                                src: MachineValue::Reg(MachineReg(8)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].as_ref().expect("entry");
        let executable = module
            .native_code_buffer()
            .expect("native code buffer should exist");
        let code = unsafe { core::slice::from_raw_parts(executable.as_ptr(), entry.text_len) };
        let words = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<vec::Vec<_>>();

        let fp0 = crate::vm::arch::arm64::abi::fp_machine_reg(0).expect("fp0");
        let fp1 = crate::vm::arch::arm64::abi::fp_machine_reg(1).expect("fp1");
        let fp2 = crate::vm::arch::arm64::abi::fp_machine_reg(2).expect("fp2");

        assert!(words.contains(&enc::fcvt_s_from_d(fp1, fp0)));
        assert!(words.contains(&enc::fcvt_d_from_s(fp2, fp1)));
        assert!(!words.contains(&enc::fmov_gp_from_d(crate::vm::arch::arm64::abi::SCRATCH0, fp0)));
        assert!(!words.contains(&enc::fmov_gp_from_s(crate::vm::arch::arm64::abi::SCRATCH0, fp1)));
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_simple_add_function_in_arm64_code() {
        let function = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 7,
                reg_count: 7,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: MachineReg(4),
                                src: MachineValue::Imm64(40),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: MachineReg(5),
                                src: MachineValue::Imm64(2),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::IntBinary {
                                width: MachineIntWidth::I32,
                                op: MachineIntBinaryOp::Add,
                                dst: MachineReg(6),
                                lhs: MachineValue::Reg(MachineReg(4)),
                                rhs: MachineValue::Reg(MachineReg(5)),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: MachineAddr {
                                    base: MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: MachineMemWidth::U64,
                                src: MachineValue::Reg(MachineReg(6)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(stack[0], 42);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_store_imm64_directly() {
        let function = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 7,
                reg_count: 7,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::Store {
                            ty: MachineStorageType::GpWord,
                            addr: MachineAddr {
                                base: MACHINE_FP_REG,
                                offset: 0,
                            },
                            width: MachineMemWidth::U64,
                            src: MachineValue::Imm64(42),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 4,
                    call_scratch: Some(MachineFrameRegion {
                        base_slot: 1,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: Some(MachineFrameRegion {
                        base_slot: 0,
                        slots: 1,
                    }),
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(
            stack[0], 42,
            "Store with Imm64(42) should write 42 to fp[0]"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_function_with_unsupported_neighbor_stub() {
        let supported = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 4,
                reg_count: 4,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let unsupported = MachineFunction {
            id: MachineFuncId(1),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 5,
                reg_count: 5,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![MachineInst {
                        kind: MachineInstKind::FloatUnary {
                            width: MachineFloatWidth::F32,
                            op: MachineFloatUnaryOp::Abs,
                            dst: MachineReg(4),
                            src: MachineValue::Imm64(0),
                        },
                    }],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![supported, unsupported],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_multiblock_empty_root_function() {
        let function = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 4,
                reg_count: 4,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Jump(
                            MachineEdge {
                                target: MachineBlockId(1),
                                args: vec![],
                            },
                        ),
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Jump(
                            MachineEdge {
                                target: MachineBlockId(2),
                                args: vec![],
                            },
                        ),
                    },
                    MachineBlock {
                        id: MachineBlockId(2),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![function],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![MachineFunctionRuntime {
                    id: MachineFuncId(0),
                    frame_prefix_slots: 0,
                    total_frame_slots: 3,
                    call_scratch: Some(MachineFrameRegion {
                        base_slot: 0,
                        slots: 3,
                    }),
                    helper_scratch: None,
                    return_results: None,
                }],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_nonfirst_function_entry() {
        let dummy = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 4,
                reg_count: 4,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let target = MachineFunction {
            id: MachineFuncId(1),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 7,
                reg_count: 7,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![
                        MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: MachineReg(4),
                                src: MachineValue::Imm64(9),
                            },
                        },
                        MachineInst {
                            kind: MachineInstKind::Store {
                                ty: MachineStorageType::GpWord,
                                addr: MachineAddr {
                                    base: MACHINE_FP_REG,
                                    offset: 0,
                                },
                                width: MachineMemWidth::U64,
                                src: MachineValue::Reg(MachineReg(4)),
                            },
                        },
                    ],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![dummy, target],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 4,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 1,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 1,
                        }),
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[1].clone().expect("entry");

        let mut stack = [0u64; 4];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[1] = entry.root_return as u64;
        stack[2] = stack.as_mut_ptr() as u64;
        stack[3] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert_eq!(stack[0], 9);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_with_jump_table_neighbor() {
        let empty = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 4,
                reg_count: 4,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let jumpy = MachineFunction {
            id: MachineFuncId(1),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 5,
                reg_count: 5,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::JumpTable {
                            index: MachineValue::Imm64(0),
                            entries: vec![
                                MachineEdge {
                                    target: MachineBlockId(1),
                                    args: vec![],
                                },
                                MachineEdge {
                                    target: MachineBlockId(2),
                                    args: vec![],
                                },
                            ],
                        },
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                    MachineBlock {
                        id: MachineBlockId(2),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![empty, jumpy],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn executes_empty_root_with_indirect_call_neighbor() {
        let empty = MachineFunction {
            id: MachineFuncId(0),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 4,
                reg_count: 4,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![MachineBlock {
                    id: MachineBlockId(0),
                    params: vec![],
                    ops: vec![],
                    terminator: MachineTerminator::Return,
                }],
            },
        };
        let indirect = MachineFunction {
            id: MachineFuncId(1),
            program: MachineProgram {
                entry: MachineBlockId(0),
                first_fp_reg: 5,
                reg_count: 5,

                fp_transient_count: 0,

                fp_reg_init_widths: vec![],

                blocks: vec![
                    MachineBlock {
                        id: MachineBlockId(0),
                        params: vec![],
                        ops: vec![MachineInst {
                            kind: MachineInstKind::Move {
                                ty: MachineStorageType::GpWord,
                                dst: MachineReg(4),
                                src: MachineValue::Reg(
                                    MACHINE_FP_REG,
                                ),
                            },
                        }],
                        terminator: MachineTerminator::CallIndirect {
                            callee_target: MachineValue::Imm64(0),
                            callee_frame_base: MachineReg(4),
                            arg_slots: 0,
                            caller_result_base: 0,
                            continuation: MachineBlockId(1),
                        },
                    },
                    MachineBlock {
                        id: MachineBlockId(1),
                        params: vec![],
                        ops: vec![],
                        terminator: MachineTerminator::Return,
                    },
                ],
            },
        };
        let compiled = CompiledNativeModule::new(
            crate::vm::arch::NativeBackend::Arm64,
            BackendConfig::new(3, 4, 2, 2),
            MachineModule {
                functions: vec![empty, indirect],
                consts: vec![],
                externs: vec![],
            },
            MachineRuntimeContract {
                call_link: MachineCallLinkLayout {
                    slot_count: 3,
                    continuation_offset: 0,
                    caller_frame_offset: 8,
                    caller_result_base_offset: 16,
                },
                functions: vec![
                    MachineFunctionRuntime {
                        id: MachineFuncId(0),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                    MachineFunctionRuntime {
                        id: MachineFuncId(1),
                        frame_prefix_slots: 0,
                        total_frame_slots: 3,
                        call_scratch: Some(MachineFrameRegion {
                            base_slot: 0,
                            slots: 3,
                        }),
                        helper_scratch: None,
                        return_results: None,
                    },
                ],
            },
        )
        .expect("compiled module");

        let module = ModuleInst::new(
            String::from("m"),
            TypeContext::new(vec![Rc::new(FunctionType::new(vec![], vec![]))]),
        );
        let entries = compile_module(&module, &compiled).expect("arm64 compile should succeed");
        let entry = entries[0].clone().expect("entry");

        let mut stack = [0u64; 3];
        let mut store = Box::new(Store::new(module));
        let stack_end = unsafe { stack.as_mut_ptr().add(stack.len()) };
        let mut ctx = NativeContext::new(store.as_mut() as *mut Store, stack_end);
        stack[0] = entry.root_return as u64;
        stack[1] = stack.as_mut_ptr() as u64;
        stack[2] = 0;
        let status = unsafe { (entry.entry)(&mut ctx, stack.as_mut_ptr()) };
        assert_eq!(status, 0);
        assert!(ctx.error.is_none());
    }
}
