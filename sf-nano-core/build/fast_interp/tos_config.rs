//! TOS (Top-of-Stack) register configuration.
//! Single source of truth for TOS register count across all code generation.

/// Number of TOS registers in the fast interpreter.
/// Changing this value and rebuilding will update all generated code.
pub const TOS_REGISTER_COUNT: usize = 4;

/// Generate register names: ["t0", "t1", ..., "t{N-1}"]
pub fn tos_register_names() -> Vec<String> {
    (0..TOS_REGISTER_COUNT).map(|i| format!("t{}", i)).collect()
}

/// Generate variant names: ["D1", "D2", ..., "DN"]
pub fn tos_variant_names() -> Vec<String> {
    (1..=TOS_REGISTER_COUNT)
        .map(|i| format!("D{}", i))
        .collect()
}

/// Generate C-style pointer parameters for all TOS registers: "uint64_t*, uint64_t*, ..."
pub fn tos_all_ptr_params() -> String {
    (0..TOS_REGISTER_COUNT)
        .map(|_| "uint64_t*")
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate C-style address arguments for all TOS registers: "&t0, &t1, &t2, &t3"
pub fn tos_all_register_args() -> String {
    tos_register_names()
        .iter()
        .map(|r| format!("&{}", r))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate register argument string for pop/push pattern at variant index.
///
/// Formula: For position P at depth D, register index = (D - P) % TOS_REGISTER_COUNT
/// variant_idx: 0 = D1 (depth%N=1), 1 = D2 (depth%N=2), etc.
pub fn get_register_args_for_pattern(pop: u8, push: u8, variant_idx: usize) -> String {
    let n = TOS_REGISTER_COUNT;
    // D1 means depth%N=1, D2 means depth%N=2, etc.
    // variant_idx 0 (D1) -> depth_mod = 1, variant_idx 3 (D4) -> depth_mod = 0
    let depth_mod = (variant_idx + 1) % n;

    // Helper to get register name for position P at depth D
    let reg = |pos: usize| -> String {
        let idx = (depth_mod + n - pos) % n;
        format!("&t{}", idx)
    };

    match (pop, push) {
        (2, 1) => {
            // pop2_push1: lhs (pos2), rhs (pos1), dst (pos2)
            format!("{}, {}, {}", reg(2), reg(1), reg(2))
        }
        (1, 1) => {
            // pop1_push1: src/dst same register
            format!("{}, {}", reg(1), reg(1))
        }
        (0, 1) => {
            // pop0_push1: dst at new top (pos1 after push)
            reg(1)
        }
        (2, 0) => {
            // pop2_push0: addr (pos2), val (pos1)
            format!("{}, {}", reg(2), reg(1))
        }
        (1, 0) => {
            // pop1_push0: src
            reg(1)
        }
        (3, 1) => {
            // pop3_push1: val1 (pos3), val2 (pos2), cond (pos1), dst (pos3)
            format!("{}, {}, {}, {}", reg(3), reg(2), reg(1), reg(3))
        }
        (3, 0) => {
            // pop3_push0: a (pos3), b (pos2), c (pos1)
            format!("{}, {}, {}", reg(3), reg(2), reg(1))
        }
        (4, 0) => {
            // pop4_push0: a (pos4), b (pos3), c (pos2), d (pos1)
            format!("{}, {}, {}, {}", reg(4), reg(3), reg(2), reg(1))
        }
        (0, 2) => {
            // pop0_push2: dst0 (pos2), dst1 (pos1)
            format!("{}, {}", reg(2), reg(1))
        }
        (0, 3) => {
            // pop0_push3: dst0 (pos3), dst1 (pos2), dst2 (pos1)
            format!("{}, {}, {}", reg(3), reg(2), reg(1))
        }
        (0, 4) => {
            // pop0_push4: dst0 (pos4), dst1 (pos3), dst2 (pos2), dst3 (pos1)
            format!("{}, {}, {}, {}", reg(4), reg(3), reg(2), reg(1))
        }
        (1, 2) => {
            // pop1_push2: src (original pos1, now pos2), dst0 (pos2), dst1 (pos1)
            format!("{}, {}, {}", reg(2), reg(2), reg(1))
        }
        (2, 2) => {
            // pop2_push2: lhs (pos2), rhs (pos1), dst0 (pos2), dst1 (pos1)
            format!("{}, {}, {}, {}", reg(2), reg(1), reg(2), reg(1))
        }
        _ => panic!("Unsupported pop/push pattern: ({}, {})", pop, push),
    }
}
