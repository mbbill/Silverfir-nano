;; x86_64 folds a load whose only reader is the block's branch compare into
;; the compare (`cmp reg, [mem]`, `cmp [mem], reg`, `cmp [mem], imm`). The
;; flags must match the register compare for every predicate and width, an
;; immediate that does not fit a sign-extended imm32 must still compare
;; correctly, a loaded value read again later must stay in its register, and
;; a folded Wasm memory access must trap at exactly the same bound.

(module
  (memory 1)

  (func (export "setup")
    (i32.store (i32.const 0) (i32.const 10))
    (i32.store (i32.const 4) (i32.const -5))
    (i64.store (i32.const 8) (i64.const 0x1_0000_0000))
    (i64.store (i32.const 16) (i64.const -3))
    (i32.store (i32.const 24) (i32.const -9))
    (i32.store (i32.const 28) (i32.const 3))
    (i32.store (i32.const 32) (i32.const 500))
    (i32.store (i32.const 36) (i32.const 7))
    (i32.store (i32.const 40) (i32.const 5))
    (i32.store (i32.const 44) (i32.const 500)))

  ;; Each returns 1 when the branch is taken.
  (func (export "lt_u_rhs") (param $x i32) (param $p i32) (result i32)
    (block $t
      (br_if $t (i32.lt_u (local.get $x) (i32.load (local.get $p))))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "lt_s_lhs") (param $x i32) (param $p i32) (result i32)
    (block $t
      (br_if $t (i32.lt_s (i32.load (local.get $p)) (local.get $x)))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "ge_s_imm") (param $p i32) (result i32)
    (block $t
      (br_if $t (i32.ge_s (i32.load offset=4 (local.get $p)) (i32.const -5)))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "gt_u_imm") (param $p i32) (result i32)
    (block $t
      (br_if $t (i32.gt_u (i32.load offset=4 (local.get $p)) (i32.const 7)))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "i64_eq_big") (param $p i32) (result i32)
    (block $t
      (br_if $t (i64.eq (i64.load offset=8 (local.get $p)) (i64.const 0x1_0000_0000)))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "i64_lt_s_imm") (param $p i32) (result i32)
    (block $t
      (br_if $t (i64.lt_s (i64.load offset=16 (local.get $p)) (i64.const -2)))
      (return (i32.const 0)))
    (i32.const 1))
  (func (export "i64_gt_u_reg") (param $x i64) (param $p i32) (result i32)
    (block $t
      (br_if $t (i64.gt_u (local.get $x) (i64.load offset=8 (local.get $p))))
      (return (i32.const 0)))
    (i32.const 1))

  ;; The loaded value is returned after the branch.
  (func (export "reuse") (param $p i32) (result i32)
    (local $v i32)
    (block $t
      (br_if $t (i32.lt_u (local.tee $v (i32.load (local.get $p))) (i32.const 100)))
      (return (i32.const -1)))
    (local.get $v))

  ;; A loop bound read from memory on every iteration.
  (func (export "count") (result i32)
    (local $i i32) (local $c i32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get $i) (i32.load (i32.const 36))))
        (local.set $c (i32.add (local.get $c) (i32.const 2)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
    (local.get $c))

  ;; Loops whose exit compare has the loaded value on the left of an
  ;; immediate, in the imm8 and the imm32 encoding.
  (func (export "drain") (param $p i32) (result i32)
    (local $n i32)
    (block $done
      (loop $l
        (br_if $done (i32.le_s (i32.load (local.get $p)) (i32.const 0)))
        (i32.store (local.get $p) (i32.sub (i32.load (local.get $p)) (i32.const 1)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $l)))
    (local.get $n))
  (func (export "grow") (param $p i32) (result i32)
    (local $n i32)
    (block $done
      (loop $l
        (br_if $done (i32.gt_u (i32.load (local.get $p)) (i32.const 1000)))
        (i32.store (local.get $p) (i32.add (i32.load (local.get $p)) (i32.const 100)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $l)))
    (local.get $n))

  ;; Loops whose exit compare has the loaded value on the left of a
  ;; register, at a constant address and through an index.
  (func (export "count_le") (result i32)
    (local $i i32) (local $c i32)
    (block $done
      (loop $l
        (br_if $done (i32.le_u (i32.load (i32.const 36)) (local.get $i)))
        (local.set $c (i32.add (local.get $c) (i32.const 2)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
    (local.get $c))
  (func (export "count_lt_s") (param $p i32) (result i32)
    (local $i i32) (local $c i32)
    (block $done
      (loop $l
        (br_if $done (i32.lt_s (i32.load (local.get $p)) (local.get $i)))
        (local.set $c (i32.add (local.get $c) (i32.const 3)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $l)))
    (local.get $c))
)

(invoke "setup")
(assert_return (invoke "lt_u_rhs" (i32.const 5) (i32.const 0)) (i32.const 1))
(assert_return (invoke "lt_u_rhs" (i32.const 10) (i32.const 0)) (i32.const 0))
(assert_return (invoke "lt_u_rhs" (i32.const -1) (i32.const 0)) (i32.const 0))
(assert_return (invoke "lt_s_lhs" (i32.const 0) (i32.const 0)) (i32.const 0))
(assert_return (invoke "lt_s_lhs" (i32.const 11) (i32.const 0)) (i32.const 1))
(assert_return (invoke "lt_s_lhs" (i32.const 0) (i32.const 4)) (i32.const 1))
(assert_return (invoke "ge_s_imm" (i32.const 0)) (i32.const 1))
(assert_return (invoke "ge_s_imm" (i32.const 20)) (i32.const 0))
(assert_return (invoke "gt_u_imm" (i32.const 0)) (i32.const 1))
(assert_return (invoke "gt_u_imm" (i32.const 20)) (i32.const 1))
(assert_return (invoke "gt_u_imm" (i32.const 24)) (i32.const 0))
(assert_return (invoke "i64_eq_big" (i32.const 0)) (i32.const 1))
(assert_return (invoke "i64_eq_big" (i32.const 8)) (i32.const 0))
(assert_return (invoke "i64_lt_s_imm" (i32.const 0)) (i32.const 1))
(assert_return (invoke "i64_lt_s_imm" (i32.const 8)) (i32.const 0))
(assert_return (invoke "i64_gt_u_reg" (i64.const 0x1_0000_0001) (i32.const 0)) (i32.const 1))
(assert_return (invoke "i64_gt_u_reg" (i64.const 0x1_0000_0000) (i32.const 0)) (i32.const 0))
(assert_return (invoke "i64_gt_u_reg" (i64.const -1) (i32.const 0)) (i32.const 1))
(assert_return (invoke "reuse" (i32.const 0)) (i32.const 10))
(assert_return (invoke "reuse" (i32.const 32)) (i32.const -1))
(assert_return (invoke "count") (i32.const 14))
(assert_return (invoke "count_le") (i32.const 14))
(assert_return (invoke "drain" (i32.const 40)) (i32.const 5))
(assert_return (invoke "grow" (i32.const 44)) (i32.const 6))
(assert_return (invoke "count_lt_s" (i32.const 24)) (i32.const 0))
(assert_return (invoke "count_lt_s" (i32.const 28)) (i32.const 12))

;; A folded access traps at exactly the load's bound.
(assert_return (invoke "lt_u_rhs" (i32.const 0) (i32.const 65532)) (i32.const 0))
(assert_trap (invoke "lt_u_rhs" (i32.const 0) (i32.const 65533)) "out of bounds memory access")
(assert_trap (invoke "lt_u_rhs" (i32.const 0) (i32.const -4)) "out of bounds memory access")
(assert_return (invoke "i64_gt_u_reg" (i64.const 0) (i32.const 65520)) (i32.const 0))
(assert_trap (invoke "i64_gt_u_reg" (i64.const 0) (i32.const 65521)) "out of bounds memory access")
