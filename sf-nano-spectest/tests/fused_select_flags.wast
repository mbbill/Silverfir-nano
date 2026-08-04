;; Regression: a fused compare+select must not clobber EFLAGS between the
;; compare and the CMOV. Materializing the select's zero immediate with the
;; xor zero idiom did exactly that (ZF forced to 1), so the CMOV always took
;; the true value regardless of the compare.
(module
  (func (export "sel") (param i32) (result i32)
    (select (i32.const 5) (i32.const 0) (i32.eq (local.get 0) (i32.const 5))))
  (func (export "sel_rev") (param i32) (result i32)
    (select (i32.const 0) (i32.const 7) (i32.lt_u (local.get 0) (i32.const 4))))
)
(assert_return (invoke "sel" (i32.const 5)) (i32.const 5))
(assert_return (invoke "sel" (i32.const 3)) (i32.const 0))
(assert_return (invoke "sel_rev" (i32.const 2)) (i32.const 0))
(assert_return (invoke "sel_rev" (i32.const 9)) (i32.const 7))
