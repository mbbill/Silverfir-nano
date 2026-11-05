;; A value read from a local and still on the operand stack is a snapshot of
;; the local at the time of the read. The JIT may read the local's register in
;; place of a snapshot only while the local is unchanged, so every snapshot
;; below is used after the local has been written again: later in the same
;; block, carried on a branch edge, inside loops, and in both register banks.
;; The branch-edge case (`edge`) used to return the new value.

(module
  ;; Snapshot used later in the block, after the local is rewritten.
  (func (export "set_after") (param $x i32) (result i32)
    (local.get $x)
    (local.set $x (i32.mul (local.get $x) (i32.const 3)))
    (i32.sub (local.get $x)))

  ;; local.tee rewrites the local while the snapshot is below it.
  (func (export "tee_after") (param $x i64) (result i64)
    (local.get $x)
    (i64.add (local.tee $x (i64.shl (local.get $x) (i64.const 4)))
             (local.get $x))
    (i64.sub))

  ;; The snapshot leaves the block on a branch edge.
  (func (export "edge") (param $x i32) (param $c i32) (result i32)
    (block $b (result i32)
      (local.get $x)
      (local.set $x (i32.add (local.get $x) (i32.const 100)))
      (br_if $b (local.get $c))
      (drop)
      (local.get $x))
    (i32.add (local.get $x)))

  ;; The post-increment idiom: the old counter is multiplied by the new
  ;; one after the counter has moved on, on every iteration.
  (func (export "post_inc") (param $n i32) (result i32)
    (local $i i32) (local $s i32)
    (block $done
      (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))
        (local.get $i)
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (i32.mul (local.get $i))
        (local.set $s (i32.add (local.get $s)))
        (br $l)))
    (local.get $s))

  ;; The same in the FP bank.
  (func (export "fp_prev") (param $n i32) (result f64)
    (local $x f64) (local $acc f64)
    (local.set $x (f64.const 1))
    (block $done
      (loop $l
        (br_if $done (i32.eqz (local.get $n)))
        (local.get $x)
        (local.set $x (f64.mul (local.get $x) (f64.const 2)))
        (f64.sub (local.get $x))
        (local.set $acc (f64.add (local.get $acc)))
        (local.set $n (i32.sub (local.get $n) (i32.const 1)))
        (br $l)))
    (local.get $acc))
)

(assert_return (invoke "set_after" (i32.const 5)) (i32.const -10))
(assert_return (invoke "set_after" (i32.const 0)) (i32.const 0))
(assert_return (invoke "tee_after" (i64.const 1)) (i64.const -31))
(assert_return (invoke "tee_after" (i64.const 3)) (i64.const -93))
(assert_return (invoke "edge" (i32.const 1) (i32.const 1)) (i32.const 102))
(assert_return (invoke "edge" (i32.const 1) (i32.const 0)) (i32.const 202))
(assert_return (invoke "post_inc" (i32.const 0)) (i32.const 0))
(assert_return (invoke "post_inc" (i32.const 4)) (i32.const 20))
(assert_return (invoke "fp_prev" (i32.const 0)) (f64.const 0))
(assert_return (invoke "fp_prev" (i32.const 4)) (f64.const -15))
