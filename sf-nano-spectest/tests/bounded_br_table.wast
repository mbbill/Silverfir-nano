;; A `br_table` behind an explicit unsigned range check on the same index,
;; the shape LLVM emits for a dense `switch`. When the check bounds the
;; index at or below the default's position the table is indexed without
;; clamping, so every index the guard lets through, including the default's
;; own position, must still select its own case, and an index one past the
;; default must still be clamped when the guard admits it.

(module
  ;; Four cases and the default. `$bound` is the largest index the guard
  ;; lets through to the table.
  (func $dispatch (param $x i32) (param $bound i32) (result i32) (local $i i32)
    (block $out (result i32)
      (block $default
        (block $c3
          (block $c2
            (block $c1
              (block $c0
                (local.set $i (i32.add (local.get $x) (i32.const -2)))
                (br_if $default (i32.gt_u (local.get $i) (local.get $bound)))
                (br_table $c0 $c1 $c2 $c3 $default (local.get $i)))
              (br $out (i32.const 100)))
            (br $out (i32.const 101)))
          (br $out (i32.const 102)))
        (br $out (i32.const 103)))
      (i32.const 199)))

  ;; The guard stops below the default.
  (func (export "below") (param $x i32) (result i32)
    (local $i i32)
    (block $out (result i32)
      (block $default
        (block $c3
          (block $c2
            (block $c1
              (block $c0
                (local.set $i (i32.add (local.get $x) (i32.const -2)))
                (br_if $default (i32.gt_u (local.get $i) (i32.const 3)))
                (br_table $c0 $c1 $c2 $c3 $default (local.get $i)))
              (br $out (i32.const 100)))
            (br $out (i32.const 101)))
          (br $out (i32.const 102)))
        (br $out (i32.const 103)))
      (i32.const 199)))

  ;; The guard admits the default's own position.
  (func (export "at_default") (param $x i32) (result i32)
    (local $i i32)
    (block $out (result i32)
      (block $default
        (block $c3
          (block $c2
            (block $c1
              (block $c0
                (local.set $i (i32.add (local.get $x) (i32.const -2)))
                (br_if $default (i32.gt_u (local.get $i) (i32.const 4)))
                (br_table $c0 $c1 $c2 $c3 $default (local.get $i)))
              (br $out (i32.const 100)))
            (br $out (i32.const 101)))
          (br $out (i32.const 102)))
        (br $out (i32.const 103)))
      (i32.const 199)))

  ;; The guard admits one index past the default, which must be clamped.
  (func (export "past_default") (param $x i32) (result i32)
    (local $i i32)
    (block $out (result i32)
      (block $default
        (block $c3
          (block $c2
            (block $c1
              (block $c0
                (local.set $i (i32.add (local.get $x) (i32.const -2)))
                (br_if $default (i32.gt_u (local.get $i) (i32.const 5)))
                (br_table $c0 $c1 $c2 $c3 $default (local.get $i)))
              (br $out (i32.const 100)))
            (br $out (i32.const 101)))
          (br $out (i32.const 102)))
        (br $out (i32.const 103)))
      (i32.const 199)))

  ;; A `>=` guard that admits the default's own position.
  (func (export "lt_guard") (param $x i32) (result i32)
    (local $i i32)
    (block $out (result i32)
      (block $default
        (block $c3
          (block $c2
            (block $c1
              (block $c0
                (local.set $i (i32.add (local.get $x) (i32.const -2)))
                (br_if $default (i32.ge_u (local.get $i) (i32.const 5)))
                (br_table $c0 $c1 $c2 $c3 $default (local.get $i)))
              (br $out (i32.const 100)))
            (br $out (i32.const 101)))
          (br $out (i32.const 102)))
        (br $out (i32.const 103)))
      (i32.const 199)))

  ;; A guard against a run-time bound proves nothing.
  (func (export "runtime_bound") (param $x i32) (param $bound i32) (result i32)
    (call $dispatch (local.get $x) (local.get $bound)))
)

(assert_return (invoke "below" (i32.const 2)) (i32.const 100))
(assert_return (invoke "below" (i32.const 3)) (i32.const 101))
(assert_return (invoke "below" (i32.const 4)) (i32.const 102))
(assert_return (invoke "below" (i32.const 5)) (i32.const 103))
(assert_return (invoke "below" (i32.const 6)) (i32.const 199))
(assert_return (invoke "below" (i32.const 1)) (i32.const 199))
(assert_return (invoke "below" (i32.const 0x7fffffff)) (i32.const 199))

(assert_return (invoke "at_default" (i32.const 2)) (i32.const 100))
(assert_return (invoke "at_default" (i32.const 5)) (i32.const 103))
(assert_return (invoke "at_default" (i32.const 6)) (i32.const 199))
(assert_return (invoke "at_default" (i32.const 7)) (i32.const 199))
(assert_return (invoke "at_default" (i32.const 1)) (i32.const 199))

(assert_return (invoke "past_default" (i32.const 2)) (i32.const 100))
(assert_return (invoke "past_default" (i32.const 5)) (i32.const 103))
(assert_return (invoke "past_default" (i32.const 6)) (i32.const 199))
(assert_return (invoke "past_default" (i32.const 7)) (i32.const 199))
(assert_return (invoke "past_default" (i32.const 8)) (i32.const 199))

(assert_return (invoke "lt_guard" (i32.const 4)) (i32.const 102))
(assert_return (invoke "lt_guard" (i32.const 6)) (i32.const 199))
(assert_return (invoke "lt_guard" (i32.const 7)) (i32.const 199))
(assert_return (invoke "lt_guard" (i32.const 8)) (i32.const 199))

(assert_return (invoke "runtime_bound" (i32.const 3) (i32.const 10)) (i32.const 101))
(assert_return (invoke "runtime_bound" (i32.const 8) (i32.const 10)) (i32.const 199))
(assert_return (invoke "runtime_bound" (i32.const 100) (i32.const 1000)) (i32.const 199))
(assert_return (invoke "runtime_bound" (i32.const 4) (i32.const 1)) (i32.const 199))
