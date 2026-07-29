;; RuntimeWorld migration baseline: a funcref stored in a GC container by one
;; instance must retain its identity when another instance reads and calls it.
(module
  (type $ft (func (result i32)))
  (type $arr (array (mut funcref)))
  (global $refs (ref $arr)
    (array.new_default $arr (i32.const 1)))
  (func $padding (type $ft) (result i32)
    i32.const -1)
  ;; Local index 1, matching the consumer's decoy below. A missed
  ;; cross-instance conversion therefore calls a signature-compatible wrong
  ;; function instead of trapping.
  (func $target (type $ft) (result i32)
    i32.const 324508639)
  (elem declare func $target)
  (func (export "get_refs") (result (ref $arr))
    global.get $refs)
  (func (export "write")
    (array.set $arr
      (global.get $refs)
      (i32.const 0)
      (ref.func $target))))
(register "producer")
(assert_return (invoke "write"))

(module
  (type $ft (func (result i32)))
  (type $arr (array (mut funcref)))
  (func $get_refs (import "producer" "get_refs")
    (result (ref $arr)))
  (func $decoy (type $ft) (result i32)
    i32.const 610839776)
  (func (export "read_and_call") (result i32)
    (call_ref $ft
      (ref.cast (ref $ft)
        (array.get $arr
          (call $get_refs)
          (i32.const 0))))))
(assert_return
  (invoke "read_and_call")
  (i32.const 324508639))
