;; Effective addresses of loads and stores: base + index (the fused form
;; the interpreter folds into one dispatch) wraps at 32 bits before the
;; static offset is added, and the static offset never wraps. Covers
;; in-range accesses, accesses that wrap to the bottom of memory, and
;; out-of-bounds accesses on either side of the last valid address.
(module
  (memory 1)
  (data (i32.const 0) "\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f\10\11\12\13\14")
  (func (export "ld") (param $a i32) (param $b i32) (result i32)
    (i32.load offset=2 (i32.add (local.get $a) (local.get $b))))
  (func (export "ld8") (param $a i32) (param $b i32) (result i32)
    (i32.load8_u offset=1 (i32.add (local.get $a) (local.get $b))))
  (func (export "st") (param $a i32) (param $b i32) (param $v i32) (result i32)
    (i32.store offset=3 (i32.add (local.get $a) (local.get $b)) (local.get $v))
    (i32.load offset=3 (i32.add (local.get $a) (local.get $b))))
  (func (export "ldc") (param $a i32) (result i32)
    (i32.load offset=4 (local.get $a)))
  (func (export "ldk") (result i32)
    (i32.load offset=1 (i32.const 6)))
)
(assert_return (invoke "ld" (i32.const 1) (i32.const 2)) (i32.const 0x09080706))
(assert_return (invoke "ld" (i32.const 0xfffffff0) (i32.const 0x11)) (i32.const 0x07060504))
(assert_trap (invoke "ld" (i32.const 0xffff) (i32.const 0)) "out of bounds memory access")
(assert_trap (invoke "ld" (i32.const 0x10000) (i32.const 0xfffffffe)) "out of bounds memory access")
(assert_return (invoke "ld" (i32.const 0xfffb) (i32.const 0xffffffff)) (i32.const 0))
(assert_trap (invoke "ld" (i32.const 0xfffd) (i32.const 0xffffffff)) "out of bounds memory access")
(assert_return (invoke "ld8" (i32.const 0xffffffff) (i32.const 1)) (i32.const 2))
(assert_return (invoke "st" (i32.const 0x20) (i32.const 0xfffffff0) (i32.const 0x55aa)) (i32.const 0x55aa))
(assert_return (invoke "ld" (i32.const 0x20) (i32.const 0xfffffff1)) (i32.const 0x000055aa))
(assert_trap (invoke "st" (i32.const 0xfffc) (i32.const 1) (i32.const 7)) "out of bounds memory access")
(assert_return (invoke "ldc" (i32.const 3)) (i32.const 0x0b0a0908))
(assert_trap (invoke "ldc" (i32.const 0xfffffffc)) "out of bounds memory access")
(assert_trap (invoke "ldc" (i32.const 0xfff9)) "out of bounds memory access")
(assert_return (invoke "ldk") (i32.const 0x0b0a0908))
