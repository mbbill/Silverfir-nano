;; `memory.copy` / `memory.fill` on a 32-bit memory 0 become a direct libc
;; call after inline bounds checks. Values held in caller-saved registers
;; must survive the call, the offsets must be zero-extended before the range
;; check, both ranges of a copy must be proven before any byte moves, and the
;; fill value is truncated to a byte.

(module
  (memory 1)

  ;; Integer and float values computed before the bulk operations and only
  ;; consumed after them.
  (func (export "live") (param $a i32) (param $b i64) (param $x f64) (param $y f32)
                        (result f64)
    (local $c i32) (local $d i64) (local $z f64) (local $w f64)
    (local $e i32) (local $f i32) (local $g i32) (local $h i32)
    (local.set $c (i32.mul (local.get $a) (i32.const 3)))
    (local.set $d (i64.add (local.get $b) (i64.const 7)))
    (local.set $z (f64.mul (local.get $x) (f64.const 1.5)))
    (local.set $w (f64.sub (local.get $x) (f64.const 0.25)))
    (local.set $e (i32.add (local.get $a) (i32.const 11)))
    (local.set $f (i32.xor (local.get $a) (i32.const 0x55)))
    (local.set $g (i32.sub (local.get $a) (i32.const 2)))
    (local.set $h (i32.or (local.get $a) (i32.const 0x100)))
    (memory.fill (i32.const 16) (local.get $f) (i32.const 64))
    (memory.copy (i32.const 200) (i32.const 16) (i32.const 64))
    (f64.add
      (f64.add
        (f64.add (local.get $z) (local.get $w))
        (f64.promote_f32 (local.get $y)))
      (f64.convert_i64_s
        (i64.add
          (local.get $d)
          (i64.extend_i32_u
            (i32.add
              (i32.add (local.get $c) (local.get $e))
              (i32.add
                (i32.add (local.get $g) (local.get $h))
                (i32.load8_u (i32.const 263)))))))))

  ;; A loop whose counter and accumulator are live across the call on the
  ;; backedge.
  (func (export "loop") (param $n i32) (result i32)
    (local $i i32) (local $acc i32)
    (loop $l
      (memory.fill (local.get $i) (local.get $i) (i32.const 1))
      (local.set $acc (i32.add (local.get $acc) (i32.load8_u (local.get $i))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $l (i32.lt_u (local.get $i) (local.get $n))))
    (local.get $acc))

  (func (export "fill") (param $d i32) (param $v i32) (param $n i32)
    (memory.fill (local.get $d) (local.get $v) (local.get $n)))
  (func (export "copy") (param $d i32) (param $s i32) (param $n i32)
    (memory.copy (local.get $d) (local.get $s) (local.get $n)))
  (func (export "copy_to_zero") (param $s i32) (param $n i32)
    (memory.copy (i32.const 0) (local.get $s) (local.get $n)))
  (func (export "load8") (param $p i32) (result i32)
    (i32.load8_u (local.get $p)))
)

(assert_return (invoke "live" (i32.const 10) (i64.const 100) (f64.const 2.0) (f32.const 0.5))
  (f64.const 532.25))
(assert_return (invoke "loop" (i32.const 10)) (i32.const 45))
(assert_return (invoke "loop" (i32.const 300)) (i32.const 33586))

;; The fill value is truncated to its low byte.
(invoke "fill" (i32.const 1000) (i32.const 0x1ab) (i32.const 4))
(assert_return (invoke "load8" (i32.const 1003)) (i32.const 0xab))

;; Ranges ending exactly at the memory end are in bounds, including empty
;; ones starting there.
(invoke "fill" (i32.const 65535) (i32.const 7) (i32.const 1))
(assert_return (invoke "load8" (i32.const 65535)) (i32.const 7))
(invoke "fill" (i32.const 65536) (i32.const 9) (i32.const 0))
(invoke "copy" (i32.const 65536) (i32.const 65536) (i32.const 0))
(invoke "copy" (i32.const 65535) (i32.const 1003) (i32.const 1))
(assert_return (invoke "load8" (i32.const 65535)) (i32.const 0xab))

;; One byte past the end traps for either side, before anything is written.
(invoke "fill" (i32.const 0) (i32.const 0x11) (i32.const 8))
(assert_trap (invoke "fill" (i32.const 65535) (i32.const 0x22) (i32.const 2))
  "out of bounds memory access")
(assert_return (invoke "load8" (i32.const 65535)) (i32.const 0xab))
(assert_trap (invoke "copy" (i32.const 0) (i32.const 65530) (i32.const 7))
  "out of bounds memory access")
(assert_return (invoke "load8" (i32.const 0)) (i32.const 0x11))
(assert_trap (invoke "copy" (i32.const 65530) (i32.const 0) (i32.const 7))
  "out of bounds memory access")
(assert_return (invoke "load8" (i32.const 65530)) (i32.const 0))
(assert_trap (invoke "copy_to_zero" (i32.const 100) (i32.const 65437))
  "out of bounds memory access")
(assert_return (invoke "load8" (i32.const 0)) (i32.const 0x11))

;; Offsets and lengths are unsigned: a sign-extended 0xffffffff would wrap
;; the range check around to a small sum.
(assert_trap (invoke "fill" (i32.const -1) (i32.const 0x33) (i32.const 2))
  "out of bounds memory access")
(assert_trap (invoke "copy" (i32.const -1) (i32.const 0) (i32.const 2))
  "out of bounds memory access")
(assert_trap (invoke "copy" (i32.const 0) (i32.const -1) (i32.const 2))
  "out of bounds memory access")
(assert_trap (invoke "fill" (i32.const 0) (i32.const 0x44) (i32.const -1))
  "out of bounds memory access")
(assert_return (invoke "load8" (i32.const 0)) (i32.const 0x11))

;; Overlapping copies move like memmove in both directions.
(invoke "fill" (i32.const 2000) (i32.const 1) (i32.const 1))
(invoke "fill" (i32.const 2001) (i32.const 2) (i32.const 1))
(invoke "fill" (i32.const 2002) (i32.const 3) (i32.const 1))
(invoke "fill" (i32.const 2003) (i32.const 4) (i32.const 1))
(invoke "copy" (i32.const 2001) (i32.const 2000) (i32.const 3))
(assert_return (invoke "load8" (i32.const 2001)) (i32.const 1))
(assert_return (invoke "load8" (i32.const 2003)) (i32.const 3))
(invoke "copy" (i32.const 2000) (i32.const 2001) (i32.const 3))
(assert_return (invoke "load8" (i32.const 2000)) (i32.const 1))
(assert_return (invoke "load8" (i32.const 2002)) (i32.const 3))
