
target/armv7-unknown-linux-musleabihf/release/sf-nano-mandelbrot-armv7:	file format elf32-littlearm

Disassembly of section .text:

00028e7c <bench_render_i64>:
   28e7c: 2d e9 f0 4f  	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
   28e80: 83 b0        	sub	sp, #12
   28e82: 4c f6 cd 4e  	movw	lr, #52429
   28e86: 02 92        	str	r2, [sp, #8]
   28e88: 02 46        	mov	r2, r0
   28e8a: 00 20        	movs	r0, #0
   28e8c: cf f6 fe 7e  	movt	lr, #65534
   28e90: 00 23        	movs	r3, #0
   28e92: 40 f2 00 0a  	movw	r10, #0
   28e96: 4f f0 00 09  	mov.w	r9, #0
   28e9a: cf f6 fe 7a  	movt	r10, #65534
   28e9e: 01 33        	adds	r3, #1
   28ea0: 01 93        	str	r3, [sp, #4]
   28ea2: 09 f1 01 09  	add.w	r9, r9, #1
   28ea6: 00 23        	movs	r3, #0
   28ea8: 00 24        	movs	r4, #0
   28eaa: 4f f0 00 0c  	mov.w	r12, #0
   28eae: 84 fb 04 56  	smull	r5, r6, r4, r4
   28eb2: 8c fb 0c b8  	smull	r11, r8, r12, r12
   28eb6: 2d 0c        	lsrs	r5, r5, #16
   28eb8: 45 ea 06 45  	orr.w	r5, r5, r6, lsl #16
   28ebc: 4f ea 1b 46  	lsr.w	r6, r11, #16
   28ec0: 46 ea 08 46  	orr.w	r6, r6, r8, lsl #16
   28ec4: 77 19        	adds	r7, r6, r5
   28ec6: b7 f5 80 2f  	cmp.w	r7, #262144
   28eca: 13 dc        	bgt	0x28ef4 <bench_render_i64+0x78> @ imm = #38
   28ecc: 8c fb 04 74  	smull	r7, r4, r12, r4
   28ed0: 01 33        	adds	r3, #1
   28ed2: 40 2b        	cmp	r3, #64
   28ed4: 4f ea d7 37  	lsr.w	r7, r7, #15
   28ed8: 47 ea 44 47  	orr.w	r7, r7, r4, lsl #17
   28edc: 0a eb 05 04  	add.w	r4, r10, r5
   28ee0: 27 f0 01 07  	bic	r7, r7, #1
   28ee4: a4 eb 06 04  	sub.w	r4, r4, r6
   28ee8: 07 eb 0e 0c  	add.w	r12, r7, lr
   28eec: df d1        	bne	0x28eae <bench_render_i64+0x32> @ imm = #-66
   28eee: 00 24        	movs	r4, #0
   28ef0: 00 23        	movs	r3, #0
   28ef2: 08 e0        	b	0x28f06 <bench_render_i64+0x8a> @ imm = #16
   28ef4: 13 4c        	ldr	r4, [pc, #76]           @ 0x28f44 <$d>
   28ef6: 02 9f        	ldr	r7, [sp, #8]
   28ef8: 7c 44        	add	r4, pc
   28efa: 3b 44        	add	r3, r7
   28efc: 03 f0 1f 03  	and	r3, r3, #31
   28f00: 34 f8 13 40  	ldrh.w	r4, [r4, r3, lsl #1]
   28f04: 23 0a        	lsrs	r3, r4, #8
   28f06: 88 42        	cmp	r0, r1
   28f08: 12 d2        	bhs	0x28f30 <bench_render_i64+0xb4> @ imm = #36
   28f0a: 13 54        	strb	r3, [r2, r0]
   28f0c: 43 1c        	adds	r3, r0, #1
   28f0e: 8b 42        	cmp	r3, r1
   28f10: 12 d2        	bhs	0x28f38 <bench_render_i64+0xbc> @ imm = #36
   28f12: 0a f2 cc 4a  	addw	r10, r10, #1228
   28f16: 02 30        	adds	r0, #2
   28f18: b9 f1 a0 0f  	cmp.w	r9, #160
   28f1c: d4 54        	strb	r4, [r2, r3]
   28f1e: c0 d1        	bne	0x28ea2 <bench_render_i64+0x26> @ imm = #-128
   28f20: 01 9b        	ldr	r3, [sp, #4]
   28f22: 0e f2 cc 4e  	addw	lr, lr, #1228
   28f26: 80 2b        	cmp	r3, #128
   28f28: b3 d1        	bne	0x28e92 <bench_render_i64+0x16> @ imm = #-154
   28f2a: 03 b0        	add	sp, #12
   28f2c: bd e8 f0 8f  	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
   28f30: 05 4a        	ldr	r2, [pc, #20]           @ 0x28f48 <$d+0x4>
   28f32: 7a 44        	add	r2, pc
   28f34: 02 f0 98 ef  	blx	0x2be68 <_ZN4core9panicking18panic_bounds_check17h0a434307a5fa3d9aE> @ imm = #12080
   28f38: 04 4a        	ldr	r2, [pc, #16]           @ 0x28f4c <$d+0x8>
   28f3a: 18 46        	mov	r0, r3
   28f3c: 7a 44        	add	r2, pc
   28f3e: 02 f0 94 ef  	blx	0x2be68 <_ZN4core9panicking18panic_bounds_check17h0a434307a5fa3d9aE> @ imm = #12072
   28f42: 00 bf        	nop
