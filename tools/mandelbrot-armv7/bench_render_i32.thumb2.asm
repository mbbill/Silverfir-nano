
target/armv7-unknown-linux-musleabihf/release/sf-nano-mandelbrot-armv7:	file format elf32-littlearm

Disassembly of section .text:

00028dbc <bench_render_i32>:
   28dbc: 2d e9 f0 4f  	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
   28dc0: 81 b0        	sub	sp, #4
   28dc2: 4b f2 34 3e  	movw	lr, #45876
   28dc6: 84 46        	mov	r12, r0
   28dc8: 00 20        	movs	r0, #0
   28dca: cf f6 ff 7e  	movt	lr, #65535
   28dce: 6f f0 01 0a  	mvn	r10, #1
   28dd2: 00 23        	movs	r3, #0
   28dd4: 48 f2 00 0b  	movw	r11, #32768
   28dd8: 4f f0 00 09  	mov.w	r9, #0
   28ddc: cf f6 ff 7b  	movt	r11, #65535
   28de0: 01 33        	adds	r3, #1
   28de2: 00 93        	str	r3, [sp]
   28de4: 09 f1 01 09  	add.w	r9, r9, #1
   28de8: 00 23        	movs	r3, #0
   28dea: 00 25        	movs	r5, #0
   28dec: 00 26        	movs	r6, #0
   28dee: 06 fb 06 f4  	mul	r4, r6, r6
   28df2: 05 fb 05 f7  	mul	r7, r5, r5
   28df6: a4 13        	asrs	r4, r4, #14
   28df8: 04 eb a7 38  	add.w	r8, r4, r7, asr #14
   28dfc: b8 f5 80 3f  	cmp.w	r8, #65536
   28e00: 0e dc        	bgt	0x28e20 <bench_render_i32+0x64> @ imm = #28
   28e02: 6e 43        	muls	r6, r5, r6
   28e04: bf 13        	asrs	r7, r7, #14
   28e06: 07 eb 0b 05  	add.w	r5, r7, r11
   28e0a: 2d 1b        	subs	r5, r5, r4
   28e0c: 01 33        	adds	r3, #1
   28e0e: 40 2b        	cmp	r3, #64
   28e10: 0a ea 66 34  	and.w	r4, r10, r6, asr #13
   28e14: 04 eb 0e 06  	add.w	r6, r4, lr
   28e18: e9 d1        	bne	0x28dee <bench_render_i32+0x32> @ imm = #-46
   28e1a: 00 25        	movs	r5, #0
   28e1c: 00 23        	movs	r3, #0
   28e1e: 07 e0        	b	0x28e30 <bench_render_i32+0x74> @ imm = #14
   28e20: 13 4c        	ldr	r4, [pc, #76]           @ 0x28e70 <$d>
   28e22: 13 44        	add	r3, r2
   28e24: 03 f0 1f 03  	and	r3, r3, #31
   28e28: 7c 44        	add	r4, pc
   28e2a: 34 f8 13 50  	ldrh.w	r5, [r4, r3, lsl #1]
   28e2e: 2b 0a        	lsrs	r3, r5, #8
   28e30: 88 42        	cmp	r0, r1
   28e32: 14 d2        	bhs	0x28e5e <bench_render_i32+0xa2> @ imm = #40
   28e34: 0c f8 00 30  	strb.w	r3, [r12, r0]
   28e38: 43 1c        	adds	r3, r0, #1
   28e3a: 8b 42        	cmp	r3, r1
   28e3c: 13 d2        	bhs	0x28e66 <bench_render_i32+0xaa> @ imm = #38
   28e3e: 0b f2 33 1b  	addw	r11, r11, #307
   28e42: 02 30        	adds	r0, #2
   28e44: b9 f1 a0 0f  	cmp.w	r9, #160
   28e48: 0c f8 03 50  	strb.w	r5, [r12, r3]
   28e4c: ca d1        	bne	0x28de4 <bench_render_i32+0x28> @ imm = #-108
   28e4e: 00 9b        	ldr	r3, [sp]
   28e50: 0e f2 33 1e  	addw	lr, lr, #307
   28e54: 80 2b        	cmp	r3, #128
   28e56: bd d1        	bne	0x28dd4 <bench_render_i32+0x18> @ imm = #-134
   28e58: 01 b0        	add	sp, #4
   28e5a: bd e8 f0 8f  	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
   28e5e: 05 4a        	ldr	r2, [pc, #20]           @ 0x28e74 <$d+0x4>
   28e60: 7a 44        	add	r2, pc
   28e62: 03 f0 02 e8  	blx	0x2be68 <_ZN4core9panicking18panic_bounds_check17h0a434307a5fa3d9aE> @ imm = #12292
   28e66: 04 4a        	ldr	r2, [pc, #16]           @ 0x28e78 <$d+0x8>
   28e68: 18 46        	mov	r0, r3
   28e6a: 7a 44        	add	r2, pc
   28e6c: 02 f0 fc ef  	blx	0x2be68 <_ZN4core9panicking18panic_bounds_check17h0a434307a5fa3d9aE> @ imm = #12280
