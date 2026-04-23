
/tmp/sf-native-post-rnzalu_a/native_code.elf:	file format elf32-littlearm

Disassembly of section .text:

00000000 <_binary__tmp_mandel_i64_dump_native_code_bin_start>:
     90c: ad f2 04 0d  	subw	sp, sp, #4
     910: 2d e9 f0 4f  	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
     914: 2d ed 10 8b  	vpush	{d8, d9, d10, d11, d12, d13, d14, d15}
     918: 80 46        	mov	r8, r0
     91a: 00 bf        	nop
     91c: 8a 46        	mov	r10, r1
     91e: 00 bf        	nop
     920: d8 f8 04 b0  	ldr.w	r11, [r8, #4]
     924: d8 f8 08 40  	ldr.w	r4, [r8, #8]
     928: ad f2 08 0d  	subw	sp, sp, #8
     92c: cd f8 00 a0  	str.w	r10, [sp]
     930: cd f8 04 a0  	str.w	r10, [sp, #4]
     934: 00 f0 08 f8  	bl	0x948 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0x948> @ imm = #16
     938: bd ec 10 8b  	vpop	{d8, d9, d10, d11, d12, d13, d14, d15}
     93c: bd e8 f0 4f  	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
     940: 0d f2 04 0d  	addw	sp, sp, #4
     944: 70 47        	bx	lr
     946: 00 bf        	nop
     948: ad f2 08 0d  	subw	sp, sp, #8
     94c: cd f8 00 e0  	str.w	lr, [sp]
     950: 4f f0 00 03  	mov.w	r3, #0
     954: 4c f6 cd 4c  	movw	r12, #52429
     958: cf f6 fe 7c  	movt	r12, #65534
     95c: ca f8 10 c0  	str.w	r12, [r10, #16]
     960: 4f f0 00 0e  	mov.w	lr, #0
     964: ca f8 14 e0  	str.w	lr, [r10, #20]
     968: 4f f0 00 0c  	mov.w	r12, #0
     96c: ca f8 18 c0  	str.w	r12, [r10, #24]
     970: 4f f0 00 0e  	mov.w	lr, #0
     974: ca f8 1c e0  	str.w	lr, [r10, #28]
     978: ca f8 08 30  	str.w	r3, [r10, #8]
     97c: 4f f0 00 0c  	mov.w	r12, #0
     980: ca f8 0c c0  	str.w	r12, [r10, #12]
     984: 4c f6 cd 43  	movw	r3, #52429
     988: cf f6 fe 73  	movt	r3, #65534
     98c: 4f f0 00 09  	mov.w	r9, #0
     990: 00 f0 00 b8  	b.w	0x994 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0x994> @ imm = #0
     994: 09 f2 01 09  	addw	r9, r9, #1
     998: 40 f2 00 00  	movw	r0, #0
     99c: cf f6 fe 70  	movt	r0, #65534
     9a0: 4f f0 00 01  	mov.w	r1, #0
     9a4: ca f8 10 30  	str.w	r3, [r10, #16]
     9a8: 4f f0 00 0e  	mov.w	lr, #0
     9ac: ca f8 14 e0  	str.w	lr, [r10, #20]
     9b0: ca f8 18 90  	str.w	r9, [r10, #24]
     9b4: 4f f0 00 0c  	mov.w	r12, #0
     9b8: ca f8 1c c0  	str.w	r12, [r10, #28]
     9bc: ca f8 28 10  	str.w	r1, [r10, #40]
     9c0: 4f f0 00 0e  	mov.w	lr, #0
     9c4: ca f8 2c e0  	str.w	lr, [r10, #44]
     9c8: da f8 08 30  	ldr.w	r3, [r10, #8]
     9cc: 00 f0 00 b8  	b.w	0x9d0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0x9d0> @ imm = #0
     9d0: da f8 28 20  	ldr.w	r2, [r10, #40]
     9d4: 02 f2 01 02  	addw	r2, r2, #1
     9d8: ca f8 28 20  	str.w	r2, [r10, #40]
     9dc: 4f f0 00 0c  	mov.w	r12, #0
     9e0: ca f8 2c c0  	str.w	r12, [r10, #44]
     9e4: 4f f0 00 0e  	mov.w	lr, #0
     9e8: ca f8 30 e0  	str.w	lr, [r10, #48]
     9ec: 4f f0 00 0c  	mov.w	r12, #0
     9f0: ca f8 34 c0  	str.w	r12, [r10, #52]
     9f4: 4f f0 00 09  	mov.w	r9, #0
     9f8: 4f f0 00 01  	mov.w	r1, #0
     9fc: ca f8 08 30  	str.w	r3, [r10, #8]
     a00: 4f f0 00 0e  	mov.w	lr, #0
     a04: ca f8 0c e0  	str.w	lr, [r10, #12]
     a08: ca f8 20 00  	str.w	r0, [r10, #32]
     a0c: 4f f0 00 0c  	mov.w	r12, #0
     a10: ca f8 24 c0  	str.w	r12, [r10, #36]
     a14: ca f8 40 10  	str.w	r1, [r10, #64]
     a18: 4f f0 00 0e  	mov.w	lr, #0
     a1c: ca f8 44 e0  	str.w	lr, [r10, #68]
     a20: 00 f0 00 b8  	b.w	0xa24 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xa24> @ imm = #0
     a24: da f8 40 30  	ldr.w	r3, [r10, #64]
     a28: 4f ea e3 70  	asr.w	r0, r3, #31
     a2c: ca f8 48 30  	str.w	r3, [r10, #72]
     a30: ca f8 4c 00  	str.w	r0, [r10, #76]
     a34: da f8 48 30  	ldr.w	r3, [r10, #72]
     a38: da f8 4c 00  	ldr.w	r0, [r10, #76]
     a3c: a3 fb 03 ce  	umull	r12, lr, r3, r3
     a40: 03 fb 00 ee  	mla	lr, r3, r0, lr
     a44: 00 fb 03 ee  	mla	lr, r0, r3, lr
     a48: 63 46        	mov	r3, r12
     a4a: 00 bf        	nop
     a4c: 70 46        	mov	r0, lr
     a4e: 00 bf        	nop
     a50: 9c 46        	mov	r12, r3
     a52: 00 bf        	nop
     a54: 86 46        	mov	lr, r0
     a56: 00 bf        	nop
     a58: 4f ea 1c 43  	lsr.w	r3, r12, #16
     a5c: 43 ea 0e 43  	orr.w	r3, r3, lr, lsl #16
     a60: 4f ea 1e 40  	lsr.w	r0, lr, #16
     a64: ca f8 40 30  	str.w	r3, [r10, #64]
     a68: 4f f0 00 0c  	mov.w	r12, #0
     a6c: ca f8 44 c0  	str.w	r12, [r10, #68]
     a70: 48 46        	mov	r0, r9
     a72: 00 bf        	nop
     a74: 4f ea e9 71  	asr.w	r1, r9, #31
     a78: ca f8 50 00  	str.w	r0, [r10, #80]
     a7c: ca f8 54 10  	str.w	r1, [r10, #84]
     a80: da f8 50 00  	ldr.w	r0, [r10, #80]
     a84: da f8 54 10  	ldr.w	r1, [r10, #84]
     a88: a0 fb 00 ec  	umull	lr, r12, r0, r0
     a8c: 00 fb 01 cc  	mla	r12, r0, r1, r12
     a90: 01 fb 00 cc  	mla	r12, r1, r0, r12
     a94: 70 46        	mov	r0, lr
     a96: 00 bf        	nop
     a98: 61 46        	mov	r1, r12
     a9a: 00 bf        	nop
     a9c: 86 46        	mov	lr, r0
     a9e: 00 bf        	nop
     aa0: 8c 46        	mov	r12, r1
     aa2: 00 bf        	nop
     aa4: 4f ea 1e 40  	lsr.w	r0, lr, #16
     aa8: 40 ea 0c 40  	orr.w	r0, r0, r12, lsl #16
     aac: 4f ea 1c 41  	lsr.w	r1, r12, #16
     ab0: 03 eb 00 03  	add.w	r3, r3, r0
     ab4: b3 f5 80 2f  	cmp.w	r3, #262144
     ab8: 00 f3 02 80  	bgt.w	0xac0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xac0> @ imm = #4
     abc: 00 f0 76 b9  	b.w	0xdac <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdac> @ imm = #748
     ac0: da f8 08 00  	ldr.w	r0, [r10, #8]
     ac4: da f8 20 30  	ldr.w	r3, [r10, #32]
     ac8: da f8 30 20  	ldr.w	r2, [r10, #48]
     acc: da f8 00 50  	ldr.w	r5, [r10]
     ad0: 02 eb 05 02  	add.w	r2, r2, r5
     ad4: 02 f0 1f 02  	and	r2, r2, #31
     ad8: 4f ea 42 02  	lsl.w	r2, r2, #1
     adc: 02 f5 80 41  	add.w	r1, r2, #16384
     ae0: b1 f5 80 4f  	cmp.w	r1, #16384
     ae4: c0 f0 de 81  	blo.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #956
     ae8: 01 f2 02 05  	addw	r5, r1, #2
     aec: b5 f1 02 0f  	cmp.w	r5, #2
     af0: c0 f0 d8 81  	blo.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #944
     af4: 25 45        	cmp	r5, r4
     af6: 00 bf        	nop
     af8: 00 f2 d4 81  	bhi.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #936
     afc: 0b eb 01 01  	add.w	r1, r11, r1
     b00: b1 f8 00 10  	ldrh.w	r1, [r1]
     b04: 4f ea 11 29  	lsr.w	r9, r1, #8
     b08: 00 f0 00 b8  	b.w	0xb0c <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xb0c> @ imm = #0
     b0c: 49 f6 ff 7e  	movw	lr, #40959
     b10: 70 45        	cmp	r0, lr
     b12: 00 bf        	nop
     b14: 00 f2 c8 80  	bhi.w	0xca8 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xca8> @ imm = #400
     b18: 00 f0 00 b8  	b.w	0xb1c <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xb1c> @ imm = #0
     b1c: 02 46        	mov	r2, r0
     b1e: 00 bf        	nop
     b20: 44 f2 40 0c  	movw	r12, #16448
     b24: 02 eb 0c 02  	add.w	r2, r2, r12
     b28: 44 f2 40 0e  	movw	lr, #16448
     b2c: 72 45        	cmp	r2, lr
     b2e: 00 bf        	nop
     b30: c0 f0 b8 81  	blo.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #880
     b34: 02 f2 01 02  	addw	r2, r2, #1
     b38: b2 f1 01 0f  	cmp.w	r2, #1
     b3c: c0 f0 b2 81  	blo.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #868
     b40: 22 45        	cmp	r2, r4
     b42: 00 bf        	nop
     b44: 00 f2 ae 81  	bhi.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #860
     b48: a2 f2 01 02  	subw	r2, r2, #1
     b4c: 0b eb 02 02  	add.w	r2, r11, r2
     b50: 82 f8 00 90  	strb.w	r9, [r2]
     b54: 49 f6 ff 7c  	movw	r12, #40959
     b58: 60 45        	cmp	r0, r12
     b5a: 00 bf        	nop
     b5c: 00 f0 6e 80  	beq.w	0xc3c <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xc3c> @ imm = #220
     b60: 00 f0 00 b8  	b.w	0xb64 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xb64> @ imm = #0
     b64: 44 f2 41 0e  	movw	lr, #16449
     b68: 00 eb 0e 02  	add.w	r2, r0, lr
     b6c: 02 f2 01 05  	addw	r5, r2, #1
     b70: b5 f1 01 0f  	cmp.w	r5, #1
     b74: c0 f0 96 81  	blo.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #812
     b78: 25 45        	cmp	r5, r4
     b7a: 00 bf        	nop
     b7c: 00 f2 92 81  	bhi.w	0xea4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xea4> @ imm = #804
     b80: 0b eb 02 02  	add.w	r2, r11, r2
     b84: 82 f8 00 10  	strb.w	r1, [r2]
     b88: 40 f2 cc 4c  	movw	r12, #1228
     b8c: 03 eb 0c 03  	add.w	r3, r3, r12
     b90: 00 f2 02 00  	addw	r0, r0, #2
     b94: da f8 28 20  	ldr.w	r2, [r10, #40]
     b98: b2 f1 a0 0f  	cmp.w	r2, #160
     b9c: 40 f0 0a 81  	bne.w	0xdb4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdb4> @ imm = #532
     ba0: 00 f0 00 b8  	b.w	0xba4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xba4> @ imm = #0
     ba4: ca f8 08 00  	str.w	r0, [r10, #8]
     ba8: 4f f0 00 0e  	mov.w	lr, #0
     bac: ca f8 0c e0  	str.w	lr, [r10, #12]
     bb0: ca f8 38 90  	str.w	r9, [r10, #56]
     bb4: 4f f0 00 0c  	mov.w	r12, #0
     bb8: ca f8 3c c0  	str.w	r12, [r10, #60]
     bbc: ca f8 40 10  	str.w	r1, [r10, #64]
     bc0: 4f f0 00 0e  	mov.w	lr, #0
     bc4: ca f8 44 e0  	str.w	lr, [r10, #68]
     bc8: da f8 10 90  	ldr.w	r9, [r10, #16]
     bcc: da f8 18 00  	ldr.w	r0, [r10, #24]
     bd0: da f8 28 10  	ldr.w	r1, [r10, #40]
     bd4: 40 f2 cc 4c  	movw	r12, #1228
     bd8: 09 eb 0c 09  	add.w	r9, r9, r12
     bdc: b0 f1 80 0f  	cmp.w	r0, #128
     be0: 40 f0 f0 80  	bne.w	0xdc4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdc4> @ imm = #480
     be4: 00 f0 00 b8  	b.w	0xbe8 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xbe8> @ imm = #0
     be8: ca f8 10 90  	str.w	r9, [r10, #16]
     bec: 4f f0 00 0e  	mov.w	lr, #0
     bf0: ca f8 14 e0  	str.w	lr, [r10, #20]
     bf4: ca f8 20 30  	str.w	r3, [r10, #32]
     bf8: 4f f0 00 0c  	mov.w	r12, #0
     bfc: ca f8 24 c0  	str.w	r12, [r10, #36]
     c00: da f8 08 30  	ldr.w	r3, [r10, #8]
     c04: 44 f2 40 0e  	movw	lr, #16448
     c08: ca f8 98 e0  	str.w	lr, [r10, #152]
     c0c: 4f f0 00 0c  	mov.w	r12, #0
     c10: ca f8 9c c0  	str.w	r12, [r10, #156]
     c14: dd f8 08 e0  	ldr.w	lr, [sp, #8]
     c18: da f8 98 c0  	ldr.w	r12, [r10, #152]
     c1c: ce f8 00 c0  	str.w	r12, [lr]
     c20: da f8 9c c0  	ldr.w	r12, [r10, #156]
     c24: ce f8 04 c0  	str.w	r12, [lr, #4]
     c28: dd f8 00 e0  	ldr.w	lr, [sp]
     c2c: dd f8 0c a0  	ldr.w	r10, [sp, #12]
     c30: 0d f2 10 0d  	addw	sp, sp, #16
     c34: 4f f0 00 00  	mov.w	r0, #0
     c38: 70 47        	bx	lr
     c3a: 00 bf        	nop
     c3c: ca f8 38 90  	str.w	r9, [r10, #56]
     c40: 4f f0 00 0e  	mov.w	lr, #0
     c44: ca f8 3c e0  	str.w	lr, [r10, #60]
     c48: ca f8 40 10  	str.w	r1, [r10, #64]
     c4c: 4f f0 00 0c  	mov.w	r12, #0
     c50: ca f8 44 c0  	str.w	r12, [r10, #68]
     c54: 4f f4 20 4e  	mov.w	lr, #40960
     c58: ca f8 98 e0  	str.w	lr, [r10, #152]
     c5c: 4f f0 00 0c  	mov.w	r12, #0
     c60: ca f8 9c c0  	str.w	r12, [r10, #156]
     c64: 0a f2 98 03  	addw	r3, r10, #152
     c68: d8 f8 00 90  	ldr.w	r9, [r8]
     c6c: a9 f2 48 09  	subw	r9, r9, #72
     c70: 4b 45        	cmp	r3, r9
     c72: 00 bf        	nop
     c74: 00 f2 b6 80  	bhi.w	0xde4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xde4> @ imm = #364
     c78: 0a f2 98 09  	addw	r9, r10, #152
     c7c: ad f2 08 0d  	subw	sp, sp, #8
     c80: cd f8 00 90  	str.w	r9, [sp]
     c84: cd f8 04 a0  	str.w	r10, [sp, #4]
     c88: 9a 46        	mov	r10, r3
     c8a: 00 bf        	nop
     c8c: 46 f2 3d 7e  	movw	lr, #26429
     c90: c3 f6 38 7e  	movt	lr, #16184
     c94: f0 47        	blx	lr
     c96: 00 bf        	nop
     c98: b0 f1 00 0f  	cmp.w	r0, #0
     c9c: 40 f0 98 80  	bne.w	0xdd0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdd0> @ imm = #304
     ca0: 00 f0 00 b8  	b.w	0xca4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xca4> @ imm = #0
     ca4: 00 f0 5e b9  	b.w	0xf64 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xf64> @ imm = #700
     ca8: ca f8 38 90  	str.w	r9, [r10, #56]
     cac: 4f f0 00 0c  	mov.w	r12, #0
     cb0: ca f8 3c c0  	str.w	r12, [r10, #60]
     cb4: ca f8 40 10  	str.w	r1, [r10, #64]
     cb8: 4f f0 00 0e  	mov.w	lr, #0
     cbc: ca f8 44 e0  	str.w	lr, [r10, #68]
     cc0: ca f8 98 00  	str.w	r0, [r10, #152]
     cc4: 4f f0 00 0c  	mov.w	r12, #0
     cc8: ca f8 9c c0  	str.w	r12, [r10, #156]
     ccc: 0a f2 98 03  	addw	r3, r10, #152
     cd0: d8 f8 00 90  	ldr.w	r9, [r8]
     cd4: a9 f2 48 09  	subw	r9, r9, #72
     cd8: 4b 45        	cmp	r3, r9
     cda: 00 bf        	nop
     cdc: 00 f2 82 80  	bhi.w	0xde4 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xde4> @ imm = #260
     ce0: 0a f2 98 09  	addw	r9, r10, #152
     ce4: ad f2 08 0d  	subw	sp, sp, #8
     ce8: cd f8 00 90  	str.w	r9, [sp]
     cec: cd f8 04 a0  	str.w	r10, [sp, #4]
     cf0: 9a 46        	mov	r10, r3
     cf2: 00 bf        	nop
     cf4: 46 f2 3d 7e  	movw	lr, #26429
     cf8: c3 f6 38 7e  	movt	lr, #16184
     cfc: f0 47        	blx	lr
     cfe: 00 bf        	nop
     d00: b0 f1 00 0f  	cmp.w	r0, #0
     d04: 40 f0 64 80  	bne.w	0xdd0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdd0> @ imm = #200
     d08: 00 f0 00 b8  	b.w	0xd0c <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xd0c> @ imm = #0
     d0c: 00 f0 2a b9  	b.w	0xf64 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xf64> @ imm = #596
     d10: da f8 20 30  	ldr.w	r3, [r10, #32]
     d14: 03 eb 09 03  	add.w	r3, r3, r9
     d18: da f8 40 00  	ldr.w	r0, [r10, #64]
     d1c: a3 eb 00 09  	sub.w	r9, r3, r0
     d20: da f8 48 30  	ldr.w	r3, [r10, #72]
     d24: da f8 4c 00  	ldr.w	r0, [r10, #76]
     d28: da f8 50 10  	ldr.w	r1, [r10, #80]
     d2c: da f8 54 20  	ldr.w	r2, [r10, #84]
     d30: a3 fb 01 ce  	umull	r12, lr, r3, r1
     d34: 03 fb 02 ee  	mla	lr, r3, r2, lr
     d38: 00 fb 01 ee  	mla	lr, r0, r1, lr
     d3c: 63 46        	mov	r3, r12
     d3e: 00 bf        	nop
     d40: 70 46        	mov	r0, lr
     d42: 00 bf        	nop
     d44: 9c 46        	mov	r12, r3
     d46: 00 bf        	nop
     d48: 86 46        	mov	lr, r0
     d4a: 00 bf        	nop
     d4c: 4f ea dc 33  	lsr.w	r3, r12, #15
     d50: 43 ea 4e 43  	orr.w	r3, r3, lr, lsl #17
     d54: 4f ea de 30  	lsr.w	r0, lr, #15
     d58: 4f f6 fe 7c  	movw	r12, #65534
     d5c: cf f6 ff 7c  	movt	r12, #65535
     d60: 03 ea 0c 03  	and.w	r3, r3, r12
     d64: da f8 10 00  	ldr.w	r0, [r10, #16]
     d68: 03 eb 00 03  	add.w	r3, r3, r0
     d6c: ca f8 40 30  	str.w	r3, [r10, #64]
     d70: 4f f0 00 0e  	mov.w	lr, #0
     d74: ca f8 44 e0  	str.w	lr, [r10, #68]
     d78: da f8 30 30  	ldr.w	r3, [r10, #48]
     d7c: 03 f2 01 03  	addw	r3, r3, #1
     d80: ca f8 30 30  	str.w	r3, [r10, #48]
     d84: 4f f0 00 0c  	mov.w	r12, #0
     d88: ca f8 34 c0  	str.w	r12, [r10, #52]
     d8c: b3 f1 40 0f  	cmp.w	r3, #64
     d90: 7f f4 48 ae  	bne.w	0xa24 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xa24> @ imm = #-880
     d94: 00 f0 00 b8  	b.w	0xd98 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xd98> @ imm = #0
     d98: da f8 08 00  	ldr.w	r0, [r10, #8]
     d9c: da f8 20 30  	ldr.w	r3, [r10, #32]
     da0: 4f f0 00 01  	mov.w	r1, #0
     da4: 4f f0 00 09  	mov.w	r9, #0
     da8: ff f7 b0 be  	b.w	0xb0c <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xb0c> @ imm = #-672
     dac: 81 46        	mov	r9, r0
     dae: 00 bf        	nop
     db0: ff f7 ae bf  	b.w	0xd10 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xd10> @ imm = #-164
     db4: 9e 46        	mov	lr, r3
     db6: 00 bf        	nop
     db8: 03 46        	mov	r3, r0
     dba: 00 bf        	nop
     dbc: 70 46        	mov	r0, lr
     dbe: 00 bf        	nop
     dc0: ff f7 06 be  	b.w	0x9d0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0x9d0> @ imm = #-1012
     dc4: 4b 46        	mov	r3, r9
     dc6: 00 bf        	nop
     dc8: 81 46        	mov	r9, r0
     dca: 00 bf        	nop
     dcc: ff f7 e2 bd  	b.w	0x994 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0x994> @ imm = #-1084
     dd0: dd f8 00 e0  	ldr.w	lr, [sp]
     dd4: 0d f2 08 0d  	addw	sp, sp, #8
     dd8: dd f8 04 a0  	ldr.w	r10, [sp, #4]
     ddc: 0d f2 08 0d  	addw	sp, sp, #8
     de0: 70 47        	bx	lr
     de2: 00 bf        	nop
     de4: 40 46        	mov	r0, r8
     de6: 00 bf        	nop
     de8: 4f f0 08 01  	mov.w	r1, #8
     dec: ca f8 80 40  	str.w	r4, [r10, #128]
     df0: ca f8 84 80  	str.w	r8, [r10, #132]
     df4: ca f8 88 a0  	str.w	r10, [r10, #136]
     df8: ca f8 8c b0  	str.w	r11, [r10, #140]
     dfc: ca f8 90 c0  	str.w	r12, [r10, #144]
     e00: ca f8 94 e0  	str.w	lr, [r10, #148]
     e04: 5e ec 13 cb  	vmov	r12, lr, d3
     e08: ca f8 58 c0  	str.w	r12, [r10, #88]
     e0c: ca f8 5c e0  	str.w	lr, [r10, #92]
     e10: 5e ec 14 cb  	vmov	r12, lr, d4
     e14: ca f8 60 c0  	str.w	r12, [r10, #96]
     e18: ca f8 64 e0  	str.w	lr, [r10, #100]
     e1c: 5e ec 15 cb  	vmov	r12, lr, d5
     e20: ca f8 68 c0  	str.w	r12, [r10, #104]
     e24: ca f8 6c e0  	str.w	lr, [r10, #108]
     e28: 5e ec 16 cb  	vmov	r12, lr, d6
     e2c: ca f8 70 c0  	str.w	r12, [r10, #112]
     e30: ca f8 74 e0  	str.w	lr, [r10, #116]
     e34: 5e ec 17 cb  	vmov	r12, lr, d7
     e38: ca f8 78 c0  	str.w	r12, [r10, #120]
     e3c: ca f8 7c e0  	str.w	lr, [r10, #124]
     e40: 42 f2 98 4c  	movw	r12, #9368
     e44: c0 f2 16 0c  	movt	r12, #22
     e48: e0 47        	blx	r12
     e4a: 00 bf        	nop
     e4c: da f8 58 e0  	ldr.w	lr, [r10, #88]
     e50: da f8 5c c0  	ldr.w	r12, [r10, #92]
     e54: 4c ec 13 eb  	vmov	d3, lr, r12
     e58: da f8 60 e0  	ldr.w	lr, [r10, #96]
     e5c: da f8 64 c0  	ldr.w	r12, [r10, #100]
     e60: 4c ec 14 eb  	vmov	d4, lr, r12
     e64: da f8 68 e0  	ldr.w	lr, [r10, #104]
     e68: da f8 6c c0  	ldr.w	r12, [r10, #108]
     e6c: 4c ec 15 eb  	vmov	d5, lr, r12
     e70: da f8 70 e0  	ldr.w	lr, [r10, #112]
     e74: da f8 74 c0  	ldr.w	r12, [r10, #116]
     e78: 4c ec 16 eb  	vmov	d6, lr, r12
     e7c: da f8 78 e0  	ldr.w	lr, [r10, #120]
     e80: da f8 7c c0  	ldr.w	r12, [r10, #124]
     e84: 4c ec 17 eb  	vmov	d7, lr, r12
     e88: da f8 80 40  	ldr.w	r4, [r10, #128]
     e8c: da f8 84 80  	ldr.w	r8, [r10, #132]
     e90: da f8 88 a0  	ldr.w	r10, [r10, #136]
     e94: da f8 8c b0  	ldr.w	r11, [r10, #140]
     e98: da f8 90 e0  	ldr.w	lr, [r10, #144]
     e9c: da f8 94 c0  	ldr.w	r12, [r10, #148]
     ea0: ff f7 96 bf  	b.w	0xdd0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdd0> @ imm = #-212
     ea4: 40 46        	mov	r0, r8
     ea6: 00 bf        	nop
     ea8: 4f f0 01 01  	mov.w	r1, #1
     eac: ca f8 80 40  	str.w	r4, [r10, #128]
     eb0: ca f8 84 80  	str.w	r8, [r10, #132]
     eb4: ca f8 88 a0  	str.w	r10, [r10, #136]
     eb8: ca f8 8c b0  	str.w	r11, [r10, #140]
     ebc: ca f8 90 e0  	str.w	lr, [r10, #144]
     ec0: ca f8 94 c0  	str.w	r12, [r10, #148]
     ec4: 5c ec 13 eb  	vmov	lr, r12, d3
     ec8: ca f8 58 e0  	str.w	lr, [r10, #88]
     ecc: ca f8 5c c0  	str.w	r12, [r10, #92]
     ed0: 5c ec 14 eb  	vmov	lr, r12, d4
     ed4: ca f8 60 e0  	str.w	lr, [r10, #96]
     ed8: ca f8 64 c0  	str.w	r12, [r10, #100]
     edc: 5c ec 15 eb  	vmov	lr, r12, d5
     ee0: ca f8 68 e0  	str.w	lr, [r10, #104]
     ee4: ca f8 6c c0  	str.w	r12, [r10, #108]
     ee8: 5c ec 16 eb  	vmov	lr, r12, d6
     eec: ca f8 70 e0  	str.w	lr, [r10, #112]
     ef0: ca f8 74 c0  	str.w	r12, [r10, #116]
     ef4: 5c ec 17 eb  	vmov	lr, r12, d7
     ef8: ca f8 78 e0  	str.w	lr, [r10, #120]
     efc: ca f8 7c c0  	str.w	r12, [r10, #124]
     f00: 42 f2 98 4e  	movw	lr, #9368
     f04: c0 f2 16 0e  	movt	lr, #22
     f08: f0 47        	blx	lr
     f0a: 00 bf        	nop
     f0c: da f8 58 c0  	ldr.w	r12, [r10, #88]
     f10: da f8 5c e0  	ldr.w	lr, [r10, #92]
     f14: 4e ec 13 cb  	vmov	d3, r12, lr
     f18: da f8 60 c0  	ldr.w	r12, [r10, #96]
     f1c: da f8 64 e0  	ldr.w	lr, [r10, #100]
     f20: 4e ec 14 cb  	vmov	d4, r12, lr
     f24: da f8 68 c0  	ldr.w	r12, [r10, #104]
     f28: da f8 6c e0  	ldr.w	lr, [r10, #108]
     f2c: 4e ec 15 cb  	vmov	d5, r12, lr
     f30: da f8 70 c0  	ldr.w	r12, [r10, #112]
     f34: da f8 74 e0  	ldr.w	lr, [r10, #116]
     f38: 4e ec 16 cb  	vmov	d6, r12, lr
     f3c: da f8 78 c0  	ldr.w	r12, [r10, #120]
     f40: da f8 7c e0  	ldr.w	lr, [r10, #124]
     f44: 4e ec 17 cb  	vmov	d7, r12, lr
     f48: da f8 80 40  	ldr.w	r4, [r10, #128]
     f4c: da f8 84 80  	ldr.w	r8, [r10, #132]
     f50: da f8 88 a0  	ldr.w	r10, [r10, #136]
     f54: da f8 8c b0  	ldr.w	r11, [r10, #140]
     f58: da f8 90 c0  	ldr.w	r12, [r10, #144]
     f5c: da f8 94 e0  	ldr.w	lr, [r10, #148]
     f60: ff f7 36 bf  	b.w	0xdd0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdd0> @ imm = #-404
     f64: 40 46        	mov	r0, r8
     f66: 00 bf        	nop
     f68: 4f f0 00 01  	mov.w	r1, #0
     f6c: ca f8 80 40  	str.w	r4, [r10, #128]
     f70: ca f8 84 80  	str.w	r8, [r10, #132]
     f74: ca f8 88 a0  	str.w	r10, [r10, #136]
     f78: ca f8 8c b0  	str.w	r11, [r10, #140]
     f7c: ca f8 90 c0  	str.w	r12, [r10, #144]
     f80: ca f8 94 e0  	str.w	lr, [r10, #148]
     f84: 5e ec 13 cb  	vmov	r12, lr, d3
     f88: ca f8 58 c0  	str.w	r12, [r10, #88]
     f8c: ca f8 5c e0  	str.w	lr, [r10, #92]
     f90: 5e ec 14 cb  	vmov	r12, lr, d4
     f94: ca f8 60 c0  	str.w	r12, [r10, #96]
     f98: ca f8 64 e0  	str.w	lr, [r10, #100]
     f9c: 5e ec 15 cb  	vmov	r12, lr, d5
     fa0: ca f8 68 c0  	str.w	r12, [r10, #104]
     fa4: ca f8 6c e0  	str.w	lr, [r10, #108]
     fa8: 5e ec 16 cb  	vmov	r12, lr, d6
     fac: ca f8 70 c0  	str.w	r12, [r10, #112]
     fb0: ca f8 74 e0  	str.w	lr, [r10, #116]
     fb4: 5e ec 17 cb  	vmov	r12, lr, d7
     fb8: ca f8 78 c0  	str.w	r12, [r10, #120]
     fbc: ca f8 7c e0  	str.w	lr, [r10, #124]
     fc0: 42 f2 98 4c  	movw	r12, #9368
     fc4: c0 f2 16 0c  	movt	r12, #22
     fc8: e0 47        	blx	r12
     fca: 00 bf        	nop
     fcc: da f8 58 e0  	ldr.w	lr, [r10, #88]
     fd0: da f8 5c c0  	ldr.w	r12, [r10, #92]
     fd4: 4c ec 13 eb  	vmov	d3, lr, r12
     fd8: da f8 60 e0  	ldr.w	lr, [r10, #96]
     fdc: da f8 64 c0  	ldr.w	r12, [r10, #100]
     fe0: 4c ec 14 eb  	vmov	d4, lr, r12
     fe4: da f8 68 e0  	ldr.w	lr, [r10, #104]
     fe8: da f8 6c c0  	ldr.w	r12, [r10, #108]
     fec: 4c ec 15 eb  	vmov	d5, lr, r12
     ff0: da f8 70 e0  	ldr.w	lr, [r10, #112]
     ff4: da f8 74 c0  	ldr.w	r12, [r10, #116]
     ff8: 4c ec 16 eb  	vmov	d6, lr, r12
     ffc: da f8 78 e0  	ldr.w	lr, [r10, #120]
    1000: da f8 7c c0  	ldr.w	r12, [r10, #124]
    1004: 4c ec 17 eb  	vmov	d7, lr, r12
    1008: da f8 80 40  	ldr.w	r4, [r10, #128]
    100c: da f8 84 80  	ldr.w	r8, [r10, #132]
    1010: da f8 88 a0  	ldr.w	r10, [r10, #136]
    1014: da f8 8c b0  	ldr.w	r11, [r10, #140]
    1018: da f8 90 e0  	ldr.w	lr, [r10, #144]
    101c: da f8 94 c0  	ldr.w	r12, [r10, #148]
    1020: ff f7 d6 be  	b.w	0xdd0 <_binary__tmp_mandel_i64_dump_native_code_bin_start+0xdd0> @ imm = #-596
