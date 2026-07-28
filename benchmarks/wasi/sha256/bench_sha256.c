/*
 * SHA-256 benchmark for WASI
 * Standalone implementation — no external dependencies.
 * Hashes a large buffer repeatedly and reports throughput.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../common/bench.h"
#include <stdint.h>

/* ---- SHA-256 implementation ---- */

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

typedef struct {
    uint32_t state[8];
    uint64_t bitcount;
    uint8_t buffer[64];
    uint32_t buflen;
} sha256_ctx;

static void sha256_transform(sha256_ctx *ctx, const uint8_t *data) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];
    }

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (int i = 0; i < 64; i++) {
        t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(sha256_ctx *ctx) {
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
    ctx->bitcount = 0;
    ctx->buflen = 0;
}

static void sha256_update(sha256_ctx *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->buffer[ctx->buflen++] = data[i];
        if (ctx->buflen == 64) {
            sha256_transform(ctx, ctx->buffer);
            ctx->bitcount += 512;
            ctx->buflen = 0;
        }
    }
}

static void sha256_final(sha256_ctx *ctx, uint8_t hash[32]) {
    uint32_t i = ctx->buflen;
    ctx->buffer[i++] = 0x80;
    if (i > 56) {
        while (i < 64) ctx->buffer[i++] = 0;
        sha256_transform(ctx, ctx->buffer);
        i = 0;
    }
    while (i < 56) ctx->buffer[i++] = 0;

    ctx->bitcount += ctx->buflen * 8;
    ctx->buffer[56] = (uint8_t)(ctx->bitcount >> 56);
    ctx->buffer[57] = (uint8_t)(ctx->bitcount >> 48);
    ctx->buffer[58] = (uint8_t)(ctx->bitcount >> 40);
    ctx->buffer[59] = (uint8_t)(ctx->bitcount >> 32);
    ctx->buffer[60] = (uint8_t)(ctx->bitcount >> 24);
    ctx->buffer[61] = (uint8_t)(ctx->bitcount >> 16);
    ctx->buffer[62] = (uint8_t)(ctx->bitcount >> 8);
    ctx->buffer[63] = (uint8_t)(ctx->bitcount);
    sha256_transform(ctx, ctx->buffer);

    for (int j = 0; j < 8; j++) {
        hash[j * 4]     = (uint8_t)(ctx->state[j] >> 24);
        hash[j * 4 + 1] = (uint8_t)(ctx->state[j] >> 16);
        hash[j * 4 + 2] = (uint8_t)(ctx->state[j] >> 8);
        hash[j * 4 + 3] = (uint8_t)(ctx->state[j]);
    }
}

/* ---- Benchmark ---- */

/* Batch unit: one hash of this block. Kept small (see bench.h) so even an
 * interpreter under qemu finishes one unit well inside the time target.
 * sha256 is compute-bound, so the block size does not change what is
 * being measured. */
#define DATA_SIZE (64 * 1024)

static void generate_data(unsigned char *buf, int size) {
    unsigned int state = 0x12345678;
    for (int i = 0; i < size; i++) {
        state = state * 1103515245 + 12345;
        buf[i] = (unsigned char)(state >> 16);
    }
}

static unsigned char *g_input;

static void hash_batch(long n, void *ctx_) {
    uint8_t hash[32];
    (void)ctx_;
    for (long i = 0; i < n; i++) {
        sha256_ctx ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, g_input, DATA_SIZE);
        sha256_final(&ctx, hash);
    }
}

int main(int argc, char **argv) {
    uint8_t hash[32];

    g_input = malloc(DATA_SIZE);
    if (!g_input) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    generate_data(g_input, DATA_SIZE);

    /* Validation: one fixed hash, independent of the timed workload. */
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, g_input, DATA_SIZE);
    sha256_final(&ctx, hash);
    printf("sha256 benchmark: %d KB input, hash = ", DATA_SIZE / 1024);
    for (int i = 0; i < 8; i++) printf("%02x", hash[i]);
    printf("...\n");

    bench_result result = bench_run(hash_batch, 0, argc, argv);
    if (result.n == 0) return 2;
    printf("sha256: throughput = %.2f MB/s\n",
           result.rate * (double)DATA_SIZE / (1024.0 * 1024.0));

    free(g_input);
    return 0;
}
