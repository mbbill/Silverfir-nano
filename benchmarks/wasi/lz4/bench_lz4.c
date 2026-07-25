/*
 * LZ4 compression/decompression benchmark for WASI
 * Compresses and decompresses a buffer repeatedly, reports throughput.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../common/bench.h"
#include "lz4.h"

/* Batch unit: one compress (or decompress) of this block. Kept small (see
 * bench.h) so one unit fits inside the target even on a slow engine, and
 * so the roundtrip validation below stays cheap. */
#define DATA_SIZE (64 * 1024)

/* Generate pseudo-random but compressible data (English-like text patterns) */
static void generate_data(unsigned char *buf, int size) {
    /* Simulate text-like data with repeated words and patterns */
    const char *words[] = {
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
        "hello ", "world ", "benchmark ", "compression ", "data ", "test ",
        "performance ", "wasm ", "runtime ", "function ", "value ", "return ",
    };
    int nwords = 20;
    unsigned int state = 0xCAFEBABE;
    int pos = 0;
    while (pos < size) {
        state = state * 1103515245 + 12345;
        const char *w = words[(state >> 16) % nwords];
        while (*w && pos < size) {
            buf[pos++] = *w++;
        }
    }
}

struct lz_ctx_fwd {
    char *input; char *compressed; char *decompressed;
    int comp_size; int max_compressed;
};
static void lz_compress_batch(long n, void *p) {
    struct lz_ctx_fwd *c = (struct lz_ctx_fwd *)p;
    for (long i = 0; i < n; i++)
        LZ4_compress_default(c->input, c->compressed, DATA_SIZE, c->max_compressed);
}
static void lz_decompress_batch(long n, void *p) {
    struct lz_ctx_fwd *c = (struct lz_ctx_fwd *)p;
    for (long i = 0; i < n; i++)
        LZ4_decompress_safe(c->compressed, c->decompressed, c->comp_size, DATA_SIZE);
}

int main(int argc, char **argv) {
    unsigned char *input = malloc(DATA_SIZE);
    int max_compressed = LZ4_compressBound(DATA_SIZE);
    char *compressed = malloc(max_compressed);
    char *decompressed = malloc(DATA_SIZE);

    if (!input || !compressed || !decompressed) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    generate_data(input, DATA_SIZE);

    /* Warm up and verify */
    int comp_size = LZ4_compress_default((char *)input, compressed, DATA_SIZE, max_compressed);
    if (comp_size <= 0) {
        fprintf(stderr, "LZ4 compress failed\n");
        return 1;
    }

    int decomp_size = LZ4_decompress_safe(compressed, decompressed, comp_size, DATA_SIZE);
    if (decomp_size != DATA_SIZE) {
        fprintf(stderr, "LZ4 decompress failed: got %d expected %d\n", decomp_size, DATA_SIZE);
        return 1;
    }
    if (memcmp(input, decompressed, DATA_SIZE) != 0) {
        fprintf(stderr, "LZ4 roundtrip mismatch!\n");
        return 1;
    }

    printf("lz4 benchmark: %d KB input -> %d KB compressed (%.1fx)\n",
           DATA_SIZE / 1024, comp_size / 1024, (double)DATA_SIZE / comp_size);

    struct lz_ctx_fwd lc = { (char *)input, compressed, decompressed, comp_size,
                             max_compressed };
    /* Two timed phases share one budget, so the whole run honours the target. */
    double phase = bench_target(argc, argv) / 2.0;
    double crate = bench_ramp(lz_compress_batch, &lc, phase, 0);
    printf("lz4 compress: throughput = %.2f MB/s\n",
           crate * (double)DATA_SIZE / (1024.0 * 1024.0));
    double drate = bench_ramp(lz_decompress_batch, &lc, phase, 0);
    printf("lz4 decompress: throughput = %.2f MB/s\n",
           drate * (double)DATA_SIZE / (1024.0 * 1024.0));

    free(input);
    free(compressed);
    free(decompressed);
    return 0;
}
