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
    int correctness_only = bench_correctness_only(argc, argv);
    double phase_target = bench_target(argc, argv) / 2.0;
    long compress_workload = correctness_only
        ? 1 : bench_calibrate(lz_compress_batch, &lc, phase_target);
    long decompress_workload = correctness_only
        ? 1 : bench_calibrate(lz_decompress_batch, &lc, phase_target);
    printf("BENCH_WORKLOAD_COMPRESS=%ld\n", compress_workload);
    printf("BENCH_WORKLOAD_DECOMPRESS=%ld\n", decompress_workload);

    bench_result compress;
    bench_result decompress;
    if (correctness_only) {
        lz_compress_batch(1, &lc);
        lz_decompress_batch(1, &lc);
        compress = (bench_result){1, 1.0, 1.0};
        decompress = (bench_result){1, 1.0, 1.0};
    } else {
        compress = bench_measure(
            lz_compress_batch, &lc, compress_workload);
        decompress = bench_measure(
            lz_decompress_batch, &lc, decompress_workload);
    }
    printf("lz4 compress: throughput = %.2f MB/s\n",
           compress.rate * (double)DATA_SIZE / (1024.0 * 1024.0));
    printf("lz4 decompress: throughput = %.2f MB/s\n",
           decompress.rate * (double)DATA_SIZE / (1024.0 * 1024.0));

    free(input);
    free(compressed);
    free(decompressed);
    return 0;
}
