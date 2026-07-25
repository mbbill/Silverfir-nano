/*
 * bzip2 compression benchmark for WASI
 * Compresses a synthetic data buffer repeatedly and reports throughput.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../common/bench.h"
#include "bzlib.h"

/* Batch unit: one compression of this block. bzip2 is by far the slowest
 * per byte in the suite, so its unit is the smallest (see bench.h) — one
 * unit must still fit inside the target on an interpreter under qemu. */
#define DATA_SIZE (32 * 1024)

/* Generate pseudo-random but compressible data (English-like text patterns) */
static void generate_data(unsigned char *buf, int size) {
    const char *words[] = {
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ",
        "hello ", "world ", "benchmark ", "compression ", "data ", "test ",
        "performance ", "wasm ", "runtime ", "function ", "value ", "return ",
    };
    int nwords = 20;
    unsigned int state = 0xDEADBEEF;
    int pos = 0;
    while (pos < size) {
        state = state * 1103515245 + 12345;
        const char *w = words[(state >> 16) % nwords];
        while (*w && pos < size) {
            buf[pos++] = *w++;
        }
    }
}

struct bz_ctx_fwd { char *input; char *output; unsigned out_size; };
static void bz_batch(long n, void *p) {
    struct bz_ctx_fwd *c = (struct bz_ctx_fwd *)p;
    for (long i = 0; i < n; i++) {
        unsigned dest_len = c->out_size;
        BZ2_bzBuffToBuffCompress(c->output, &dest_len, c->input, DATA_SIZE, 9, 0, 30);
    }
}

int main(int argc, char **argv) {
    unsigned char *input = malloc(DATA_SIZE);
    /* bzip2 output can be slightly larger than input in worst case */
    unsigned int out_size = DATA_SIZE + DATA_SIZE / 100 + 600;
    char *output = malloc(out_size);

    if (!input || !output) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }

    generate_data(input, DATA_SIZE);

    /* Warm up */
    unsigned int dest_len = out_size;
    int rc = BZ2_bzBuffToBuffCompress(output, &dest_len, (char *)input, DATA_SIZE, 9, 0, 30);
    if (rc != BZ_OK) {
        fprintf(stderr, "bzip2 compress failed: %d\n", rc);
        return 1;
    }

    printf("bzip2 benchmark: %d KB input -> %u KB compressed (%.1fx)\n",
           DATA_SIZE / 1024, dest_len / 1024, (double)DATA_SIZE / dest_len);

    struct bz_ctx_fwd bc = {
        (char *)input, output, out_size };
    double rate = bench_ramp(bz_batch, &bc, bench_target(argc, argv), 0);
    printf("bzip2: throughput = %.2f MB/s\n",
           rate * (double)DATA_SIZE / (1024.0 * 1024.0));

    free(input);
    free(output);
    return 0;
}
