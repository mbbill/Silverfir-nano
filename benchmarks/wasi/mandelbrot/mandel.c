// Mandelbrot: a float-heavy benchmark and the source of truth for
// debugging FP codegen. Two modes:
//
//   mandel.wasm [seconds]        self-timing benchmark (default 2.0s)
//   mandel.wasm <size> <mag>     render a PPM image to stdout (reference)
//
// Image mode prints "Elapsed time: N ms" to stderr and is kept for md5
// comparison against known-good output. Benchmark mode sizes its own
// workload to the time target so it costs the same on any engine.
//
// Compile with: wasi-sdk clang -O2 -o mandel.wasm mandel.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "../common/bench.h"

static int iterate(double cx, double cy, int max_iter) {
    double x = 0.0, y = 0.0;
    int i;
    for (i = 0; i < max_iter; i++) {
        double xx = x * x;
        double yy = y * y;
        if (xx + yy > 4.0) return i;
        double x_new = xx - yy + cx;
        y = 2.0 * x * y + cy;
        x = x_new;
    }
    return max_iter;
}

// ---- benchmark mode ----
//
// The batch unit is one pass over a fixed GRID x GRID set of sample
// points spread across the classic view. A whole grid (not a single
// pixel) is the unit so that every unit costs exactly the same — pixels
// differ wildly in iteration count, so a partial grid would make the
// reported rate depend on the batch size. GRID is kept small so one unit
// stays cheap enough for a slow engine to regulate against (see bench.h).
#define GRID 8

static uint32_t g_checksum;

// volatile so the compiler must reload them on every batch iteration and
// cannot hoist the (loop-invariant) grid computation out of the k loop.
// The values never change, so every unit still costs exactly the same.
static volatile double v_cx0 = -0.743643887037151;
static volatile double v_cy0 = 0.131825904205330;

static void mandel_batch(long n, void *ctx) {
    const int max_iter = 1000;
    // Sample the same extent the 128x128 reference image covers.
    const double scale = 4.0 / 6e6 / 128.0;
    const int stride = 128 / GRID;
    uint32_t sum = 0;
    (void)ctx;
    for (long k = 0; k < n; k++) {
        double cx0 = v_cx0, cy0 = v_cy0;
        for (int gy = 0; gy < GRID; gy++) {
            double cy = cy0 + ((double)(gy * stride) - 64.0) * scale;
            for (int gx = 0; gx < GRID; gx++) {
                double cx = cx0 + ((double)(gx * stride) - 64.0) * scale;
                sum = sum * 31u + (uint32_t)iterate(cx, cy, max_iter);
            }
        }
    }
    g_checksum = sum;
}

static int bench_main(int argc, char **argv) {
    // Validation: one grid, so the checksum is independent of the target.
    mandel_batch(1, 0);
    printf("mandel: checksum = %08x\n", g_checksum);
    double rate = bench_ramp(mandel_batch, 0, bench_target(argc, argv), 0);
    printf("mandel: rate = %.2f Kpixel/s\n", rate * (GRID * GRID) / 1000.0);
    return 0;
}

int main(int argc, char **argv) {
    int size = 1024;
    double mag = 400000.0;
    // Image mode needs both <size> and <mag>; anything less is bench mode
    // (where a lone argument is the time target).
    if (argc < 3) return bench_main(argc, argv);
    size = atoi(argv[1]);
    mag = strtod(argv[2], NULL);

    // Center at a classic spot
    const double cx0 = -0.743643887037151;
    const double cy0 = 0.131825904205330;
    double scale = 4.0 / mag / (double)size;
    int max_iter = 1000;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    printf("P6 %d %d 255\n", size, size);
    for (int py = 0; py < size; py++) {
        double cy = cy0 + ((double)py - (double)size * 0.5) * scale;
        for (int px = 0; px < size; px++) {
            double cx = cx0 + ((double)px - (double)size * 0.5) * scale;
            int it = iterate(cx, cy, max_iter);
            unsigned char r, g, b;
            if (it == max_iter) {
                r = g = b = 0;
            } else {
                r = (unsigned char)(it * 3 & 0xff);
                g = (unsigned char)(it * 5 & 0xff);
                b = (unsigned char)(it * 7 & 0xff);
            }
            putchar(r);
            putchar(g);
            putchar(b);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                      + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    fprintf(stderr, "Elapsed time: %.2f ms\n", elapsed_ms);
    return 0;
}
