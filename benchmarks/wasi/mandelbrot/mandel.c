// Minimal mandelbrot reference that matches the existing mandel.wasm behavior:
//   usage: mandel.wasm <size> <magnification>
// Renders a PPM image to stdout at <size>x<size> with the given magnification.
// Prints "Elapsed time: N ms" to stderr.
//
// Used as the source of truth for debugging FP codegen. Compile with:
//   wasi-sdk clang -O2 -o mandel.wasm mandel.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

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

int main(int argc, char **argv) {
    int size = 1024;
    double mag = 400000.0;
    if (argc >= 2) size = atoi(argv[1]);
    if (argc >= 3) mag = strtod(argv[2], NULL);

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
