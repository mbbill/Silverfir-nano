/*
 * Shared self-timing driver for the WASI benchmarks.
 *
 * The semantic work unit is fixed by each benchmark. Calibration changes
 * only how many identical units are repeated, so runtimes with very different
 * speeds can run for roughly the same wall-clock duration and still report a
 * comparable work/second rate.
 *
 * Calibration and measurement are deliberately separate. Probe batches grow
 * from one unit until their duration is trustworthy, then extrapolate an
 * iteration count for the requested time. A fresh batch with that count is
 * the only sample used for the reported metric.
 *
 * Keep one unit small enough for the slowest supported engine. Setup,
 * validation, data-set size, algorithm complexity, and working-set size must
 * not depend on the calibrated iteration count.
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BENCH_DEFAULT_SEC
#define BENCH_DEFAULT_SEC 2.0
#endif

#define BENCH_MIN_DT 1e-3
#define BENCH_N_MAX (1L << 28)

static inline double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline double bench_clock_resolution(void) {
    struct timespec resolution;
    if (clock_getres(CLOCK_MONOTONIC, &resolution) != 0) return 0.0;
    return (double)resolution.tv_sec + (double)resolution.tv_nsec * 1e-9;
}

static inline double bench_measurement_start(double resolution) {
    double start = bench_now();
    if (resolution < 0.1) return start;
    /* Align a coarse clock to a tick edge so the first partial tick cannot
     * dominate a short sample. */
    for (;;) {
        double current = bench_now();
        if (current != start) return current;
    }
}

static inline double bench_target(int argc, char **argv) {
    if (argc > 1) {
        double value = atof(argv[argc - 1]);
        if (value > 0.0) return value;
    }
    return BENCH_DEFAULT_SEC;
}

static inline int bench_correctness_only(int argc, char **argv) {
    int index;
    for (index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--bench-correctness") == 0) return 1;
    }
    return 0;
}

typedef struct {
    long n;
    double dt;
    double rate;
} bench_result;

/*
 * Use short probes to estimate how many fixed work units fill `target`.
 * Requiring target/8 of observable work keeps calibration cheap while
 * avoiding an estimate based on timer noise.
 */
static inline long bench_calibrate(void (*batch)(long, void *), void *ctx,
                                   double target) {
    double resolution = bench_clock_resolution();
    double probe_target = target / 8.0;
    long n = 1;

    if (resolution >= 0.1) {
        double start;
        double dt = 0.0;
        long total = 0;
        if (target < 2.0 * resolution) target = 2.0 * resolution;
        start = bench_measurement_start(resolution);
        while (dt < resolution && total < BENCH_N_MAX) {
            long remaining = BENCH_N_MAX - total;
            if (n > remaining) n = remaining;
            batch(n, ctx);
            total += n;
            dt = bench_now() - start;
            if (dt < resolution && n < BENCH_N_MAX / 8) n *= 8;
        }
        if (dt <= 0.0 || total <= 0) return 1;
        {
            double estimate = (double)total * target / dt;
            if (estimate < 1.0) return 1;
            if (estimate >= (double)BENCH_N_MAX) return BENCH_N_MAX;
            return (long)(estimate + 0.5);
        }
    }

    if (probe_target < BENCH_MIN_DT) probe_target = BENCH_MIN_DT;

    for (;;) {
        double t0 = bench_now();
        batch(n, ctx);
        double dt = bench_now() - t0;
        long next;

        if (dt >= probe_target || n >= BENCH_N_MAX) {
            double estimate;
            if (dt <= 0.0) return n;
            estimate = (double)n * target / dt;
            if (estimate < 1.0) return 1;
            if (estimate >= (double)BENCH_N_MAX) return BENCH_N_MAX;
            return (long)(estimate + 0.5);
        }

        if (dt < BENCH_MIN_DT) {
            next = n <= BENCH_N_MAX / 8 ? n * 8 : BENCH_N_MAX;
        } else {
            double estimate = (double)n * probe_target * 1.05 / dt;
            next = estimate >= (double)BENCH_N_MAX
                       ? BENCH_N_MAX
                       : (long)estimate;
            if (n <= BENCH_N_MAX / 8 && next > n * 8) next = n * 8;
            if (next <= n) next = n + 1;
        }
        n = next;
    }
}

static inline bench_result bench_measure(
    void (*batch)(long, void *), void *ctx, long workload) {
    bench_result result;
    double t0 = bench_measurement_start(bench_clock_resolution());
    batch(workload, ctx);
    result.dt = bench_now() - t0;
    if (result.dt <= 0.0) result.dt = 1e-9;
    result.n = workload;
    result.rate = (double)workload / result.dt;
    return result;
}

static inline bench_result bench_run(
    void (*batch)(long, void *), void *ctx, int argc, char **argv) {
    if (bench_correctness_only(argc, argv)) {
        bench_result result;
        batch(1, ctx);
        printf("BENCH_WORKLOAD=1 (correctness only)\n");
        result.n = 1;
        result.dt = 1.0;
        result.rate = 1.0;
        return result;
    }
    double target = bench_target(argc, argv);
    long workload = bench_calibrate(batch, ctx, target);
    printf("BENCH_WORKLOAD=%ld\n", workload);
    return bench_measure(batch, ctx, workload);
}

#endif
