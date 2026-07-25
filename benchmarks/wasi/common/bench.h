/*
 * Shared self-timing driver for the WASI benchmarks.
 *
 * Why: these benchmarks run on engines whose speed spans four orders of
 * magnitude — native JIT, our interpreter, and an interpreter under qemu
 * for armv7/armv7m. A fixed workload that takes 2s on the JIT takes many
 * minutes there. Every benchmark therefore sizes its own workload at run
 * time to hit a wall-clock target.
 *
 * Geometric ramp: run 1 batch unit, then grow the batch until one batch
 * alone meets the target. The ramp doubles as warmup, the clock is read
 * once per BATCH (never per unit, so host-call overhead stays out of the
 * measurement), and the reported metric is a RATE taken from the final
 * batch only. Total cost is bounded at ~3x the target on any engine.
 *
 * THE ONE RULE FOR CALLERS — keep the batch unit small.
 * The ramp cannot regulate below the cost of a SINGLE unit: if one unit
 * already exceeds the target, the benchmark overshoots and no target can
 * bound it. Size the unit so the SLOWEST engine you care about finishes
 * one in well under the target (~1ms or less on a fast native engine).
 * In practice that means chunking the work: hash a 64 KB block, not a
 * 1 MB buffer; render a tile, not a frame.
 *
 * Validation must be cheap for the same reason. Verify correctness on a
 * tiny fixed workload (one unit) and print an iteration-count-independent
 * checksum; never validate over a workload that scales with the target.
 *
 * The target comes from the last argv argument when it parses as a
 * positive number (seconds), else BENCH_DEFAULT_SEC.
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdlib.h>
#include <time.h>

#ifndef BENCH_DEFAULT_SEC
#define BENCH_DEFAULT_SEC 2.0
#endif

/* Smallest batch time trusted for extrapolation. Below this, clock
 * resolution and one-off noise dominate, so the ramp grows blindly. */
#define BENCH_MIN_DT 1e-3

/* Hard ceiling on total ramp cost, as a multiple of the target. Stops a
 * mis-extrapolation (noise, a thermally throttling machine, an engine
 * that warms up mid-ramp) from looping indefinitely. */
#define BENCH_MAX_TOTAL 3.0

/* Batch-size ceiling. `long` is 32-bit on wasm32, so every growth step
 * clamps to this and n can never overflow. */
#define BENCH_N_MAX (1L << 28)

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double bench_target(int argc, char **argv) {
    if (argc > 1) {
        double v = atof(argv[argc - 1]);
        if (v > 0.0) return v;
    }
    return BENCH_DEFAULT_SEC;
}

typedef struct {
    long   n;     /* units in the measured batch */
    double dt;    /* seconds that batch took */
    double rate;  /* units per second */
} bench_result;

/* Run batch(n, ctx) with a geometrically growing n until one batch meets
 * `target` seconds, then report that batch's rate. */
static bench_result bench_run(void (*batch)(long, void *), void *ctx,
                              double target) {
    bench_result r;
    double total = 0.0;
    long n = 1;
    for (;;) {
        double t0 = bench_now();
        batch(n, ctx);
        double dt = bench_now() - t0;
        total += dt;

        int good = dt >= target;
        /* Out of budget but the sample is still usable: take it rather
         * than spend another, larger batch chasing the target exactly. */
        int out_of_budget = total >= BENCH_MAX_TOTAL * target &&
                            dt >= target / 8.0;
        if (good || out_of_budget || n >= BENCH_N_MAX) {
            if (dt <= 0.0) dt = 1e-9; /* clock never advanced */
            r.n = n;
            r.dt = dt;
            r.rate = (double)n / dt;
            return r;
        }

        if (dt < BENCH_MIN_DT || dt < target / 8.0) {
            /* Too short to extrapolate from; grow blindly. */
            n = (n <= BENCH_N_MAX / 8) ? n * 8 : BENCH_N_MAX;
        } else {
            /* Aim slightly past the target so one more batch suffices. */
            double est = (double)n * target * 1.25 / dt;
            long next = est >= (double)BENCH_N_MAX ? BENCH_N_MAX : (long)est;
            n = next > n ? next : n + 1;
        }
    }
}

/* Convenience wrapper: just the rate. */
static double bench_ramp(void (*batch)(long, void *), void *ctx, double target,
                         long *out_n) {
    bench_result r = bench_run(batch, ctx, target);
    if (out_n) *out_n = r.n;
    return r.rate;
}

/* For best-of-N benchmarks (STREAM), where the metric is the best single
 * trial rather than a rate over a batch. Given how long one trial took,
 * return how many trials fit the remaining budget, clamped to [min,max].
 * The calibration trial is a real trial, so nothing is wasted. */
static long bench_plan(double dt_one, double target, long min_trials,
                       long max_trials) {
    long n;
    if (dt_one <= 0.0) return max_trials;
    n = (long)(target / dt_one);
    if (n < min_trials) n = min_trials;
    if (n > max_trials) n = max_trials;
    return n;
}

#endif
