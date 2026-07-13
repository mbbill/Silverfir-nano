// freqprobe: measure effective core frequency via a dependent-add chain
// (1 add/cycle on every Apple core, P or E). Prints GHz.
// Usage: freqprobe [duration_ms]   (default 200)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <mach/mach_time.h>

int main(int argc, char **argv) {
    double target_ms = argc > 1 ? atof(argv[1]) : 200.0;
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    uint64_t t0 = mach_absolute_time();
    uint64_t blocks = 0;
    uint64_t x = 0;
    for (;;) {
        __asm__ volatile(".rept 16384\n\tadd %0, %0, #1\n\t.endr\n" : "+r"(x));
        blocks++;
        uint64_t t1 = mach_absolute_time();
        double ns = (double)(t1 - t0) * tb.numer / tb.denom;
        if (ns > target_ms * 1e6) {
            // dependent adds retire 1/cycle -> adds per ns == GHz
            printf("freqprobe: %.3f GHz  (%llu Madds / %.1f ms)\n",
                   (double)(blocks * 16384) / ns,
                   (unsigned long long)(blocks * 16384 / 1000000), ns / 1e6);
            break;
        }
    }
    return (int)(x & 1); // defeat optimizer
}
