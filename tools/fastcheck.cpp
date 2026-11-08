// tools/fastcheck.cpp: fastnoise.hpp's vector-lane noise setup against standalone.hpp's NormalNoise::init on 20,000
// random seeds (24 samples each, every noise): the count of mismatching samples must be 0.
#include "../src/fastnoise.hpp"
#include <cstdio>
#include <random>
#include <x86intrin.h>
using namespace standalone;
int main() {
    std::mt19937_64 g(7); long bad = 0; uint64_t cs = 0, cf = 0;
    for (int t = 0; t < 20000; t++) {
        const uint64_t seed = g(); Sampler a(seed), b(seed);
        uint64_t t0 = __rdtsc(); for (int c = 0; c < 6; c++) a.get(Channel(c)); uint64_t t1 = __rdtsc(); fastnoise::ready_all(b); uint64_t t2 = __rdtsc(); cs += t1 - t0; cf += t2 - t1;
        for (int c = 0; c < 6; c++) for (int k = 0; k < 4; k++) { const double x = (double) (int64_t) (g() % 4000000) - 2e6, z = (double) (int64_t) (g() % 4000000) - 2e6;
            if (a.get(Channel(c)).sample(x, 0, z) != b.get(Channel(c)).sample(x, 0, z)) bad++; }
    }
    printf("20000 seeds: %ld mismatching samples; all six noises %.0f cycles (NormalNoise::init) vs %.0f (interleaved)\n", bad, (double) cs / 20000, (double) cf / 20000);
}
