// origin.cpp: the largest Snowy Slopes containing the world origin (default world type, y 256, 1:4 cells, 4-connected,
// bridged as MC-Measure: see snowy.hpp). Seed indices START + k, k < COUNT (COUNT 0: no end). Index k is the seed
// (k G) ^ X, G = 0x9e3779b97f4a7c15, X = 0x6a09e667f3bcc909: xSetSeed mixes t = k G and t + G, and t + G is index k+1's
// t, so the gate mixes once a seed. Per seed, cheapest first:
//   1. phase gate (RNG only): the y-phase of erosion octave -9, humidity octave -8 and temperature octave -10, both
//      noise stacks, within PHE, PHH, PHT of mid-cell (where each layer swings most). It runs on the GPU through Vulkan
//      compute (gpu/gate.comp, bit-identical; a feeder thread queues the gated indices) and on the CPU threads whenever
//      that queue is empty (AVX2 block cascade: each window test on full vectors of the survivors of the one before).
//   2. tail cuts at the origin: the temperature octave -10 sum (both stacks) <= T10, built 16 gated seeds at a time in
//      vector lanes, the stack with the larger predicted spread first and the other skipped when the first is more than
//      TZ predicted sds above the cut; then the erosion octave -9 sum <= E9 (built before the rest of erosion).
//   3. snowy slopes at quart (0, 0) with temperature <= TMAX (cheap bounds first, then the exact climate)
//   4. the patch on a lattice of 32 quarts (8-connected, a screen); from R32 (M blocks2) on a lattice of 16, from R16 on
//      8, from R8 on 4, from R4 the exact count (lattice 1, the same count as MC-Measure).
// Output: PREFIX.csv (one line per seed that reaches step 4: seed,area8,area4,exact,area16 in M blocks2, 0 = that flood
// did not run; area8 is the lattice-16 size where no lattice-8 flood ran), CANDIDATE lines on stderr for exact counts
// from CAND (M), progress every 60 s. MC-Measure starts from the biome at block (0, 256, 0), which the block-level zoom
// can take from a neighbouring cell, so a candidate can measure as another biome there: verify every candidate
// (scripts/run_origin.sh does).
// usage: origin START COUNT PREFIX THREADS [key=value ...]   defaults (the record run): phT=0.24 phH=0.22 phE=0.22
//   (ph= sets all three) tmax=-0.5 t10=-0.62 tz=2.0 e9=-0.25 r32=0.7 r16=1.5 r8=2.5 r4=3.6 cand=4.6 gpu=1 gper=256
//   ggroups=4096 gqueue=200000 cpugate=1 (gpu=0: CPU gate only; cpugate=0: CPU threads wait for the GPU's seeds)
#include "snowy.hpp"
#include "fastnoise.hpp"
#ifndef NO_VULKAN
#include "gpu/vkgate.hpp"
#endif
#include <deque>
#include <immintrin.h>
#include <x86intrin.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using namespace snowy;
using Clock = std::chrono::steady_clock;
static volatile sig_atomic_t stop_ = 0;
static void on_signal(int) { stop_ = 1; }
static inline uint64_t splitmix(uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
struct Hash { uint64_t lo, hi; };
// noise name hashes and octave name hashes (../caves/name_hashes.h), in gate order: erosion -9 (its phase matters most
// and its window is the narrowest: tested first, in vectors), vegetation -8, temperature -10
static const Hash NOISE[3] = {{0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL}, {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL}, {0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL}};
static const Hash OCT[3] = {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL}};
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t nl(uint64_t &l, uint64_t &h) { const uint64_t a = l; uint64_t b = h; const uint64_t n = rotl(a + b, 17) + a; b ^= a; l = rotl(a, 49) ^ b ^ (b << 21); h = rotl(b, 28); return n; }
// AVX2, 4 seeds a lane-set: the world state and the first noise's both stacks in vectors; the rest scalar
static inline __m256i vrotl(__m256i x, int k) { return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k)); }
static inline __m256i vmul(__m256i a, __m256i b) {   // low 64 bits of a * b
    __m256i lo = _mm256_mul_epu32(a, b), ahi = _mm256_srli_epi64(a, 32), bhi = _mm256_srli_epi64(b, 32);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(ahi, b), _mm256_mul_epu32(a, bhi)); return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}
static inline __m256i vnl(__m256i &l, __m256i &h) { __m256i a = l, b = h; __m256i n = _mm256_add_epi64(vrotl(_mm256_add_epi64(a, b), 17), a); b = _mm256_xor_si256(b, a); l = _mm256_xor_si256(_mm256_xor_si256(vrotl(a, 49), b), _mm256_slli_epi64(b, 21)); h = vrotl(b, 28); return n; }
// the spread (sd) of one Perlin octave at the point with offsets f (cell fractions) over its 16 random gradients: the
// eight corners are independent, E[g g^T] = diag(0.625, 0.75, 0.625), and the fade weights factor per axis
static inline double perlin_var(double fx, double fy, double fz) {
    auto ax = [](double f, double &W, double &S) { const double t = f * f * f * (f * (f * 6 - 15) + 10), a = (1 - t) * (1 - t), b = t * t; W = a + b; S = a * f * f + b * (f - 1) * (f - 1); };
    double Wx, Sx, Wy, Sy, Wz, Sz; ax(fx, Wx, Sx); ax(fy, Wy, Sy); ax(fz, Wz, Sz);
    return 0.625 * Sx * Wy * Wz + 0.75 * Wx * Sy * Wz + 0.625 * Wx * Wy * Sz;
}
// AVX2, 8 seeds: two independent lane-sets so one's multiply latency hides behind the other's work
static inline int gate_avx8(uint64_t s0, uint64_t lim, uint64_t *wlo, uint64_t *whi) {
    const __m256i XH = _mm256_set1_epi64x(0x6a09e667f3bcc909LL), XL = _mm256_set1_epi64x((long long) 0x9e3779b97f4a7c15ULL), A = _mm256_set1_epi64x((long long) 0xbf58476d1ce4e5b9ULL), B = _mm256_set1_epi64x((long long) 0x94d049bb133111ebULL);
    const __m256i MASK45 = _mm256_set1_epi64x((1LL << 45) - 1), MID = _mm256_set1_epi64x(1LL << 44), LIM = _mm256_set1_epi64x((long long) lim), HI = _mm256_add_epi64(MID, LIM), LO = _mm256_sub_epi64(MID, LIM);
    __m256i l[2], h[2], rl[2], rh[2]; int mask[2] = {15, 15};
    for (int g = 0; g < 2; g++) { const uint64_t b0 = s0 + 4 * g; __m256i s = _mm256_set_epi64x((long long) (b0 + 3), (long long) (b0 + 2), (long long) (b0 + 1), (long long) b0); l[g] = _mm256_xor_si256(s, XH); h[g] = _mm256_add_epi64(l[g], XL); }
    for (int g = 0; g < 2; g++) { l[g] = vmul(_mm256_xor_si256(l[g], _mm256_srli_epi64(l[g], 30)), A); h[g] = vmul(_mm256_xor_si256(h[g], _mm256_srli_epi64(h[g], 30)), A); }
    for (int g = 0; g < 2; g++) { l[g] = vmul(_mm256_xor_si256(l[g], _mm256_srli_epi64(l[g], 27)), B); h[g] = vmul(_mm256_xor_si256(h[g], _mm256_srli_epi64(h[g], 27)), B); }
    for (int g = 0; g < 2; g++) { l[g] = _mm256_xor_si256(l[g], _mm256_srli_epi64(l[g], 31)); h[g] = _mm256_xor_si256(h[g], _mm256_srli_epi64(h[g], 31)); }
    for (int g = 0; g < 2; g++) { __m256i lo = vnl(l[g], h[g]), hi = vnl(l[g], h[g]); _mm256_storeu_si256((__m256i *) (wlo + 4 * g), lo); _mm256_storeu_si256((__m256i *) (whi + 4 * g), hi);
        rl[g] = _mm256_xor_si256(lo, _mm256_set1_epi64x((long long) NOISE[0].lo)); rh[g] = _mm256_xor_si256(hi, _mm256_set1_epi64x((long long) NOISE[0].hi)); }
    for (int half = 0; half < 2; half++) {
        for (int g = 0; g < 2; g++) {
            __m256i a = vnl(rl[g], rh[g]), b = vnl(rl[g], rh[g]); __m256i ol = _mm256_xor_si256(a, _mm256_set1_epi64x((long long) OCT[0].lo)), oh = _mm256_xor_si256(b, _mm256_set1_epi64x((long long) OCT[0].hi));
            vnl(ol, oh); __m256i v = _mm256_and_si256(_mm256_srli_epi64(vnl(ol, oh), 11), MASK45); __m256i bad = _mm256_or_si256(_mm256_cmpgt_epi64(v, HI), _mm256_cmpgt_epi64(LO, v));
            mask[g] &= ~_mm256_movemask_pd(_mm256_castsi256_pd(bad));
        }
        if (!(mask[0] | mask[1])) return 0;
    }
    return mask[0] | mask[1] << 4;
}

// erosion: octave -9 of both stacks first (E's first octave), its sum at the origin against the tail cut E9; then the
// other six octaves as NormalNoise::init would make them (same RNG per octave) and the noise marked ready
static inline bool erosion_staged(Sampler &s, double e9) {
    NormalNoise &N = s.noise[Erosion]; const Config &c = configs[Erosion]; Random r(s.a ^ c.a, s.b ^ c.b);
    uint64_t ra[2], rb[2]; for (int side = 0; side < 2; side++) { ra[side] = r.next(); rb[side] = r.next(); }
    const double w0 = double(1 << (c.len - 1)) / ((1 << c.len) - 1); int lo = 0, hi = c.len - 1; while (!c.amplitude[lo]) lo++; while (!c.amplitude[hi]) hi--;
    const double factor = (5. * (hi - lo + 1)) / (3. * (hi - lo + 2));
    Perlin *p[8]; Random R[8] = {Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0)}; double f[8], w[8]; int m = 0;
    for (int side = 0; side < 2; side++) { const auto &salt = octave_salt[c.first + 10]; p[m] = &N.octave[side][0]; R[m] = Random(ra[side] ^ salt[0], rb[side] ^ salt[1]); f[m] = std::ldexp(1., c.first); w[m] = w0 * c.amplitude[0]; m++; }
    fastnoise::perlin_init_many(p, R, f, w, 2);
    const double v9 = (N.octave[0][0].weight * N.octave[0][0].sample(0, 0, 0) + N.octave[1][0].weight * N.octave[1][0].sample(0, 0, 0)) * factor;
    if (v9 > e9) return false;
    m = 0;
    for (int side = 0; side < 2; side++) { int j = 1; double fq = std::ldexp(1., c.first), wq = w0;
        for (int i = 0; i < c.len; i++, fq *= 2, wq *= .5) if (c.amplitude[i] && i > 0) { const auto &salt = octave_salt[c.first + i + 10]; p[m] = &N.octave[side][j++]; R[m] = Random(ra[side] ^ salt[0], rb[side] ^ salt[1]); f[m] = fq; w[m] = wq * c.amplitude[i]; m++; } else if (c.amplitude[i]) { } }
    fastnoise::perlin_init_many(p, R, f, w, m);
    int n = 0; for (int i = 0; i < c.len; i++) if (c.amplitude[i]) n++;
    N.n = n; N.factor = factor; s.ready[Erosion] = true;
    return true;
}
// ---- block cascade: every test on full vectors of survivors only ----
// pack table: for a 4-bit lane mask, the 32-bit lane indices that move the kept 64-bit lanes to the front
struct PackLUT { alignas(32) int32_t idx[16][8]; PackLUT() { for (int m = 0; m < 16; m++) { int o = 0; for (int k = 0; k < 4; k++) if (m >> k & 1) { idx[m][2 * o] = 2 * k; idx[m][2 * o + 1] = 2 * k + 1; o++; } for (; o < 4; o++) { idx[m][2 * o] = 0; idx[m][2 * o + 1] = 1; } } } };
static const PackLUT LUT;
static inline void pack_store(uint64_t *dst, int at, __m256i v, int m) { _mm256_storeu_si256((__m256i *) (dst + at), _mm256_permutevar8x32_epi32(v, _mm256_load_si256((const __m256i *) LUT.idx[m]))); }
static inline __m256i ld(const uint64_t *p) { return _mm256_loadu_si256((const __m256i *) p); }
// one stack test on 4 lanes: RNG state (rl, rh) -> fork (two draws, the state moves on) -> octave y phase within LIM
static inline int stack_test(__m256i &rl, __m256i &rh, const Hash &oct, __m256i LO, __m256i HI) {
    __m256i a = vnl(rl, rh), b = vnl(rl, rh); __m256i ol = _mm256_xor_si256(a, _mm256_set1_epi64x((long long) oct.lo)), oh = _mm256_xor_si256(b, _mm256_set1_epi64x((long long) oct.hi));
    vnl(ol, oh); const __m256i v = _mm256_and_si256(_mm256_srli_epi64(vnl(ol, oh), 11), _mm256_set1_epi64x((1LL << 45) - 1));
    return ~_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_or_si256(_mm256_cmpgt_epi64(v, HI), _mm256_cmpgt_epi64(LO, v)))) & 15;
}
struct BlockGate {
    static const int B = 1024;
    alignas(32) uint64_t sd[2][B + 16], lo[2][B + 16], hi[2][B + 16], rl[2][B + 16], rh[2][B + 16];
    __m256i LO[3], HI[3];
    void set(const uint64_t *lims) { for (int n = 0; n < 3; n++) { const __m256i MID = _mm256_set1_epi64x(1LL << 44), L = _mm256_set1_epi64x((long long) lims[n]); LO[n] = _mm256_sub_epi64(MID, L); HI[n] = _mm256_add_epi64(MID, L); } }
    // seeds s0 .. s0 + B - 1 (B a multiple of 8): the ones passing all six windows go to out
    void run(uint64_t s0, std::vector<uint64_t> &out) {
        const __m256i XH = _mm256_set1_epi64x(0x6a09e667f3bcc909LL), XL = _mm256_set1_epi64x((long long) 0x9e3779b97f4a7c15ULL), A = _mm256_set1_epi64x((long long) 0xbf58476d1ce4e5b9ULL), Bm = _mm256_set1_epi64x((long long) 0x94d049bb133111ebULL);
        int n = 0;   // stage 0: world state and erosion side 0 (two 4-lane groups interleaved). Index k is seed (k G) ^ X:
        // xSetSeed mixes t = k G and t + G, and t + G is seed k+1's t, so each mix serves two seeds
        const uint64_t G1 = 0x9e3779b97f4a7c15ULL;
        auto mixv = [&](__m256i z) { z = vmul(_mm256_xor_si256(z, _mm256_srli_epi64(z, 30)), A); z = vmul(_mm256_xor_si256(z, _mm256_srli_epi64(z, 27)), Bm); return _mm256_xor_si256(z, _mm256_srli_epi64(z, 31)); };
        auto shift1 = [](__m256i a, __m256i b) { return _mm256_blend_epi32(_mm256_permute4x64_epi64(a, _MM_SHUFFLE(0, 3, 2, 1)), _mm256_permute4x64_epi64(b, 0), 0xC0); };
        const __m256i STEP4 = _mm256_set1_epi64x((long long) (4 * G1));
        __m256i T = _mm256_add_epi64(_mm256_set1_epi64x((long long) (s0 * G1)), _mm256_set_epi64x((long long) (3 * G1), (long long) (2 * G1), (long long) G1, 0));
        __m256i Mprev = mixv(T); (void) XH; (void) XL;
        for (int k = 0; k < B; k += 8) {
            __m256i s[2], l[2], h[2], wl[2], wh[2], r0[2], r1[2]; int m[2];
            const __m256i T1 = _mm256_add_epi64(T, STEP4), T2 = _mm256_add_epi64(T1, STEP4), M1 = mixv(T1), M2 = mixv(T2);
            l[0] = Mprev; h[0] = shift1(Mprev, M1); l[1] = M1; h[1] = shift1(M1, M2); Mprev = M2; T = T2;
            for (int g = 0; g < 2; g++) { const uint64_t b0 = s0 + k + 4 * g; s[g] = _mm256_set_epi64x((long long) (b0 + 3), (long long) (b0 + 2), (long long) (b0 + 1), (long long) b0); }
            for (int g = 0; g < 2; g++) { wl[g] = vnl(l[g], h[g]); wh[g] = vnl(l[g], h[g]); r0[g] = _mm256_xor_si256(wl[g], _mm256_set1_epi64x((long long) NOISE[0].lo)); r1[g] = _mm256_xor_si256(wh[g], _mm256_set1_epi64x((long long) NOISE[0].hi)); }
            for (int g = 0; g < 2; g++) m[g] = stack_test(r0[g], r1[g], OCT[0], LO[0], HI[0]);
            for (int g = 0; g < 2; g++) if (m[g]) { pack_store(sd[0], n, s[g], m[g]); pack_store(lo[0], n, wl[g], m[g]); pack_store(hi[0], n, wh[g], m[g]); pack_store(rl[0], n, r0[g], m[g]); pack_store(rh[0], n, r1[g], m[g]); n += __builtin_popcount(m[g]); }
        }
        // stages 1-5: erosion side 1, humidity sides 0 and 1, temperature sides 0 and 1, each over the packed survivors
        int cur = 0;
        for (int st = 1; st < 6; st++) {
            const int noise = st >> 1, half = st & 1, nxt = cur ^ 1; int m2 = 0;
            for (int k = 0; k < n; k += 8) {
                __m256i S[2], WL[2], WH[2], R0[2], R1[2]; int M[2];
                for (int g = 0; g < 2; g++) { const int kk = k + 4 * g; S[g] = ld(sd[cur] + kk); WL[g] = ld(lo[cur] + kk); WH[g] = ld(hi[cur] + kk);
                    if (half) { R0[g] = ld(rl[cur] + kk); R1[g] = ld(rh[cur] + kk); } else { R0[g] = _mm256_xor_si256(WL[g], _mm256_set1_epi64x((long long) NOISE[noise].lo)); R1[g] = _mm256_xor_si256(WH[g], _mm256_set1_epi64x((long long) NOISE[noise].hi)); } }
                for (int g = 0; g < 2; g++) M[g] = stack_test(R0[g], R1[g], OCT[noise], LO[noise], HI[noise]);
                for (int g = 0; g < 2; g++) {
                    const int kk = k + 4 * g; int m = kk >= n ? 0 : n - kk < 4 ? M[g] & ((1 << (n - kk)) - 1) : M[g];
                    if (!m) continue;
                    if (st == 5) { alignas(32) uint64_t SS[4]; _mm256_store_si256((__m256i *) SS, S[g]); while (m) { const int j = __builtin_ctz(m); m &= m - 1; out.push_back(SS[j]); } continue; }
                    pack_store(sd[nxt], m2, S[g], m); pack_store(lo[nxt], m2, WL[g], m); pack_store(hi[nxt], m2, WH[g], m); pack_store(rl[nxt], m2, R0[g], m); pack_store(rh[nxt], m2, R1[g], m); m2 += __builtin_popcount(m);
                }
            }
            n = m2; cur = nxt; if (!n) break;
        }
    }
};
// the most a Perlin octave can add at a point, from its offsets alone (RNG only, no permutation): each corner adds its
// fade weight times g.d, and |g.d| <= the two largest |d| components for Minecraft's gradients (all weights >= 0)
static inline double perlin_bound(Random r) {
    const double ox = r.uniform() * 256, oy = r.uniform() * 256, oz = r.uniform() * 256;
    const double f[3] = {ox - std::floor(ox), oy - std::floor(oy), oz - std::floor(oz)}, t[3] = {fade(f[0]), fade(f[1]), fade(f[2])};
    double b = 0;
    for (int c = 0; c < 8; c++) {
        double d[3], w = 1; for (int k = 0; k < 3; k++) { const int e = (c >> k) & 1; d[k] = std::fabs(f[k] - e); w *= e ? t[k] : 1 - t[k]; }
        const double m = std::max({d[0] + d[1], d[0] + d[2], d[1] + d[2]}); b += w * m;
    }
    return b;
}
// temperature at quart (0, 0) without the shift. Octave -10 first, side A alone: if even side B at its most (bounded
// from its offsets, RNG only) cannot bring the -10 sum to T10 or below, side B's permutation is never made. Then the
// -10 sum itself must be <= T10 (the tail cut: the patches of 2.5M+ sit at the coldest -10 values, see tail notes),
// and the -10 sum with the -8 octaves at their most must reach THR. Otherwise the noise is finished as NormalNoise::init
// makes it (same RNG order) and marked ready, and the full value is tested against THR.
static inline bool temp_staged(Sampler &s, double thr, double t10, uint64_t *stage) {
    NormalNoise &N = s.noise[Temperature]; const Config &c = configs[Temperature]; Random r(s.a ^ c.a, s.b ^ c.b);
    uint64_t ra[2], rb[2]; for (int side = 0; side < 2; side++) { ra[side] = r.next(); rb[side] = r.next(); }
    const double w = 32. / 63., factor = 1.25;   // len 6: w = 2^5 / (2^6 - 1); amplitudes {1.5, 0, 1}: span 3
    const double cut = std::min(t10, thr);   // (the -8 octaves can also help: the THR test below allows for them)
    N.octave[0][0].init(Random(ra[0] ^ octave_salt[0][0], rb[0] ^ octave_salt[0][1]), std::ldexp(1., -10), w * 1.5);
    const double vA = N.octave[0][0].weight * N.octave[0][0].sample(0, 0, 0) * factor;
    const double bB = factor * w * 1.5 * perlin_bound(Random(ra[1] ^ octave_salt[0][0], rb[1] ^ octave_salt[0][1])) * (1 + 1e-9);
    stage[0]++;
    if (vA - bB > t10) return false;
    N.octave[1][0].init(Random(ra[1] ^ octave_salt[0][0], rb[1] ^ octave_salt[0][1]), std::ldexp(1., -10), w * 1.5);
    const double v10 = (N.octave[0][0].weight * N.octave[0][0].sample(0, 0, 0) + N.octave[1][0].weight * N.octave[1][0].sample(0, 0, 0)) * factor;
    stage[1]++;
    if (v10 > t10) return false;
    double rest = 0; for (int side = 0; side < 2; side++) rest += perlin_bound(Random(ra[side] ^ octave_salt[2][0], rb[side] ^ octave_salt[2][1]));
    if (v10 - factor * w * 0.25 * rest * (1 + 1e-9) > thr) return false;
    stage[2]++; (void) cut;
    for (int side = 0; side < 2; side++) N.octave[side][1].init(Random(ra[side] ^ octave_salt[2][0], rb[side] ^ octave_salt[2][1]), std::ldexp(1., -8), w * 0.25);
    N.n = 2; N.factor = factor; s.ready[Temperature] = true;
    return N.sample(0, 0, 0) <= thr;
}
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: origin START COUNT PREFIX THREADS [key=value ...] (keys: ph phT phH phE tmax t10 tz e9 r32 r16 r8 r4 cand gpu gper ggroups gqueue cpugate)\n"); return 2; }
    const uint64_t start = std::stoull(argv[1]), count = std::stoull(argv[2]); const std::string prefix = argv[3]; const int threads = std::stoi(argv[4]);
    double ph = -1, phT = -1, phH = -1, phE = -1, tmax = -0.5, t10 = -0.62, tz = 2.0, r32 = 0.7, e9 = -0.25, gpu = 1, gper = 256, ggroups = 4096, gqueue = 200000, cpugate = 1, r16 = 1.5, r8 = 2.5, r4 = 3.6, cand = 4.6;
    for (int i = 5; i < argc; i++) { std::string kv = argv[i]; auto e = kv.find('='); std::string k = kv.substr(0, e); double v = std::stod(kv.substr(e + 1));
        if (k == "ph") ph = v; else if (k == "t10") t10 = v; else if (k == "tz") tz = v; else if (k == "r32") r32 = v; else if (k == "e9") e9 = v; else if (k == "gpu") gpu = v; else if (k == "gper") gper = v; else if (k == "ggroups") ggroups = v; else if (k == "gqueue") gqueue = v; else if (k == "cpugate") cpugate = v; else if (k == "phT") phT = v; else if (k == "phH") phH = v; else if (k == "phE") phE = v; else if (k == "tmax") tmax = v; else if (k == "r16") r16 = v; else if (k == "r8") r8 = v; else if (k == "r4") r4 = v; else if (k == "cand") cand = v; else { fprintf(stderr, "unknown key %s\n", k.c_str()); return 2; } }
    if (ph >= 0) { if (phT < 0) phT = ph; if (phH < 0) phH = ph; if (phE < 0) phE = ph; }
    if (phT < 0) phT = 0.24;
    if (phH < 0) phH = 0.22;
    if (phE < 0) phE = 0.22;
    fprintf(stderr, "gate: erosion -9 within %.3f, humidity -8 within %.3f, temperature -10 within %.3f of mid-cell (both stacks): %.3g of the seeds\n", phE, phH, phT, pow(2 * phE, 2) * pow(2 * phH, 2) * pow(2 * phT, 2));
    FILE *out = fopen((prefix + ".csv").c_str(), "a"); if (!out) { fprintf(stderr, "cannot open %s.csv\n", prefix.c_str()); return 2; }
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
    static SurfaceGrid grid; std::atomic<uint64_t> next(0), gated(0), hits(0), floods(0), sA(0), sB(0), s8(0); std::mutex mu; double best = 0; const uint64_t CHUNK = 1 << 16;
    // cycles spent per stage (all threads): gate, origin test, lattice 16, 8, 4, 1
    std::atomic<uint64_t> cyc[6] = {};
    auto t0 = Clock::now(), last = t0;
    std::vector<std::thread> pool;
    // GPU feeder: blocks of GGROUPS x 64 x GPER indices from the same counter as the CPU chunks; gated indices queued
    // (no new dispatch while the queue holds GQUEUE or more, so a GPU that outruns the CPU leaves it the power)
    std::mutex qmu; std::deque<uint64_t> gq; std::atomic<bool> feeding(false); std::atomic<uint64_t> gpuSeeds(0); std::thread feeder;
    const uint64_t glims[3] = {(uint64_t) (phE * (double) (1ULL << 45)), (uint64_t) (phH * (double) (1ULL << 45)), (uint64_t) (phT * (double) (1ULL << 45))};
#ifndef NO_VULKAN
    if (gpu > 0) {
        static VkGate vg;
        try { vg.init(std::string(argv[0]).substr(0, std::string(argv[0]).rfind('/') + 1) + "gate.spv", 1u << 22); fprintf(stderr, "GPU gate on %s\n", vg.name.c_str()); feeding = true; }
        catch (const std::exception &e) { fprintf(stderr, "no GPU gate (%s): CPU only\n", e.what()); }
        if (feeding) feeder = std::thread([&] {
            const uint64_t blk = (uint64_t) ggroups * 64 * (uint64_t) gper; std::vector<uint64_t> got; got.reserve(1 << 20);
            while (!stop_) {
                { std::lock_guard<std::mutex> l(qmu); if (gq.size() >= (size_t) gqueue) { } }
                size_t qs; { std::lock_guard<std::mutex> l(qmu); qs = gq.size(); }
                if (qs >= (size_t) gqueue) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
                const uint64_t base = next.fetch_add(blk); if (count && base >= count) break;
                got.clear(); if (!vg.run(start + base, (uint32_t) ggroups, (uint32_t) gper, glims, got)) fprintf(stderr, "GPU output overflow\n");
                std::sort(got.begin(), got.end()); gpuSeeds += blk;
                std::lock_guard<std::mutex> l(qmu); for (const uint64_t idx : got) if (!count || idx - start < count) gq.push_back(idx);
            }
            feeding = false;
        });
    }
#else
    if (gpu > 0) fprintf(stderr, "built without Vulkan: CPU gate only\n");
#endif
    for (int t = 0; t < threads; t++) pool.emplace_back([&] {
        std::vector<uint8_t> st; std::vector<int> stack;
        // gated seeds wait in PEND until BATCH of them: their temperature -10 octaves (side A, then side B of the
        // survivors) are built together in vector lanes (fastnoise), then each seed goes on alone
        constexpr int BATCH = 16; uint64_t pend[BATCH]; int np = 0; static thread_local Perlin PS[2][BATCH];
        uint64_t c[6] = {0, 0, 0, 0, 0, 0}, tdown = 0, nh = 0, stg[3] = {0, 0, 0};
        const Config &cT = configs[Temperature]; const double W = 32. / 63., FACTOR = 1.25;
        auto run_seed = [&](uint64_t seed, int j, const uint64_t *ra, const uint64_t *rb) {
            uint64_t ta = __rdtsc();
            auto reject = [&] { const uint64_t d = __rdtsc() - ta; c[1] += d; tdown += d; };
            if ((PS[0][j].weight * PS[0][j].sample(0, 0, 0) + PS[1][j].weight * PS[1][j].sample(0, 0, 0)) * FACTOR > t10) { stg[1]++; return reject(); }
            SurfaceSampler S(seed, grid); State ss{S};
            // temperature: the -10 sum against the tail cut, then the -8 octaves (bounded, then built) against THR
            NormalNoise &N = S.s.noise[Temperature]; N.octave[0][0] = PS[0][j]; N.octave[1][0] = PS[1][j];
            const double v10 = (N.octave[0][0].weight * N.octave[0][0].sample(0, 0, 0) + N.octave[1][0].weight * N.octave[1][0].sample(0, 0, 0)) * FACTOR;
            stg[1]++; if (v10 > t10) return reject();
            double rest = 0; for (int side = 0; side < 2; side++) rest += perlin_bound(Random(ra[side] ^ octave_salt[2][0], rb[side] ^ octave_salt[2][1]));
            const double thr = tmax + 0.025;
            if (v10 - FACTOR * W * 0.25 * rest * (1 + 1e-9) > thr) return reject();
            stg[2]++;
            if (!erosion_staged(S.s, e9)) return reject();   // the second tail cut
            if (fastnoise::get(S.s, Humidity).sample(0, 0, 0) > -0.1 + 0.06) return reject();   // (humidity first: it drops 56% for 4 octaves)
            for (int side = 0; side < 2; side++) N.octave[side][1].init(Random(ra[side] ^ octave_salt[2][0], rb[side] ^ octave_salt[2][1]), std::ldexp(1., -8), W * 0.25);
            N.n = 2; N.factor = FACTOR; S.s.ready[Temperature] = true;
            // pre-stages: each value at the origin without the shift noise (which moves it by at most 0.017 for T,
            // 0.05 for H and E, 0.11 for C over 200,000 seeds); they only drop seeds the exact test
            // below would drop too
            if (N.sample(0, 0, 0) > thr || fastnoise::get(S.s, Humidity).sample(0, 0, 0) > -0.1 + 0.06 ||
                fastnoise::get(S.s, Erosion).sample(0, 0, 0) > -0.375 + 0.06 || fastnoise::get(S.s, Continentalness).sample(0, 0, 0) < -0.19 - 0.13) return reject();
            fastnoise::ready_all(S.s);
            if (ss(0, 0) != 2) return reject();
            int64_t q[6]; S.params(0, 0, q); if (q[0] > int64_t(10000 * tmax)) return reject();
            nh++; { uint64_t d = __rdtsc() - ta; c[1] += d; tdown += d; } ta = __rdtsc();
            auto f = [&](int x, int z) { return ss(x, z); };
            // lattice 32, 8-connected (bridges the necks a coarse 4-connected lattice cuts): from R32 the lattice of 16
            double a16 = dense_flood_bridged(f, 0, 0, 32, 1500 / 32, st, stack, true).cells * 1024 * 16 / 1e6, a8 = 0, a4 = 0, a1 = 0;
            if (a16 >= r32) a16 = dense_flood_bridged(f, 0, 0, 16, 1500 / 16, st, stack).cells * 256 * 16 / 1e6;
            { uint64_t t = __rdtsc(); c[2] += t - ta; tdown += t - ta; ta = t; }
            if (a16 >= r16) a8 = dense_flood_bridged(f, 0, 0, 8, 1500 / 8, st, stack).cells * 64 * 16 / 1e6; else a8 = a16;
            { uint64_t t = __rdtsc(); c[3] += t - ta; tdown += t - ta; ta = t; }
            if (a8 >= r8) { a4 = dense_flood_bridged(f, 0, 0, 4, 1500 / 4, st, stack).cells * 16 * 16 / 1e6; floods++; }
            { uint64_t t = __rdtsc(); c[4] += t - ta; tdown += t - ta; ta = t; }
            if (a4 >= r4) a1 = dense_flood_bridged(f, 0, 0, 1, 1500, st, stack).cells * 16 / 1e6;
            { uint64_t t = __rdtsc(); c[5] += t - ta; tdown += t - ta; ta = t; }
            std::lock_guard<std::mutex> l(mu);
            fprintf(out, "%lld,%.4f,%.4f,%.6f,%.4f\n", (long long) seed, a8, a4, a1, a16);
            if (a1 > 0) { fflush(out); if (a1 >= cand) fprintf(stderr, "CANDIDATE %lld %.0f (step 8: %.3fM, step 4: %.3fM)\n", (long long) seed, a1 * 1e6, a8, a4); }
            if (a1 > best) best = a1;
        };
        auto run_batch = [&] {
            const uint64_t ta = __rdtsc();
            uint64_t ra[BATCH][2], rb[BATCH][2]; int first[BATCH]; double sd[BATCH][2];
            Random R1[BATCH] = {Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0)}, R2[BATCH] = {Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0)};
            Perlin *p1[BATCH], *p2[BATCH]; double f[BATCH], w[BATCH];
            for (int j = 0; j < np; j++) {
                Random world(pend[j]); const uint64_t a = world.next(), b = world.next(); Random r(a ^ cT.a, b ^ cT.b);
                for (int side = 0; side < 2; side++) { ra[j][side] = r.next(); rb[j][side] = r.next();
                    Random o(ra[j][side] ^ octave_salt[0][0], rb[j][side] ^ octave_salt[0][1]); double fr[3]; for (int k = 0; k < 3; k++) { const double u = o.uniform() * 256; fr[k] = u - std::floor(u); }
                    sd[j][side] = FACTOR * W * 1.5 * std::sqrt(perlin_var(fr[0], fr[1], fr[2])); }
                first[j] = sd[j][0] >= sd[j][1] ? 0 : 1; const int s1 = first[j];
                R1[j] = Random(ra[j][s1] ^ octave_salt[0][0], rb[j][s1] ^ octave_salt[0][1]); p1[j] = &PS[s1][j]; f[j] = std::ldexp(1., -10); w[j] = W * 1.5;
            }
            fastnoise::perlin_init_many(p1, R1, f, w, np);
            int keep[BATCH], nk = 0;
            for (int j = 0; j < np; j++) {
                const int s1 = first[j], s2 = 1 - s1;
                const double v1 = PS[s1][j].weight * PS[s1][j].sample(0, 0, 0) * FACTOR;
                const double b2 = FACTOR * W * 1.5 * perlin_bound(Random(ra[j][s2] ^ octave_salt[0][0], rb[j][s2] ^ octave_salt[0][1])) * (1 + 1e-9);
                stg[0]++; if (v1 - std::min(b2, tz * sd[j][s2]) > t10) continue;
                R2[nk] = Random(ra[j][s2] ^ octave_salt[0][0], rb[j][s2] ^ octave_salt[0][1]); p2[nk] = &PS[s2][j]; keep[nk++] = j;
            }
            fastnoise::perlin_init_many(p2, R2, f, w, nk);
            { const uint64_t d = __rdtsc() - ta; c[1] += d; tdown += d; }
            for (int i = 0; i < nk; i++) run_seed(pend[keep[i]], keep[i], ra[keep[i]], rb[keep[i]]);
            np = 0;
        };
        static thread_local BlockGate G; std::vector<uint64_t> gl; gl.reserve(4096);
        const uint64_t lims[3] = {(uint64_t) (phE * (double) (1ULL << 45)), (uint64_t) (phH * (double) (1ULL << 45)), (uint64_t) (phT * (double) (1ULL << 45))}; G.set(lims);
        for (;;) {
            uint64_t ng = 0; const uint64_t tg = __rdtsc();
            // gated indices from the GPU first
            uint64_t take[256]; int nt = 0;
            { std::lock_guard<std::mutex> l(qmu); while (nt < 256 && !gq.empty()) { take[nt++] = gq.front(); gq.pop_front(); } }
            uint64_t base = 0; bool cpu = false;
            if (nt == 0 && cpugate <= 0 && feeding) { std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }   // GPU-only gating: wait for it
            if (nt == 0) {
                if (stop_) break;
                base = next.fetch_add(CHUNK);
                if (count && base >= count) { if (feeding) { next -= 0; std::this_thread::sleep_for(std::chrono::milliseconds(1)); bool empty; { std::lock_guard<std::mutex> l(qmu); empty = gq.empty(); } if (!empty || feeding) continue; } break; }
                cpu = true;
            }
            for (int i = 0; i < nt; i++) { pend[np++] = (take[i] * 0x9e3779b97f4a7c15ULL) ^ 0x6a09e667f3bcc909ULL; ng++; if (np == BATCH) run_batch(); }
            for (uint64_t k0 = base; cpu && k0 < base + CHUNK && (count == 0 || k0 < count); k0 += BlockGate::B) {
              gl.clear(); G.run(start + k0, gl);
              for (const uint64_t idx : gl) {
                if (count && idx - start >= count) continue;
                pend[np++] = (idx * 0x9e3779b97f4a7c15ULL) ^ 0x6a09e667f3bcc909ULL; ng++;
                if (np == BATCH) run_batch();
              }
            }
            if (np) run_batch();
            gated += ng; hits += nh; sA += stg[0]; sB += stg[1]; s8 += stg[2]; c[0] = __rdtsc() - tg - tdown; for (int i = 0; i < 6; i++) cyc[i] += c[i];
            nh = 0; tdown = 0; for (int i = 0; i < 6; i++) c[i] = 0; for (int i = 0; i < 3; i++) stg[i] = 0;
            std::lock_guard<std::mutex> l(mu);
            auto now = Clock::now();
            if (std::chrono::duration<double>(now - last).count() >= 60) {
                last = now; const double s = std::chrono::duration<double>(now - t0).count(); const uint64_t done = std::min<uint64_t>(next.load(), count ? count : ~0ULL);
                double tot = 0; for (int i = 0; i < 6; i++) tot += cyc[i].load();
                fprintf(stderr, "%.0f s: %.3e seeds (%.3e/s), %llu gated (side B built %.3f, octave -8 %.3f), %llu hits, %llu step-4 floods, best exact %.3fM; time: gate %.0f%%, origin %.0f%%, lattice 16 %.0f%%, 8 %.0f%%, 4 %.0f%%, exact %.0f%%; next index %llu\n", s, (double) done, done / s, (unsigned long long) gated.load(), (double) sB.load() / std::max<uint64_t>(1, sA.load()), (double) s8.load() / std::max<uint64_t>(1, sA.load()), (unsigned long long) hits.load(), (unsigned long long) floods.load(), best,
                        100 * cyc[0] / tot, 100 * cyc[1] / tot, 100 * cyc[2] / tot, 100 * cyc[3] / tot, 100 * cyc[4] / tot, 100 * cyc[5] / tot, (unsigned long long) (start + done));
                fflush(out);
            }
        }
    });
    for (auto &t : pool) t.join();
    stop_ = 1; if (feeder.joinable()) feeder.join();
    fprintf(stderr, "GPU gated %.3e of the seeds\n", (double) gpuSeeds.load());
    const double s = std::chrono::duration<double>(Clock::now() - t0).count(); const uint64_t done = std::min<uint64_t>(next.load(), count ? count : ~0ULL);
    fprintf(stderr, "done: %.3e seeds in %.0f s, %llu gated, %llu hits, best exact %.3fM; next index %llu\n", (double) done, s, (unsigned long long) gated.load(), (unsigned long long) hits.load(), best, (unsigned long long) (start + done));
    fclose(out);
}
