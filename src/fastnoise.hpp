// fastnoise.hpp: NormalNoise::init with the Perlin permutations' RNG run in vector lanes (3 or 4 octaves a step
// with AVX2: 1.44x on a P-core, 1.18x on an E-core); the swaps stay scalar. Same draws in the same order per octave, so the same noise (tools/fastcheck.cpp).
#pragma once
#include "standalone.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
namespace fastnoise {
using namespace standalone;
struct V4 { using T = __m256i; static constexpr int W = 4;
    static T load(const uint64_t *p) { return _mm256_loadu_si256((const __m256i *) p); } static void store(uint64_t *p, T v) { _mm256_storeu_si256((__m256i *) p, v); }
    static T add(T a, T b) { return _mm256_add_epi64(a, b); } static T x(T a, T b) { return _mm256_xor_si256(a, b); } static T band(T a, T b) { return _mm256_and_si256(a, b); }
    template <int k> static T shl(T a) { return _mm256_slli_epi64(a, k); } template <int k> static T shr(T a) { return _mm256_srli_epi64(a, k); }
    template <int k> static T rotl(T a) { return _mm256_or_si256(_mm256_slli_epi64(a, k), _mm256_srli_epi64(a, 64 - k)); }
    static T mul32(T a, T b) { return _mm256_mul_epu32(a, b); } static T set1(uint64_t v) { return _mm256_set1_epi64x((long long) v); }
    static int lt(T a, T b) { return _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(b, a))); } };
struct V2 { using T = __m128i; static constexpr int W = 2;
    static T load(const uint64_t *p) { return _mm_loadu_si128((const __m128i *) p); } static void store(uint64_t *p, T v) { _mm_storeu_si128((__m128i *) p, v); }
    static T add(T a, T b) { return _mm_add_epi64(a, b); } static T x(T a, T b) { return _mm_xor_si128(a, b); } static T band(T a, T b) { return _mm_and_si128(a, b); }
    template <int k> static T shl(T a) { return _mm_slli_epi64(a, k); } template <int k> static T shr(T a) { return _mm_srli_epi64(a, k); }
    template <int k> static T rotl(T a) { return _mm_or_si128(_mm_slli_epi64(a, k), _mm_srli_epi64(a, 64 - k)); }
    static T mul32(T a, T b) { return _mm_mul_epu32(a, b); } static T set1(uint64_t v) { return _mm_set1_epi64x((long long) v); }
    static int lt(T a, T b) { return _mm_movemask_pd(_mm_castsi128_pd(_mm_cmpgt_epi64(b, a))); } };
static inline void perlin_head(Perlin *p, Random &r, double f, double w) {
    p->ox = r.uniform() * 256; p->oy = r.uniform() * 256; p->oz = r.uniform() * 256; p->frequency = f; p->weight = w;
    for (int i = 0; i < 256; i++) p->permutation[i] = (uint8_t) i;
}
static inline void perlin_tail(Perlin *p) { p->iy = int(std::floor(p->oy)); p->fy = p->oy - std::floor(p->oy); p->vy = fade(p->fy); }
// K <= V::W octaves from their RNGs r (taken by value); lanes past K run on a dummy state and are ignored. A lane whose
// draw could meet Java's rejection (low 32 bits of draw * n below n: about 1 octave in 130,000) is redone by Perlin::init.
template <class V, int K> static inline void perlin_init_v(Perlin *const *p, const Random *r, const double *f, const double *w) {
    alignas(32) uint64_t A[V::W], B[V::W];
    for (int k = 0; k < V::W; k++) { if (k < K) { Random q = r[k]; perlin_head(p[k], q, f[k], w[k]); A[k] = q.a; B[k] = q.b; } else { A[k] = 0x9e3779b97f4a7c15ULL; B[k] = 1; } }
    typename V::T a = V::load(A), b = V::load(B); const typename V::T LOW = V::set1(0xffffffffULL); int odd = 0;
    uint8_t *pm[K]; for (int k = 0; k < K; k++) pm[k] = p[k]->permutation.data();
    for (int i = 0; i < 256; i++) {
        const typename V::T n = V::set1(256 - i);
        const typename V::T out = V::add(V::template rotl<17>(V::add(a, b)), a);
        b = V::x(b, a); a = V::x(V::x(V::template rotl<49>(a), b), V::template shl<21>(b)); b = V::template rotl<28>(b);
        const typename V::T m = V::mul32(out, n);
        odd |= V::lt(V::band(m, LOW), n);
        alignas(32) uint64_t J[V::W]; V::store(J, V::template shr<32>(m));
        for (int k = 0; k < K; k++) { const uint32_t j = i + (uint32_t) J[k]; const uint8_t t = pm[k][i]; pm[k][i] = pm[k][j]; pm[k][j] = t; }
    }
    for (int k = 0; k < K; k++) { if (odd >> k & 1) p[k]->init(r[k], f[k], w[k]); else perlin_tail(p[k]); }
}

// 8 octaves: two 4-lane vectors stepped in the same loop (more independent work per step for a wide core)
static inline void perlin_init_v8(Perlin *const *p, const Random *r, const double *f, const double *w) {
    alignas(32) uint64_t A[8], B[8];
    for (int k = 0; k < 8; k++) { Random q = r[k]; perlin_head(p[k], q, f[k], w[k]); A[k] = q.a; B[k] = q.b; }
    __m256i a0 = V4::load(A), b0 = V4::load(B), a1 = V4::load(A + 4), b1 = V4::load(B + 4); const __m256i LOW = V4::set1(0xffffffffULL); int odd = 0;
    uint8_t *pm[8]; for (int k = 0; k < 8; k++) pm[k] = p[k]->permutation.data();
    for (int i = 0; i < 256; i++) {
        const __m256i n = V4::set1(256 - i);
        const __m256i o0 = V4::add(V4::rotl<17>(V4::add(a0, b0)), a0), o1 = V4::add(V4::rotl<17>(V4::add(a1, b1)), a1);
        b0 = V4::x(b0, a0); a0 = V4::x(V4::x(V4::rotl<49>(a0), b0), V4::shl<21>(b0)); b0 = V4::rotl<28>(b0);
        b1 = V4::x(b1, a1); a1 = V4::x(V4::x(V4::rotl<49>(a1), b1), V4::shl<21>(b1)); b1 = V4::rotl<28>(b1);
        const __m256i m0 = V4::mul32(o0, n), m1 = V4::mul32(o1, n);
        odd |= V4::lt(V4::band(m0, LOW), n) | V4::lt(V4::band(m1, LOW), n) << 4;
        alignas(32) uint64_t J[8]; V4::store(J, V4::shr<32>(m0)); V4::store(J + 4, V4::shr<32>(m1));
        for (int k = 0; k < 8; k++) { const uint32_t j = i + (uint32_t) J[k]; const uint8_t t = pm[k][i]; pm[k][i] = pm[k][j]; pm[k][j] = t; }
    }
    for (int k = 0; k < 8; k++) { if (odd >> k & 1) p[k]->init(r[k], f[k], w[k]); else perlin_tail(p[k]); }
}
inline bool use8 = true;
static inline void perlin_init_many(Perlin *const *p, const Random *r, const double *f, const double *w, int m) {
    for (int j = 0; j < m;) {
        if (m - j >= 8 && use8) { perlin_init_v8(p + j, r + j, f + j, w + j); j += 8; continue; }
        const int k = std::min(4, m - j);
        if (k == 4) perlin_init_v<V4, 4>(p + j, r + j, f + j, w + j); else if (k == 3) perlin_init_v<V4, 3>(p + j, r + j, f + j, w + j);
        else { for (int q = 0; q < k; q++) p[j + q]->init(r[j + q], f[j + q], w[j + q]); }   // (2 lanes of SSE gained nothing)
        j += k;
    }
}
static inline void noise_init(NormalNoise &N, uint64_t a, uint64_t b, Channel channel) {
    const auto &c = configs[channel]; Random r(a ^ c.a, b ^ c.b);
    Perlin *ps[18]; Random rs[18] = {Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0), Random(0, 0)};
    double fs[18], ws[18]; int m = 0;
    for (int side = 0; side < 2; side++) {
        const uint64_t ra = r.next(), rb = r.next(); N.n = 0;
        double f = std::ldexp(1., c.first), w = double(1 << (c.len - 1)) / ((1 << c.len) - 1);
        for (int i = 0; i < c.len; i++, f *= 2, w *= .5) if (c.amplitude[i]) {
            const auto &salt = octave_salt[c.first + i + 10];
            ps[m] = &N.octave[side][N.n++]; rs[m] = Random(ra ^ salt[0], rb ^ salt[1]); fs[m] = f; ws[m] = w * c.amplitude[i]; m++;
        }
    }
    perlin_init_many(ps, rs, fs, ws, m);
    int lo = 0, hi = c.len - 1; while (!c.amplitude[lo]) lo++; while (!c.amplitude[hi]) hi--;
    const int span = hi - lo + 1; N.factor = (5. * span) / (3. * (span + 1));
}
static inline const NormalNoise &get(Sampler &s, Channel c) { if (!s.ready[c]) { noise_init(s.noise[c], s.a, s.b, c); s.ready[c] = true; } return s.noise[c]; }
static inline void ready_all(Sampler &s) { for (int c = 0; c < 6; c++) get(s, Channel(c)); }
}
