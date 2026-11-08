// anywhere.cpp: the largest Snowy Slopes anywhere in the world (default world type, y 256, 1:4 cells, 4-connected,
// bridged as MC-Measure: see snowy.hpp). Seeds START + k, k < COUNT (COUNT 0: no end). Per seed:
//   1. phase gate (RNG only, 8 seeds at a time): y-phases of erosion -9, humidity -8, temperature -10, both stacks,
//      within PHE, PHH, PHT of mid-cell. Narrow: a world costs ~0.1 s after the gate, the gate ~10 ns a seed.
//   2. a coarse linear climate score (COARSE terms: octave, coefficient; the broad octaves only) tabulated on each noise
//      stack over one period: stack A of every octave repeats every P = 262144 quarts, tabulated at R; stack B samples
//      x * 337/331, so it repeats every P * 331/337 in x and is tabulated in its own coordinate u = x * 337/331 at RB.
//   3. world scan (|x|, |z| <= 7.49M quarts): the A cells in A's top FA fraction, each at its ~57^2 world copies with
//      B looked up there (nearest RB point). The best N1 places, one per BUCKET^2 quarts, go on.
//   4. the fine score (FINE terms: the logistic fit of 1.5M+ patches against random points on nine low octaves, both
//      stacks summed exactly) at the best of a 5x5 grid (step 64) around each place; the best SITES places go on.
//   5. per site: a snowy slopes cell on a 9x9 probe (step 16) around it, then the bridged patch on a lattice of 16,
//      from R16 on 8, from R8 on 4, from R4 exact (lattice 1).
// Output: PREFIX.csv (seed,x,z,coarse,fine,a16,a8,a4,exact: x z in quarts, areas in M blocks2; one line per site with
// a patch), PREFIX.worlds.csv (seed,best coarse,best fine,best a16,best exact), CANDIDATE lines on stderr for exact
// >= CAND (M) with block coordinates, progress every 60 s.
// usage: anywhere START COUNT PREFIX THREADS [key=value ...]   (see Params)
#include "snowy.hpp"
#include "fastnoise.hpp"
#include <immintrin.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <climits>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace snowy;
using Clock = std::chrono::steady_clock;
static volatile sig_atomic_t stop_ = 0;
static void on_signal(int) { stop_ = 1; }

// ---- phase gate (as origin.cpp's CPU gate, seeds START + k) ----
struct Hash { uint64_t lo, hi; };
static const Hash NOISE[3] = {{0xd02491e6058f6fd8ULL, 0x4792512c94c17a80ULL}, {0x81bb4d22e8dc168eULL, 0xf1c8b4bea16303cdULL}, {0x5c7e6b29735f0d7fULL, 0xf7d86f1bbc734988ULL}};
static const Hash OCT[3] = {{0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL}, {0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL}, {0x36d326eed40efeb2ULL, 0x5be9ce18223c636aULL}};
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t nl(uint64_t &l, uint64_t &h) { const uint64_t a = l; uint64_t b = h; const uint64_t n = rotl(a + b, 17) + a; b ^= a; l = rotl(a, 49) ^ b ^ (b << 21); h = rotl(b, 28); return n; }
static inline __m256i vrotl(__m256i x, int k) { return _mm256_or_si256(_mm256_slli_epi64(x, k), _mm256_srli_epi64(x, 64 - k)); }
static inline __m256i vmul(__m256i a, __m256i b) {
    __m256i lo = _mm256_mul_epu32(a, b), ahi = _mm256_srli_epi64(a, 32), bhi = _mm256_srli_epi64(b, 32);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(ahi, b), _mm256_mul_epu32(a, bhi)); return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}
static inline __m256i vnl(__m256i &l, __m256i &h) { __m256i a = l, b = h; __m256i n = _mm256_add_epi64(vrotl(_mm256_add_epi64(a, b), 17), a); b = _mm256_xor_si256(b, a); l = _mm256_xor_si256(_mm256_xor_si256(vrotl(a, 49), b), _mm256_slli_epi64(b, 21)); h = vrotl(b, 28); return n; }
static inline bool gate_rest(uint64_t lo, uint64_t hi, const uint64_t *lims) {
    for (int n = 1; n < 3; n++) { const uint64_t lim = lims[n]; uint64_t rl = lo ^ NOISE[n].lo, rh = hi ^ NOISE[n].hi;
        for (int half = 0; half < 2; half++) { const uint64_t a = nl(rl, rh), b = nl(rl, rh); uint64_t ol = a ^ OCT[n].lo, oh = b ^ OCT[n].hi;
            nl(ol, oh); const uint64_t v = (nl(ol, oh) >> 11) & ((1ULL << 45) - 1); const uint64_t d = v > (1ULL << 44) ? v - (1ULL << 44) : (1ULL << 44) - v; if (d > lim) return false; } }
    return true;
}
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

// ---- parameters ----
struct Term { Channel c; int octave; double coef; };
struct Params {
    double phE = 0.05, phH = 0.06, phT = 0.08;
    int R = 256, RB = 128;       // table resolutions, quarts (divide 262144)
    double FA = 0.01;            // fraction of A's best cells enumerated
    int N1 = 512, bucket = 1024, sites = 96;
    double r16 = 2.5, r8 = 3.8, r4 = 4.5, cand = 5.0;   // lattice 8 >= 0.74 x lattice 16, 4 >= 0.94 x 8, exact >= 0.94 x 4 (origin-search data)
    std::vector<Term> coarse, fine;
};
static const int P = 262144;
static const double S = 337.0 / 331.0, WORLD = 7490000;   // quarts
static std::vector<Term> parse_terms(const std::string &text) {
    std::vector<Term> out;
    for (size_t from = 0; from < text.size();) {
        size_t to = text.find(',', from); if (to == std::string::npos) to = text.size();
        char c; int oct; double coef;
        if (sscanf(text.substr(from, to - from).c_str(), "%c%d:%lf", &c, &oct, &coef) == 3)
            out.push_back({c == 'T' ? Temperature : c == 'H' ? Humidity : c == 'E' ? Erosion : c == 'C' ? Continentalness : Weirdness, oct, coef});
        from = to + 1;
    }
    return out;
}
struct Oct { const Perlin *o; double k; };
static std::vector<Oct> octaves(const Sampler &s, const std::vector<Term> &terms, int side) {
    std::vector<Oct> octs;
    for (const Term &t : terms) {
        const NormalNoise &N = s.noise[t.c]; const Config &cf = configs[t.c]; int j = 0;
        for (int i = 0; i < cf.len; i++) if (cf.amplitude[i]) { if (cf.first + i == t.octave) octs.push_back({&N.octave[side][j], t.coef * N.factor * N.octave[side][j].weight}); j++; }
    }
    return octs;
}
// one octave (y = 0) on a regular grid, added times K into v: per row and lattice cell, the y and z interpolation is folded
// into two lines in x (G dx + K per x corner), so a sample costs a fade and two lerps. Same values as Perlin::sample up to
// rounding (tablecheck).
static void add_octave_row(const Perlin &o, double k, int res, int n, int j, double *row) {
    static constexpr double gx[16] = {1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, 0, -1, 0};
    static constexpr double gy[16] = {1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1};
    static constexpr double gz[16] = {0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 1, 0, -1};
    const double step = res * o.frequency, dy = o.fy, ty = o.vy; const int jy = o.iy; const uint8_t *pm = o.permutation.data();
    const double z = j * step + o.oz, bz = std::floor(z), fz = z - bz, tz = fade(fz); const int iz = (int) bz;
    const double w[2][2] = {{(1 - ty) * (1 - tz), (1 - ty) * tz}, {ty * (1 - tz), ty * tz}};   // [ey][ez]
    int cell = INT32_MIN; double G[2] = {0, 0}, K[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        const double x = i * step + o.ox, bx = std::floor(x), dx = x - bx; const int ix = (int) bx;
        if (ix != cell) { cell = ix;
            for (int ex = 0; ex < 2; ex++) { const int a = pm[(ix + ex) & 255]; G[ex] = K[ex] = 0;
                for (int ey = 0; ey < 2; ey++) { const int b = pm[(a + jy + ey) & 255];
                    for (int ez = 0; ez < 2; ez++) { const int c = pm[(b + iz + ez) & 255] & 15; G[ex] += w[ey][ez] * gx[c]; K[ex] += w[ey][ez] * (gy[c] * (dy - ey) + gz[c] * (fz - ez)); } } } }
        const double l0 = G[0] * dx + K[0], l1 = G[1] * (dx - 1) + K[1];
        row[i] += k * (l0 + fade(dx) * (l1 - l0));
    }
}
// one stack's score over a period at RES: v[j * n + i] at (i RES, j RES) in that stack's own coordinate (each row summed
// in a small buffer and written once: the tables are 4-17 MB a thread, so memory traffic is what costs)
static void build_table(const Sampler &s, const std::vector<Term> &terms, int side, int res, std::vector<float> &v) {
    const int n = P / res; v.resize(size_t(n) * n); const auto octs = octaves(s, terms, side); std::vector<double> row(n);
    for (int j = 0; j < n; j++) {
        std::fill(row.begin(), row.end(), 0.0);
        for (const Oct &o : octs) add_octave_row(*o.o, o.k, res, n, j, row.data());
        float *dst = v.data() + size_t(j) * n; for (int i = 0; i < n; i++) dst[i] = (float) row[i];
    }
}
// the score summed exactly at world quart (x, z), both stacks
static inline double point_score(const std::vector<Oct> &a, const std::vector<Oct> &b, double x, double z) {
    double v = 0;
    for (const Oct &o : a) v += o.k * o.o->sample(x * o.o->frequency, 0, z * o.o->frequency);
    for (const Oct &o : b) v += o.k * o.o->sample(x * S * o.o->frequency, 0, z * S * o.o->frequency);
    return v;
}
struct Site { float score; int x, z; float fine; int fx, fz; };
static inline int64_t floordiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b) && ((a < 0) != (b < 0))) q--; return q; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: anywhere START COUNT PREFIX THREADS [phE= phH= phT= R= RB= FA= N1= bucket= sites= r16= r8= r4= cand= coarse=T-10:-10,... fine=... tablecheck=N]\n"); return 2; }
    const uint64_t start = std::stoull(argv[1]), count = std::stoull(argv[2]); const std::string prefix = argv[3]; const int threads = std::stoi(argv[4]);
    int tablecheck = 0; Params p;
    std::string coarse = "T-10:-10,E-9:-11.9,C-9:5.1", fine = "T-10:-10,T-8:-6.3,H-8:-8.9,H-7:-6,E-9:-11.9,E-8:-7.5,C-9:5.1,C-8:3.2,C-7:1.9";
    for (int i = 5; i < argc; i++) { std::string kv = argv[i]; auto e = kv.find('='); std::string k = kv.substr(0, e), v = kv.substr(e + 1);
        if (k == "coarse") { coarse = v; continue; } if (k == "fine") { fine = v; continue; } if (k == "tablecheck") { tablecheck = std::stoi(v); continue; }
        const double d = std::stod(v);
        if (k == "phE") p.phE = d; else if (k == "phH") p.phH = d; else if (k == "phT") p.phT = d; else if (k == "R") p.R = (int) d; else if (k == "RB") p.RB = (int) d; else if (k == "FA") p.FA = d;
        else if (k == "N1") p.N1 = (int) d; else if (k == "bucket") p.bucket = (int) d; else if (k == "sites") p.sites = (int) d;
        else if (k == "r16") p.r16 = d; else if (k == "r8") p.r8 = d; else if (k == "r4") p.r4 = d; else if (k == "cand") p.cand = d;
        else { fprintf(stderr, "unknown key %s\n", k.c_str()); return 2; } }
    p.coarse = parse_terms(coarse); p.fine = parse_terms(fine);
    if (P % p.R || P % p.RB) { fprintf(stderr, "R and RB must divide %d\n", P); return 2; }
    fprintf(stderr, "gate: erosion %.3f, humidity %.3f, temperature %.3f: %.3g of the seeds; R %d, RB %d, FA %.3f, %d places, %d sites a world\n  coarse %s\n  fine %s\n", p.phE, p.phH, p.phT,
            pow(2 * p.phE, 2) * pow(2 * p.phH, 2) * pow(2 * p.phT, 2), p.R, p.RB, p.FA, p.N1, p.sites, coarse.c_str(), fine.c_str());

    // world copies along one axis: for A cell i, the world x = i R + k P inside the world, and B's nearest column there
    const int nA = P / p.R, nB = P / p.RB;
    std::vector<std::vector<std::pair<int, int>>> copyA(nA);
    for (int i = 0; i < nA; i++) for (int64_t k = floordiv(int64_t(-WORLD) - int64_t(i) * p.R, P); int64_t(i) * p.R + k * P <= WORLD; k++) {
        const int64_t x = int64_t(i) * p.R + k * P; if (x < -WORLD) continue;
        const int64_t bu = (int64_t) std::llround(double(x) * S / p.RB); copyA[i].push_back({(int) x, (int) (((bu % nB) + nB) % nB)}); }

    if (tablecheck) {   // coarse table lookups (A at its grid point, B at the nearest RB point) vs the exact sum at the world point
        Sampler s(start); fastnoise::ready_all(s); std::vector<float> TA, TB; build_table(s, p.coarse, 0, p.R, TA); build_table(s, p.coarse, 1, p.RB, TB);
        const auto oa = octaves(s, p.coarse, 0), ob = octaves(s, p.coarse, 1);
        uint64_t rng = 12345; double worst = 0, sum = 0, sd = 0;
        for (int t = 0; t < tablecheck; t++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; const int ix = int((rng >> 33) % nA); rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; const int iz = int((rng >> 33) % nA);
            const auto &cx = copyA[ix][(rng >> 20) % copyA[ix].size()], &cz = copyA[iz][(rng >> 40) % copyA[iz].size()];
            const double direct = point_score(oa, ob, cx.first, cz.first), tab = TA[size_t(iz) * nA + ix] + TB[size_t(cz.second) * nB + cx.second];
            worst = std::max(worst, std::fabs(tab - direct)); sum += std::fabs(tab - direct); sd += direct * direct;
        }
        printf("%d points: |table - exact| mean %.3f, max %.3f; the score's rms %.3f\n", tablecheck, sum / tablecheck, worst, std::sqrt(sd / tablecheck)); return 0;
    }
    FILE *out = fopen((prefix + ".csv").c_str(), "a"), *wout = fopen((prefix + ".worlds.csv").c_str(), "a");
    if (!out || !wout) { fprintf(stderr, "cannot open output\n"); return 2; }
    signal(SIGINT, on_signal); signal(SIGTERM, on_signal);

    static SurfaceGrid grid; std::atomic<uint64_t> next(0), gated(0), worlds(0), sitesTried(0), sitesHit(0); std::mutex mu; double best = 0; const uint64_t CHUNK = 1 << 18;
    std::atomic<uint64_t> cyc[7] = {};   // gate, tables, scan, fine, probe+16, 8+4, exact
    auto t0 = Clock::now(), last = t0;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back([&] {
        std::vector<uint8_t> st; std::vector<int> stack; std::vector<float> TA, TB, tmp; std::vector<Site> cand;
        const uint64_t lims[3] = {(uint64_t) (p.phE * (double) (1ULL << 45)), (uint64_t) (p.phH * (double) (1ULL << 45)), (uint64_t) (p.phT * (double) (1ULL << 45))};
        for (uint64_t base; !stop_ && (base = next.fetch_add(CHUNK)) < (count ? count : ~0ULL);) {
            uint64_t c[7] = {}, tg = __rdtsc(), tdown = 0, ng = 0;
            auto lap = [&](int i, uint64_t &ta) { const uint64_t u = __rdtsc(); c[i] += u - ta; tdown += u - ta; ta = u; };
            for (uint64_t k0 = base; k0 < base + CHUNK && (count == 0 || k0 < count) && !stop_; k0 += 8) {
                alignas(32) uint64_t wl[8], wh[8]; int m = gate_avx8(start + k0, lims[0], wl, wh);
                while (m) {
                    const int lane = __builtin_ctz(m); m &= m - 1;
                    if (count && k0 + lane >= count) continue;
                    if (!gate_rest(wl[lane], wh[lane], lims)) continue;
                    const uint64_t seed = start + k0 + lane; ng++;
                    uint64_t ta = __rdtsc();
                    SurfaceSampler Sm(seed, grid); fastnoise::ready_all(Sm.s); State ss{Sm};
                    build_table(Sm.s, p.coarse, 0, p.R, TA); build_table(Sm.s, p.coarse, 1, p.RB, TB);
                    lap(1, ta);
                    tmp.assign(TA.begin(), TA.end()); const size_t q = size_t((1 - p.FA) * tmp.size()); std::nth_element(tmp.begin(), tmp.begin() + q, tmp.end()); const float qa = tmp[q];
                    const float bmax = *std::max_element(TB.begin(), TB.end());
                    cand.clear(); float bar = -1e30f; const size_t keep = size_t(p.N1) * 16;
                    auto push = [&](float sc, int x, int z) {
                        cand.push_back({sc, x, z, 0, 0, 0});
                        if (cand.size() >= 2 * keep) { std::nth_element(cand.begin(), cand.begin() + keep, cand.end(), [](const Site &a, const Site &b) { return a.score > b.score; }); cand.resize(keep);
                            bar = std::min_element(cand.begin(), cand.end(), [](const Site &a, const Site &b) { return a.score < b.score; })->score; }
                    };
                    std::vector<int> top;
                    for (int jz = 0; jz < nA; jz++) {
                        top.clear(); const float *arow = &TA[size_t(jz) * nA]; float rowmax = -1e30f;
                        for (int ix = 0; ix < nA; ix++) if (arow[ix] >= qa) { top.push_back(ix); rowmax = std::max(rowmax, arow[ix]); }
                        if (top.empty() || rowmax + bmax <= bar) continue;
                        for (const auto &cz : copyA[jz]) { const float *brow = &TB[size_t(cz.second) * nB];
                            for (const int ix : top) { const float a = arow[ix]; if (a + bmax <= bar) continue;
                                for (const auto &cx : copyA[ix]) { const float sc = a + brow[cx.second]; if (sc > bar) push(sc, cx.first, cz.first); } } }
                    }
                    std::sort(cand.begin(), cand.end(), [](const Site &a, const Site &b) { return a.score > b.score; });
                    std::vector<Site> places; std::unordered_map<int64_t, int> used;
                    for (const Site &s : cand) { if ((int) places.size() >= p.N1) break;
                        const int64_t key = (int64_t(floordiv(s.x, p.bucket)) << 32) ^ (uint32_t) floordiv(s.z, p.bucket); if (used.count(key)) continue; used[key] = 1; places.push_back(s); }
                    lap(2, ta);
                    // fine score: the best of a 5x5 grid (step 64) around each place
                    const auto fa = octaves(Sm.s, p.fine, 0), fb = octaves(Sm.s, p.fine, 1);
                    for (Site &s : places) { s.fine = -1e30f;
                        for (int dz = -2; dz <= 2; dz++) for (int dx = -2; dx <= 2; dx++) { const int x = s.x + 64 * dx, z = s.z + 64 * dz; const float v = (float) point_score(fa, fb, x, z); if (v > s.fine) { s.fine = v; s.fx = x; s.fz = z; } } }
                    std::sort(places.begin(), places.end(), [](const Site &a, const Site &b) { return a.fine > b.fine; });
                    if ((int) places.size() > p.sites) places.resize(p.sites);
                    lap(3, ta);
                    double wbest16 = 0, wbest = 0; std::vector<std::pair<int64_t, int>> done;   // flooded patches (lattice-16 box, cells), to skip repeats
                    for (const Site &s : places) {
                        sitesTried++;
                        int sx = 0, sz = 0; bool found = false;
                        for (int r = 0; r <= 4 && !found; r++) for (int dz = -r; dz <= r && !found; dz++) for (int dx = -r; dx <= r && !found; dx++) {
                            if (std::max(std::abs(dx), std::abs(dz)) != r) continue;
                            if (ss(s.fx + 16 * dx, s.fz + 16 * dz) == 2) { sx = s.fx + 16 * dx; sz = s.fz + 16 * dz; found = true; } }
                        if (!found) { lap(4, ta); continue; }
                        auto f = [&](int x, int z) { return ss(x, z); };
                        const auto d16 = dense_flood_bridged(f, sx, sz, 16, 1500 / 16, st, stack);
                        const double a16 = d16.cells * 256 * 16 / 1e6; double a8 = a16, a4 = 0, a1 = 0;
                        lap(4, ta); sitesHit++;
                        const int64_t box = (int64_t(d16.minx) * 1000003 + d16.maxx) * 1000003 + d16.minz * 31 + d16.maxz; bool repeat = false;
                        for (auto &dn : done) if (dn.first == box && dn.second == (int) d16.cells) repeat = true;
                        if (repeat) continue;
                        done.push_back({box, (int) d16.cells});
                        if (a16 >= p.r16) a8 = dense_flood_bridged(f, sx, sz, 8, 1500 / 8, st, stack).cells * 64 * 16 / 1e6;
                        if (a8 >= p.r8) a4 = dense_flood_bridged(f, sx, sz, 4, 1500 / 4, st, stack).cells * 16 * 16 / 1e6;
                        lap(5, ta);
                        if (a4 >= p.r4) a1 = dense_flood_bridged(f, sx, sz, 1, 1500, st, stack).cells * 16 / 1e6;
                        lap(6, ta);
                        wbest16 = std::max(wbest16, a8); wbest = std::max(wbest, a1);
                        std::lock_guard<std::mutex> l(mu);
                        fprintf(out, "%llu,%d,%d,%.3f,%.3f,%.4f,%.4f,%.4f,%.6f\n", (unsigned long long) seed, sx, sz, s.score, s.fine, a16, a8, a4, a1);
                        if (a1 >= p.cand) { fflush(out); fprintf(stderr, "CANDIDATE %llu %d %d %.0f (step 8: %.3fM, step 4: %.3fM)\n", (unsigned long long) seed, 4 * sx, 4 * sz, a1 * 1e6, a8, a4); }
                        if (a1 > best) best = a1;
                    }
                    worlds++;
                    std::lock_guard<std::mutex> l(mu);
                    fprintf(wout, "%llu,%.3f,%.3f,%.4f,%.6f\n", (unsigned long long) seed, cand.empty() ? 0.0 : cand[0].score, places.empty() ? 0.0 : places[0].fine, wbest16, wbest);
                }
            }
            gated += ng; c[0] = __rdtsc() - tg - tdown; for (int i = 0; i < 7; i++) cyc[i] += c[i];
            std::lock_guard<std::mutex> l(mu);
            auto now = Clock::now();
            if (std::chrono::duration<double>(now - last).count() >= 60) {
                last = now; const double s = std::chrono::duration<double>(now - t0).count(); const uint64_t dn = std::min<uint64_t>(next.load(), count ? count : ~0ULL);
                double tot = 0; for (int i = 0; i < 7; i++) tot += cyc[i].load();
                fprintf(stderr, "%.0f s: %.3e seeds, %llu worlds (%.2f/s), %llu sites, %llu with snowy slopes, best exact %.3fM; time: gate %.0f%%, tables %.0f%%, scan %.0f%%, fine %.0f%%, probe+16 %.0f%%, 8+4 %.0f%%, exact %.0f%%; next index %llu\n",
                        s, (double) dn, (unsigned long long) worlds.load(), worlds.load() / s, (unsigned long long) sitesTried.load(), (unsigned long long) sitesHit.load(), best,
                        100 * cyc[0] / tot, 100 * cyc[1] / tot, 100 * cyc[2] / tot, 100 * cyc[3] / tot, 100 * cyc[4] / tot, 100 * cyc[5] / tot, 100 * cyc[6] / tot, (unsigned long long) (start + dn));
                fflush(out); fflush(wout);
            }
        }
    });
    for (auto &t : pool) t.join();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count(); const uint64_t dn = std::min<uint64_t>(next.load(), count ? count : ~0ULL);
    fprintf(stderr, "done: %.3e seeds in %.0f s, %llu worlds, best exact %.3fM; next index %llu\n", (double) dn, s, (unsigned long long) worlds.load(), best, (unsigned long long) (start + dn));
    fclose(out); fclose(wout);
}
