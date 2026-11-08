// tools/gpucheck.cpp: the GPU gate (bin/gate.spv) against a scalar CPU gate on an index range, then its speed.
// usage: gpucheck [PER=64] [GROUPS=16384]
#include "../src/gpu/vkgate.hpp"
#include <string>
#include <algorithm>
#include <chrono>
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t nl(uint64_t &l, uint64_t &h) { const uint64_t a = l; uint64_t b = h; const uint64_t n = rotl(a + b, 17) + a; b ^= a; l = rotl(a, 49) ^ b ^ (b << 21); h = rotl(b, 28); return n; }
static const uint64_t NLO[3] = {0xd02491e6058f6fd8ULL, 0x81bb4d22e8dc168eULL, 0x5c7e6b29735f0d7fULL}, NHI[3] = {0x4792512c94c17a80ULL, 0xf1c8b4bea16303cdULL, 0xf7d86f1bbc734988ULL};
static const uint64_t OLO[3] = {0x082fe255f8be6631ULL, 0x0ef68ec68504005eULL, 0x36d326eed40efeb2ULL}, OHI[3] = {0x4e96119e22dedc81ULL, 0x48b6bf93a2789640ULL, 0x5be9ce18223c636aULL};
static bool ref(uint64_t seed, const uint64_t *lims) {
    uint64_t l = seed ^ 0x6a09e667f3bcc909ULL, h = l + 0x9e3779b97f4a7c15ULL;
    l = (l ^ (l >> 30)) * 0xbf58476d1ce4e5b9ULL; h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL; l = (l ^ (l >> 27)) * 0x94d049bb133111ebULL; h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL; l ^= l >> 31; h ^= h >> 31;
    const uint64_t lo = nl(l, h), hi = nl(l, h);
    for (int n = 0; n < 3; n++) { uint64_t rl = lo ^ NLO[n], rh = hi ^ NHI[n];
        for (int s = 0; s < 2; s++) { const uint64_t a = nl(rl, rh), b = nl(rl, rh); uint64_t ol = a ^ OLO[n], oh = b ^ OHI[n];
            nl(ol, oh); const uint64_t v = (nl(ol, oh) >> 11) & ((1ULL << 45) - 1); const uint64_t d = v > (1ULL << 44) ? v - (1ULL << 44) : (1ULL << 44) - v; if (d > lims[n]) return false; } }
    return true;
}
int main(int argc, char **argv) {
    const uint32_t per = argc > 1 ? atoi(argv[1]) : 64, groups = argc > 2 ? atoi(argv[2]) : 16384;
    const uint64_t lims[3] = {(uint64_t) (0.22 * (double) (1ULL << 45)), (uint64_t) (0.22 * (double) (1ULL << 45)), (uint64_t) (0.24 * (double) (1ULL << 45))};
    VkGate g; g.init(std::string(argv[0]).substr(0, std::string(argv[0]).rfind('/') + 1) + "gate.spv", 1u << 22); printf("device: %s\n", g.name.c_str());
    const uint64_t base = 123456789ULL * 1024, n = (uint64_t) groups * 64 * per; std::vector<uint64_t> got, want;
    g.run(base, groups, per, lims, got); std::sort(got.begin(), got.end());
    for (uint64_t k = 0; k < n; k++) if (ref(((base + k) * 0x9e3779b97f4a7c15ULL) ^ 0x6a09e667f3bcc909ULL, lims)) want.push_back(base + k);
    printf("%llu indices: GPU %zu pass, CPU reference %zu, identical %s\n", (unsigned long long) n, got.size(), want.size(), got == want ? "yes" : "NO");
    auto t0 = std::chrono::steady_clock::now(); uint64_t total = 0; std::vector<uint64_t> tmp;
    for (int r = 0; r < 10; r++) { tmp.clear(); g.run(base + (r + 1) * n, groups, per, lims, tmp); total += n; }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("GPU: %.3e indices in %.2f s = %.3e a second (per %u, %u groups of 64)\n", (double) total, s, total / s, per, groups);
}
