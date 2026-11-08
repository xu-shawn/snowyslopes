// snowy.hpp: Snowy Slopes at y 256 on the 1:4 grid, as MC-Measure counts it, with the standalone climate sampler
// (standalone.hpp, bit-exact with Cubiomes) and the surface-box lookup (surface.hpp).
//   state(x, z): 2 = snowy slopes, 1 = a river, frozen river, snowy plains or ice spikes cell whose weirdness is within
//   0.05 of 0 and whose biome with the weirdness pushed to +-0.0501 is snowy slopes (crossed, not counted), 0 = other.
#pragma once
#include "surface.hpp"
#include "dense_flood.hpp"
#include <cmath>
namespace snowy {
using namespace standalone;
constexpr int SNOWY_SLOPES = 179, SNOWY_PLAINS = 12, ICE_SPIKES = 140, RIVER = 7, FROZEN_RIVER = 11;
struct State {
    SurfaceSampler &S;
    int operator()(int x, int z) {
        auto &s = S.s; const auto &sh = s.get(Shift);
        double px = x + 4. * sh.sample(x, 0, z), pz = z + 4. * sh.sample(z, x, 0);
        float t = s.get(Temperature).sample(px, 0, pz); if (int64_t(10000.f * t) > 2000) return 0;      // every snowy slopes box: T <= 0.2
        float h = s.get(Humidity).sample(px, 0, pz); if (int64_t(10000.f * h) > -1000) return 0;        // H <= -0.1
        float e = s.get(Erosion).sample(px, 0, pz); if (int64_t(10000.f * e) > -3750) return 0;         // E <= -0.375
        float c = s.get(Continentalness).sample(px, 0, pz); if (int64_t(10000.f * c) < -1900) return 0; // C >= -0.19
        float w = s.get(Weirdness).sample(px, 0, pz);
        int64_t q[6] = {int64_t(10000.f * t), int64_t(10000.f * h), int64_t(10000.f * c), int64_t(10000.f * e), -10000, int64_t(10000.f * w)};
        bool exact = false;
        auto pick = [&]() {
            int b = S.g.lookup(q); if (b >= 0) return b;
            if (!exact) { float v[] = {c, e, -(std::fabs(std::fabs(w) - 0.6666667f) - 0.33333334f) * 3.f, w}; double off = spline(0, v) + 0.015f; q[4] = int64_t(10000.f * float(1. - (64 * 4) / 128. - 83. / 160. + off)); exact = true; }
            return classify(q);
        };
        int b = pick();
        if (b == SNOWY_SLOPES) return 2;
        if (b != RIVER && b != FROZEN_RIVER && b != SNOWY_PLAINS && b != ICE_SPIKES) return 0;
        if (q[5] > 500 || q[5] < -500) return 0;
        q[5] = q[5] >= 0 ? 501 : -501;
        return pick() == SNOWY_SLOPES ? 1 : 0;
    }
};
}
