#pragma once
// Dense-bitmap flood fill (4-connectivity) on a lattice of spacing `step` quarts.
// step=1 reproduces the sheet metric exactly, bounded by a window.
#include "standalone.hpp"
#include <algorithm>
#include <vector>

namespace standalone {
struct DenseArea {
    uint64_t cells = 0, sampled = 0;
    int minx = 0, maxx = 0, minz = 0, maxz = 0;   // quart bounds of accepted samples
    bool edge = false, border = false;            // hit window edge / world border
};
// Window is (2*half+1)^2 samples centered at (qx,qz). Caller passes reusable buffers.
// conn8: also step to diagonal neighbours (screening only: bridges gaps that a 4-connected
// coarse lattice would cut; the sheet metric is always 4-connected at step 1).
template <class Pred>
inline DenseArea dense_flood(Pred &&pred, int qx, int qz, int step, int half, std::vector<uint8_t> &state,
                             std::vector<int> &stack, bool conn8 = false) {
    const int n = 2 * half + 1;
    state.assign(size_t(n) * n, 0);
    stack.clear();
    DenseArea r;
    r.minx = r.maxx = qx; r.minz = r.maxz = qz;
    auto touch = [&](int i, int j) {
        if (i < 0 || j < 0 || i >= n || j >= n) { r.edge = true; return; }
        uint8_t &s = state[size_t(j) * n + i];
        if (s) return;
        s = 1;
        int x = qx + (i - half) * step, z = qz + (j - half) * step;
        if (x < -7499996 || x > 7499996 || z < -7499996 || z > 7499996) { r.border = true; return; }
        r.sampled++;
        if (pred(x, z)) {
            s = 2;
            r.cells++;
            stack.push_back(j * n + i);
            r.minx = std::min(r.minx, x); r.maxx = std::max(r.maxx, x);
            r.minz = std::min(r.minz, z); r.maxz = std::max(r.maxz, z);
        }
    };
    touch(half, half);
    while (!stack.empty()) {
        int k = stack.back(); stack.pop_back();
        int i = k % n, j = k / n;
        touch(i - 1, j); touch(i + 1, j); touch(i, j - 1); touch(i, j + 1);
        if (conn8) { touch(i - 1, j - 1); touch(i + 1, j - 1); touch(i - 1, j + 1); touch(i + 1, j + 1); }
    }
    return r;
}
// Flood with MC-Measure-style bridging: pred returns 2 (counted), 1 (crossed, not counted: a river
// whose underlying biome is the target) or 0 (boundary).
template <class Pred>
inline DenseArea dense_flood_bridged(Pred &&pred, int qx, int qz, int step, int half, std::vector<uint8_t> &state,
                                     std::vector<int> &stack, bool conn8 = false) {
    const int n = 2 * half + 1;
    state.assign(size_t(n) * n, 0);
    stack.clear();
    DenseArea r;
    r.minx = r.maxx = qx; r.minz = r.maxz = qz;
    auto touch = [&](int i, int j) {
        if (i < 0 || j < 0 || i >= n || j >= n) { r.edge = true; return; }
        uint8_t &s = state[size_t(j) * n + i];
        if (s) return;
        s = 1;
        int x = qx + (i - half) * step, z = qz + (j - half) * step;
        if (x < -7499996 || x > 7499996 || z < -7499996 || z > 7499996) { r.border = true; return; }
        r.sampled++;
        int v = pred(x, z);
        if (v == 0) return;
        s = uint8_t(1 + v);
        if (v == 2) {
            r.cells++;
            r.minx = std::min(r.minx, x); r.maxx = std::max(r.maxx, x);
            r.minz = std::min(r.minz, z); r.maxz = std::max(r.maxz, z);
        }
        stack.push_back(j * n + i);
    };
    touch(half, half);
    while (!stack.empty()) {
        int k = stack.back(); stack.pop_back();
        int i = k % n, j = k / n;
        touch(i - 1, j); touch(i + 1, j); touch(i, j - 1); touch(i, j + 1);
        if (conn8) { touch(i - 1, j - 1); touch(i + 1, j - 1); touch(i - 1, j + 1); touch(i + 1, j + 1); }
    }
    return r;
}
}
