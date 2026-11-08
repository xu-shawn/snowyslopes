# snowyslopes

Seed searches for the largest **Snowy Slopes** biomes in Minecraft Java 26.3 (default world type). Patches are measured
the way the Large Biomes spreadsheet does: MC-Measure's 4-connected area on the 1:4 biome grid at y 256, rivers
bridged, times 16. There are two searches:

- **at (0, 0)**: the patch containing the world origin. `src/origin.cpp` runs on a laptop (i5-1235U and its Iris Xe
  integrated GPU) at 3.7 × 10⁸ seeds/s. It found **5,196,288 blocks²**; the spreadsheet's record was 4,068,576.
- **anywhere**: the largest patch anywhere in a world. `src/anywhere.cpp` found **6,491,200 blocks²**; the
  spreadsheet's record was 5,873,360.

## Results

Every result was verified with `bin/mcmeasure` (MC-Measure) at y 256.

| | Blocks² | 1:4 cells | Seed | X | Z |
|---|---:|---:|---:|---:|---:|
| at (0, 0) | **5,196,288** | 324,768 | 7420745560948894716 | 0 | 0 |
| at (0, 0) | 5,128,608 | 320,538 | 6095040073850818261 | 0 | 0 |
| at (0, 0) | 5,021,792 | 313,862 | 4098352742734554736 | 0 | 0 |
| anywhere | **6,491,200** | 405,700 | 140000520764451 | 27837888 | 28535488 |
| anywhere | 6,084,112 | 380,257 | 130000226908221 | 26294784 | -6514176 |
| anywhere | 6,052,432 | 378,277 | 140001292435022 | 9472 | -4689920 |

Full lists:
- [`results/origin_over_4.8M.txt`](results/origin_over_4.8M.txt): 12 patches at (0, 0)
- [`results/anywhere_over_4.8M.txt`](results/anywhere_over_4.8M.txt): 24 patches anywhere
- [`results/finds.jsonl`](results/finds.jsonl): every verified find

## Quick start

Requirements:
- x86-64 with AVX2
- g++ with C++20 (tested with GCC 14)
- optional: Vulkan (`libvulkan-dev`, `pkg-config`) for the GPU gate. It needs a GPU with 64-bit shader integers.
- optional: `glslc`, only to rebuild the shader

```sh
git clone --recursive https://github.com/xu-shawn/snowyslopes && cd snowyslopes
make              # bin/{origin,anywhere,mcmeasure,fastcheck} (+ bin/gate.spv, bin/gpucheck with Vulkan)
make check        # self-checks, the four records measured, each search finds its record again (about 15 s)
bin/mcmeasure SEED X Z                              # exact area of the biome at block (X, 256, Z)
scripts/run_origin.sh o1 0 $(nproc)                 # (0, 0) search from seed index 0, unbounded
scripts/run_anywhere.sh a1 1000000000000 $(nproc)   # anywhere search from seed 1e12, unbounded
tools/report.py o1 --x 5.2,5.5                      # rates, tail and expected time to 5.2M / 5.5M
```

`make VULKAN=0` builds without Vulkan, and `origin` then gates on the CPU only. If there is no usable GPU at run time,
`origin` falls back to the CPU gate on its own.

The run scripts write their output to `runs/`:
- `NAME.csv`: the search's own output
- `NAME.log`: progress every 60 s, plus `CANDIDATE` lines
- `NAME.verified`: each candidate as MC-Measure measures it

To stop a run, press Ctrl-C or kill the search process. The last line of the log gives the index to resume from.
The defaults are the record-run settings. `src/origin.cpp` (header) and `struct Params` in `src/anywhere.cpp` list
the other `key=value` options.

**Seed numbering.**
- `origin`: index k is the seed `(k · 0x9e3779b97f4a7c15) ^ 0x6a09e667f3bcc909` (mod 2⁶⁴). The record run covered
  indices 100,353,404,059,648 to 109,469,744,025,600.
- `anywhere`: searches seeds `START + k` directly.
- Give each machine its own range.

**Disk use.** The (0, 0) search's csv grows by about 0.5 GB an hour.

**Verify every candidate.** MC-Measure starts from the biome at *block* (0, 256, 0). Minecraft's block-level biome
zoom can take that block's biome from a neighbouring 1:4 cell. The (0, 0) search checks cell (0, 0) itself, so a
candidate on the edge of its patch can measure as another biome. This happened to 1 of the record run's 20
candidates, and to 1 of 50 anywhere candidates whose block landed on a bridged cell. The run scripts verify every
candidate.

## How the (0, 0) search works

Snowy Slopes at y 256 needs temperature ≤ 0.2, humidity ≤ −0.1, erosion ≤ −0.375 and continentalness ≥ −0.19. Big
patches are frozen throughout (temperature < −0.45), where slopes also take the low and valley slices. The patch size
is set mostly by the broadest octaves of three noises: temperature −10, humidity −8 and erosion −9.

Per seed, cheapest first:

1. **Phase gate** (RNG only). Each broad octave is a 2D slice of 3D Perlin noise at y 0. Its y-phase sets how far
   the slice can swing, and it swings most at mid-cell. The gate keeps seeds whose erosion −9, humidity −8 and
   temperature −10 y-phases (both noise stacks) are within ±0.22, ±0.22 and ±0.24 of mid-cell. That keeps 0.86% of
   seeds. The window widths were fitted on the tail: their effect grows with patch size.

   The gate runs as a Vulkan compute shader on the integrated GPU (`src/gpu/gate.comp`, about 10⁹ seeds/s when idle).
   The CPU threads run an AVX2 version whenever the GPU's queue is empty. Both are bit-identical to a scalar
   reference (`bin/gpucheck`).
2. **Tail cuts at the origin.**
   - The temperature −10 sum (both stacks) must be ≤ −0.62. This is the strongest single predictor of a big patch:
     it keeps 5.2% of gated seeds but about 80% of those that reach 2.5M and 86% of 3M+.
   - Those octaves are built 16 seeds at a time in vector lanes (`src/fastnoise.hpp`, bit-identical to the plain
     setup; checked by `bin/fastcheck`). The stack with the larger predicted spread is built first. The second is
     skipped when the first already sits more than 2 predicted standard deviations above the cut. The prediction
     uses the exact variance of a Perlin octave at a point.
   - The erosion −9 sum must be ≤ −0.25 (keeps about 98% of the tail).
3. **Snowy Slopes at cell (0, 0)** with temperature ≤ −0.5. Cheap bounds come first, then the exact climate
   (`src/standalone.hpp`, bit-exact with Cubiomes).
4. **Flood cascade.** The patch is filled on a lattice of 32 cells (8-connected), then 16, 8 and 4. A seed continues
   to the next, finer lattice only if the coarser estimate reaches 0.7, 1.5, 2.5 and 3.6M blocks² respectively. The
   final exact count (lattice 1, river bridging) equals MC-Measure's.

The record run took 6.8 hours: 9.08 × 10¹² seeds at 3.7 × 10⁸ seeds/s, with 9 CPU threads plus the integrated GPU.

- **Rates:** about 3,640 patches of 2.5M+ per hour, 660 of 3M+ and 110 of 3.5M+.
- **Tail:** above 2.5M it is exponential with an e-fold of 0.294M (n = 24,544).
- **Expected wait:** one ≥ 5.2M patch every ~2.6 h and one ≥ 5.5M every ~7.3 h on this laptop.

### Development history

Every step was checked to give the same seeds as the step before it, with the new test off.

| Step | Change | Effect |
|---|---|---|
| 1 | Phase gate, floods 16 → 8 → 4 → exact | 8× the 2M+ patches per seed vs no gate |
| 2 | AVX2 gate (8 seeds), staged temperature build, vector-lane permutation build, per-noise gate widths | 1.32× seeds/s at the same gate; 1.3× the 2M+ per second from the widths; 2,022 of 2.5M+ per hour |
| 3 | Temperature −10 tail cut, batched builds, larger-spread stack first | 1.12× the 2.5M+ per hour (2,274) |
| 4 | Gate as a block cascade on packed survivors | 1.09× |
| 5 | Lattice-32 8-connected screen before lattice 16 | 1.08× |
| 6 | Seeds on an arithmetic progression (one mix per seed in the gate) | 1.05× |
| 7 | Erosion −9 tail cut, erosion built in stages | 1.05× |
| 8 | Gate on the integrated GPU (Vulkan), CPU threads take its queue | 1.30× (3.6 × 10⁸ seeds/s); 3,640 of 2.5M+ per hour |

Tried without gain:
- choosing seeds by the predicted spread at the origin from the x/z offsets
- ring or shape tests of temperature beyond the origin
- phase tests on any other octave
- a two-kernel GPU gate

## How the anywhere search works

`src/anywhere.cpp`, per seed:

1. A narrow phase gate (erosion ±0.05, humidity ±0.06, temperature ±0.08; keeps 3.7 × 10⁻⁶ of seeds).
2. For each gated world, a coarse linear climate score is tabulated over one period of each noise stack. Every octave
   of stack A repeats every 262,144 cells. Stack B is sampled at x · 337/331 and is tabulated in that coordinate.
3. The best cells of stack A are checked at each of their ~57² copies across the world, with stack B looked up
   there. The best 512 places, one per 1024² cells, go on.
4. A fine logistic score on nine low octaves (fitted on 1.5M+ patches against random points) keeps the best 96
   sites.
5. Each site gets a probe for a Snowy Slopes cell, then the same flood cascade as above.

It handles about 15 gated worlds per second on 12 threads. The best patch of a gated world has a median of 2.7M, and
1% of worlds reach 4.2M.

## Layout

| Path | What |
|---|---|
| `src/origin.cpp`, `src/anywhere.cpp` | the two searches |
| `src/snowy.hpp` | the Snowy Slopes flood state with MC-Measure's river bridging |
| `src/standalone.hpp`, `src/worldgen_data.hpp`, `src/surface.hpp`, `src/dense_flood.hpp` | standalone climate sampler (bit-exact with Cubiomes), surface-biome lookup, bitmap flood fill |
| `src/fastnoise.hpp` | noise setup with the permutation RNG in vector lanes |
| `src/gpu/` | the Vulkan gate: shader source, prebuilt SPIR-V (`make spv` rebuilds it), host code |
| `tools/` | `fastcheck`, `gpucheck`, `report.py` |
| `scripts/` | run scripts with candidate verification |
| `MC-Measure/` | submodule: the reference measuring tool |

## License

MIT (see [LICENSE](LICENSE)). Third-party code and credits: [THIRD_PARTY.md](THIRD_PARTY.md).
