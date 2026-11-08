# Third-party code and data

| Component | Source | License | Use |
|---|---|---|---|
| MC-Measure (`MC-Measure/`, git submodule) | [ImAchievers/MC-Measure](https://github.com/ImAchievers/MC-Measure) @ `a58650e` | MIT, © 2026 ImAchievers | `bin/mcmeasure`: exact patch area with river bridging, the Large Biomes spreadsheet's metric. Built from the submodule with only `THREADS` substituted. |
| Cubiomes (inside `MC-Measure/lib/`) | fork of [Cubitect/cubiomes](https://github.com/Cubitect/cubiomes) shipped with MC-Measure | MIT, © 2020 Cubitect | Reference generator inside `bin/mcmeasure`; source of `make data` |
| `src/worldgen_data.hpp` | Generated from Cubiomes by `src/export_constants.c` (`make data`, byte-identical) | MIT (Cubiomes) | Minecraft 26.3 biome parameter tree and terrain-offset spline data |

Algorithmic ideas adapted from [COMMISSION](https://github.com/MinecraftAtHome/COMMISSION) by MinecraftAtHome:
y-phase seed screening and scanning world translations of the periodic first noise stack. No COMMISSION
code is included.
