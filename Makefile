# -ffp-contract=off is required: the standalone sampler must round exactly like Java/Cubiomes.
# The searches need AVX2 (x86-64); the GPU gate needs Vulkan with 64-bit shader integers (shaderInt64).
CC       ?= gcc
CXX      ?= g++
OPT      ?= -O3 -march=native -ffp-contract=off
THREADS  ?= $(shell nproc)
GLSLC    ?= glslc
VULKAN   ?= $(shell pkg-config --exists vulkan 2>/dev/null && echo 1 || echo 0)
MCM      := MC-Measure
CFLAGS   := $(OPT) -fwrapv -I$(MCM)/lib
CXXFLAGS := $(OPT) -std=c++20 -pthread -Isrc
LIBOBJ   := $(addprefix build/mcm_,$(addsuffix .o,biomenoise biomes generator layers noise terrainnoise))
HEADERS  := src/standalone.hpp src/worldgen_data.hpp src/surface.hpp src/dense_flood.hpp src/snowy.hpp src/fastnoise.hpp

ifeq ($(VULKAN),1)
GPU_BINS := bin/gate.spv bin/gpucheck
ORIGIN_FLAGS := $(shell pkg-config --cflags vulkan)
ORIGIN_DEPS  := src/gpu/vkgate.hpp
ORIGIN_LIBS  := $(shell pkg-config --libs vulkan)
else
ORIGIN_FLAGS := -DNO_VULKAN
endif

.PHONY: all check spv data clean
all: bin/origin bin/anywhere bin/mcmeasure bin/fastcheck $(GPU_BINS)

build bin:
	mkdir -p $@

# MC-Measure (git submodule): the Large Biomes spreadsheet's measuring tool, the reference for every result
build/mcm_%.o: $(MCM)/lib/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@
bin/mcmeasure: $(MCM)/measure.c $(LIBOBJ) | bin
	sed 's/^#define THREADS .*/#define THREADS $(THREADS)/' $< > build/measure.c
	$(CC) $(CFLAGS) -I$(MCM) build/measure.c $(LIBOBJ) -lm -pthread -o $@

bin/origin: src/origin.cpp $(HEADERS) $(ORIGIN_DEPS) | bin
	$(CXX) $(CXXFLAGS) $(ORIGIN_FLAGS) $< $(ORIGIN_LIBS) -o $@
bin/anywhere: src/anywhere.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@
bin/fastcheck: tools/fastcheck.cpp $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@

# The GPU gate's SPIR-V is committed (src/gpu/gate.spv); `make spv` rebuilds it from gate.comp with glslc.
bin/gate.spv: src/gpu/gate.spv | bin
	cp $< $@
bin/gpucheck: tools/gpucheck.cpp src/gpu/vkgate.hpp | bin
	$(CXX) $(CXXFLAGS) $(shell pkg-config --cflags vulkan) $< $(shell pkg-config --libs vulkan) -o $@
spv:
	$(GLSLC) --target-env=vulkan1.2 -O src/gpu/gate.comp -o src/gpu/gate.spv

# Regenerate the biome parameter tree and terrain spline tables (src/worldgen_data.hpp) from MC-Measure's Cubiomes.
data: build/export_constants
	build/export_constants > src/worldgen_data.hpp
build/export_constants: src/export_constants.c $(LIBOBJ) | build
	$(CC) $(CFLAGS) $< $(LIBOBJ) -lm -o $@

# The sampler's fast noise setup, the GPU gate (when built) and both searches against MC-Measure: the two records and
# the spreadsheet's earlier ones measured, then each search run on its record seed alone (CANDIDATE lines with the
# same sizes).
check: all
	bin/fastcheck
	$(if $(GPU_BINS),bin/gpucheck 64 4096)
	bin/mcmeasure 7420745560948894716 0 0
	bin/mcmeasure 1555000016785099680 0 0
	bin/mcmeasure 140000520764451 27837888 28535488
	bin/mcmeasure 1106000000892216272 -26500860 28490312
	rm -f build/check_origin.csv build/check_anywhere.csv build/check_anywhere.worlds.csv
	bin/origin 104527398953825 1 build/check_origin 1 gpu=0 2>&1 | grep CANDIDATE
	bin/anywhere 140000520764451 1 build/check_anywhere 1 2>&1 | grep CANDIDATE

clean:
	rm -rf bin build
