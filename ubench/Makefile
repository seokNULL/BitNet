# Microbenchmark for the ggml_gemm_i2_i8_s kernel.
# Requires the project to be built first (the ggml shared libraries must
# exist in ../build/bin): python setup_env.py ...  or  cmake --build build

CC      ?= cc
CFLAGS  ?= -O3 -march=native -std=gnu11 -Wall -Wextra
LIBDIR  := $(abspath ../build/bin)
LDFLAGS := -L$(LIBDIR) -lggml-cpu -lggml-base -lm -Wl,-rpath,$(LIBDIR)

BIN := gemm_i2s_ubench

all: $(BIN)

$(BIN): gemm_i2s_ubench.c $(LIBDIR)/libggml-cpu.so
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(LIBDIR)/libggml-cpu.so:
	$(error libggml-cpu.so not found in $(LIBDIR) - build the project first: cmake --build build)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) gemm_i2s_ubench.csv

.PHONY: all run clean
