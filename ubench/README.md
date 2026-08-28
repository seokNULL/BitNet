# ubench — GEMM kernel microbenchmark

Standalone microbenchmark for the `ggml_gemm_i2_i8_s` kernel (the I2_S
ternary-weight x int8-activation GEMM) exported by `libggml-cpu.so`.

Unlike `utils/test_gemm_kernel.sh`, this calls the actual library kernel used
during inference (no local copy of the kernel source), sweeps a configurable
size grid, and verifies the kernel output against a scalar reference before
benchmarking.

## Build

The project must be built first so the ggml shared libraries exist in
`../build/bin` (run `python setup_env.py ...` or `cmake --build build`), then:

```bash
cd ubench
make
```

## Run

```bash
./gemm_i2s_ubench                 # default grid, results in gemm_i2s_ubench.csv
./gemm_i2s_ubench -h              # all options
```

Size parameters (kernel terms):

| flag | meaning | default |
|---|---|---|
| `-n` | inner dimension (multiple of 128) | `1024,2048,4096,8192` |
| `-m` | weight rows = projection output dim | `1024,2048,4096,8192` |
| `-b` | activation columns = batch / tokens | `1,32,128,512` |

`b=1` corresponds to single-token decode (GEMV, memory-bound); large `b`
corresponds to prefill/batched GEMM (compute-bound).

Other options: `-i` max iterations per config, `-t` time budget per config in
ms, `-o` CSV output path.

Example: measure a specific projection shape (Llama3-8B gate/up, m=14336,
k=4096) across batch sizes:

```bash
./gemm_i2s_ubench -n 4096 -m 14336 -b 1,32,128,512 -o llama3_gate_up.csv
```

## Output

Per config: `avg_ms`, `min_ms`, `stddev_ms`, `GFLOPS` (2·n·m·b / time) and
effective `GB/s` (weights + activations + output bytes per pass / time).
The benchmark is single-threaded by design — it measures the kernel itself,
not ggml's thread-level parallelization.
