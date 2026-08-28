# ubench — GEMM kernel microbenchmark

Microbenchmark for the `ggml_gemm_i2_i8_s` kernel (I2_S ternary-weight ×
int8-activation GEMM) exported by `libggml-cpu.so` — the code actually
executed during inference.

Runs one (n, m, b) configuration and reports per-iteration execution time and
RAPL energy split into **package** and **DRAM** domains. Threads partition the
weight-row (m) dimension, the same axis ggml parallelizes during inference.

## Build

The project must be built first so the ggml shared libraries exist in
`../build/bin` (run `python setup_env.py ...` or `cmake --build build`), then:

```bash
cd ubench
make
```

## Run

```bash
./gemm_i2s_ubench -n 4096 -m 4096 -b 1 -i 100 -t 8
```

| flag | meaning | default |
|---|---|---|
| `-n` | inner dimension (multiple of 128) | required |
| `-m` | weight rows = projection output dim | required |
| `-b` | batch / tokens per call (1 = decode GEMV) | required |
| `-i` | iterations | 100 |
| `-t` | threads | 1 |

Example output:

```
config : n=4096 m=4096 b=1 threads=8 iters=100
time   : avg 0.0713 ms | min 0.0691 | max 0.0898 | stddev 0.0031  (total 7.2 ms)
perf   : 470.65 GFLOPS
energy : package 1.203 J total, 12.031 mJ/iter, 41.85 W avg
         dram    0.312 J total,  3.118 mJ/iter, 10.84 W avg
```

## Energy measurement notes

Energy comes from Intel RAPL counters under `/sys/class/powercap/intel-rapl*`
(works on AMD for the package domain too). If the counters are not readable
the benchmark still runs and prints `energy : unavailable`:

- `energy_uj` is often root-only — run with `sudo`, or
  `sudo chmod a+r /sys/class/powercap/intel-rapl*/energy_uj*`
- Virtual machines / containers usually do not expose RAPL at all.
- RAPL measures the whole package/DRAM, so the numbers include idle power of
  everything else on the socket; keep the machine otherwise quiet and compare
  configurations rather than reading absolute values.
