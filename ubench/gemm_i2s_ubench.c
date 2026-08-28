// Microbenchmark for the ggml_gemm_i2_i8_s kernel exported by libggml-cpu.so.
//
// Runs one (n, m, b) configuration for a given iteration and thread count and
// reports per-iteration execution time plus RAPL energy split into package
// and DRAM domains.
//
//   n = inner dimension (elements per row, multiple of 128)
//   m = weight rows -> projection output dimension
//   b = activation columns -> batch / tokens processed at once
//
// Threads partition the m (weight row) dimension, the same axis ggml uses to
// parallelize this kernel during inference.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// From ggml/src/ggml-cpu/ggml-cpu-i2s.h (exported by libggml-cpu.so):
//   vx = weights (I2_S packed, nc rows of n/4 bytes), vy = activations
//   (int8, nr columns of n bytes), s[col * bs + row]
extern void ggml_gemm_i2_i8_s(int n, float * s, size_t bs,
                              const void * vx, const void * vy, int nr, int nc);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// ---------------------------------------------------------------- RAPL energy

#define MAX_RAPL 16

typedef struct {
    char   path[256];
    double max_range_uj;
    double start_uj;
    int    is_dram;   // 0 = package domain, 1 = dram subdomain
} rapl_domain;

static rapl_domain rapl[MAX_RAPL];
static int rapl_count = 0;

static double read_file_double(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return -1.0;
    double v = -1.0;
    if (fscanf(f, "%lf", &v) != 1) v = -1.0;
    fclose(f);
    return v;
}

static void rapl_add(const char * dir, int is_dram) {
    if (rapl_count >= MAX_RAPL) return;
    rapl_domain * d = &rapl[rapl_count];
    snprintf(d->path, sizeof(d->path), "%s/energy_uj", dir);
    if (read_file_double(d->path) < 0) return;   // no read permission or absent
    char range_path[256];
    snprintf(range_path, sizeof(range_path), "%s/max_energy_range_uj", dir);
    d->max_range_uj = read_file_double(range_path);
    d->is_dram = is_dram;
    rapl_count++;
}

// Discover package domains (/sys/class/powercap/intel-rapl:P) and their dram
// subdomains (intel-rapl:P:S with name "dram").
static void rapl_init(void) {
    for (int p = 0; p < 8; p++) {
        char dir[192], name_path[256], name[64] = {0};
        snprintf(dir, sizeof(dir), "/sys/class/powercap/intel-rapl:%d", p);
        snprintf(name_path, sizeof(name_path), "%s/name", dir);
        FILE * f = fopen(name_path, "r");
        if (!f) break;
        if (!fgets(name, sizeof(name), f)) name[0] = 0;
        fclose(f);
        if (strncmp(name, "package", 7) == 0 || strncmp(name, "psys", 4) == 0) {
            rapl_add(dir, 0);
        }
        for (int s = 0; s < 8; s++) {
            char sdir[224];
            snprintf(sdir, sizeof(sdir), "%s:%d", dir, s);
            snprintf(name_path, sizeof(name_path), "%s/name", sdir);
            f = fopen(name_path, "r");
            if (!f) break;
            if (!fgets(name, sizeof(name), f)) name[0] = 0;
            fclose(f);
            if (strncmp(name, "dram", 4) == 0) rapl_add(sdir, 1);
        }
    }
}

static void rapl_start(void) {
    for (int i = 0; i < rapl_count; i++) {
        rapl[i].start_uj = read_file_double(rapl[i].path);
    }
}

// Accumulated joules since rapl_start(), split by domain type.
static void rapl_stop(double * pkg_j, double * dram_j) {
    *pkg_j = *dram_j = 0.0;
    for (int i = 0; i < rapl_count; i++) {
        double end = read_file_double(rapl[i].path);
        double delta = end - rapl[i].start_uj;
        if (delta < 0 && rapl[i].max_range_uj > 0) delta += rapl[i].max_range_uj;
        if (rapl[i].is_dram) *dram_j += delta / 1e6;
        else                 *pkg_j  += delta / 1e6;
    }
}

// ------------------------------------------------------------------ benchmark

// One GEMM call with the m (weight row) dimension split across threads.
static void gemm_mt(int n, float * S, int m, const uint8_t * X, const int8_t * Y,
                    int b, int nthreads) {
    #pragma omp parallel num_threads(nthreads)
    {
        int t  = omp_get_thread_num();
        int nt = omp_get_num_threads();
        // contiguous row chunks, aligned to 4 (the kernel's row tile)
        int chunk = ((m / nt) / 4) * 4;
        int r0 = t * chunk;
        int r1 = (t == nt - 1) ? m : r0 + chunk;
        if (r1 > r0) {
            ggml_gemm_i2_i8_s(n, S + r0, m, X + (size_t)r0 * n / 4, Y, b, r1 - r0);
        }
    }
}

static void usage(const char * prog) {
    printf("Usage: %s -n <inner_dim> -m <out_dim> -b <batch> [-i <iters>] [-t <threads>]\n", prog);
    printf("  -n   inner dimension, multiple of 128\n");
    printf("  -m   weight rows (projection output dim)\n");
    printf("  -b   batch / tokens per call\n");
    printf("  -i   iterations (default: 100)\n");
    printf("  -t   threads (default: 1)\n");
}

int main(int argc, char ** argv) {
    int n = 0, m = 0, b = 0, iters = 100, nthreads = 1;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) m = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) b = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) nthreads = atoi(argv[++i]);
        else { usage(argv[0]); return !!strcmp(argv[i], "-h"); }
    }
    if (n <= 0 || m <= 0 || b <= 0 || iters <= 0 || nthreads <= 0) {
        usage(argv[0]);
        return 1;
    }
    if (n % 128 != 0) {
        fprintf(stderr, "error: n must be a multiple of 128\n");
        return 1;
    }

    size_t x_bytes = (size_t)m * n / 4;
    size_t y_bytes = (size_t)b * n;
    size_t s_bytes = (size_t)m * b * sizeof(float);
    uint8_t * X = aligned_alloc(64, (x_bytes + 63) / 64 * 64);
    int8_t  * Y = aligned_alloc(64, (y_bytes + 63) / 64 * 64);
    float   * S = aligned_alloc(64, (s_bytes + 63) / 64 * 64);
    if (!X || !Y || !S) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    srand(42);
    for (size_t i = 0; i < x_bytes; i++) X[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < y_bytes; i++) Y[i] = (int8_t)((rand() % 256) - 128);
    memset(S, 0, s_bytes);

    rapl_init();

    // warmup
    for (int i = 0; i < 3; i++) gemm_mt(n, S, m, X, Y, b, nthreads);

    double sum = 0.0, sum2 = 0.0, tmin = 1e30, tmax = 0.0;
    rapl_start();
    double t_all0 = now_ms();
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        gemm_mt(n, S, m, X, Y, b, nthreads);
        double dt = now_ms() - t0;
        sum += dt; sum2 += dt * dt;
        if (dt < tmin) tmin = dt;
        if (dt > tmax) tmax = dt;
    }
    double total_ms = now_ms() - t_all0;
    double pkg_j, dram_j;
    rapl_stop(&pkg_j, &dram_j);

    double avg = sum / iters;
    double var = sum2 / iters - avg * avg;
    double sd  = var > 0 ? sqrt(var) : 0.0;
    double gflops = 2.0 * n * m * b / (avg * 1e6);

    printf("config : n=%d m=%d b=%d threads=%d iters=%d\n", n, m, b, nthreads, iters);
    printf("time   : avg %.4f ms | min %.4f | max %.4f | stddev %.4f  (total %.1f ms)\n",
           avg, tmin, tmax, sd, total_ms);
    printf("perf   : %.2f GFLOPS\n", gflops);

    if (rapl_count == 0) {
        printf("energy : unavailable (no readable /sys/class/powercap/intel-rapl* domain;\n");
        printf("         try running with sudo, or on bare metal)\n");
    } else {
        printf("energy : package %.3f J total, %.3f mJ/iter, %.2f W avg\n",
               pkg_j, pkg_j * 1e3 / iters, pkg_j / (total_ms / 1e3));
        if (dram_j > 0) {
            printf("         dram    %.3f J total, %.3f mJ/iter, %.2f W avg\n",
                   dram_j, dram_j * 1e3 / iters, dram_j / (total_ms / 1e3));
        } else {
            printf("         dram    domain not exposed on this CPU\n");
        }
    }

    free(X); free(Y); free(S);
    return 0;
}
