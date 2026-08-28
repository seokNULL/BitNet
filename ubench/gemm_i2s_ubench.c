// Microbenchmark for the ggml_gemm_i2_i8_s kernel exported by libggml-cpu.so.
//
// Sweeps a grid of (n, m, b) sizes where, in kernel terms:
//   n = inner dimension (elements per row, must be a multiple of 128)
//   m = weight rows  (kernel arg nc) -> output dimension of the projection
//   b = activation columns (kernel arg nr) -> batch / tokens processed at once
//
// For each size it reports avg/min/stddev time, GFLOPS and effective memory
// bandwidth, and appends a row to a CSV file.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// From ggml/src/ggml-cpu/ggml-cpu-i2s.h (libggml-cpu.so exports it):
//   vx = weight data (I2_S, 2-bit packed, nc rows of n/4 bytes)
//   vy = activation data (int8, nr columns of n bytes)
//   s  = output, s[col * bs + row]
extern void ggml_gemm_i2_i8_s(int n, float * s, size_t bs,
                              const void * vx, const void * vy, int nr, int nc);

#define MAX_LIST 32

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

// Scalar reference for one output element, replicating the kernel's unpacking
// order: element (bl*128 + g*32 + k) of a row is bits (6-2g)..(7-2g) of packed
// byte (bl*32 + k), used as an unsigned 2-bit code multiplied by signed int8.
static float ref_dot(const uint8_t * x_row, const int8_t * y_col, int n) {
    int32_t acc = 0;
    for (int bl = 0; bl < n / 128; bl++) {
        for (int g = 0; g < 4; g++) {
            for (int k = 0; k < 32; k++) {
                int code = (x_row[bl * 32 + k] >> (6 - 2 * g)) & 3;
                acc += code * (int32_t)y_col[bl * 128 + g * 32 + k];
            }
        }
    }
    return (float)acc;
}

static int self_test(void) {
    const int n = 256, m = 8, b = 8;
    uint8_t * X = aligned_alloc(64, (size_t)m * n / 4);
    int8_t  * Y = aligned_alloc(64, (size_t)b * n);
    float   * S = aligned_alloc(64, (size_t)m * b * sizeof(float));
    if (!X || !Y || !S) return 0;

    srand(12345);
    for (int i = 0; i < m * n / 4; i++) X[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < b * n;     i++) Y[i] = (int8_t)((rand() % 256) - 128);
    memset(S, 0, (size_t)m * b * sizeof(float));

    ggml_gemm_i2_i8_s(n, S, m, X, Y, b, m);

    int ok = 1;
    for (int c = 0; c < b && ok; c++) {
        for (int r = 0; r < m && ok; r++) {
            float want = ref_dot(X + (size_t)r * n / 4, Y + (size_t)c * n, n);
            if (S[c * m + r] != want) {
                fprintf(stderr, "self-test MISMATCH at row=%d col=%d: got %.0f want %.0f\n",
                        r, c, S[c * m + r], want);
                ok = 0;
            }
        }
    }
    free(X); free(Y); free(S);
    return ok;
}

static void bench_one(int n, int m, int b, int max_iters, double budget_ms, FILE * csv) {
    size_t x_bytes = (size_t)m * n / 4;
    size_t y_bytes = (size_t)b * n;
    size_t s_bytes = (size_t)m * b * sizeof(float);

    uint8_t * X = aligned_alloc(64, (x_bytes + 63) / 64 * 64);
    int8_t  * Y = aligned_alloc(64, (y_bytes + 63) / 64 * 64);
    float   * S = aligned_alloc(64, (s_bytes + 63) / 64 * 64);
    if (!X || !Y || !S) {
        fprintf(stderr, "allocation failed for n=%d m=%d b=%d\n", n, m, b);
        free(X); free(Y); free(S);
        return;
    }
    for (size_t i = 0; i < x_bytes; i++) X[i] = (uint8_t)(rand() & 0xFF);
    for (size_t i = 0; i < y_bytes; i++) Y[i] = (int8_t)((rand() % 256) - 128);
    memset(S, 0, s_bytes);

    // warmup
    for (int i = 0; i < 3; i++) {
        ggml_gemm_i2_i8_s(n, S, m, X, Y, b, m);
    }

    // timed iterations: stop at max_iters or when the time budget is spent
    double sum = 0.0, sum2 = 0.0, tmin = 1e30;
    int iters = 0;
    double t_begin = now_ms();
    while (iters < max_iters && (iters < 5 || now_ms() - t_begin < budget_ms)) {
        double t0 = now_ms();
        ggml_gemm_i2_i8_s(n, S, m, X, Y, b, m);
        double dt = now_ms() - t0;
        sum += dt; sum2 += dt * dt;
        if (dt < tmin) tmin = dt;
        iters++;
    }

    double avg = sum / iters;
    double var = sum2 / iters - avg * avg;
    double sd  = var > 0 ? sqrt(var) : 0.0;

    double flops  = 2.0 * n * m * b;                       // multiply-adds
    double bytes  = (double)x_bytes + y_bytes + s_bytes;   // one full pass
    double gflops = flops / (avg * 1e6);
    double gbps   = bytes / (avg * 1e6);

    printf("%7d %7d %5d | %5d it | %9.4f %9.4f %8.4f | %8.2f %8.2f\n",
           n, m, b, iters, avg, tmin, sd, gflops, gbps);
    if (csv) {
        fprintf(csv, "%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f\n",
                n, m, b, iters, avg, tmin, sd, gflops, gbps);
    }

    free(X); free(Y); free(S);
}

static int parse_list(const char * arg, int * out, int max) {
    int count = 0;
    char * copy = strdup(arg);
    for (char * tok = strtok(copy, ","); tok && count < max; tok = strtok(NULL, ",")) {
        out[count++] = atoi(tok);
    }
    free(copy);
    return count;
}

static void usage(const char * prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -n <list>   inner dimensions, comma-separated (default: 1024,2048,4096,8192)\n");
    printf("              each must be a multiple of 128\n");
    printf("  -m <list>   weight rows / output dim (default: 1024,2048,4096,8192)\n");
    printf("  -b <list>   batch / activation columns (default: 1,32,128,512)\n");
    printf("  -i <num>    max iterations per config (default: 200)\n");
    printf("  -t <ms>     time budget per config in ms (default: 250)\n");
    printf("  -o <path>   output CSV file (default: gemm_i2s_ubench.csv)\n");
    printf("  -h          show this help\n");
}

int main(int argc, char ** argv) {
    int ns[MAX_LIST] = {1024, 2048, 4096, 8192};
    int ms[MAX_LIST] = {1024, 2048, 4096, 8192};
    int bs[MAX_LIST] = {1, 32, 128, 512};
    int n_count = 4, m_count = 4, b_count = 4;
    int max_iters = 200;
    double budget_ms = 250.0;
    const char * csv_path = "gemm_i2s_ubench.csv";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) n_count = parse_list(argv[++i], ns, MAX_LIST);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) m_count = parse_list(argv[++i], ms, MAX_LIST);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) b_count = parse_list(argv[++i], bs, MAX_LIST);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) max_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) budget_ms = atof(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    printf("ggml_gemm_i2_i8_s microbenchmark (single thread)\n\n");

    if (!self_test()) {
        fprintf(stderr, "self-test FAILED, aborting\n");
        return 1;
    }
    printf("self-test passed (kernel output matches scalar reference)\n\n");

    FILE * csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "cannot open %s for writing\n", csv_path);
        return 1;
    }
    fprintf(csv, "n,m,b,iters,avg_ms,min_ms,stddev_ms,gflops,gbps\n");

    printf("%7s %7s %5s | %8s | %9s %9s %8s | %8s %8s\n",
           "n", "m", "b", "iters", "avg_ms", "min_ms", "sd_ms", "GFLOPS", "GB/s");
    printf("---------------------------------------------------------------------------------\n");

    srand(42);
    for (int i = 0; i < n_count; i++) {
        if (ns[i] % 128 != 0) {
            fprintf(stderr, "skipping n=%d (must be a multiple of 128)\n", ns[i]);
            continue;
        }
        for (int j = 0; j < m_count; j++) {
            for (int k = 0; k < b_count; k++) {
                bench_one(ns[i], ms[j], bs[k], max_iters, budget_ms, csv);
            }
        }
    }

    fclose(csv);
    printf("\nresults written to %s\n", csv_path);
    return 0;
}
