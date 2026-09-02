#!/bin/bash
# Sweep gemm_i2s_ubench over n / m / b / thread combinations and collect results.
#
#   ./sweep.sh -n 1024,2048,4096 -m 4096 -t 1,2,4,8
#   ./sweep.sh -n 4096 -m 14336 -b 1,32,128 -t 8 -i 200 -o llama3_ffn.csv

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/gemm_i2s_ubench"

N_LIST="1024,2048,4096,8192"
M_LIST="1024,2048,4096,8192"
B_LIST="1"
T_LIST="1"
ITERS=100
OUT="sweep.csv"

usage() {
    cat << EOF
Usage: $0 [options]
  -n <list>  inner dimensions, comma-separated, multiples of 128 (default: $N_LIST)
  -m <list>  weight rows / output dims (default: $M_LIST)
  -b <list>  batch sizes (default: $B_LIST)
  -t <list>  thread counts (default: $T_LIST)
  -i <num>   iterations per run (default: $ITERS)
  -o <path>  output CSV (default: $OUT)
  -h         show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n) N_LIST="$2"; shift 2 ;;
        -m) M_LIST="$2"; shift 2 ;;
        -b) B_LIST="$2"; shift 2 ;;
        -t) T_LIST="$2"; shift 2 ;;
        -i) ITERS="$2";  shift 2 ;;
        -o) OUT="$2";    shift 2 ;;
        -h) usage; exit 0 ;;
        *)  echo "unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ ! -x "$BIN" ]; then
    echo "$BIN not found - run 'make' first" >&2
    exit 1
fi

# Pull the numbers out of one benchmark run and emit them as CSV fields:
# avg,min,max,stddev,gflops,pkg_j,pkg_mj,pkg_w,dram_j,dram_mj,dram_w
extract() {
    awk '
        function num(v) { return (v ~ /^[0-9.]+$/) ? v : "" }
        /^time/          { avg=$4; tmin=$8; tmax=$11; sd=$14 }
        /^perf/          { gflops=$3 }
        /package/        { pj=num($4); pmj=num($7); pw=num($9) }
        /^ *dram/        { dj=num($2); dmj=num($5); dw=num($7) }
        END { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                     avg, tmin, tmax, sd, gflops, pj, pmj, pw, dj, dmj, dw }
    '
}

echo "n,m,b,threads,iters,avg_ms,min_ms,max_ms,stddev_ms,gflops,pkg_j,pkg_mj_per_iter,pkg_w,dram_j,dram_mj_per_iter,dram_w" > "$OUT"

total=0
for n in ${N_LIST//,/ }; do for m in ${M_LIST//,/ }; do
for b in ${B_LIST//,/ }; do for t in ${T_LIST//,/ }; do
    total=$((total + 1))
done; done; done; done

echo "$total configurations, $ITERS iterations each"
echo
printf "%7s %7s %5s %4s | %9s %8s %9s | %10s %10s\n" \
       n m b thr avg_ms sd_ms GFLOPS "pkg_mJ/it" "dram_mJ/it"
printf -- "---------------------------------------------------------------------------------\n"

done_count=0
failed=0
for n in ${N_LIST//,/ }; do
for m in ${M_LIST//,/ }; do
for b in ${B_LIST//,/ }; do
for t in ${T_LIST//,/ }; do
    output=$("$BIN" -n "$n" -m "$m" -b "$b" -i "$ITERS" -t "$t" 2>&1)
    if [ $? -ne 0 ] || ! grep -q "^time" <<< "$output"; then
        printf "%7s %7s %5s %4s | FAILED: %s\n" "$n" "$m" "$b" "$t" \
               "$(tail -n1 <<< "$output")"
        failed=$((failed + 1))
        continue
    fi

    fields=$(extract <<< "$output")
    echo "$n,$m,$b,$t,$ITERS,$fields" >> "$OUT"
    done_count=$((done_count + 1))

    IFS=',' read -r avg tmin tmax sd gflops pj pmj pw dj dmj dw <<< "$fields"
    printf "%7s %7s %5s %4s | %9s %8s %9s | %10s %10s\n" \
           "$n" "$m" "$b" "$t" "$avg" "$sd" "$gflops" "${pmj:--}" "${dmj:--}"
done; done; done; done

echo
if [ "$failed" -gt 0 ]; then
    echo "$done_count results written to $OUT ($failed failed)"
else
    echo "$done_count results written to $OUT"
fi
[ "$failed" -eq 0 ]
