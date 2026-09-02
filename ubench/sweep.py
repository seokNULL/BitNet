#!/usr/bin/env python3
"""Sweep gemm_i2s_ubench over n / m / b / thread combinations and collect results.

Examples:
    ./sweep.py -n 1024,2048,4096 -m 4096 -t 1,2,4,8
    ./sweep.py -n 4096 -m 14336 -b 1,32,128 -t 8 -i 200 -o llama3_ffn.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_i2s_ubench")

FIELDS = ["n", "m", "b", "threads", "iters",
          "avg_ms", "min_ms", "max_ms", "stddev_ms", "gflops",
          "pkg_j", "pkg_mj_per_iter", "pkg_w",
          "dram_j", "dram_mj_per_iter", "dram_w"]

RE_TIME = re.compile(
    r"avg\s+([\d.]+)\s+ms\s+\|\s+min\s+([\d.]+)\s+\|\s+max\s+([\d.]+)\s+\|\s+stddev\s+([\d.]+)")
RE_PERF = re.compile(r"perf\s*:\s*([\d.]+)\s+GFLOPS")
RE_PKG = re.compile(r"package\s+([\d.]+)\s+J total,\s+([\d.]+)\s+mJ/iter,\s+([\d.]+)\s+W")
RE_DRAM = re.compile(r"dram\s+([\d.]+)\s+J total,\s+([\d.]+)\s+mJ/iter,\s+([\d.]+)\s+W")


def parse_output(text):
    """Pull the numbers out of one benchmark run; returns a dict of metrics."""
    row = {}
    t = RE_TIME.search(text)
    if not t:
        return None
    row["avg_ms"], row["min_ms"], row["max_ms"], row["stddev_ms"] = t.groups()

    p = RE_PERF.search(text)
    row["gflops"] = p.group(1) if p else ""

    pkg = RE_PKG.search(text)
    if pkg:
        row["pkg_j"], row["pkg_mj_per_iter"], row["pkg_w"] = pkg.groups()
    dram = RE_DRAM.search(text)
    if dram:
        row["dram_j"], row["dram_mj_per_iter"], row["dram_w"] = dram.groups()
    return row


def int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int_list, default=[1024, 2048, 4096, 8192],
                    help="inner dimensions, comma-separated (multiples of 128)")
    ap.add_argument("-m", type=int_list, default=[1024, 2048, 4096, 8192],
                    help="weight rows / output dims, comma-separated")
    ap.add_argument("-b", type=int_list, default=[1],
                    help="batch sizes, comma-separated (default: 1)")
    ap.add_argument("-t", "--threads", type=int_list, default=[1],
                    help="thread counts, comma-separated (default: 1)")
    ap.add_argument("-i", "--iters", type=int, default=100, help="iterations per run")
    ap.add_argument("-o", "--output", default="sweep.csv", help="output CSV path")
    args = ap.parse_args()

    if not os.path.exists(BIN):
        sys.exit(f"{BIN} not found - run 'make' first")

    combos = [(n, m, b, t) for n in args.n for m in args.m
              for b in args.b for t in args.threads]
    print(f"{len(combos)} configurations, {args.iters} iterations each\n")

    hdr = f"{'n':>7} {'m':>7} {'b':>5} {'thr':>4} | {'avg_ms':>9} {'sd_ms':>8} {'GFLOPS':>9}"
    hdr += f" | {'pkg_mJ/it':>10} {'dram_mJ/it':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    failed = 0
    for n, m, b, t in combos:
        cmd = [BIN, "-n", str(n), "-m", str(m), "-b", str(b),
               "-i", str(args.iters), "-t", str(t)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"{n:>7} {m:>7} {b:>5} {t:>4} | FAILED: "
                  f"{(proc.stderr or proc.stdout).strip().splitlines()[-1:]}")
            failed += 1
            continue

        row = parse_output(proc.stdout)
        if row is None:
            print(f"{n:>7} {m:>7} {b:>5} {t:>4} | could not parse output")
            failed += 1
            continue

        row.update(n=n, m=m, b=b, threads=t, iters=args.iters)
        rows.append(row)
        print(f"{n:>7} {m:>7} {b:>5} {t:>4} | {row['avg_ms']:>9} {row['stddev_ms']:>8} "
              f"{row['gflops']:>9} | {row.get('pkg_mj_per_iter', '-'):>10} "
              f"{row.get('dram_mj_per_iter', '-'):>10}")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} results written to {args.output}"
          + (f" ({failed} failed)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
