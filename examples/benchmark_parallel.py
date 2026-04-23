"""Benchmark script comparing original and parallel allantools functions.

Run from the repo root with::

    python examples/benchmark_parallel.py

The script compares execution times for:
  - Original allantools function
  - Parallel version with parallel=False (serial fallback)
  - Parallel version with parallel=True

Results are printed in a simple table.
"""
import time
import numpy as np

import allantools
from allantools.allantools_parallel import (
    padev, poadev, pmdev, phdev, pohdev, ptotdev,
    pmtotdev, phtotdev, ptheo1, pmtie, ptierms,
    pgradev, ppdev, ptdev, pttotdev,
)


def benchmark(func, *args, n_runs=3, **kwargs):
    """Return median execution time over *n_runs* repetitions."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2]


def print_row(name, t_orig, t_serial, t_para):
    """Pretty-print a timing comparison row."""
    if t_orig is None:
        t_orig_str = "    --"
    else:
        t_orig_str = f"{t_orig:6.3f}"

    if t_serial is None:
        t_serial_str = "    --"
    else:
        t_serial_str = f"{t_serial:6.3f}"

    if t_para is None:
        t_para_str = "    --"
    else:
        t_para_str = f"{t_para:6.3f}"

    speedup = ""
    if t_orig is not None and t_para is not None and t_para > 0:
        speedup = f"  ({t_orig / t_para:4.1f}x)"

    print(
        f"  {name:12s}  {t_orig_str}>  "
        f"{t_serial_str}>  {t_para_str}{speedup}")


def main():
    np.random.seed(0)
    N = 50_000
    data = np.cumsum(np.random.randn(N))
    taus = "octave"
    n_runs = 3

    print(f"Benchmark: N={N}, taus={taus!r}, median of {n_runs} runs")
    print(
        f"  {'Function':12s}  {'Orig':>6s}  "
        f"{'Serial':>6s}  {'Parallel':>6s}")
    print("-" * 50)

    # --- ThreadPool functions ---
    funcs_thread = [
        ("adev",   allantools.adev,   padev),
        ("oadev",  allantools.oadev,  poadev),
        ("mdev",   allantools.mdev,   pmdev),
        ("hdev",   allantools.hdev,   phdev),
        ("ohdev",  allantools.ohdev,  pohdev),
        ("totdev", allantools.totdev, ptotdev),
        ("tierms", allantools.tierms, ptierms),
        ("mtie",   allantools.mtie,   pmtie),
        ("gradev", allantools.gradev, pgradev),
        ("tdev",   allantools.tdev,   ptdev),
    ]

    for name, orig, para in funcs_thread:
        try:
            t_orig = benchmark(orig, data, taus=taus, n_runs=n_runs)
        except Exception as exc:
            print(f"  {name:12s}  ORIG FAILED: {exc}")
            continue
        t_serial = benchmark(para, data, taus=taus, parallel=False,
                             n_runs=n_runs)
        t_para = benchmark(para, data, taus=taus, parallel=True,
                           n_runs=n_runs)
        print_row(name, t_orig, t_serial, t_para)

    # --- ProcessPool functions (heavier, fewer runs) ---
    print("-" * 50)
    print("ProcessPool functions (pure-Python loops):")
    funcs_process = [
        ("mtotdev", allantools.mtotdev, pmtotdev),
        ("htotdev", allantools.htotdev, phtotdev),
        ("theo1",   allantools.theo1,   ptheo1),
        ("pdev",    allantools.pdev,    ppdev),
        ("ttotdev", allantools.ttotdev, pttotdev),
    ]

    for name, orig, para in funcs_process:
        try:
            t_orig = benchmark(orig, data, taus=taus, n_runs=n_runs)
        except Exception as exc:
            print(f"  {name:12s}  ORIG FAILED: {exc}")
            continue
        t_serial = benchmark(para, data, taus=taus, parallel=False,
                             n_runs=n_runs)
        t_para = benchmark(para, data, taus=taus, parallel=True,
                           n_runs=n_runs)
        print_row(name, t_orig, t_serial, t_para)

    print("-" * 50)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
