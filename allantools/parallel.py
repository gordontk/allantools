"""Parallel computation utilities for allantools.

This module provides infrastructure for parallelizing allantools deviation
functions using concurrent.futures. ThreadPoolExecutor is used for NumPy-
intensive functions (NumPy releases the GIL), and ProcessPoolExecutor for
pure-Python loop-heavy functions.
"""
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def _get_n_workers(n_workers):
    """Return sensible number of workers.

    Parameters
    ----------
    n_workers : int
        Requested worker count. -1 means auto-detect.

    Returns
    -------
    int
        Actual worker count to use.
    """
    if n_workers == -1:
        return min(6, os.cpu_count() or 1)
    return max(1, n_workers)


def _should_parallelize(n_taus, data_size, threshold_taus=4,
                        threshold_size=5000):
    """Auto-disable parallelization for small problems.

    The overhead of spawning threads/processes can exceed the benefit for
    small datasets or few tau values.

    Parameters
    ----------
    n_taus : int
        Number of tau values to compute.
    data_size : int
        Length of the input data array.
    threshold_taus : int, optional
        Minimum number of taus to consider parallelization.
    threshold_size : int, optional
        Minimum data size to consider parallelization.

    Returns
    -------
    bool
        True if parallelization is expected to be beneficial.
    """
    return n_taus >= threshold_taus and data_size >= threshold_size


def parallel_map(func, items, parallel=False, n_workers=-1,
                 use_processes=False):
    """Map *func* over *items*, optionally in parallel.

    Parameters
    ----------
    func : callable
        Function to apply to each item.
    items : iterable
        Items to process.
    parallel : bool, optional
        If True, use parallel execution. Default is False (serial).
    n_workers : int, optional
        Number of workers. -1 means auto-detect (up to 6 or CPU count).
    use_processes : bool, optional
        If True, use ProcessPoolExecutor instead of ThreadPoolExecutor.
        Required for functions with heavy pure-Python inner loops to bypass
        the GIL. Note: *func* must be pickleable (no lambdas; use
        ``functools.partial`` or top-level functions).

    Returns
    -------
    list
        Results in the same order as *items*.
    """
    items = list(items)
    if not parallel or len(items) <= 1:
        return [func(item) for item in items]

    n_workers = min(_get_n_workers(n_workers), len(items))
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    try:
        with Executor(max_workers=n_workers) as exe:
            return list(exe.map(func, items))
    except OSError:
        if not use_processes:
            raise

    # Restricted Windows environments can block ProcessPool named pipes.
    # Fall back to threads to preserve correctness and API behaviour.
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        return list(exe.map(func, items))


def parallel_map_selective(func, items, parallel_mask, parallel=False,
                           n_workers=-1, use_processes=False):
    """Map *func* with only selected items sent to a parallel executor.

    ``parallel_mask`` must have one boolean per item. Items marked False are
    computed serially, which avoids executor overhead for small tau workloads
    while preserving result order.
    """
    items = list(items)
    parallel_mask = list(parallel_mask)
    if len(items) != len(parallel_mask):
        raise ValueError("items and parallel_mask must have the same length")
    if not items:
        return []
    if not parallel:
        return [func(item) for item in items]

    results = [None] * len(items)
    parallel_jobs = []
    for idx, item in enumerate(items):
        if parallel_mask[idx]:
            parallel_jobs.append((idx, item))
        else:
            results[idx] = func(item)

    if not parallel_jobs:
        return results

    mapped = parallel_map(
        func, [item for _, item in parallel_jobs], parallel=True,
        n_workers=n_workers, use_processes=use_processes)
    for (idx, _), value in zip(parallel_jobs, mapped):
        results[idx] = value
    return results
