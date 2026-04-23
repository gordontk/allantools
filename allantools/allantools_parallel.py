"""Parallel accelerated version of allantools core functions.

This module provides parallelized versions of allantools deviation functions.
All functions reuse the original calculation kernels from ``allantools.py``.

Naming convention: prefix ``p`` to the original function name.
  ``oadev``  -> ``poadev``
  ``adev``   -> ``padev``
  ``mdev``   -> ``pmdev``
  ...

Usage
-----
Import and use exactly like the original functions, with optional
``parallel=True`` to enable parallel execution::

    from allantools.allantools_parallel import poadev
    taus, devs, errs, ns = poadev(data, rate=1.0, taus="octave", parallel=True)

Parallelization is automatically disabled for small problems where the
overhead would exceed the benefit.
"""
from functools import partial
import numpy as np

# Reuse original helper functions and calculation kernels
from .allantools import (
    input_to_phase,
    tau_generator,
    remove_small_ns,
    calc_adev_phase,
    calc_hdev_phase,
    calc_mtotdev_phase,
    calc_htotdev_freq,
    calc_gradev_phase,
    calc_gcodev_phase,
    calc_pdev_phase,
    frequency2phase,
    phase2frequency,
)
from .parallel import (
    parallel_map_selective,
    _should_parallelize,
)


MIN_PARALLEL_TAU_WORK = 250000


def _unpack_results(results, arrays):
    """Unpack a list of (dev, err, n) tuples into pre-allocated arrays.

    Parameters
    ----------
    results : list of tuple
        Each tuple is (dev, err, n).
    arrays : tuple of np.ndarray
        Three arrays: (devs, errs, ns).
    """
    devs, errs, ns = arrays
    for idx, (dev, err, n) in enumerate(results):
        devs[idx] = dev
        errs[idx] = err
        ns[idx] = n


def _unpack_results_gradev(results, arrays):
    """Unpack gradev results where error is a list of two arrays."""
    devs, errs_l, errs_h, ns = arrays
    for idx, (dev, err, n) in enumerate(results):
        devs[idx] = dev
        errs_l[idx] = err[0]
        errs_h[idx] = err[1]
        ns[idx] = n


def _calc_htotdev_worker(args):
    """Pickleable ProcessPool worker for htotdev tau values."""
    phase, freq, rate, mj = args
    mj = int(mj)
    if mj == 1:
        return calc_hdev_phase(phase, rate, mj, 1)
    return calc_htotdev_freq(freq, mj)


def _calc_theo1_worker(args):
    """Pickleable ProcessPool worker for theo1 tau values."""
    phase, rate, m = args
    tau0 = 1.0 / rate
    n_phase = len(phase)
    m = int(m)
    assert m % 2 == 0
    dev = 0
    n = 0
    half_m = int(m / 2)
    for i in range(int(n_phase - m)):
        s = 0
        for d in range(half_m):
            pre = 1.0 / (float(m) / 2 - float(d))
            s += pre * pow(
                phase[i] - phase[i - d + half_m] +
                phase[i + m] - phase[i + d + half_m], 2)
            n = n + 1
        dev += s
    dev = dev / (0.75 * (n_phase - m) * pow(m * tau0, 2))
    dev_val = np.sqrt(dev)
    return dev_val, dev_val / np.sqrt(n_phase - m), n


def _oadev_tau_work(data_size, mj):
    """Return the approximate number of samples used by one oadev tau."""
    return max(0, int(data_size) - 2 * int(mj))


def _stride_tau_work(data_size, mj, order, stride):
    """Approximate samples touched by stride-based variance kernels."""
    data_size = int(data_size)
    mj = int(mj)
    stride = int(stride)
    if stride <= 1:
        return max(0, data_size - order * mj)
    return max(0, (data_size - order * mj) // stride)


def _mdev_tau_work(data_size, mj):
    """Approximate samples touched by modified Allan deviation."""
    return max(0, int(data_size) - 3 * int(mj))


def _totdev_tau_work(data_size, mj):
    """Approximate samples touched by total deviation."""
    return max(0, int(data_size) - 2)


def _tierms_tau_work(data_size, mj):
    """Approximate samples touched by TIE RMS."""
    return max(0, int(data_size) - int(mj))


def _mtie_tau_work(data_size, mj):
    """Approximate window work for MTIE."""
    return max(0, int(data_size) - int(mj)) * max(1, int(mj))


def _mtotdev_tau_work(data_size, mj):
    """Approximate inner-loop work for modified total deviation."""
    mj = int(mj)
    return max(0, int(data_size) - 3 * mj + 1) * max(1, mj)


def _htotdev_tau_work(data_size, mj):
    """Approximate inner-loop work for Hadamard total deviation."""
    mj = int(mj)
    if mj == 1:
        return _stride_tau_work(data_size, mj, order=3, stride=1)
    return max(0, int(data_size) - 3 * mj + 1) * max(1, mj)


def _theo1_tau_work(data_size, mj):
    """Approximate nested-loop work for Theo1."""
    mj = int(mj)
    return max(0, int(data_size) - mj) * max(1, int(mj / 2))


def _tau_parallel_mask(items, data_size, work_func,
                       min_work=MIN_PARALLEL_TAU_WORK):
    """Return a mask selecting tau items large enough to parallelize."""
    return [work_func(data_size, item) >= min_work for item in items]


def _task_parallel_mask(tasks, data_size, work_func,
                        min_work=MIN_PARALLEL_TAU_WORK):
    """Return a boolean mask for task tuples with tau in the last position."""
    return [work_func(data_size, task[-1]) >= min_work for task in tasks]


###############################################################################
# ThreadPool functions (NumPy-intensive, GIL released)
###############################################################################

def padev(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated Allan deviation.

    See ``allantools.adev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    ad = np.zeros_like(taus_used)
    ade = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_adev(mj):
            return calc_adev_phase(phase, rate, mj, mj)
        parallel_mask = _tau_parallel_mask(
            m, len(phase),
            lambda data_size, mj: _stride_tau_work(
                data_size, mj, order=2, stride=mj))
        results = parallel_map_selective(
            _calc_adev, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (ad, ade, adn))
    else:
        for idx, mj in enumerate(m):
            (ad[idx], ade[idx], adn[idx]) = calc_adev_phase(
                phase, rate, mj, mj)

    return remove_small_ns(taus_used, ad, ade, adn)


def poadev(data, rate=1.0, data_type="phase", taus=None,
           parallel=False, n_workers=-1):
    """Parallel accelerated overlapping Allan deviation.

    See ``allantools.oadev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    ad = np.zeros_like(taus_used)
    ade = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        _calc = partial(calc_adev_phase, phase, rate, stride=1)
        parallel_mask = _tau_parallel_mask(m, len(phase), _oadev_tau_work)
        results = parallel_map_selective(
            _calc, m, parallel_mask, parallel=True, n_workers=n_workers)
        _unpack_results(results, (ad, ade, adn))
    else:
        for idx, mj in enumerate(m):
            (ad[idx], ade[idx], adn[idx]) = calc_adev_phase(phase, rate, mj, 1)

    return remove_small_ns(taus_used, ad, ade, adn)


def pmdev(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated modified Allan deviation.

    See ``allantools.mdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, ms, taus_used) = tau_generator(phase, rate, taus=taus)
    phase_arr = np.array(phase)

    md = np.zeros_like(ms)
    mderr = np.zeros_like(ms)
    ns = np.zeros_like(ms)

    if parallel and _should_parallelize(len(ms), len(phase_arr)):
        # mdev inner loop is not a simple helper, so we define a closure
        def _calc_mdev(mj):
            m = int(mj)
            tau = mj / float(rate)
            d0 = phase_arr[0:m]
            d1 = phase_arr[m:2 * m]
            d2 = phase_arr[2 * m:3 * m]
            e = min(len(d0), len(d1), len(d2))
            v = np.sum(d2[:e] - 2 * d1[:e] + d0[:e])
            s = v * v

            d3 = phase_arr[3 * m:]
            d2 = phase_arr[2 * m:]
            d1 = phase_arr[1 * m:]
            d0 = phase_arr[0:]

            e = min(len(d0), len(d1), len(d2), len(d3))
            n = e + 1

            v_arr = v + np.cumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])
            s = s + np.sum(v_arr * v_arr)
            s /= 2.0 * m * m * tau * tau * n
            s = np.sqrt(s)
            return s, s / np.sqrt(n), n

        parallel_mask = _tau_parallel_mask(ms, len(phase_arr), _mdev_tau_work)
        results = parallel_map_selective(
            _calc_mdev, ms, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (md, mderr, ns))
    else:
        for idx, m in enumerate(ms):
            m = int(m)
            tau = taus_used[idx]
            d0 = phase_arr[0:m]
            d1 = phase_arr[m:2 * m]
            d2 = phase_arr[2 * m:3 * m]
            e = min(len(d0), len(d1), len(d2))
            v = np.sum(d2[:e] - 2 * d1[:e] + d0[:e])
            s = v * v

            d3 = phase_arr[3 * m:]
            d2 = phase_arr[2 * m:]
            d1 = phase_arr[1 * m:]
            d0 = phase_arr[0:]

            e = min(len(d0), len(d1), len(d2), len(d3))
            n = e + 1

            v_arr = v + np.cumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])
            s = s + np.sum(v_arr * v_arr)
            s /= 2.0 * m * m * tau * tau * n
            s = np.sqrt(s)

            md[idx] = s
            mderr[idx] = s / np.sqrt(n)
            ns[idx] = n

    return remove_small_ns(taus_used, md, mderr, ns)


def phdev(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated Hadamard deviation.

    See ``allantools.hdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    hdevs = np.zeros_like(taus_used)
    hdeverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_hdev(mj):
            return calc_hdev_phase(phase, rate, mj, mj)
        parallel_mask = _tau_parallel_mask(
            m, len(phase),
            lambda data_size, mj: _stride_tau_work(
                data_size, mj, order=3, stride=mj))
        results = parallel_map_selective(
            _calc_hdev, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (hdevs, hdeverrs, ns))
    else:
        for idx, mj in enumerate(m):
            (hdevs[idx],
             hdeverrs[idx],
             ns[idx]) = calc_hdev_phase(phase, rate, mj, mj)

    return remove_small_ns(taus_used, hdevs, hdeverrs, ns)


def pohdev(data, rate=1.0, data_type="phase", taus=None,
           parallel=False, n_workers=-1):
    """Parallel accelerated overlapping Hadamard deviation.

    See ``allantools.ohdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    hdevs = np.zeros_like(taus_used)
    hdeverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        _calc = partial(calc_hdev_phase, phase, rate, stride=1)
        parallel_mask = _tau_parallel_mask(
            m, len(phase),
            lambda data_size, mj: _stride_tau_work(
                data_size, mj, order=3, stride=1))
        results = parallel_map_selective(
            _calc, m, parallel_mask, parallel=True, n_workers=n_workers)
        _unpack_results(results, (hdevs, hdeverrs, ns))
    else:
        for idx, mj in enumerate(m):
            (hdevs[idx],
             hdeverrs[idx],
             ns[idx]) = calc_hdev_phase(phase, rate, mj, 1)

    return remove_small_ns(taus_used, hdevs, hdeverrs, ns)


def ptotdev(data, rate=1.0, data_type="phase", taus=None,
            parallel=False, n_workers=-1):
    """Parallel accelerated total deviation.

    See ``allantools.totdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    N = len(phase)

    x1 = 2.0 * phase[0] * np.ones((N - 2,))
    x1 = x1 - phase[1:-1]
    x1 = x1[::-1]

    x2 = 2.0 * phase[-1] * np.ones((N - 2,))
    x2 = x2 - phase[1:-1][::-1]

    assert len(x1) + len(phase) + len(x2) == 3 * N - 4
    x = np.zeros((3 * N - 4))
    x[0:N - 2] = x1
    x[N - 2:2 * (N - 2) + 2] = phase
    x[2 * (N - 2) + 2:] = x2

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)
    mid = len(x1)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_totdev(mj):
            mj = int(mj)
            d0 = x[mid + 1:]
            d1 = x[mid + mj + 1:]
            d1n = x[mid - mj + 1:]
            e = min(len(d0), len(d1), len(d1n))
            v_arr = d1n[:e] - 2.0 * d0[:e] + d1[:e]
            dev = np.sum(v_arr[:mid] * v_arr[:mid])
            dev /= float(2 * pow(mj / rate, 2) * (N - 2))
            dev = np.sqrt(dev)
            return dev, dev / np.sqrt(mid), mid

        parallel_mask = _tau_parallel_mask(m, len(phase), _totdev_tau_work)
        results = parallel_map_selective(
            _calc_totdev, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        for idx, mj in enumerate(m):
            mj = int(mj)
            d0 = x[mid + 1:]
            d1 = x[mid + mj + 1:]
            d1n = x[mid - mj + 1:]
            e = min(len(d0), len(d1), len(d1n))

            v_arr = d1n[:e] - 2.0 * d0[:e] + d1[:e]
            dev = np.sum(v_arr[:mid] * v_arr[:mid])

            dev /= float(2 * pow(mj / rate, 2) * (N - 2))
            dev = np.sqrt(dev)
            devs[idx] = dev
            deverrs[idx] = dev / np.sqrt(mid)
            ns[idx] = mid

    return remove_small_ns(taus_used, devs, deverrs, ns)


def ptierms(data, rate=1.0, data_type="phase", taus=None,
            parallel=False, n_workers=-1):
    """Parallel accelerated time interval error RMS.

    See ``allantools.tierms`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (data, m, taus_used) = tau_generator(phase, rate, taus)
    count = len(phase)

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_tierms(mj):
            mj = int(mj)
            phases = np.column_stack((phase[:-mj], phase[mj:]))
            p_max = np.max(phases, axis=1)
            p_min = np.min(phases, axis=1)
            phases = p_max - p_min
            tie = np.sqrt(np.mean(phases * phases))
            ncount = count - mj
            return tie, 0.0 / np.sqrt(ncount), ncount

        parallel_mask = _tau_parallel_mask(m, len(phase), _tierms_tau_work)
        results = parallel_map_selective(
            _calc_tierms, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        for idx, mj in enumerate(m):
            mj = int(mj)
            phases = np.column_stack((phase[:-mj], phase[mj:]))
            p_max = np.max(phases, axis=1)
            p_min = np.min(phases, axis=1)
            phases = p_max - p_min
            tie = np.sqrt(np.mean(phases * phases))
            ncount = count - mj
            devs[idx] = tie
            deverrs[idx] = 0 / np.sqrt(ncount)
            ns[idx] = ncount

    return remove_small_ns(taus_used, devs, deverrs, ns)


def pmtie(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated maximum time interval error.

    See ``allantools.mtie`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_mtie(mj):
            mj = int(mj)
            try:
                from .allantools import mtie_rolling_window
                rw = mtie_rolling_window(phase, int(mj + 1))
                win_max = np.max(rw, axis=1)
                win_min = np.min(rw, axis=1)
                tie = win_max - win_min
                dev = np.max(tie)
            except ValueError:
                if int(mj + 1) < 1:
                    raise ValueError("`window` must be at least 1.")
                if int(mj + 1) > phase.shape[-1]:
                    raise ValueError("`window` is too long.")
                mj = int(mj)
                currMax = np.max(phase[0:mj])
                currMin = np.min(phase[0:mj])
                dev = currMax - currMin
                for winStartIdx in range(1, int(phase.shape[0] - mj)):
                    winEndIdx = mj + winStartIdx
                    if currMax == phase[winStartIdx - 1]:
                        currMax = np.max(phase[winStartIdx:winEndIdx])
                    elif currMax < phase[winEndIdx]:
                        currMax = phase[winEndIdx]
                    if currMin == phase[winStartIdx - 1]:
                        currMin = np.min(phase[winStartIdx:winEndIdx])
                    elif currMin > phase[winEndIdx]:
                        currMin = phase[winEndIdx]
                    if dev < currMax - currMin:
                        dev = currMax - currMin
            ncount = phase.shape[0] - mj
            return dev, dev / np.sqrt(ncount), ncount

        parallel_mask = _tau_parallel_mask(m, len(phase), _mtie_tau_work)
        results = parallel_map_selective(
            _calc_mtie, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        for idx, mj in enumerate(m):
            try:
                from .allantools import mtie_rolling_window
                rw = mtie_rolling_window(phase, int(mj + 1))
                win_max = np.max(rw, axis=1)
                win_min = np.min(rw, axis=1)
                tie = win_max - win_min
                dev = np.max(tie)
            except ValueError:
                if int(mj + 1) < 1:
                    raise ValueError("`window` must be at least 1.")
                if int(mj + 1) > phase.shape[-1]:
                    raise ValueError("`window` is too long.")
                mj = int(mj)
                currMax = np.max(phase[0:mj])
                currMin = np.min(phase[0:mj])
                dev = currMax - currMin
                for winStartIdx in range(1, int(phase.shape[0] - mj)):
                    winEndIdx = mj + winStartIdx
                    if currMax == phase[winStartIdx - 1]:
                        currMax = np.max(phase[winStartIdx:winEndIdx])
                    elif currMax < phase[winEndIdx]:
                        currMax = phase[winEndIdx]
                    if currMin == phase[winStartIdx - 1]:
                        currMin = np.min(phase[winStartIdx:winEndIdx])
                    elif currMin > phase[winEndIdx]:
                        currMin = phase[winEndIdx]
                    if dev < currMax - currMin:
                        dev = currMax - currMin
            ncount = phase.shape[0] - mj
            devs[idx] = dev
            deverrs[idx] = dev / np.sqrt(ncount)
            ns[idx] = ncount

    return remove_small_ns(taus_used, devs, deverrs, ns)


def pgradev(data, rate=1.0, data_type="phase", taus=None,
            ci=0.9, noisetype='wp', parallel=False, n_workers=-1):
    """Parallel accelerated gap resistant overlapping Allan deviation.

    See ``allantools.gradev`` for full documentation.
    """
    if data_type == "freq":
        print("Warning : phase data is preferred as input to gradev()")
    phase = input_to_phase(data, rate, data_type)
    (data, m, taus_used) = tau_generator(phase, rate, taus)

    ad = np.zeros_like(taus_used)
    ade_l = np.zeros_like(taus_used)
    ade_h = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        def _calc_gradev(mj):
            (dev, deverr, n) = calc_gradev_phase(
                data, rate, mj, 1, ci, noisetype)
            return dev, deverr, n

        parallel_mask = _tau_parallel_mask(m, len(phase), _oadev_tau_work)
        results = parallel_map_selective(
            _calc_gradev, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results_gradev(results, (ad, ade_l, ade_h, adn))
    else:
        for idx, mj in enumerate(m):
            (dev, deverr, n) = calc_gradev_phase(
                data, rate, mj, 1, ci, noisetype)
            ad[idx] = dev
            ade_l[idx] = deverr[0]
            ade_h[idx] = deverr[1]
            adn[idx] = n

    return remove_small_ns(taus_used, ad, [ade_l, ade_h], adn)


def pgcodev(data_1, data_2, rate=1.0, data_type="phase", taus=None,
            parallel=False, n_workers=-1):
    """Parallel accelerated gap resistant overlapping Allan codeviation.

    See ``allantools.gcodev`` for full documentation.
    """
    phase_1 = input_to_phase(data_1, rate, data_type)
    phase_2 = input_to_phase(data_2, rate, data_type)
    (phase_1, m, taus_used) = tau_generator(phase_1, rate, taus)
    (phase_2, m, taus_used) = tau_generator(phase_2, rate, taus)

    gd = np.zeros_like(taus_used)
    gde = np.zeros_like(taus_used)
    gdn = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase_1)):
        def _calc_gcodev(mj):
            return calc_gcodev_phase(phase_1, phase_2, rate, mj, stride=1)

        parallel_mask = _tau_parallel_mask(m, len(phase_1), _oadev_tau_work)
        results = parallel_map_selective(
            _calc_gcodev, m, parallel_mask, parallel=True,
            n_workers=n_workers)
        _unpack_results(results, (gd, gde, gdn))
    else:
        for idx, mj in enumerate(m):
            (gd[idx], gde[idx], gdn[idx]) = calc_gcodev_phase(
                phase_1, phase_2, rate, mj, stride=1)

    return remove_small_ns(taus_used, gd, gde, gdn)


###############################################################################
# ProcessPool functions (pure-Python loops, GIL-bound)
###############################################################################

def pmtotdev(data, rate=1.0, data_type="phase", taus=None,
             parallel=False, n_workers=-1):
    """Parallel accelerated modified total deviation.

    See ``allantools.mtotdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, ms, taus_used) = tau_generator(
        phase, rate, taus, maximum_m=float(len(phase)) / 3.0)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(ms), len(phase)):
        _calc = partial(calc_mtotdev_phase, phase, rate)
        parallel_mask = _tau_parallel_mask(ms, len(phase), _mtotdev_tau_work)
        results = parallel_map_selective(
            _calc, ms, parallel_mask, parallel=True,
            n_workers=n_workers, use_processes=True)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        for idx, mj in enumerate(ms):
            devs[idx], deverrs[idx], ns[idx] = calc_mtotdev_phase(
                phase, rate, mj)

    return remove_small_ns(taus_used, devs, deverrs, ns)


def phtotdev(data, rate=1.0, data_type="phase", taus=None,
             parallel=False, n_workers=-1):
    """Parallel accelerated Hadamard total deviation.

    See ``allantools.htotdev`` for full documentation.
    """
    if data_type == "phase":
        phase = data
        freq = phase2frequency(phase, rate)
    elif data_type == "freq":
        phase = frequency2phase(data, rate)
        freq = data
    else:
        raise Exception("unknown data_type: " + data_type)

    rate = float(rate)
    (freq, ms, taus_used) = tau_generator(
        freq, rate, taus, maximum_m=float(len(freq)) / 3.0)
    phase = np.array(phase)
    freq = np.array(freq)
    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(ms), len(phase)):
        tasks = [(phase, freq, rate, mj) for mj in ms]
        parallel_mask = _task_parallel_mask(tasks, len(phase),
                                            _htotdev_tau_work)
        results = parallel_map_selective(
            _calc_htotdev_worker, tasks, parallel_mask, parallel=True,
            n_workers=n_workers, use_processes=True)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        for idx, mj in enumerate(ms):
            if int(mj) == 1:
                (devs[idx],
                 deverrs[idx],
                 ns[idx]) = calc_hdev_phase(phase, rate, mj, 1)
            else:
                (devs[idx],
                 deverrs[idx],
                 ns[idx]) = calc_htotdev_freq(freq, mj)

    return remove_small_ns(taus_used, devs, deverrs, ns)


def ptheo1(data, rate=1.0, data_type="phase", taus=None,
           parallel=False, n_workers=-1):
    """Parallel accelerated Theo1 deviation.

    See ``allantools.theo1`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    tau0 = 1.0 / rate
    (phase, ms, taus_used) = tau_generator(phase, rate, taus, even=True)

    devs = np.zeros_like(taus_used)
    deverrs = np.zeros_like(taus_used)
    ns = np.zeros_like(taus_used)
    if parallel and _should_parallelize(len(ms), len(phase)):
        tasks = [(phase, rate, m) for m in ms]
        parallel_mask = _task_parallel_mask(tasks, len(phase),
                                            _theo1_tau_work)
        results = parallel_map_selective(
            _calc_theo1_worker, tasks, parallel_mask, parallel=True,
            n_workers=n_workers, use_processes=True)
        _unpack_results(results, (devs, deverrs, ns))
    else:
        N = len(phase)
        for idx, m in enumerate(ms):
            m = int(m)
            assert m % 2 == 0
            dev = 0
            n = 0
            for i in range(int(N - m)):
                s = 0
                for d in range(int(m / 2)):
                    pre = 1.0 / (float(m) / 2 - float(d))
                    s += pre * pow(
                        phase[i] - phase[i - d + int(m / 2)] +
                        phase[i + m] - phase[i + d + int(m / 2)], 2)
                    n = n + 1
                dev += s
            dev = dev / (0.75 * (N - m) * pow(m * tau0, 2))
            devs[idx] = np.sqrt(dev)
            deverrs[idx] = devs[idx] / np.sqrt(N - m)
            ns[idx] = n

    return remove_small_ns(taus_used, devs, deverrs, ns)


def ppdev(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated parabolic deviation.

    See ``allantools.pdev`` for full documentation.
    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, taus)
    ad = np.zeros_like(taus_used)
    ade = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    if parallel and _should_parallelize(len(m), len(phase)):
        _calc = partial(calc_pdev_phase, phase, rate)
        parallel_mask = _tau_parallel_mask(
            m, len(phase),
            lambda data_size, mj: _stride_tau_work(
                data_size, mj, order=2, stride=1) * max(1, int(mj)))
        results = parallel_map_selective(
            _calc, m, parallel_mask, parallel=True,
            n_workers=n_workers, use_processes=True)
        _unpack_results(results, (ad, ade, adn))
    else:
        for idx, mj in enumerate(m):
            (ad[idx], ade[idx], adn[idx]) = calc_pdev_phase(phase, rate, mj)

    return remove_small_ns(taus_used, ad, ade, adn)


###############################################################################
# Passthrough functions
###############################################################################

def ptdev(data, rate=1.0, data_type="phase", taus=None,
          parallel=False, n_workers=-1):
    """Parallel accelerated time deviation (delegates to pmdev).

    See ``allantools.tdev`` for full documentation.
    """
    (taus, md, mde, ns) = pmdev(data, rate=rate, data_type=data_type,
                                taus=taus, parallel=parallel,
                                n_workers=n_workers)
    td = taus * md / np.sqrt(3.0)
    tde = td / np.sqrt(ns)
    return taus, td, tde, ns


def pttotdev(data, rate=1.0, data_type="phase", taus=None,
             parallel=False, n_workers=-1):
    """Parallel accelerated total time deviation (delegates to pmtotdev).

    See ``allantools.ttotdev`` for full documentation.
    """
    (taus, mtotdevs, mde, ns) = pmtotdev(
        data, rate=rate, data_type=data_type, taus=taus,
        parallel=parallel, n_workers=n_workers)
    td = taus * mtotdevs / np.sqrt(3.0)
    tde = td / np.sqrt(ns)
    return taus, td, tde, ns
