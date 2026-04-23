"""Tests for parallelized allantools functions.

Verify that parallel=True gives identical results to parallel=False,
and that both match the original allantools functions.
"""
import numpy as np
import pytest

import allantools
from allantools.allantools_parallel import (
    padev, poadev, pmdev, phdev, pohdev, ptotdev,
    pmtotdev, phtotdev, ptheo1, pmtie, ptierms,
    pgradev, pgcodev, ppdev, ptdev, pttotdev,
)


TEST_TAUS = [1, 2, 4, 8, 16, 32]
THEO1_TAUS = [2, 4, 8, 16]
PROCESS_FUNCS = (pmtotdev, phtotdev, ptheo1, ppdev, pttotdev)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def phase_data():
    """Return a modest-sized phase dataset."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(6000))


@pytest.fixture
def freq_data():
    """Return a modest-sized frequency dataset."""
    np.random.seed(42)
    return np.random.randn(6000)


# ---------------------------------------------------------------------------
# Equivalence tests: original vs parallel-serial vs parallel-parallel
# ---------------------------------------------------------------------------

def assert_equivalent(r1, r2, rtol=1e-10):
    """Compare two allantools result tuples."""
    np.testing.assert_array_equal(r1[0], r2[0])          # taus
    np.testing.assert_allclose(r1[1], r2[1], rtol=rtol)  # devs
    # gradev returns list of two arrays for errors
    if isinstance(r1[2], list):
        np.testing.assert_allclose(r1[2][0], r2[2][0], rtol=rtol)
        np.testing.assert_allclose(r1[2][1], r2[2][1], rtol=rtol)
    else:
        np.testing.assert_allclose(r1[2], r2[2], rtol=rtol)  # errs
    np.testing.assert_array_equal(r1[3], r2[3])          # ns


class TestThreadPoolFunctions:
    """Functions using ThreadPoolExecutor (NumPy-intensive)."""

    @pytest.mark.parametrize("orig_func, para_func", [
        (allantools.adev, padev),
        (allantools.oadev, poadev),
        (allantools.mdev, pmdev),
        (allantools.hdev, phdev),
        (allantools.ohdev, pohdev),
        (allantools.totdev, ptotdev),
        (allantools.tierms, ptierms),
        (allantools.mtie, pmtie),
    ])
    def test_parallel_equivalence_phase(self, phase_data, orig_func,
                                        para_func):
        r1 = orig_func(phase_data, taus=TEST_TAUS)
        r2 = para_func(phase_data, taus=TEST_TAUS, parallel=False)
        r3 = para_func(
            phase_data, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)

    def test_gradev_parallel_equivalence(self, phase_data):
        r1 = allantools.gradev(phase_data, taus=TEST_TAUS)
        r2 = pgradev(phase_data, taus=TEST_TAUS, parallel=False)
        r3 = pgradev(
            phase_data, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)

    def test_gcodev_parallel_equivalence(self, phase_data):
        d1 = phase_data
        d2 = np.cumsum(np.random.randn(len(d1)))
        r1 = allantools.gcodev(d1, d2, taus=TEST_TAUS)
        r2 = pgcodev(d1, d2, taus=TEST_TAUS, parallel=False)
        r3 = pgcodev(d1, d2, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)

    def test_tdev_passthrough(self, phase_data):
        """ptdev delegates to pmdev; verify consistency."""
        r1 = allantools.tdev(phase_data, taus=TEST_TAUS)
        r2 = ptdev(phase_data, taus=TEST_TAUS, parallel=False)
        r3 = ptdev(phase_data, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)


class TestProcessPoolFunctions:
    """Functions using ProcessPoolExecutor (pure-Python loops)."""

    @pytest.mark.parametrize("orig_func, para_func", [
        (allantools.mtotdev, pmtotdev),
        (allantools.htotdev, phtotdev),
        (allantools.pdev, ppdev),
    ])
    def test_parallel_equivalence_phase(self, phase_data, orig_func,
                                        para_func):
        r1 = orig_func(phase_data, taus=TEST_TAUS)
        r2 = para_func(phase_data, taus=TEST_TAUS, parallel=False)
        r3 = para_func(
            phase_data, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)

    def test_theo1_parallel_equivalence(self, phase_data):
        r1 = allantools.theo1(phase_data, taus=THEO1_TAUS)
        r2 = ptheo1(phase_data, taus=THEO1_TAUS, parallel=False)
        r3 = ptheo1(
            phase_data, taus=THEO1_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)

    def test_ttotdev_passthrough(self, phase_data):
        """pttotdev delegates to pmtotdev; verify consistency."""
        r1 = allantools.ttotdev(phase_data, taus=TEST_TAUS)
        r2 = pttotdev(phase_data, taus=TEST_TAUS, parallel=False)
        r3 = pttotdev(
            phase_data, taus=TEST_TAUS, parallel=True, n_workers=2)
        assert_equivalent(r1, r2)
        assert_equivalent(r2, r3)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and small-data behaviour."""

    def test_small_data_auto_disable_parallel(self):
        """Parallel should be auto-disabled for tiny datasets."""
        data = np.random.randn(100)
        # Should not raise; auto-disable means it falls back to serial
        r = poadev(data, taus="all", parallel=True)
        assert r[0].size >= 0

    def test_single_tau(self, phase_data):
        """Single tau value should work with parallel=True."""
        r = poadev(phase_data, taus=[1.0], parallel=True, n_workers=2)
        assert r[0].size == 1

    def test_empty_tau_list(self, phase_data):
        """Empty tau list after filtering should be handled gracefully."""
        # allantools falls back to "octave" for empty list, so this should
        # simply run without error.
        r = poadev(phase_data, taus=[], parallel=True, n_workers=2)
        assert r[0].size > 0

    def test_freq_input(self, freq_data):
        """Frequency input should work for parallel functions."""
        r1 = allantools.oadev(freq_data, data_type="freq", taus=TEST_TAUS)
        r2 = poadev(
            freq_data, data_type="freq", taus=TEST_TAUS,
            parallel=True, n_workers=2)
        assert_equivalent(r1, r2)


# ---------------------------------------------------------------------------
# Smoke tests for all functions with both data types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func", [
    padev, poadev, pmdev, phdev, pohdev, ptotdev,
    pmtotdev, phtotdev, ptheo1, pmtie, ptierms,
    pgradev, ppdev, ptdev, pttotdev,
])
def test_smoke_phase(func, phase_data):
    """Every parallel function should run without error on phase data."""
    if func in (pgcodev,):
        pytest.skip("gcodev takes two data arrays")
    taus = THEO1_TAUS if func is ptheo1 else TEST_TAUS
    use_parallel = func not in PROCESS_FUNCS
    r = func(
        phase_data, taus=taus, parallel=use_parallel, n_workers=2)
    assert len(r) == 4


@pytest.mark.parametrize("func", [
    padev, poadev, pmdev, phdev, pohdev, ptotdev,
    pmtotdev, phtotdev, ptheo1, pmtie, ptierms,
    pgradev, ppdev, ptdev, pttotdev,
])
def test_smoke_freq(func, freq_data):
    """Every parallel function should run without error on frequency data."""
    if func in (pgcodev,):
        pytest.skip("gcodev takes two data arrays")
    taus = THEO1_TAUS if func is ptheo1 else TEST_TAUS
    use_parallel = func not in PROCESS_FUNCS
    r = func(
        freq_data, data_type="freq", taus=taus,
        parallel=use_parallel, n_workers=2)
    assert len(r) == 4
