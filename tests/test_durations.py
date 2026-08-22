"""Unit tests for pymaslow.durations."""

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

import numpy as np
import pytest

import pymaslow
from pymaslow import durations

# ---------------------------------------------------------------------
# Embedded data
# ---------------------------------------------------------------------


def test_embedded_data_structure():
    d = durations.data
    assert set(d.keys()) == {"moment", "mhns", "activity", "duration"}
    n = len(d["moment"])
    assert n > 1000
    for k in ("mhns", "activity", "duration"):
        assert len(d[k]) == n
    # moments are seconds of day; durations positive seconds
    assert d["moment"].min() >= 0 and d["moment"].max() < 86400
    assert d["duration"].min() > 0


def test_embedded_mhns_valid():
    for mhn in durations.data["mhns"][:200]:
        levels = pymaslow.parse_mhn(str(mhn))
        assert all(1 <= lv <= 5 for lv in levels)


# ---------------------------------------------------------------------
# PositiveCircularKDE
# ---------------------------------------------------------------------


def _toy_model():
    rng = np.random.default_rng(0)
    d = rng.lognormal(mean=7.0, sigma=0.8, size=300)  # durations in seconds
    t = rng.uniform(0, 24, 300)
    return durations.PositiveCircularKDE(d, t)


def test_model_validation():
    with pytest.raises(ValueError):
        durations.PositiveCircularKDE([1.0, -2.0], [1.0, 2.0])  # d <= 0
    with pytest.raises(ValueError):
        durations.PositiveCircularKDE([1.0], [24.5])  # t out of range
    with pytest.raises(ValueError):
        durations.PositiveCircularKDE([1.0, 2.0], [1.0])  # length mismatch
    with pytest.raises(ValueError):
        durations.PositiveCircularKDE([], [])


def test_pdf_and_marginal_positive():
    model = _toy_model()
    vals = model.pdf([500.0, 1000.0], [8.0, 18.0])
    assert vals.shape == (2,)
    assert np.all(vals >= 0)
    marg = model.marginal_t(np.linspace(0, 23.9, 50))
    assert marg.shape == (50,)
    assert np.all(marg >= 0)


def test_sample_conditional_shapes_and_reproducibility():
    model = _toy_model()
    s1 = model.sample_conditional(8.0, n_samples=50, random_state=42)
    s2 = model.sample_conditional(8.0, n_samples=50, random_state=42)
    assert s1.shape == (50,)
    np.testing.assert_array_equal(s1, s2)
    assert np.all(s1 > 0)  # durations are positive

    s_multi = model.sample_conditional([8.0, 18.0], n_samples=10, random_state=1)
    assert s_multi.shape == (2, 10)


def test_sample_conditional_time_localization():
    # durations at 8h should dominate p(d|t=8); those at 13h dominate p(d|t=13)
    # (clusters are not diametrically opposite, so the time kernel localizes)
    t = np.concatenate([np.full(250, 8.0), np.full(250, 13.0)])
    d = np.concatenate([np.full(250, 500.0), np.full(250, 5000.0)])
    model = durations.PositiveCircularKDE(d, t)
    s8 = model.sample_conditional(8.0, n_samples=200, random_state=0)
    s13 = model.sample_conditional(13.0, n_samples=200, random_state=0)
    assert s8.mean() < 1500  # near the 500s cluster
    assert s13.mean() > 3500  # near the 5000s cluster


# ---------------------------------------------------------------------
# fit() and module-level sample_conditional()
# ---------------------------------------------------------------------


def test_fit_default_embedded():
    model = durations.fit()
    assert isinstance(model, durations.PositiveCircularKDE)
    assert model.N > 1000  # after dropping log-duration <= 0
    assert model.bw_d > 0 and model.bw_t > 0
    # samples are on the log-duration scale -> exp() gives seconds
    samples = model.sample_conditional(8.0, n_samples=100, random_state=7)
    assert np.all(samples > 0)
    seconds = np.exp(samples)
    assert seconds.mean() > 60  # at least a minute on average


def test_fit_custom_data():
    rng = np.random.default_rng(3)
    d = rng.lognormal(6.5, 0.5, 200)
    t = rng.uniform(0, 24, 200)
    model = durations.fit(d, t, log_duration=False)
    assert model.N == 200
    with pytest.raises(ValueError):
        durations.fit(d)  # t_data required with d_data


def test_module_sample_conditional():
    s = durations.sample_conditional(18.0, n_samples=10, random_state=5)
    assert s.shape == (10,)
    assert np.all(s > 0)
    s2 = durations.sample_conditional([8.0, 18.0], n_samples=5, random_state=5)
    assert s2.shape == (2, 5)


# ---------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------


def test_plot_default_and_save(tmp_path):
    _fig, axes = durations.plot(save_path=str(tmp_path / "dur.png"))
    assert axes.shape == (2, 2)
    assert (tmp_path / "dur.png").exists()


def test_plot_log_duration(tmp_path):
    _fig, axes = durations.plot(log_duration=True)
    assert "log(Duration)" in axes[0, 0].get_title()


def test_plot_length_mismatch():
    with pytest.raises(ValueError):
        durations.plot(moments=[1.0, 2.0], duration=[1.0])


def test_top_level_exposure():
    assert pymaslow.PositiveCircularKDE is durations.PositiveCircularKDE
