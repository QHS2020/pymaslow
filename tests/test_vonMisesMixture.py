"""Unit tests for the pymaslow.vonMisesMixture module."""

import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

import numpy as np
import pytest
from scipy.integrate import trapezoid

import pymaslow
from pymaslow import (
    VonMisesMixture,
    fit_vmmm_dictionary,
    sample_joint_vmmm,
    sample_vonmises_mixture,
)
from pymaslow import vonMisesMixture as vmmm


def _bimodal_hours(seed=42):
    rng = np.random.default_rng(seed)
    return (
        np.concatenate([rng.normal(8.0, 0.5, 400), rng.normal(20.0, 0.5, 400)]) % 24.0
    )


# ---------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------


def test_from_parameters_and_pdf_normalization():
    mix = VonMisesMixture.from_parameters(
        weights=[0.6, 0.4], mu=[1.0, 4.0], kappa=[5.0, 3.0]
    )
    grid = np.linspace(0, 2 * np.pi, 20001)
    np.testing.assert_allclose(trapezoid(mix.pdf(grid), grid), 1.0, atol=1e-6)
    assert mix.n_components == 2
    # weighted components sum to the mixture density
    total = mix.weighted_component_pdf(grid, 0) + mix.weighted_component_pdf(grid, 1)
    np.testing.assert_allclose(total, mix.pdf(grid), rtol=1e-10)


def test_from_parameters_validation():
    with pytest.raises(ValueError):
        VonMisesMixture.from_parameters([0.5, 0.6], [1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        VonMisesMixture.from_parameters([1.0], [1.0], [-1.0])
    with pytest.raises(ValueError):
        VonMisesMixture(n_components=0)


def test_em_fit_recovers_bimodal_peaks():
    theta = pymaslow.hours_to_rad(_bimodal_hours())
    model = VonMisesMixture(n_components=2, random_state=42).fit(theta)
    peaks = sorted(model.peak_times())
    assert abs(peaks[0] - 8.0) < 1.0
    assert abs(peaks[1] - 20.0) < 1.0
    np.testing.assert_allclose(model.weights.sum(), 1.0)


def test_single_component_fit():
    theta = pymaslow.hours_to_rad(np.random.default_rng(1).normal(12.0, 1.0, 200) % 24)
    model = VonMisesMixture(n_components=1, random_state=0).fit(theta)
    assert model.n_components == 1
    assert abs(model.peak_times()[0] - 12.0) < 0.5


# ---------------------------------------------------------------------
# Dictionary fitting with model selection
# ---------------------------------------------------------------------


def test_fit_vmmm_dictionary():
    data = {
        "a": _bimodal_hours(seed=1),
        "b": np.random.default_rng(2).normal(13, 1, 300) % 24,
    }
    p_x, models, best_k, table = fit_vmmm_dictionary(
        data, k_max=4, criterion="bic", random_state=42, verbose=False
    )
    assert set(models) == {"a", "b"}
    np.testing.assert_allclose(sum(p_x.values()), 1.0)
    np.testing.assert_allclose(p_x["a"], 800 / 1100)
    assert 1 <= best_k["a"] <= 4
    # bimodal class should need more components than the unimodal one
    assert best_k["a"] >= best_k["b"]
    # the fitting results table summarizes every class
    assert list(table.columns) == [
        "class", "n", "p_x", "K", "logL", "AIC", "BIC", "peak_times",
    ]
    assert len(table) == 2
    row_a = table.loc[table["class"] == "a"].iloc[0]
    assert row_a["n"] == 800
    assert row_a["K"] == best_k["a"]
    np.testing.assert_allclose(row_a["p_x"], 800 / 1100)
    # BIC penalizes parameters more than AIC for n > e^2, so BIC > AIC here
    assert row_a["BIC"] > row_a["AIC"]
    assert isinstance(row_a["peak_times"], str) and ":" in row_a["peak_times"]


def test_fit_vmmm_dictionary_single_observation():
    p_x, models, best_k, table = fit_vmmm_dictionary(
        {"only": np.array([10.0])}, verbose=False
    )
    assert best_k["only"] == 1
    assert p_x["only"] == 1.0
    peak = models["only"].peak_times()[0]
    assert abs(peak - 10.0) < 0.1
    row = table.loc[table["class"] == "only"].iloc[0]
    assert row["K"] == 1 and row["n"] == 1


# ---------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------


def test_sample_reproducible_and_in_range():
    model = VonMisesMixture.from_parameters([0.5, 0.5], [1.0, 4.0], [5.0, 5.0])
    s1 = sample_vonmises_mixture(model, 500, seed=42, return_radians=False)
    s2 = sample_vonmises_mixture(model, 500, seed=42, return_radians=False)
    np.testing.assert_array_equal(s1, s2)
    assert s1.min() >= 0.0 and s1.max() < 24.0

    s3 = sample_vonmises_mixture(model, 100, seed=1, return_radians=True)
    assert s3.min() >= 0.0 and s3.max() < 2 * np.pi


def test_sample_method_on_model():
    model = VonMisesMixture.from_parameters([1.0], [np.pi], [10.0])
    samples = model.sample(200, seed=7, return_radians=False)
    assert len(samples) == 200
    # concentrated component near 12h
    assert abs(np.median(samples) - 12.0) < 1.5


def test_sample_joint_vmmm_matches_prior():
    classes, times = sample_joint_vmmm(vmmm.p_x, vmmm.models, n_samples=20000, seed=123)
    assert len(classes) == len(times) == 20000
    labels, counts = np.unique(classes, return_counts=True)
    freq = dict(zip(labels.tolist(), (counts / counts.sum()).tolist(), strict=True))
    for c, p in vmmm.p_x.items():
        assert freq[c] == pytest.approx(p, abs=0.03)
    assert times.min() >= 0.0 and times.max() < 24.0


def test_sample_vmmm_dictionary():
    out = vmmm.sample_vmmm_dictionary(vmmm.models, n_samples=50, seed=0)
    assert set(out.keys()) == set(vmmm.models.keys())
    assert all(len(v) == 50 for v in out.values())


# ---------------------------------------------------------------------
# Embedded assets
# ---------------------------------------------------------------------


def test_embedded_data_attribute():
    data = vmmm.data
    assert set(data.keys()) == {"1", "2", "3", "4", "5"}
    total = sum(len(v) for v in data.values())
    assert total > 15000
    for arr in data.values():
        assert arr.min() > -1.0 and arr.max() < 25.0  # hours of day


def test_embedded_fitted_parameters_loaded_at_import():
    # p_x is a valid prior over the five hierarchies
    np.testing.assert_allclose(sum(vmmm.p_x.values()), 1.0)
    assert set(vmmm.p_x) == {"1", "2", "3", "4", "5"}
    # models are fitted VonMisesMixture objects consistent with best_k
    for c, model in vmmm.models.items():
        assert isinstance(model, VonMisesMixture)
        assert model.n_components == vmmm.best_k[c]
        grid = np.linspace(0, 2 * np.pi, 2001)
        assert np.all(model.pdf(grid) >= 0)


def test_embedded_model_peak_times_sensible():
    # H1 (physiological) should have peaks near typical meal times
    peaks_h1 = sorted(vmmm.models["1"].peak_times())
    assert any(abs(p - 8.0) < 1.5 for p in peaks_h1)  # breakfast
    assert any(abs(p - 12.0) < 1.5 for p in peaks_h1)  # lunch


def test_load_functions():
    data = vmmm.load_resampled_data()
    assert set(data) == {"1", "2", "3", "4", "5"}
    p_x, models, best_k = vmmm.load_fitted_models()
    np.testing.assert_allclose(sum(p_x.values()), 1.0)
    assert set(models) == set(best_k) == {"1", "2", "3", "4", "5"}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def test_plot_vmmm_results(tmp_path):
    _fig, axes = vmmm.plot_vmmm_results(
        vmmm.data,
        vmmm.p_x,
        vmmm.models,
        vmmm.best_k,
        save_path=str(tmp_path / "vmmm.png"),
    )
    assert len(axes) == 3
    assert (tmp_path / "vmmm.png").exists()
    # panel titles convey the meaning of each panel
    assert "Observed" in axes[0].get_title()
    assert "p(t | class)" in axes[1].get_title()
    assert "Joint" in axes[2].get_title()
