"""Tests for the pymaslow.hmm module."""

import numpy as np
import pytest

from pymaslow import ExponentialHMM, LognormalHMM


def _toy_data():
    hierarchy = [[1, 1, 2, 3], [2, 2, 1], [3, 1, 2]]
    durations = [
        [1800.0, 900.0, 3600.0, 1200.0],
        [600.0, 300.0, 2400.0],
        [500.0, 2000.0, 3500.0],
    ]
    return hierarchy, durations


@pytest.mark.parametrize("cls", [ExponentialHMM, LognormalHMM])
def test_fit_and_predict(cls):
    hierarchy, durations = _toy_data()
    model = cls(n_states=5, random_state=42)
    ll = model.fit_supervised(hierarchy, durations, verbose=False)
    assert np.isfinite(ll)

    pred = model.predict([1500.0, 700.0, 3000.0])
    assert len(pred) == 3

    params = model.get_params()
    assert params["trans_probs"].shape == (5, 5)
    np.testing.assert_allclose(params["trans_probs"].sum(axis=1), 1.0)
    np.testing.assert_allclose(params["start_probs"].sum(), 1.0)


@pytest.mark.parametrize("cls", [ExponentialHMM, LognormalHMM])
def test_unobserved_states_tracked(cls):
    hierarchy, durations = _toy_data()
    model = cls(n_states=5, random_state=42)
    model.fit_supervised(hierarchy, durations, verbose=False)
    assert model.get_observed_states() == {1, 2, 3}
    assert model.get_unobserved_states() == {0, 4}


@pytest.mark.parametrize("cls", [ExponentialHMM, LognormalHMM])
def test_expected_duration(cls):
    hierarchy, durations = _toy_data()
    model = cls(n_states=3, random_state=42)
    model.fit_supervised(hierarchy, durations, verbose=False)
    ed = model.expected_duration(1)
    assert ed > 0
    with pytest.raises(ValueError):
        model.expected_duration(99)


def test_exponential_mle_recovers_scale():
    # State 1 durations ~ Exp(mean=1000) -> fitted scale ~= 1000
    hierarchy = [[1] * 200]
    rng = np.random.default_rng(0)
    durations = [rng.exponential(1000.0, 200).tolist()]
    model = ExponentialHMM(n_states=2, random_state=42)
    model.fit_supervised(hierarchy, durations, verbose=False)
    assert model.expected_duration(1) == pytest.approx(1000.0, rel=0.15)


def test_lognormal_mle_recovers_params():
    # State 1 durations ~ Lognormal(mu=ln(1000), sigma=0.3)
    mu_true, sigma_true = np.log(1000.0), 0.3
    hierarchy = [[1] * 500]
    rng = np.random.default_rng(0)
    durations = [rng.lognormal(mu_true, sigma_true, 500).tolist()]
    model = LognormalHMM(n_states=2, random_state=42)
    model.fit_supervised(hierarchy, durations, verbose=False)
    params = model.get_params()["emission_params"][1]
    assert params["mu"] == pytest.approx(mu_true, rel=0.05)
    assert params["sigma"] == pytest.approx(sigma_true, rel=0.15)
    expected = np.exp(mu_true + sigma_true**2 / 2)
    assert model.expected_duration(1) == pytest.approx(expected, rel=0.10)


def test_invalid_init():
    with pytest.raises(ValueError):
        ExponentialHMM(n_states=0)
    with pytest.raises(ValueError):
        LognormalHMM(n_states=3, state_ids=[1, 2])


def test_predict_empty():
    model = ExponentialHMM(n_states=2)
    assert model.predict([]) == []
