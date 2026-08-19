"""Tests for the pymaslow.dirichlet module."""

import numpy as np

from pymaslow import dirichlet


def _simplex_sample(n, alphas, seed=42):
    return np.random.default_rng(seed).dirichlet(alphas, size=n)


def test_mle_recovers_parameters():
    true_alphas = np.array([5.0, 3.0, 2.0, 4.0, 1.0])
    data = _simplex_sample(2000, true_alphas)
    est = dirichlet.mle(data)
    np.testing.assert_allclose(est, true_alphas, rtol=0.25)


def test_mle_fixedpoint_matches_meanprecision():
    data = _simplex_sample(500, [2.0, 4.0, 1.0, 3.0, 5.0])
    a_mp = dirichlet.mle(data, method="meanprecision")
    a_fp = dirichlet.mle(data, method="fixedpoint")
    np.testing.assert_allclose(a_mp, a_fp, rtol=1e-3)


def test_meanprecision():
    a = np.array([2.0, 3.0, 5.0])
    mean, precision = dirichlet.meanprecision(a)
    np.testing.assert_allclose(mean, [0.2, 0.3, 0.5])
    assert precision == 10.0


def test_loglikelihood_finite():
    data = _simplex_sample(100, [1.0, 2.0, 3.0, 4.0, 5.0])
    a = dirichlet.mle(data)
    ll = dirichlet.loglikelihood(data, a)
    assert np.isfinite(ll)


def test_pdf_positive_on_simplex():
    a = np.array([2.0, 3.0, 4.0, 1.0, 5.0])
    pdf_fn = dirichlet.pdf(a)
    xs = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.1, 0.1, 0.1, 0.2]])
    vals = pdf_fn(xs)
    assert vals.shape == (2,)
    assert np.all(vals > 0)


def test_likelihood_ratio_test():
    d1 = _simplex_sample(300, [5.0, 1.0, 1.0, 1.0, 1.0], seed=1)
    d2 = _simplex_sample(300, [1.0, 5.0, 1.0, 1.0, 1.0], seed=2)
    D, p, a0, a1, a2 = dirichlet.test(d1, d2)
    assert D > 0
    assert 0 <= p <= 1
    # Strongly different distributions should be detected
    assert p < 0.05
    assert a0.shape == a1.shape == a2.shape == (5,)


def test_fit_temporal_profile():
    data = _simplex_sample(500, [3.0, 3.0, 2.0, 2.0, 1.0])
    alphas, mean, precision = dirichlet.fit_temporal_profile(data)
    assert alphas.shape == (5,)
    np.testing.assert_allclose(mean.sum(), 1.0)
    np.testing.assert_allclose(precision, alphas.sum())
