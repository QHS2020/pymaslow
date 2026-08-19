"""Unit tests for the pymaslow.circularkde module."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from pymaslow import CircularKDE, fit_circular_kde, sec_to_rad


def test_kde_integrates_to_one():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 200)
    kde = fit_circular_kde(theta)
    grid = np.linspace(0, 2 * np.pi, 20001)
    np.testing.assert_allclose(trapezoid(kde.pdf(grid), grid), 1.0, atol=1e-3)


def test_midnight_boundary_continuity():
    # Data clustered just before and just after midnight should yield a
    # smooth density across the 0 / 2*pi boundary (no artificial dip).
    rng = np.random.default_rng(1)
    t = rng.normal(0, 1800, 500) % 86400  # ~00:00 +/- 30 min
    kde = fit_circular_kde(sec_to_rad(t))
    eps = 1e-6
    left = kde.pdf(np.array([2 * np.pi - eps]))[0]
    right = kde.pdf(np.array([eps]))[0]
    assert left == pytest.approx(right, rel=1e-3)
    # And the boundary region should be near the mode, not a discontinuity
    mid = kde.pdf(np.array([0.0]))[0]
    assert mid == pytest.approx(left, rel=1e-3)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        CircularKDE(np.array([]))


def test_repr():
    kde = fit_circular_kde(np.array([0.1, 0.2, 0.3]))
    assert "n=3" in repr(kde)
