"""
pymaslow.circularkde
====================

Circular kernel density estimation (KDE) for time-of-day data.

A standard Gaussian KDE leaks probability mass across the midnight boundary
(0 is identified with 2*pi on the circle). :class:`CircularKDE` corrects this
by triplicating the data -- shifting copies by ``-2*pi``, ``0`` and
``+2*pi`` -- before fitting a :class:`scipy.stats.gaussian_kde`, and rescaling
the resulting density by 3.

Adapted from ``codes/vonMises/circular_time_analysis.py`` of the companion
research repository.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde

from .timeutils import CircularTimeModel  # pyright: ignore[reportMissingImports]

__all__ = [
    "CircularKDE",
    "fit_circular_kde",
]


class CircularKDE(CircularTimeModel):
    """Circular kernel density estimator with boundary correction.

    Handles the 0/2*pi (midnight) boundary by triplicating the data
    (shifting by ``-2*pi``, ``0``, ``+2*pi``) before fitting a standard
    Gaussian KDE.

    Parameters
    ----------
    theta_data : array_like
        Observed angles in radians ``[0, 2*pi)``.
    bandwidth : str, float, or None
        Bandwidth method for :class:`scipy.stats.gaussian_kde`.
        ``None`` uses Scott's rule; ``"silverman"`` uses Silverman's rule;
        a float sets the bandwidth factor manually (smaller = sharper peaks).
    """

    def __init__(self, theta_data: npt.ArrayLike, bandwidth: str | float | None = None):
        self.theta_data = np.asarray(theta_data, dtype=float)
        if self.theta_data.size == 0:
            raise ValueError("theta_data must be non-empty")
        self.bandwidth = bandwidth

        # Circular boundary correction: triplicate the data
        theta_ext = np.concatenate(
            [
                self.theta_data - 2.0 * np.pi,
                self.theta_data,
                self.theta_data + 2.0 * np.pi,
            ]
        )
        self.kde = gaussian_kde(theta_ext, bw_method=bandwidth)

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate the circular KDE at angles ``theta`` (radians)."""
        theta = np.asarray(theta, dtype=float)
        # The KDE was trained on 3x the data mass, so rescale by 3
        return self.kde(theta) * 3.0

    def __repr__(self) -> str:
        return f"CircularKDE(n={len(self.theta_data)}, bandwidth={self.bandwidth})"


def fit_circular_kde(
    theta_data: npt.ArrayLike, bandwidth: str | float | None = None
) -> CircularKDE:
    """Fit a circular KDE with automatic midnight-boundary correction.

    Parameters
    ----------
    theta_data : array_like
        Observed angles in radians ``[0, 2*pi)`` (see
        :func:`pymaslow.sec_to_rad`).
    bandwidth : str, float, or None
        Bandwidth selection method; see :class:`CircularKDE`.

    Returns
    -------
    CircularKDE
        The fitted circular kernel density estimator.
    """
    return CircularKDE(theta_data, bandwidth=bandwidth)
