"""
pymaslow.timeutils
==================

Shared time-of-day conversion helpers and the abstract base class for
circular time probability models.

Time of day is treated as a point on the circle: seconds since midnight in
``[0, 86400)`` (or hours in ``[0, 24)``) map linearly to radians in
``[0, 2*pi)``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    "sec_to_rad",
    "rad_to_sec",
    "hours_to_rad",
    "rad_to_hours",
    "format_time",
    "CircularTimeModel",
]

#: Number of seconds in a day.
SECONDS_PER_DAY = 86400

#: Number of seconds in an hour.
SECONDS_PER_HOUR = 3600


def sec_to_rad(t: npt.ArrayLike) -> np.ndarray:
    """Map time in seconds ``[0, 86400]`` to radians ``[0, 2*pi]``."""
    return 2.0 * np.pi * np.asarray(t) / SECONDS_PER_DAY


def rad_to_sec(theta: npt.ArrayLike) -> np.ndarray:
    """Map radians ``[0, 2*pi]`` back to time in seconds ``[0, 86400]``."""
    return np.asarray(theta) * SECONDS_PER_DAY / (2.0 * np.pi)


def hours_to_rad(t: npt.ArrayLike, y_lw: float = 0.0, y_up: float = 24.0) -> np.ndarray:
    """Map time in hours ``[y_lw, y_up]`` to radians ``[0, 2*pi]``."""
    return 2.0 * np.pi * (np.asarray(t) - y_lw) / (y_up - y_lw)


def rad_to_hours(
    theta: npt.ArrayLike, y_lw: float = 0.0, y_up: float = 24.0
) -> np.ndarray:
    """Map radians ``[0, 2*pi]`` back to hours ``[y_lw, y_up]``."""
    return y_lw + np.asarray(theta) / (2.0 * np.pi) * (y_up - y_lw)


def format_time(seconds: npt.ArrayLike) -> str | list[str]:
    """Convert seconds since midnight to ``HH:MM`` string(s)."""
    seconds = np.asarray(seconds) % SECONDS_PER_DAY
    h = (seconds // 3600).astype(int)
    m = ((seconds % 3600) // 60).astype(int)
    if np.ndim(seconds) == 0:
        return f"{h:02d}:{m:02d}"
    return [f"{hh:02d}:{mm:02d}" for hh, mm in zip(h, m, strict=True)]


class CircularTimeModel:
    """Abstract base class for circular time-of-day probability models."""

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        """Probability density at angles ``theta`` (radians)."""
        raise NotImplementedError

    def pdf_time(self, t: np.ndarray) -> np.ndarray:
        """Probability density at time ``t`` in seconds, including the
        Jacobian correction ``d(theta)/dt = 2*pi / 86400``."""
        theta = np.asarray(sec_to_rad(t), dtype=float)
        return self.pdf(theta) * (2.0 * np.pi / SECONDS_PER_DAY)
