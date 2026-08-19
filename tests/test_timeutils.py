"""Unit tests for the pymaslow.timeutils module."""

import numpy as np

import pymaslow


def test_sec_rad_roundtrip():
    t = np.array([0.0, 6 * 3600, 12 * 3600, 18 * 3600, 86400.0])
    np.testing.assert_allclose(
        pymaslow.rad_to_sec(pymaslow.sec_to_rad(t)), t, rtol=1e-10
    )


def test_hours_rad_roundtrip():
    t = np.array([0.0, 6.0, 12.0, 18.0, 24.0])
    np.testing.assert_allclose(
        pymaslow.rad_to_hours(pymaslow.hours_to_rad(t)), t, rtol=1e-10
    )


def test_hours_rad_custom_bounds():
    theta = pymaslow.hours_to_rad(np.array([12.0]), y_lw=0.0, y_up=24.0)
    np.testing.assert_allclose(theta, [np.pi], rtol=1e-10)


def test_format_time():
    assert pymaslow.format_time(0) == "00:00"
    assert pymaslow.format_time(7.5 * 3600) == "07:30"
    assert pymaslow.format_time(86400 + 3600) == "01:00"  # wraps
    assert pymaslow.format_time(np.array([0.0, 12 * 3600])) == ["00:00", "12:00"]
