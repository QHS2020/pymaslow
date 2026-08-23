"""Unit tests for the embedded ETRI temporal hierarchy data."""

import numpy as np

import pymaslow
from pymaslow import data as pdata


def test_etri_temporal_hierarchy_data_exposed():
    d = pymaslow.ETRI_TemporalHierarchyData
    assert set(d.keys()) == {"1", "2", "3", "4", "5"}
    total = sum(len(v) for v in d.values())
    assert total == 19997
    for arr in d.values():
        assert isinstance(arr, np.ndarray)
        assert arr.min() > -2.0 and arr.max() < 26.0  # hours of day


def test_etri_per_hierarchy_counts():
    d = pymaslow.ETRI_TemporalHierarchyData
    expected = {"1": 3829, "2": 3360, "3": 4221, "4": 5934, "5": 2653}
    for k, n in expected.items():
        assert len(d[k]) == n


def test_loader_matches_attribute():
    d1 = pymaslow.ETRI_TemporalHierarchyData
    d2 = pdata.load_etri_temporal_hierarchy()
    for k in d1:
        np.testing.assert_array_equal(d1[k], d2[k])


def test_etri_data_usable_with_fit_vmmm_dictionary():
    # the ETRI resampled data feeds the same fitting pipeline as CAPTURE-24
    p_x, models, _best_k, table = pymaslow.fit_vmmm_dictionary(
        pymaslow.ETRI_TemporalHierarchyData, k_max=4, verbose=False
    )
    assert set(models) == {"1", "2", "3", "4", "5"}
    np.testing.assert_allclose(sum(p_x.values()), 1.0)
    assert len(table) == 5
