"""Tests for the pymaslow.hierarchy module."""

import pytest

from pymaslow import hierarchy


def test_five_levels_defined():
    assert hierarchy.N_HIERARCHIES == 5
    assert set(hierarchy.HIERARCHY_NAMES) == {1, 2, 3, 4, 5}
    assert hierarchy.HIERARCHY_NAMES[1] == "Physiological Needs"
    assert hierarchy.HIERARCHY_NAMES[5] == "Self-Actualization Needs"


def test_parse_mhn_string():
    assert hierarchy.parse_mhn("1,4,5") == (1, 4, 5)
    assert hierarchy.parse_mhn("3") == (3,)
    assert hierarchy.parse_mhn("5,1") == (1, 5)  # sorted
    assert hierarchy.parse_mhn("2,2,4") == (2, 4)  # de-duplicated


def test_parse_mhn_int_and_sequence():
    assert hierarchy.parse_mhn(2) == (2,)
    assert hierarchy.parse_mhn([4, 1]) == (1, 4)


def test_parse_mhn_invalid():
    with pytest.raises(ValueError):
        hierarchy.parse_mhn("0")
    with pytest.raises(ValueError):
        hierarchy.parse_mhn("6")
    with pytest.raises(ValueError):
        hierarchy.parse_mhn("abc")
    with pytest.raises(ValueError):
        hierarchy.parse_mhn("")


def test_format_mhn_roundtrip():
    assert hierarchy.format_mhn([4, 1, 5]) == "1,4,5"
    assert hierarchy.parse_mhn(hierarchy.format_mhn([3, 2])) == (2, 3)


def test_mhn_to_vector():
    assert hierarchy.mhn_to_vector("1,3") == [1, 0, 1, 0, 0]
    assert hierarchy.mhn_to_vector(5) == [0, 0, 0, 0, 1]


def test_build_hierarchy_prompt():
    prompt = hierarchy.build_hierarchy_prompt(
        "bicycling, mountain, uphill, vigorous", "bicycling"
    )
    assert "bicycling, mountain, uphill, vigorous" in prompt
    assert '"bicycling"' in prompt
    assert "maslow hierarchy" in prompt
    assert "number 1 to 5" in prompt
