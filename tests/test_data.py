"""Unit tests for the pymaslow.data module (embedded compendium)."""

import pymaslow


def test_load_compendium_shape_and_columns():
    df = pymaslow.load_compendium()
    assert len(df) == 823
    assert list(df.columns) == [
        "code",
        "mets",
        "major_heading",
        "activity",
        "flag",
        "mhn",
        "reason",
    ]


def test_compendium_mhn_labels_valid():
    df = pymaslow.load_compendium()
    for mhn in df["mhn"].dropna():
        levels = pymaslow.parse_mhn(str(mhn))
        assert all(1 <= lv <= 5 for lv in levels)


def test_load_compendium_returns_copy():
    df1 = pymaslow.load_compendium()
    df1.iloc[0, 0] = -999
    df2 = pymaslow.load_compendium()
    assert df2.iloc[0, 0] != -999


def test_activity_hierarchy_map():
    mapping = pymaslow.get_activity_hierarchy_map()
    assert len(mapping) > 700
    for levels in mapping.values():
        assert isinstance(levels, tuple)
        assert all(1 <= lv <= 5 for lv in levels)
