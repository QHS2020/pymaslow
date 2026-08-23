"""Unit tests for pymaslow.datautilities."""

import pandas as pd
import pytest

import pymaslow
from pymaslow import datautilities


def _toy_capture24_df():
    """A tiny CAPTURE-24-like dataframe with known codes."""
    compendium = pymaslow.load_compendium()
    codes = compendium["code"].dropna().unique()
    c0, c1 = float(codes[0]), float(codes[1])
    times = pd.to_datetime(
        ["2024-01-01 07:00:00", "2024-01-01 08:30:00",
         "2024-01-01 12:00:00", "2024-01-01 18:00:00"]
    )
    annotations = [
        f"{int(c0)} some activity;MET 1.5",   # valid code c0
        f"{int(c1)} other activity;MET 2.0",  # valid code c1
        "999999 unknown activity;MET 9.9",    # code not in compendium
        f"{int(c0)} repeated activity;MET 1.5",  # valid again
    ]
    return pd.DataFrame({"time": times, "annotation": annotations}), (c0, c1)


# ---------------------------------------------------------------------
# extract_numbers
# ---------------------------------------------------------------------


def test_extract_numbers():
    assert datautilities.extract_numbers("7030 sleeping;MET 0.95") == [7030.0, 0.95]
    assert datautilities.extract_numbers("no numbers here") == [None]
    assert datautilities.extract_numbers("10") == [10.0]


# ---------------------------------------------------------------------
# capture24_extract_time_hierarchy
# ---------------------------------------------------------------------


def test_extract_time_hierarchy_embedded_compendium():
    data, (c0, c1) = _toy_capture24_df()
    out = datautilities.capture24_extract_time_hierarchy(data)
    TS_unix, TS_seconds, FLAGS, MHNS, idxs_of_code, effective_idxs = out

    # the unknown-code row (index 2) is skipped
    assert effective_idxs == [0, 1, 3]
    assert len(TS_unix) == len(TS_seconds) == len(FLAGS) == len(MHNS) == 3
    assert len(idxs_of_code) == 3

    compendium = pymaslow.load_compendium()
    assert idxs_of_code[0] == int((compendium["code"] == c0).idxmax())
    # MHNS labels match the compendium rows
    assert MHNS[0] == str(compendium.iloc[idxs_of_code[0]]["mhn"])
    assert MHNS[1] == str(compendium.iloc[idxs_of_code[1]]["mhn"])

    # time conversions: 07:00 -> 25200 s; ms epoch checks
    assert TS_seconds[0] == pytest.approx(7 * 3600)
    assert TS_seconds[1] == pytest.approx(8.5 * 3600)
    expected_ms = int(pd.Timestamp("2024-01-01 07:00:00").value // 10**6)
    assert TS_unix[0] == expected_ms


def test_extract_time_hierarchy_custom_xlsx_schema():
    # a dataframe with the ORIGINAL xlsx column names should also work
    data, (c0, _c1) = _toy_capture24_df()
    compendium = pymaslow.load_compendium().rename(
        columns={
            "code": "CODE",
            "flag": "Flag (1: note related)",
            "mhn": "MHNs (1 Physiological Needs; 2 Safety Needs; 3 Love and "
            "Belonging Needs; 4 Esteem Needs; 5 Self-Actualization Needs)",
        }
    )
    out = datautilities.capture24_extract_time_hierarchy(data, mhn_file_pd=compendium)
    assert out[5] == [0, 1, 3]  # effective_idxs


def test_extract_all_rows_invalid():
    times = pd.to_datetime(["2024-01-01 07:00:00"])
    data = pd.DataFrame(
        {"time": times, "annotation": ["999999 unknown;MET 1.0"]}
    )
    out = datautilities.capture24_extract_time_hierarchy(data)
    assert out[5] == []  # nothing effective
    assert out[3] == []  # no MHNS


# ---------------------------------------------------------------------
# capture24_collect_moments_per_hierarchy
# ---------------------------------------------------------------------


def test_collect_moments_per_hierarchy():
    TS = [100.0, 200.0, 300.0]
    FLAGS = ["nan", "nan", "nan"]
    MHNS = ["1", "2,3", "1,2"]
    moments = datautilities.capture24_collect_moments_per_hierarchy(
        TS, FLAGS, MHNS, verbose=False
    )
    assert set(moments.keys()) == {"1", "2", "3", "4", "5"}
    assert moments["1"] == [100.0, 300.0]
    assert moments["2"] == [200.0, 300.0]
    assert moments["3"] == [200.0]
    assert moments["4"] == [] and moments["5"] == []


def test_pipeline_end_to_end():
    data, _ = _toy_capture24_df()
    out = datautilities.capture24_extract_time_hierarchy(data)
    moments = datautilities.capture24_collect_moments_per_hierarchy(
        out[1], out[2], out[3], verbose=False
    )
    total = sum(len(v) for v in moments.values())
    # every effective row contributes at least one hierarchy moment
    assert total >= len(out[5])


def test_top_level_exposure():
    assert pymaslow.datautilities is datautilities
