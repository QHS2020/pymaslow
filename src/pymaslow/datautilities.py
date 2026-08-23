"""
pymaslow.datautilities
======================

Data-processing utilities turning raw activity-tracking datasets into the
(time, hierarchy) streams consumed by the package's models.

Currently implemented for **CAPTURE-24** (the ETRI and DailyLog2016
pipelines use the LLM prompts in :mod:`pymaslow.prompts` for hierarchy
identification; their processed forms are embedded in
:mod:`pymaslow.hmm`):

- :func:`capture24_extract_time_hierarchy` -- parse one participant's raw
  CAPTURE-24 csv (``time``, ``x``, ``y``, ``z``, ``annotation`` columns),
  map each annotation (e.g. ``"7030 sleeping;MET 0.95"``) to its Compendium
  code, and attach the MHN hierarchy labels from the annotated compendium.
- :func:`capture24_collect_moments_per_hierarchy` -- split the multi-label
  hierarchy stream into per-hierarchy occurrence-time lists.

Usage (mirrors ``codes/DataProcessCAPTURE24.py`` of the companion research
repository):

>>> import pandas as pd
>>> from pymaslow import datautilities
>>> data = pd.read_csv("P001/P001.csv")                       # doctest: +SKIP
>>> data["time"] = pd.to_datetime(data["time"])               # doctest: +SKIP
>>> out = datautilities.capture24_extract_time_hierarchy(data)  # doctest: +SKIP
>>> TS_unix, TS_seconds, FLAGS, MHNS, idxs_of_code, effective_idxs = out  # doctest: +SKIP
>>> moments = datautilities.capture24_collect_moments_per_hierarchy(  # doctest: +SKIP
...     TS_seconds, FLAGS, MHNS)

Adapted from ``maslownet.DataProcess`` of the companion research repository.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import load_compendium  # pyright: ignore[reportMissingImports]

__all__ = [
    "capture24_extract_time_hierarchy",
    "capture24_collect_moments_per_hierarchy",
    "extract_numbers",
]

#: Original column names of the Compendium-with-MHNs xlsx (accepted when a
#: custom ``mhn_file_pd`` is passed).
_XLSX_CODES_COL = "CODE"
_XLSX_FLAG_COL = "Flag (1: note related)"
_XLSX_MHN_COL = (
    "MHNs (1 Physiological Needs; 2 Safety Needs; 3 Love and Belonging Needs; "
    "4 Esteem Needs; 5 Self-Actualization Needs)"
)


def extract_numbers(text: str) -> list:
    """Extract all numbers (ints and floats) from a string.

    Returns ``[None]`` when no number is present, matching the research
    prototype. Used to parse CAPTURE-24 annotations such as
    ``"7030 sleeping;MET 0.95"`` -> ``[7030.0, 0.95]``.
    """
    numbers = []
    for x in re.findall(r"-?\d+\.?\d*", str(text)):
        if not x:
            continue
        try:
            numbers.append(float(x))
        except ValueError:
            continue
    if len(numbers) == 0:
        return [None]
    return numbers


def capture24_extract_time_hierarchy(
    data_pd: pd.DataFrame,
    mhn_file_pd: pd.DataFrame | None = None,
    t_col: str = "time",
    activity_col: str = "annotation",
    codes_col: str | None = None,
    flag_col: str | None = None,
    mhn_col: str | None = None,
) -> tuple[list, list, list, list, list, list]:
    """Extract aligned (time, hierarchy) streams from one CAPTURE-24 participant.

    Each row's annotation (e.g. ``"7030 sleeping;MET 0.95"``) is parsed for
    its Compendium code; the first code found in the compendium determines
    the row's hierarchy labels. Rows whose codes are not in the compendium
    are skipped (tracked by ``effective_idxs``).

    Parameters
    ----------
    data_pd : pandas.DataFrame
        Raw participant data with a timestamp column (``t_col``, dtype
        ``datetime64``) and an annotation column (``activity_col``).
    mhn_file_pd : pandas.DataFrame or None
        The MHN-annotated compendium. If None, the package's embedded
        compendium (:func:`pymaslow.load_compendium`) is used and the
        column names default to its schema; if a dataframe with the
        original xlsx schema is passed, the column names default to the
        xlsx names (``CODE``, ``Flag (1: note related)``, ``MHNs (...)``).
    t_col, activity_col : str
        Column names in ``data_pd``.
    codes_col, flag_col, mhn_col : str or None
        Column names in ``mhn_file_pd`` (see above for the defaults).

    Returns
    -------
    TS_unix : list of int
        Millisecond Unix timestamps of the effective rows.
    TS_secondsfromdaybegining : list of float
        Seconds since midnight of the effective rows.
    FLAGS : list of str
        The compendium's note-related flag of each effective row.
    MHNS : list of str
        Multi-label hierarchy annotation of each effective row, e.g.
        ``"1,5"``.
    idxs_of_code : list of int
        Compendium row index (0..822) of each effective row.
    effective_idxs : list of int
        Row indices into ``data_pd`` whose activity code was found.
    """
    if mhn_file_pd is None:
        mhn_file_pd = load_compendium()
        codes_col = codes_col or "code"
        flag_col = flag_col or "flag"
        mhn_col = mhn_col or "mhn"
    else:
        codes_col = codes_col or _XLSX_CODES_COL
        flag_col = flag_col or _XLSX_FLAG_COL
        mhn_col = mhn_col or _XLSX_MHN_COL

    TS_unix: list = []
    TS_secondsfromdaybegining: list = []
    FLAGS: list = []
    MHNS: list = []
    idxs_of_code: list = []
    effective_idxs: list = []

    ts = data_pd[t_col].values
    annotations = data_pd[activity_col].values

    codes = list(mhn_file_pd[codes_col].values)
    flags = mhn_file_pd[flag_col].astype(str).values
    mhns = mhn_file_pd[mhn_col].astype(str).values

    for i, (t, annotation) in enumerate(zip(ts, annotations, strict=True)):
        # annotation is like '7030 sleeping;MET 0.95' -> [7030.0, 0.95]
        codes_candidate = extract_numbers(annotation)

        for c in codes_candidate:
            try:
                idx = codes.index(c)
            except ValueError:
                idx = False
            if isinstance(idx, bool):
                continue

            effective_idxs.append(i)
            FLAGS.append(flags[idx])
            MHNS.append(mhns[idx])
            idxs_of_code.append(idx)

            # millisecond Unix timestamp, and seconds since midnight
            TS_unix.append(t.astype("datetime64[ms]").astype(np.int64))
            day_start = t.astype("datetime64[D]")
            TS_secondsfromdaybegining.append((t - day_start) / np.timedelta64(1, "s"))
            break

    return (
        TS_unix,
        TS_secondsfromdaybegining,
        FLAGS,
        MHNS,
        idxs_of_code,
        effective_idxs,
    )


def capture24_collect_moments_per_hierarchy(
    TS: Sequence,
    FLAGS: Sequence,
    MHNS: Sequence,
    verbose: bool = True,
) -> dict[str, list]:
    """Split a multi-label hierarchy stream into per-hierarchy moment lists.

    Parameters
    ----------
    TS : array_like
        Occurrence times (seconds since midnight), as returned by
        :func:`capture24_extract_time_hierarchy`.
    FLAGS : array_like
        Flag stream (accepted for interface compatibility with the research
        prototype; not used by the computation).
    MHNS : array_like of str
        Multi-label hierarchy annotations, e.g. ``"1"`` or ``"2,3"``.
    verbose : bool
        Show a progress bar.

    Returns
    -------
    dict
        ``{"1": [...], "2": [...], "3": [...], "4": [...], "5": [...]}`` —
        ``moments[str(h)]`` collects every occurrence time at which
        hierarchy ``h`` was active (a multi-label moment is counted toward
        each of its hierarchies).
    """
    moments4hierarchies: dict[str, list] = {str(i): [] for i in [1, 2, 3, 4, 5]}

    for t, _f, mhn in tqdm(zip(TS, FLAGS, MHNS, strict=True), disable=not verbose):
        for h in str(mhn).split(","):
            moments4hierarchies[h].append(t)

    return moments4hierarchies
