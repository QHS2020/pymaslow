"""
pymaslow.data
=============

Access to the package's embedded data: the *Compendium of Physical Activities*
annotated with Maslow Hierarchy of Needs (MHN) labels.

Each row of the compendium is one physical activity. The ``mhn`` column holds
the multi-label hierarchy annotation inferred by a reasoning LLM (see
:mod:`pymaslow.hierarchy` for the prompt template), e.g. ``"1,4,5"`` means the
activity serves Physiological (1), Esteem (4) and Self-Actualization (5)
needs simultaneously.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from importlib.abc import Traversable

import numpy as np
import pandas as pd

from .hierarchy import parse_mhn

__all__ = [
    "COMPENDIUM_COLUMNS",
    "COMPENDIUM_RESOURCE",
    "ETRI_RESAMPLED_RESOURCE",
    "get_activity_hierarchy_map",
    "load_compendium",
    "load_etri_temporal_hierarchy",
]

#: Resource path (inside the package) of the embedded compendium CSV.
COMPENDIUM_RESOURCE = "data/compendium_mhn.csv"

#: Resource path of the embedded resampled ETRI temporal hierarchy data.
ETRI_RESAMPLED_RESOURCE = "data/resampled_ETRI.npz"

#: Column names of the embedded compendium dataframe.
COMPENDIUM_COLUMNS = (
    "code",
    "mets",
    "major_heading",
    "activity",
    "flag",
    "mhn",
    "reason",
)


def _compendium_path() -> Traversable:
    """Locate the embedded compendium CSV inside the installed package."""
    return resources.files("pymaslow").joinpath(COMPENDIUM_RESOURCE)


@lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    with resources.as_file(_compendium_path()) as path:
        return pd.read_csv(path)


def load_compendium() -> pd.DataFrame:
    """Load the embedded MHN-annotated Compendium of Physical Activities.

    Returns
    -------
    pandas.DataFrame
        Dataframe with 823 rows (one per activity) and columns:

        - ``code`` : compendium activity code (float)
        - ``mets`` : Metabolic Equivalent of Task value (float)
        - ``major_heading`` : activity category, e.g. ``"bicycling"``
        - ``activity`` : specific activity description, e.g.
          ``"bicycling, mountain, uphill, vigorous"``
        - ``flag`` : note-related flag from the original compendium
        - ``mhn`` : multi-label Maslow hierarchy annotation, e.g. ``"1,4,5"``
        - ``reason`` : the LLM's reasoning for the assigned hierarchies

    Notes
    -----
    The dataframe is cached after the first call; each call returns a copy,
    so mutating the result does not affect subsequent calls.
    """
    return _load_cached().copy()


def load_etri_temporal_hierarchy() -> dict[str, np.ndarray]:
    """Load the embedded resampled ETRI temporal hierarchy data.

    The ETRI dataset (South Korea) does not ship in raw form; this is the
    KDE-resampled occurrence-time proxy of each Maslow hierarchy level,
    mirroring the CAPTURE-24 resampled data in
    :mod:`pymaslow.vonMisesMixture`. In the source pickle
    (``datas/resampled_ETRI.pickle``) the keys are the ints 1-5; they are
    exposed here as the strings ``"1"`` to ``"5"`` for consistency with the
    rest of the package.

    Returns
    -------
    dict
        Mapping of hierarchy level (``"1"`` to ``"5"``) to a 1D array of
        occurrence times in **hours of day** (~[0, 24]; slight
        over/undershoot from the KDE resampling). 19,997 samples in total:
        H1=3829, H2=3360, H3=4221, H4=5934, H5=2653.
    """
    with resources.as_file(
        resources.files("pymaslow").joinpath(ETRI_RESAMPLED_RESOURCE)
    ) as path:
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}


def get_activity_hierarchy_map() -> dict[str, tuple[int, ...]]:
    """Map each compendium activity to its Maslow hierarchy levels.

    Returns
    -------
    dict
        Mapping of ``activity`` (specific activity description, lower-case as
        in the compendium) to a sorted tuple of hierarchy level ids, e.g.
        ``{"bicycling, mountain, uphill, vigorous": (1, 4, 5), ...}``.
    """
    df = _load_cached()
    mapping: dict[str, tuple[int, ...]] = {}
    for activity, mhn in zip(df["activity"], df["mhn"], strict=True):
        if pd.isna(activity) or pd.isna(mhn):
            continue
        mapping[str(activity)] = parse_mhn(str(mhn))
    return mapping
