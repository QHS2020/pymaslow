"""
pymaslow.sampling
=================

Generative sampling of full daily activity sequences by chaining the
package's fitted models.

Every activity serving a need hierarchy lasts some duration. Given an
initial moment ``t0`` (e.g. 02:00), the sampler repeatedly:

1. **samples the need hierarchy** at the current time from the joint von
   Mises mixture model, via the posterior

   .. math::

       p(h \\mid t) = \\frac{p(h)\\, p(t \\mid h)}{\\sum_{h'} p(h')\\, p(t \\mid h')}

   with ``p(h)`` and ``p(t | h)`` from :mod:`pymaslow.vonMisesMixture`;
2. **samples the activity** from the categorical HMM's emission
   distribution ``p(a | h)`` (:class:`pymaslow.CategoricalMaslowHMM`);
3. **samples the duration** ``tau`` from the positive circular KDE's
   conditional ``p(d | t)`` (:class:`pymaslow.durations.PositiveCircularKDE`);
4. advances time: ``t <- t + tau`` (wrapping past midnight into the next
   day), and repeats.

The result is a sequence of ``(time, hierarchy, activity, duration)``
records — a synthetic activity diary. Called without arguments,
:func:`sample_sequence` uses the package's embedded pre-fitted models
(CAPTURE-24), so it works out of the box.

The procedure mirrors ``SampleActivitySequence`` in
``codes/vonMises/utilities.py`` of the companion research repository, with
the hierarchy step added (the prototype sampled activities directly from a
discrete circular KDE).
"""

from __future__ import annotations

import numpy as np

from .hierarchy import HIERARCHY_NAMES  # pyright: ignore[reportMissingImports]
from .timeutils import (  # pyright: ignore[reportMissingImports]
    format_time,
    hours_to_rad,
)

__all__ = [
    "sample_hierarchy_given_time",
    "sample_activity_given_hierarchy",
    "sample_sequence",
]

#: Module-level caches for the embedded default models (lazy singletons).
_DEFAULT_HMM = None
_DEFAULT_DURATION_MODEL = None


def _default_hmm():
    """CategoricalMaslowHMM fitted on the embedded CAPTURE-24 data (cached)."""
    global _DEFAULT_HMM
    if _DEFAULT_HMM is None:
        from .hmm import (  # pyright: ignore[reportMissingImports]
            CategoricalMaslowHMM,
            load_hmm_data,
            train_test_split,
        )

        data = load_hmm_data("capture24")
        train, _, _ = train_test_split(
            data, test_size_not_percent=20, random_state=42
        )
        _DEFAULT_HMM = CategoricalMaslowHMM().fit_supervised(train[0], train[1])
    return _DEFAULT_HMM


def _default_duration_model():
    """PositiveCircularKDE fitted on the embedded duration table (cached)."""
    global _DEFAULT_DURATION_MODEL
    if _DEFAULT_DURATION_MODEL is None:
        from .durations import fit  # pyright: ignore[reportMissingImports]

        _DEFAULT_DURATION_MODEL = fit()
    return _DEFAULT_DURATION_MODEL


def _default_vmmm():
    """Embedded von Mises mixture prior and conditionals."""
    from .vonMisesMixture import (  # pyright: ignore[reportMissingImports]
        models,
        p_x,
    )

    return p_x, models


def _hierarchy_level(hierarchy) -> int:
    """Parse a hierarchy label (``"1"``..``"5"`` or 1..5) to its level."""
    try:
        level = int(hierarchy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid hierarchy label: {hierarchy!r}") from exc
    if not 1 <= level <= 5:
        raise ValueError(f"hierarchy must be in 1..5, got {hierarchy!r}")
    return level


def sample_hierarchy_given_time(
    t_hours: float,
    p_x: dict | None = None,
    models: dict | None = None,
    random_state=None,
) -> str:
    """Sample the need hierarchy at time ``t_hours`` from ``p(h | t)``.

    The posterior combines the hierarchy prior with the per-hierarchy
    temporal conditionals of the von Mises mixture model:
    ``p(h | t) ∝ p(h) · p(t | h)``.

    Parameters
    ----------
    t_hours : float
        Time of day in hours ``[0, 24)``.
    p_x, models : dict or None
        Hierarchy prior and per-hierarchy :class:`VonMisesMixture`
        conditionals; defaults to the embedded fitted model.
    random_state : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    str
        The sampled hierarchy level label, e.g. ``"1"``.
    """
    if p_x is None or models is None:
        p_x, models = _default_vmmm()
    rng = np.random.default_rng(random_state)

    try:
        t_value = float(t_hours)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid time of day: {t_hours!r}") from exc
    theta = hours_to_rad(t_value % 24.0)
    classes = list(models.keys())
    weights = np.array(
        [p_x[c] * models[c].pdf(theta)[()] for c in classes], dtype=float
    )
    total = weights.sum()
    if total <= 0:
        weights = np.full(len(classes), 1.0 / len(classes))
    else:
        weights = weights / total
    return str(rng.choice(classes, p=weights))


def sample_activity_given_hierarchy(
    hierarchy,
    hmm_model=None,
    random_state=None,
) -> int:
    """Sample an activity id from the HMM emission ``p(a | hierarchy)``.

    Parameters
    ----------
    hierarchy : str or int
        Need-hierarchy level (``"1"``..``"5"`` or 1..5); mapped to the
        corresponding single-level hidden state of the HMM.
    hmm_model : CategoricalMaslowHMM or None
        A fitted categorical HMM; defaults to one fitted on the embedded
        CAPTURE-24 training set.
    random_state : int or np.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    int
        The sampled activity id (Compendium row index).
    """
    if hmm_model is None:
        hmm_model = _default_hmm()
    if hmm_model.emission_prob_ is None:
        raise RuntimeError("hmm_model is not fitted; call fit_supervised() first.")

    state_id = _hierarchy_level(hierarchy) - 1  # single-level states are ids 0..4

    rng = np.random.default_rng(random_state)
    row = hmm_model.emission_prob_[state_id]
    obs_ids = np.array(sorted(row.keys()))
    probs = np.array([row[o] for o in obs_ids], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(obs_ids, p=probs).item()


def _activity_name(activity_id: int) -> str:
    """Compendium activity name for an activity id (row index)."""
    from .data import load_compendium  # pyright: ignore[reportMissingImports]

    df = load_compendium()
    if 0 <= activity_id < len(df):
        return str(df.iloc[activity_id]["activity"])
    return f"activity_{activity_id}"


def _parse_t0(t0: float | str) -> float:
    """Parse an initial time given as hours or an ``HH:MM[:SS]`` string."""
    if isinstance(t0, str):
        parts = t0.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid time format: {t0!r}. Use 'HH:MM' or 'HH:MM:SS'")
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2]) if len(parts) > 2 else 0
        except ValueError as exc:
            raise ValueError(
                f"Invalid time format: {t0!r}. Use 'HH:MM' or 'HH:MM:SS'"
            ) from exc
        t = hours + minutes / 60.0 + seconds / 3600.0
    else:
        try:
            t = float(t0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid time format: {t0!r}. Use hours or 'HH:MM[:SS]'"
            ) from exc
    if not 0 <= t < 24:
        raise ValueError(f"Initial time must be in [0, 24), got {t}")
    return t


def sample_sequence(
    t0: float | str = 2.0,
    n_activities: int | None = None,
    max_days: int = 1,
    hmm_model=None,
    duration_model=None,
    p_x: dict | None = None,
    models: dict | None = None,
    log_scale_duration: bool = True,
    include_activity_names: bool = True,
    random_state=None,
) -> list[dict]:
    """Sample a synthetic activity diary by chaining the fitted models.

    Starting from ``t0``, repeatedly samples the need hierarchy (posterior
    of the joint von Mises mixture at the current time), the activity (HMM
    emission given the hierarchy), and the duration (positive circular KDE
    conditional given the time), then advances time by the duration.

    Parameters
    ----------
    t0 : float or str
        Initial time: hours in ``[0, 24)`` or an ``"HH:MM[:SS]"`` string
        (e.g. ``"02:00"``).
    n_activities : int or None
        Stop after this many activities; None means no cap (only
        ``max_days`` applies).
    max_days : int
        Maximum number of days to simulate (midnight crossings).
    hmm_model : CategoricalMaslowHMM or None
        Fitted categorical HMM; default: fitted on embedded CAPTURE-24 data.
    duration_model : PositiveCircularKDE or None
        Fitted duration model; default: fitted on the embedded duration
        table.
    p_x, models : dict or None
        Von Mises mixture prior and conditionals; default: the embedded
        pre-fitted model.
    log_scale_duration : bool
        Whether ``duration_model`` returns log-duration samples (True for
        the default embedded model, which follows the notebook's
        ``fit(log_duration=True)`` workflow); samples are then converted
        with ``np.exp``. Set False for a model fitted on raw durations.
    include_activity_names : bool
        Resolve activity ids to Compendium activity names.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    list of dict
        One record per activity with keys: ``day``, ``start`` / ``end``
        (hours of day), ``start_time`` / ``end_time`` (``"HH:MM"``),
        ``hierarchy`` (label, e.g. ``"1"``), ``hierarchy_name``,
        ``activity_id``, ``activity`` (name, if enabled), and
        ``duration_seconds``.

    Examples
    --------
    >>> from pymaslow.sampling import sample_sequence
    >>> diary = sample_sequence(t0="02:00", max_days=1, random_state=42)
    >>> diary[0].keys()
    dict_keys([...])
    """
    if n_activities is None and max_days is None:
        raise ValueError("At least one of n_activities or max_days must bound the sampling")
    if hmm_model is None:
        hmm_model = _default_hmm()
    if duration_model is None:
        duration_model = _default_duration_model()
    if p_x is None or models is None:
        p_x, models = _default_vmmm()

    rng = np.random.default_rng(random_state)
    t_current = _parse_t0(t0)

    records: list[dict] = []
    day = 0
    while day < max_days and (n_activities is None or len(records) < n_activities):
        # 1. hierarchy from the joint von Mises mixture posterior p(h | t)
        hierarchy = sample_hierarchy_given_time(t_current, p_x, models, random_state=rng)

        # 2. activity from the HMM emission p(a | h)
        activity_id = sample_activity_given_hierarchy(hierarchy, hmm_model, random_state=rng)

        # 3. duration from the positive circular KDE p(d | t)
        duration = np.atleast_1d(
            duration_model.sample_conditional(t_current, n_samples=1, random_state=rng)
        )[0]
        duration_seconds = np.exp(duration) if log_scale_duration else duration
        duration_seconds = max(duration_seconds, 1.0)  # at least one second

        duration_hours = duration_seconds / 3600.0
        t_next = t_current + duration_hours

        record = {
            "day": day,
            "start": t_current,
            "end": min(t_next, 24.0),
            "start_time": format_time(t_current * 3600.0),
            "end_time": format_time(min(t_next, 24.0) * 3600.0),
            "hierarchy": hierarchy,
            "hierarchy_name": HIERARCHY_NAMES[_hierarchy_level(hierarchy)],
            "activity_id": activity_id,
            "duration_seconds": duration_seconds,
        }
        if include_activity_names:
            record["activity"] = _activity_name(activity_id)
        records.append(record)

        # 4. advance time, wrapping past midnight into the next day
        if t_next >= 24.0:
            day += 1
            t_current = t_next % 24.0
        else:
            t_current = t_next

    return records
