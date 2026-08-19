"""
pymaslow.hierarchy
==================

Definitions of the five-level Maslow Hierarchy of Needs (MHN), utilities for
parsing multi-label hierarchy annotations, and the large-language-model (LLM)
prompt template used to infer need hierarchies from raw activity descriptions.

The five levels follow Maslow's classical taxonomy:

===  =====================================================================
ID   Need level
===  =====================================================================
1    Physiological Needs
2    Safety Needs
3    Love and Belonging Needs
4    Esteem Needs
5    Self-Actualization Needs
===  =====================================================================

One activity may serve several levels simultaneously (e.g. sharing a meal with
friends satisfies both level 1 and level 3), so annotations are *multi-label*:
the string ``"1,3,4"`` denotes levels 1, 3 and 4 jointly.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Sequence

__all__ = [
    "HIERARCHY_DESCRIPTIONS",
    "HIERARCHY_NAMES",
    "HIERARCHY_SHORT_NAMES",
    "LLM_PROMPT_TEMPLATE",
    "LLM_SYSTEM_MESSAGE",
    "N_HIERARCHIES",
    "build_hierarchy_prompt",
    "format_mhn",
    "mhn_to_vector",
    "parse_mhn",
]

#: Number of levels in the Maslow hierarchy.
N_HIERARCHIES = 5

#: Mapping of hierarchy id (1-5) to its canonical name.
HIERARCHY_NAMES: dict[int, str] = {
    1: "Physiological Needs",
    2: "Safety Needs",
    3: "Love and Belonging Needs",
    4: "Esteem Needs",
    5: "Self-Actualization Needs",
}

#: Short labels (H1..H5) used in figures and compact outputs.
HIERARCHY_SHORT_NAMES: dict[int, str] = {
    i: f"H{i}" for i in range(1, N_HIERARCHIES + 1)
}

#: Brief description of each level.
HIERARCHY_DESCRIPTIONS: dict[int, str] = {
    1: "Basic biological requirements: food, water, sleep, shelter, homeostasis.",
    2: "Security, stability, health, employment, resources, and protection from harm.",
    3: "Social connection, friendship, intimacy, family, and a sense of belonging.",
    4: "Self-esteem, achievement, mastery, recognition, status, and respect.",
    5: "Self-fulfillment, personal growth, creativity, and realizing one's potential.",
}

#: System message paired with the user prompt when querying a chat-style LLM.
LLM_SYSTEM_MESSAGE = "You are a helpful assistant"

#: Prompt template used to infer the Maslow hierarchy of an activity with a
#: reasoning LLM (as in the companion manuscript). Placeholders:
#: ``{activity}`` -- the specific activity description, e.g. ``"bicycling,
#: mountain, uphill, vigorous"``; ``{classification}`` -- the activity's
#: major heading / category, e.g. ``"bicycling"``.
LLM_PROMPT_TEMPLATE = (
    "We know the maslow hierarchy. There are five levels. "
    "The first or bottom is Physiological Needs while the 5th layer is "
    "Self-Actualization Needs. It is possible that one activity belong to "
    "several hierarchies. For instance, eating with friends serve basic "
    "needs while also satisfy the love and belong needs. Given a specific "
    "activity and its classification, determine the hierarchy of the "
    "specific activity using number 1 to 5. The specific activity is given "
    'as "{activity}", and the classification is given by '
    '"{classification}".'
)


def parse_mhn(mhn: str | int | Sequence[int]) -> tuple[int, ...]:
    """Parse a multi-label MHN annotation into a sorted tuple of level ids.

    Parameters
    ----------
    mhn : str, int, or sequence of int
        Annotation such as ``"1,4,5"``, ``"3"``, ``3``, or ``(1, 4, 5)``.

    Returns
    -------
    tuple of int
        Sorted, de-duplicated hierarchy level ids, each in ``[1, 5]``.

    Raises
    ------
    ValueError
        If a level id is outside ``[1, 5]`` or the string is unparsable.
    """
    if isinstance(mhn, str):
        parts = [p.strip() for p in mhn.replace(";", ",").split(",") if p.strip()]
        if not parts:
            raise ValueError(f"Empty MHN annotation: {mhn!r}")
        try:
            levels = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"Unparsable MHN annotation: {mhn!r}") from exc
    elif isinstance(mhn, numbers.Integral):
        try:
            levels = [int(mhn)]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unparsable MHN annotation: {mhn!r}") from exc
    elif isinstance(mhn, Sequence):
        try:
            levels = [int(v) for v in mhn]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unparsable MHN annotation: {mhn!r}") from exc
    else:
        raise TypeError(f"Unparsable MHN annotation type: {type(mhn).__name__}")

    for lv in levels:
        if not 1 <= lv <= N_HIERARCHIES:
            raise ValueError(
                f"Hierarchy level {lv} out of range [1, {N_HIERARCHIES}] in {mhn!r}"
            )
    return tuple(sorted(set(levels)))


def format_mhn(levels: Iterable[int]) -> str:
    """Format hierarchy level ids as a compact multi-label string.

    >>> format_mhn([4, 1, 5])
    '1,4,5'
    """
    try:
        return ",".join(str(lv) for lv in sorted({int(v) for v in levels}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unparsable hierarchy levels: {levels!r}") from exc


def mhn_to_vector(mhn: str | int | Sequence[int]) -> list[int]:
    """Convert a multi-label annotation to a binary indicator vector of length 5.

    >>> mhn_to_vector("1,3")
    [1, 0, 1, 0, 0]
    """
    levels = set(parse_mhn(mhn))
    return [1 if i in levels else 0 for i in range(1, N_HIERARCHIES + 1)]


def build_hierarchy_prompt(activity: str, classification: str) -> str:
    """Build the LLM user prompt for hierarchy identification.

    Parameters
    ----------
    activity : str
        Specific activity description (e.g. ``"bicycling, mountain, uphill,
        vigorous"``).
    classification : str
        Major heading / category of the activity (e.g. ``"bicycling"``).

    Returns
    -------
    str
        The formatted prompt, ready to be sent to a reasoning LLM together
        with :data:`LLM_SYSTEM_MESSAGE`.
    """
    return LLM_PROMPT_TEMPLATE.format(
        activity=str(activity), classification=str(classification)
    )
