"""
pymaslow.prompts
================

Verbatim large-language-model (LLM) prompt templates used to infer Maslow
need-hierarchy labels for the three datasets of the Temporal Maslow
Hierarchy framework: **CAPTURE-24**, **ETRI**, and **DailyLog2016**.

Raw Human Activity Recognition datasets record *what* a person does but not
*why*; the need hierarchy of each activity is therefore inferred by a
reasoning LLM (the companion study used ``deepseek-reasoner`` with the
system message :data:`LLM_SYSTEM_MESSAGE`). Each dataset has its own prompt
template reflecting its activity nomenclature:

- **CAPTURE-24** uses the Compendium of Physical Activities nomenclature:
  a *specific activity* (e.g. ``"bicycling, mountain, uphill, vigorous"``)
  and its *classification* (major heading, e.g. ``"bicycling"``);
- **ETRI** uses ``action---condition---place`` triples
  (e.g. ``"work---WITH_ONE---other_indoor"``);
- **DailyLog2016** uses ``start-time---major activity---minor activity``
  tuples (e.g. ``"13---Socializing---Somethingelse"``), with the minor
  activity optional, and requests a structured dict answer.

The templates are the exact strings sent by the identification scripts
(``codes/LLM_identify/`` in the companion research repository); builder
functions assemble the final user-message text for one data entry.
"""

from __future__ import annotations

from .hierarchy import (  # pyright: ignore[reportMissingImports]
    LLM_PROMPT_TEMPLATE as _CAPTURE24_TEMPLATE,
)
from .hierarchy import (  # pyright: ignore[reportMissingImports]
    LLM_SYSTEM_MESSAGE,
    build_hierarchy_prompt,
)

__all__ = [  # noqa: RUF022 -- grouped by role for readability
    "LLM_SYSTEM_MESSAGE",
    "DEFAULT_MODEL",
    "DATASETS",
    "CAPTURE24_PROMPT_TEMPLATE",
    "ETRI_PROMPT_TEMPLATE",
    "DAILYLOG2016_PROMPT_TEMPLATE",
    "PROMPT_TEMPLATES",
    "build_capture24_prompt",
    "build_etri_prompt",
    "build_dailylog2016_prompt",
    "build_prompt",
    "get_prompt_template",
]

#: Default reasoning model used for hierarchy identification in the
#: companion study.
DEFAULT_MODEL = "deepseek-reasoner"

#: Supported dataset identifiers.
DATASETS = ("capture24", "etri", "dailylog2016")

#: CAPTURE-24 prompt template (Compendium of Physical Activities
#: nomenclature). Placeholders: ``{activity}``, ``{classification}``.
CAPTURE24_PROMPT_TEMPLATE = _CAPTURE24_TEMPLATE

#: ETRI prompt template. Placeholders: ``{action}``, ``{condition}``,
#: ``{place}`` -- combined into an ``action---condition---place`` triple,
#: where ``action`` is the activity (e.g. ``"work"``), ``condition`` denotes
#: whether the action is done alone (e.g. ``"WITH_ONE"``), and ``place`` is
#: where the action takes place (e.g. ``"other_indoor"``).
ETRI_PROMPT_TEMPLATE = (
    "We know the maslow hierarchy. There are five levels. "
    "The first or bottom is Physiological Needs while the 5th layer is "
    "Self-Actualization Needs. It is possible that one activity belong to "
    "several hierarchies. For instance, eating with friends serve basic "
    "needs while also satisfy the love and belong needs. Given "
    "action---condition---place tuple, where action represent activities "
    "such as work, condition represent do the action along or not alone, "
    "place represent the place that the action takes place, determine the "
    "corresponding hierarchies using number 1 to 5. The "
    'action-condition-place tuple is "{action}---{condition}---{place}"'
)

#: DailyLog2016 prompt template. Placeholders: ``{start_time}``,
#: ``{major}``, and an optional ``{minor}`` -- combined into a
#: ``start-time---major activity---minor activity`` tuple, where
#: ``start_time`` is the beginning hour (24-hour format), ``major`` the
#: major activity (e.g. ``"Housework"``) and ``minor`` its subclass
#: (e.g. ``"TidyingUp"``). The LLM is asked to answer with a dict of the
#: form ``{'hierarchies': [...], 'reason': ...}``.
DAILYLOG2016_PROMPT_TEMPLATE = (
    "\n\nWe know the maslow hierarchy. There are five levels. "
    "The first or bottom is Physiological Needs while the 5th layer is "
    "Self-Actualization Needs. It is possible that one activity belong to "
    "several hierarchies. For instance, eating with friends serve basic "
    "needs while also satisfy the love and belong needs. \n\n\n"
    "Given a tuple of start-time---major activity---minor activity, where "
    "start-time represent the begining hour (in 24 hours format), major "
    "activity represent activities such as Housework, minor activity "
    "represent subclass of major activity such as TidyingUp, determine the "
    "corresponding hierarchies using number 1 to 5. Sometimes the minor "
    "activity may be missing. Then determine the hierarchies using start "
    "time and major activity only. When answer your answer, the format "
    "must be in dict. Such as: {{'hierarchies':[1,2], "
    "'reason':explain_your_reason_here}}\n\n"
    "The start-time---major activity---minor activity or "
    "start-time---major activity tuple is:\n\n"
    "{tuple}"
)

#: Mapping of dataset identifiers to their prompt templates.
PROMPT_TEMPLATES = {
    "capture24": CAPTURE24_PROMPT_TEMPLATE,
    "etri": ETRI_PROMPT_TEMPLATE,
    "dailylog2016": DAILYLOG2016_PROMPT_TEMPLATE,
}


def build_capture24_prompt(activity: str, classification: str) -> str:
    """Build the CAPTURE-24 user prompt for one activity entry.

    Parameters
    ----------
    activity : str
        Specific activity description, e.g. ``"bicycling, mountain, uphill,
        vigorous"``.
    classification : str
        Major heading of the activity, e.g. ``"bicycling"``.

    Returns
    -------
    str
        The user-message text to send together with
        :data:`LLM_SYSTEM_MESSAGE`.
    """
    return build_hierarchy_prompt(activity, classification)


def build_etri_prompt(action: str, condition: str, place: str) -> str:
    """Build the ETRI user prompt for one ``action---condition---place`` entry.

    Parameters
    ----------
    action : str
        Activity, e.g. ``"work"``.
    condition : str
        Whether the action is done alone, e.g. ``"WITH_ONE"``.
    place : str
        Where the action takes place, e.g. ``"other_indoor"``.

    Returns
    -------
    str
        The user-message text.
    """
    return ETRI_PROMPT_TEMPLATE.format(action=action, condition=condition, place=place)


def build_dailylog2016_prompt(
    start_time: str | int, major: str, minor: str | None = None
) -> str:
    """Build the DailyLog2016 user prompt for one activity-log entry.

    Parameters
    ----------
    start_time : str or int
        Beginning hour in 24-hour format, e.g. ``"13"``.
    major : str
        Major activity, e.g. ``"Socializing"``.
    minor : str or None
        Subclass of the major activity, e.g. ``"Somethingelse"``; omit when
        unavailable.

    Returns
    -------
    str
        The user-message text.
    """
    if minor is None:
        entry = f"{start_time}---{major}."
    else:
        entry = f"{start_time}---{major}---{minor}."
    return DAILYLOG2016_PROMPT_TEMPLATE.format(tuple=entry)


def get_prompt_template(dataset: str) -> str:
    """Return the prompt template for one of :data:`DATASETS`.

    Raises
    ------
    ValueError
        If ``dataset`` is not one of ``"capture24"``, ``"etri"``,
        ``"dailylog2016"``.
    """
    key = dataset.lower()
    if key not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
    return PROMPT_TEMPLATES[key]


def build_prompt(dataset: str, **kwargs) -> str:
    """Build the user prompt for one data entry of a supported dataset.

    Parameters
    ----------
    dataset : {'capture24', 'etri', 'dailylog2016'}
        Which dataset's prompt to build.
    **kwargs
        Entry fields: ``activity=, classification=`` for CAPTURE-24;
        ``action=, condition=, place=`` for ETRI; ``start_time=, major=``
        and optional ``minor=`` for DailyLog2016.

    Returns
    -------
    str
        The user-message text.
    """
    key = dataset.lower()
    if key == "capture24":
        return build_capture24_prompt(kwargs["activity"], kwargs["classification"])
    if key == "etri":
        return build_etri_prompt(kwargs["action"], kwargs["condition"], kwargs["place"])
    if key == "dailylog2016":
        return build_dailylog2016_prompt(
            kwargs["start_time"], kwargs["major"], kwargs.get("minor")
        )
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")
