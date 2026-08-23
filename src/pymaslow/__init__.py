"""
pymaslow
========

Temporal Maslow Hierarchy in Python: computational tools for modeling the
diurnal rhythms of human motivational needs from daily activity data.

Submodules
----------
- :mod:`pymaslow.data` -- embedded MHN-annotated Compendium of Physical
  Activities.
- :mod:`pymaslow.hierarchy` -- hierarchy definitions, label parsing, and the
  LLM prompt template for hierarchy identification.
- :mod:`pymaslow.markov` -- Markov chains over need hierarchies.
- :mod:`pymaslow.vonMisesMixture` -- von Mises mixture models ``p(t | hierarchy)``
  of time-of-day, with embedded resampled CAPTURE-24 data and fitted
  parameters, plotting, and sampling.
- :mod:`pymaslow.circularkde` -- circular kernel density estimation.
- :mod:`pymaslow.dirichlet` -- Dirichlet models of compositional need profiles.
- :mod:`pymaslow.hmm` -- hidden Markov models for need-hierarchy decoding:
  duration-emission models (exponential, lognormal) and the categorical
  model over activities, with embedded CAPTURE-24/ETRI training data.
- :mod:`pymaslow.timeutils` -- time-of-day <-> angle conversion helpers.
- :mod:`pymaslow.prompts` -- verbatim LLM prompt templates for hierarchy
  identification on CAPTURE-24, ETRI, and DailyLog2016.
- :mod:`pymaslow.durations` -- joint duration/time-of-day modeling with the
  positive circular KDE, with embedded CAPTURE-24 duration data.
- :mod:`pymaslow.plotting` -- visualizations (import explicitly; pulls in
  matplotlib).
"""

from __future__ import annotations

from . import datautilities, dirichlet, durations, prompts, sampling, vonMisesMixture
from .circularkde import CircularKDE, fit_circular_kde
from .data import get_activity_hierarchy_map, load_compendium
from .durations import (  # pyright: ignore[reportMissingImports]
  PositiveCircularKDE,
)
from .hierarchy import (
  HIERARCHY_DESCRIPTIONS,
  HIERARCHY_NAMES,
  HIERARCHY_SHORT_NAMES,
  LLM_PROMPT_TEMPLATE,
  LLM_SYSTEM_MESSAGE,
  N_HIERARCHIES,
  build_hierarchy_prompt,
  format_mhn,
  mhn_to_vector,
  parse_mhn,
)
from .hmm import (  # pyright: ignore[reportMissingImports]
  CategoricalMaslowHMM,
  ExponentialHMM,
  LognormalHMM,
  estimate_hmm_parameters,
  hmm_states_definition,
  load_etri_activity_definitions,
  load_hmm_data,
  metrics_classification,
  train_test_split,
)
from .markov import (  # pyright: ignore[reportMissingImports]
  MarkovChain,
  build_transition_counts,
)
from .prompts import (  # pyright: ignore[reportMissingImports]
  CAPTURE24_PROMPT_TEMPLATE,
  DAILYLOG2016_PROMPT_TEMPLATE,
  DATASETS,
  ETRI_PROMPT_TEMPLATE,
  PROMPT_TEMPLATES,
  build_capture24_prompt,
  build_dailylog2016_prompt,
  build_etri_prompt,
  build_prompt,
  get_prompt_template,
)
from .timeutils import (  # pyright: ignore[reportMissingImports]
  SECONDS_PER_DAY,
  SECONDS_PER_HOUR,
  CircularTimeModel,
  format_time,
  hours_to_rad,
  rad_to_hours,
  rad_to_sec,
  sec_to_rad,
)
from .vonMisesMixture import (  # pyright: ignore[reportMissingImports]
  VonMisesMixture,
  fit_vmmm_dictionary,
  load_fitted_models,
  load_resampled_data,
  plot_vmmm_results,
  sample_joint_vmmm,
  sample_vmmm_dictionary,
  sample_vonmises_mixture,
)

__version__ = "0.7.0"

__all__ = [  # noqa: RUF022 -- grouped by submodule for readability
  "__version__",
  # submodules
  "datautilities",
  "dirichlet",
  "durations",
  "prompts",
  "sampling",
  "vonMisesMixture",
  # data
  "load_compendium",
  "get_activity_hierarchy_map",
  # hierarchy
  "N_HIERARCHIES",
  "HIERARCHY_NAMES",
  "HIERARCHY_SHORT_NAMES",
  "HIERARCHY_DESCRIPTIONS",
  "LLM_PROMPT_TEMPLATE",
  "LLM_SYSTEM_MESSAGE",
  "parse_mhn",
  "format_mhn",
  "mhn_to_vector",
  "build_hierarchy_prompt",
  # markov
  "MarkovChain",
  "build_transition_counts",
  # timeutils
  "SECONDS_PER_DAY",
  "SECONDS_PER_HOUR",
  "sec_to_rad",
  "rad_to_sec",
  "hours_to_rad",
  "rad_to_hours",
  "format_time",
  "CircularTimeModel",
  # vonMisesMixture
  "VonMisesMixture",
  "fit_vmmm_dictionary",
  "plot_vmmm_results",
  "sample_vonmises_mixture",
  "sample_vmmm_dictionary",
  "sample_joint_vmmm",
  "load_resampled_data",
  "load_fitted_models",
  # circularkde
  "CircularKDE",
  "fit_circular_kde",
  # hmm
  "ExponentialHMM",
  "LognormalHMM",
  "CategoricalMaslowHMM",
  "hmm_states_definition",
  "estimate_hmm_parameters",
  "train_test_split",
  "metrics_classification",
  "load_hmm_data",
  "load_etri_activity_definitions",
  # durations
  "PositiveCircularKDE",
  # prompts
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
