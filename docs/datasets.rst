Embedded datasets
=================

``pymaslow`` ships three embedded datasets so that every model in the
package works out of the box — no downloads and no access to the (very
large) raw activity collections are required. This page describes each
dataset in detail: provenance, format, and access API.

.. contents::
   :local:
   :depth: 2

Compendium of Physical Activities with MHNs
-------------------------------------------

**Access:** ``pymaslow.load_compendium()`` → :class:`pandas.DataFrame`;
also ``pymaslow.get_activity_hierarchy_map()`` → ``dict``

**Source file:** ``datas/Compendium of Physical Activities_withMHNs.xlsx``
(companion repository), embedded in the package as
``data/compendium_mhn.csv``.

**Provenance.** The `Compendium of Physical Activities
<https://pacompendium.com/>`_ (Ainsworth et al., 2011) is the standard
nomenclature of human activities used by the CAPTURE-24 dataset; each
activity carries a MET (Metabolic Equivalent of Task) value. Because the
compendium records *what* activities are but not *why* they are performed,
every activity was annotated with Maslow Hierarchy of Needs (MHN) labels by
a reasoning LLM (``deepseek-reasoner``) using the CAPTURE-24 prompt
template shipped in :mod:`pymaslow.prompts`; the model's reasoning is kept
in the ``reason`` column.

**Shape:** 823 rows × 7 columns.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Column
     - Type
     - Description
   * - ``code``
     - float
     - Compendium activity code (e.g. ``1003.0``).
   * - ``mets``
     - float
     - Metabolic Equivalent of Task, range **1.0–23.0**.
   * - ``major_heading``
     - str
     - Activity category; **21** distinct values (e.g. ``bicycling``,
       ``walking``, ``occupation``, ``self care``).
   * - ``activity``
     - str
     - Specific activity description (e.g. ``bicycling, mountain, uphill,
       vigorous``).
   * - ``flag``
     - float
     - Note-related flag from the original compendium (non-null for 73
       rows).
   * - ``mhn``
     - str
     - Multi-label Maslow hierarchy annotation, e.g. ``"1,4,5"``; parse
       with :func:`pymaslow.parse_mhn`.
   * - ``reason``
     - str
     - The LLM's reasoning for the assigned hierarchies (all 823 rows).

**Hierarchy label statistics.** One activity may serve several need levels
simultaneously:

- labels per activity: **526** activities with 1 label, **176** with 2,
  **105** with 3, **15** with 4, **1** with all 5;
- per-level counts (an activity counts toward each of its labels):
  H1 Physiological **322**, H2 Safety **197**, H3 Love & Belonging **185**,
  H4 Esteem **291**, H5 Self-Actualization **263**.

.. code-block:: python

   import pymaslow

   df = pymaslow.load_compendium()
   df.shape                      # (823, 7)
   df[["activity", "mhn"]].head()

   mapping = pymaslow.get_activity_hierarchy_map()
   mapping["bicycling, mountain, uphill, vigorous"]   # (1, 4, 5)

Resampled CAPTURE-24 occurrence times
-------------------------------------

**Access:** ``pymaslow.vonMisesMixture.data`` → ``dict[str, ndarray]``;
also ``pymaslow.load_resampled_data()``

**Source file:** ``datas/resampleddata4vonMisesMixture.pickle`` (companion
repository), embedded as ``data/resampleddata4vonMisesMixture.npz``.

**Provenance.** The raw CAPTURE-24 activity sequences are far too large to
ship. The occurrence moments of each need hierarchy were aggregated from 30
randomly sampled participants (of 152) and **KDE-resampled** to a compact
proxy preserving the temporal distribution of each hierarchy (see the
*resample data* section of ``notebooks/pymaslow.ipynb``). This is the
dataset the embedded pre-fitted von Mises mixture model
(``pymaslow.vonMisesMixture.models``) was fitted on.

**Format.** Dictionary mapping hierarchy level (``"1"`` to ``"5"``) to a 1D
array of occurrence times in **hours of day** (~[0, 24]; slight
over/undershoot from the KDE resampling). 19,997 samples in total:

.. list-table::
   :header-rows: 1
   :widths: 30 15 25

   * - Hierarchy
     - Samples
     - Hours range
   * - H1 Physiological Needs
     - 10,506
     - -0.31 – 24.26
   * - H2 Safety Needs
     - 5,302
     - 3.93 – 23.41
   * - H3 Love and Belonging Needs
     - 1,478
     - 4.06 – 23.31
   * - H4 Esteem Needs
     - 859
     - 5.42 – 23.19
   * - H5 Self-Actualization Needs
     - 1,852
     - 4.87 – 23.94

.. code-block:: python

   from pymaslow import vonMisesMixture as vmmm

   {k: len(v) for k, v in vmmm.data.items()}
   # {'1': 10506, '2': 5302, '3': 1478, '4': 859, '5': 1852}

Resampled ETRI temporal hierarchy data
--------------------------------------

**Access:** ``pymaslow.ETRI_TemporalHierarchyData`` → ``dict[str, ndarray]``
(loaded at import); also ``pymaslow.data.load_etri_temporal_hierarchy()``

**Source file:** ``datas/resampled_ETRI.pickle`` (companion repository),
embedded as ``data/resampled_ETRI.npz``.

**Provenance.** The ETRI lifelog dataset (South Korea) provides the second
cultural context of the framework. As with CAPTURE-24, the raw sequences
are too large to ship, so the occurrence moments of each need hierarchy
were KDE-resampled into a compact proxy preserving each hierarchy's
temporal distribution. In the source pickle the keys are the ints 1–5;
they are exposed as the strings ``"1"`` to ``"5"`` for consistency with the
rest of the package.

**Format.** Dictionary mapping hierarchy level to a 1D array of occurrence
times in **hours of day** (~[0, 24]). 19,997 samples in total — note the
different composition from CAPTURE-24 (esteem needs dominate in ETRI):

.. list-table::
   :header-rows: 1
   :widths: 30 15 25

   * - Hierarchy
     - Samples
     - Hours range
   * - H1 Physiological Needs
     - 3,829
     - -0.93 – 24.86
   * - H2 Safety Needs
     - 3,360
     - -1.35 – 24.67
   * - H3 Love and Belonging Needs
     - 4,221
     - -0.32 – 24.32
   * - H4 Esteem Needs
     - 5,934
     - -0.28 – 24.74
   * - H5 Self-Actualization Needs
     - 2,653
     - -0.72 – 24.47

.. code-block:: python

   import pymaslow

   {k: len(v) for k, v in pymaslow.ETRI_TemporalHierarchyData.items()}
   # {'1': 3829, '2': 3360, '3': 4221, '4': 5934, '5': 2653}

   # the ETRI data feeds the same fitting pipeline as CAPTURE-24
   p_x, models, best_k, table = pymaslow.fit_vmmm_dictionary(
       pymaslow.ETRI_TemporalHierarchyData, k_max=8
   )

CAPTURE-24 duration table
-------------------------

**Access:** ``pymaslow.durations.data`` → ``dict[str, ndarray]``; also
``pymaslow.durations.load_data()``

**Source file:** ``datas/t_mhn_activity_dAct_dMhn.pickle`` (companion
repository), embedded as ``data/t_mhn_activity_dAct_dMhn.npz``.

**Provenance.** Computed from the raw CAPTURE-24 sequences of the same 30
randomly sampled participants (see the *Duration* section of
``notebooks/pymaslow.ipynb``): for every activity episode, the start moment
and the episode duration were extracted together with the activity code and
its (multi-label) hierarchy annotation. Every activity serving a hierarchy
lasts some duration, so at any instant of the day each hierarchy is
associated with a distribution of durations — this table is the empirical
basis of :mod:`pymaslow.durations`.

**Format:** 2,021 entries × 4 keys.

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Key
     - Type
     - Description
   * - ``moment``
     - float64
     - Activity start time in **seconds since midnight**
       (≈ 0.07–23.9 hours).
   * - ``mhns``
     - str
     - Multi-label hierarchy annotation, e.g. ``"1,2"``; **16** distinct
       combinations (most frequent: ``"1,2"`` ×508, ``"5"`` ×405,
       ``"1"`` ×246, ``"3"`` ×226, ``"2"`` ×221).
   * - ``activity``
     - int64
     - Compendium activity code; **67** distinct activities.
   * - ``duration``
     - float64
     - Episode duration in **seconds**: min 1 s, median 302 s
       (≈ 5 min), mean 1,238 s (≈ 21 min), max 36,185 s (≈ 10 h);
       139 episodes exceed 1 hour, 15 exceed 6 hours.

.. code-block:: python

   from pymaslow import durations

   d = durations.data
   d["moment"] / 3600          # start time in hours
   d["duration"] / 60          # duration in minutes
   d["mhns"][:5]               # array(['1', '1', '2', '2', '5'], dtype='<U7')

Notes on scope
--------------

All three datasets derive from the **CAPTURE-24** study (United Kingdom)
and the LLM-based hierarchy annotation pipeline; the ETRI (South Korea) and
DailyLog2016 (Germany) datasets referenced by the framework are *not*
embedded due to their size and licensing, but the package's models accept
equivalent user-provided data in the same formats (see
:doc:`quickstart`).
