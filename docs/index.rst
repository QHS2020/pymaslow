pymaslow
========

**Temporal Maslow Hierarchy in Python** — computational tools for modeling
the diurnal rhythms of human motivational needs from daily activity data.

``pymaslow`` operationalizes Maslow's hierarchy of needs in the temporal
domain. Given timestamped activity logs (e.g. the CAPTURE-24, ETRI, or
DailyLog2016 datasets) whose activities have been mapped onto the five
Maslow need levels, the package characterizes the diurnal rhythm of human
motivational needs with Markov chains, mixtures of von Mises distributions,
circular kernel density estimation, Dirichlet compositional models, and
hidden Markov models.

Features
--------

- **Embedded data** — the Compendium of Physical Activities annotated with
  Maslow Hierarchy of Needs (MHN) labels (823 activities), plus
  KDE-resampled CAPTURE-24 hierarchy occurrence times and a **pre-fitted**
  von Mises mixture model ``p(hierarchy, t)`` available on import.
- **Hierarchy tools** — level definitions, multi-label parsing, and the
  LLM prompt template used to infer need hierarchies from raw activities.
- **Markov chains** — transition counts and probabilities between need
  hierarchies, stationary distributions, graph visualization.
- **von Mises mixtures** — EM-fitted circular mixtures ``p(t | hierarchy)``
  with BIC component selection, plotting, and vectorized sampling.
- **Circular KDE** — boundary-corrected kernel density estimation for
  time-of-day data.
- **Dirichlet models** — maximum-likelihood estimation for compositional
  need profiles and likelihood-ratio tests.
- **Hidden Markov models** — exponential and lognormal duration emissions
  for decoding latent need hierarchies.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: About

   citation
   license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
