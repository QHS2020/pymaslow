Quick start
===========

This page walks through the main components of ``pymaslow`` on small
examples. All examples run without any external data: the package embeds
everything needed.

Embedded data
-------------

The Compendium of Physical Activities annotated with Maslow Hierarchy of
Needs (MHN) labels:

.. code-block:: python

   import pymaslow

   compendium = pymaslow.load_compendium()
   print(compendium[["activity", "major_heading", "mhn"]].head())

   # activity -> hierarchy levels lookup
   mapping = pymaslow.get_activity_hierarchy_map()
   print(mapping["bicycling, mountain, uphill, vigorous"])  # (1, 4, 5)

Hierarchy utilities
-------------------

.. code-block:: python

   pymaslow.HIERARCHY_NAMES[1]   # 'Physiological Needs'
   pymaslow.parse_mhn("1,4,5")   # (1, 4, 5)
   pymaslow.mhn_to_vector("1,3")  # [1, 0, 1, 0, 0]

   # the LLM prompt used to annotate activities with need levels
   prompt = pymaslow.build_hierarchy_prompt(
       "bicycling, mountain, uphill, vigorous", "bicycling"
   )

LLM prompts for hierarchy identification
----------------------------------------

Raw activity datasets record *what* a person does but not *why*; the need
hierarchy of each activity is inferred by a reasoning LLM. The
:mod:`pymaslow.prompts` module ships the exact prompt templates used for
the three datasets of the framework:

.. code-block:: python

   from pymaslow import prompts

   # CAPTURE-24 (Compendium of Physical Activities nomenclature)
   prompts.build_capture24_prompt("bicycling, mountain, uphill, vigorous",
                                  "bicycling")

   # ETRI (action---condition---place triples)
   prompts.build_etri_prompt("work", "WITH_ONE", "other_indoor")

   # DailyLog2016 (start-time---major---minor tuples, minor optional)
   prompts.build_dailylog2016_prompt("13", "Socializing", "Somethingelse")

   # dataset-dispatching helper
   prompts.build_prompt("etri", action="work", condition="WITH_ONE",
                        place="other_indoor")
   prompts.get_prompt_template("capture24")
   prompts.DATASETS            # ('capture24', 'etri', 'dailylog2016')
   prompts.DEFAULT_MODEL       # 'deepseek-reasoner'

Each user prompt is sent to the reasoning model together with the system
message ``prompts.LLM_SYSTEM_MESSAGE`` (``"You are a helpful assistant"``).

Markov chains over need hierarchies
-----------------------------------

.. code-block:: python

   sequences = [["1", "1,3", "3", "4", "5"], ["2", "2", "1"]]
   mc = pymaslow.MarkovChain.from_sequences(sequences)
   print(mc.count_matrix)         # transition counts (labeled DataFrame)
   print(mc.transition_matrix)    # row-normalized probabilities
   print(mc.stationary_distribution())

   # requires the optional networkx dependency
   fig, ax = mc.plot(save_path="markov.png")

von Mises mixtures: p(t | hierarchy)
------------------------------------

The module ships the KDE-resampled CAPTURE-24 occurrence times and the
model fitted on them — both are loaded automatically at import:

.. code-block:: python

   from pymaslow import vonMisesMixture as vmmm

   vmmm.p_x       # fitted prior p(hierarchy)
   vmmm.best_k    # BIC-selected components per hierarchy
   vmmm.models["1"]   # fitted mixture for physiological needs (H1)
   vmmm.data      # resampled occurrence times per hierarchy (hours)

   # plot the fitted joint model p(hierarchy, t) = p(hierarchy) * p(t | hierarchy)
   fig, axes = vmmm.plot_vmmm_results(vmmm.data, vmmm.p_x, vmmm.models, vmmm.best_k)

Sampling from the fitted model
------------------------------

.. code-block:: python

   # sample times of day from one hierarchy's conditional p(t | hierarchy)
   times = pymaslow.sample_vonmises_mixture(
       vmmm.models["1"], n_samples=5000, seed=42, return_radians=False
   )

   # sample (hierarchy, time) pairs from the joint model
   classes, times = pymaslow.sample_joint_vmmm(
       vmmm.p_x, vmmm.models, n_samples=10000, seed=123
   )

   # batch-sample from every hierarchy
   samples = vmmm.sample_vmmm_dictionary(vmmm.models, n_samples=100, seed=0)

Fitting your own data
---------------------

.. code-block:: python

   import numpy as np

   rng = np.random.default_rng(0)
   my_data = {
       "meals":  (rng.normal([7.5, 12.3, 18.7], 0.4, (300, 3)) % 24).ravel(),
       "sleep":  (rng.normal(23.5, 1.0, 400) % 24),
   }
   p_x, models, best_k = pymaslow.fit_vmmm_dictionary(
       my_data, k_max=6, criterion="bic"
   )

Circular KDE
------------

.. code-block:: python

   t_data = rng.normal(23 * 3600, 3600, 500) % 86400   # seconds
   theta = pymaslow.sec_to_rad(t_data)

   kde = pymaslow.fit_circular_kde(theta)
   print(kde.pdf(theta[:5]))

Dirichlet models of need profiles
---------------------------------

.. code-block:: python

   freq = rng.dirichlet([5, 3, 2, 4, 1], size=200)   # (N, 5) simplex rows
   alphas, mean, precision = pymaslow.dirichlet.fit_temporal_profile(freq)
   print(alphas, mean, precision)

Hidden Markov models
--------------------

.. code-block:: python

   hmm = pymaslow.ExponentialHMM(n_states=5)
   hierarchy = [[1, 1, 2, 3], [2, 2, 1]]
   durations = [[1800.0, 900.0, 3600.0, 1200.0], [600.0, 300.0, 2400.0]]
   hmm.fit_supervised(hierarchy, durations, verbose=False)
   print(hmm.predict([1500.0, 700.0, 3000.0]))
