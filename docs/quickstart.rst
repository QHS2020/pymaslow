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

Durations: p(d, t)
------------------

Every activity serving a hierarchy lasts some duration, so at any instant
each hierarchy carries a duration distribution. The
:mod:`pymaslow.durations` module models the joint distribution of duration
and time of day with a positive circular KDE (log-normal kernel for the
duration axis, von Mises kernel for the circular time axis), fitted on the
embedded CAPTURE-24 duration table:

.. code-block:: python

   from pymaslow import durations
   import numpy as np

   durations.data        # embedded table: moment, mhns, activity, duration

   # explore the (moment, duration) relationship (2x2 grid)
   fig, axes = durations.plot()                     # raw durations
   fig, axes = durations.plot(log_duration=True)    # log-durations

   # fit the positive circular KDE (notebook preprocessing: log-duration)
   model = durations.fit()

   # sample durations at specific times of day (log-duration scale)
   samples_8am = model.sample_conditional(8.0, n_samples=1000, random_state=0)
   samples_6pm = model.sample_conditional(18.0, n_samples=1000, random_state=0)
   print(np.exp(samples_8am).mean() / 60, "minutes at 08:00")
   print(np.exp(samples_6pm).mean() / 60, "minutes at 18:00")

   # module-level shortcut (uses a default model fitted on the embedded data)
   s = durations.sample_conditional([8.0, 18.0], n_samples=100, random_state=1)

Note: with the default ``fit(log_duration=True)`` the model's positive
variable is the *log-duration*; ``np.exp(samples)`` converts samples back
to seconds.

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

``pymaslow.hmm`` provides **two** HMM approaches for decoding the latent
need hierarchy, plus the embedded CAPTURE-24/ETRI training data used in the
companion study. The hidden state is the *set* of active need levels —
``2**5 - 1 = 31`` multi-label states (:func:`pymaslow.hmm_states_definition`).

**Embedded data and splitting.** ``hmm.data_capture24`` / ``hmm.data_etri``
are loaded lazily on first access; ``train_test_split`` converts the raw
states (multi-hot vectors for CAPTURE-24, label strings for ETRI) to state
ids and splits by sample:

.. code-block:: python

   from pymaslow import hmm

   data = hmm.data_capture24            # 146 samples, loaded lazily
   train, validation, test = hmm.train_test_split(
       data, test_size_not_percent=20, validation_size_not_percent=10,
       random_state=42,
   )
   train_states, train_activities, train_durations, train_lengths = train

**Approach 1 — duration-emission HMMs** (``ExponentialHMM`` /
``LognormalHMM``): hidden states are the need hierarchies, observations are
activity *durations*; trained by supervised MLE, decoded with Viterbi:

.. code-block:: python

   model = hmm.LognormalHMM(n_states=31)
   model.fit_supervised(train_states, train_durations, verbose=False)
   predicted_states = model.predict(test[2][0])   # one duration sequence

**Approach 2 — categorical HMM** (:class:`pymaslow.CategoricalMaslowHMM`,
wrapping ``hmmlearn``): observations are *activity ids*; parameters are
estimated by supervised MLE with Laplace smoothing and injected into a
``hmmlearn.hmm.CategoricalHMM``:

.. code-block:: python

   cat = hmm.CategoricalMaslowHMM()   # 31 states, 824 activities
   cat.fit_supervised(train_states, train_activities)
   predicted = cat.predict(test[1])   # list of activity sequences

   ground_truth = [s for seq in test[0] for s in seq]
   print(hmm.metrics_classification(ground_truth, predicted))
   # {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

The ETRI variants load via ``hmm.data_etri`` and
``hmm.etri_activity_definitions`` (the 184 ``(action, condition, place)``
activity tuples).
