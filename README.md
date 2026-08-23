# pymaslow

**Temporal Maslow Hierarchy in Python** — computational tools for modeling the
diurnal rhythms of human motivational needs from daily activity data.

`pymaslow` operationalizes Maslow's hierarchy of needs in the temporal domain.
Given timestamped activity logs (e.g. the CAPTURE-24, ETRI, or DailyLog2016
datasets) whose activities have been mapped onto the five Maslow need levels,
the package provides:

- **`pymaslow.data`** — an embedded copy of the *Compendium of Physical
  Activities* annotated with Maslow Hierarchy of Needs (MHN) labels.
- **`pymaslow.hierarchy`** — hierarchy definitions, label parsing utilities,
  and the large-language-model (LLM) prompt template used to infer need
  hierarchies from raw activity descriptions.
- **`pymaslow.prompts`** — the verbatim LLM prompt templates used to infer
  need hierarchies for the CAPTURE-24, ETRI, and DailyLog2016 datasets,
  with per-dataset builder functions.
- **`pymaslow.datautilities`** — data processing from raw CAPTURE-24 csv
  files to aligned (time, hierarchy) streams: annotation parsing, compendium
  code lookup, and per-hierarchy moment collection.
- **`pymaslow.markov`** — first-order Markov chains over need hierarchies:
  transition-count matrices from (multi-label) hierarchy sequences,
  row-normalized transition probabilities, stationary distributions, and
  graph visualization.
- **`pymaslow.vonMisesMixture`** — mixtures of von Mises distributions
  modeling the occurrence-time distribution `p(t | hierarchy)` of each
  hierarchy, fitted by EM with BIC/AIC selection of the number of
  components. The module embeds the KDE-resampled CAPTURE-24 occurrence
  times (`data`) and the parameters fitted on them (`p_x`, `models`,
  `best_k`) — available immediately on import, no raw data needed — plus
  plotting (`plot_vmmm_results`) and sampling (`sample_vonmises_mixture`,
  `sample_joint_vmmm`, `sample_vmmm_dictionary`) utilities.
- **`pymaslow.durations`** — joint duration/time-of-day modeling with the
  positive circular KDE (log-normal duration kernel + von Mises time
  kernel), with the embedded CAPTURE-24 duration table, a 2x2 overview
  plot, and conditional sampling `p(d | t)`.
- **`pymaslow.circularkde`** — circular kernel density estimation of
  time-of-day distributions with boundary correction at midnight.
- **`pymaslow.dirichlet`** — maximum-likelihood estimation of Dirichlet
  distributions for the compositional (sums-to-one) need profile at each
  time of day, plus likelihood-ratio tests (a port of Thomas P. Minka's
  fastfit MATLAB code).
- **`pymaslow.hmm`** — hidden Markov models with continuous duration
  emissions (exponential and lognormal) for decoding the latent need
  hierarchy underlying observed activity/duration sequences.
- **`pymaslow.timeutils`** — time-of-day ↔ angle conversion helpers
  (`sec_to_rad`, `hours_to_rad`, `format_time`, …).
- **`pymaslow.plotting`** — publication-ready visualizations: 24-hour clock
  (polar) plots, AM/PM radar plots, linear time-density plots, and BIC
  model-selection curves.

## Installation

```bash
pip install pymaslow
```

For Markov-chain graph visualization (requires `networkx`):

```bash
pip install "pymaslow[plot]"
```

## Quick start

```python
import numpy as np
import pymaslow

# --- Embedded annotated Compendium of Physical Activities -------------------
compendium = pymaslow.load_compendium()
print(compendium.head())

# --- Hierarchy utilities ----------------------------------------------------
pymaslow.HIERARCHY_NAMES[1]        # 'Physiological Needs'
pymaslow.parse_mhn("1,4,5")        # (1, 4, 5)

# --- Markov chain over need hierarchies -------------------------------------
sequences = [["1", "1,3", "3", "4", "5"], ["2", "2", "1"]]  # multi-label steps
mc = pymaslow.MarkovChain.from_sequences(sequences)
print(mc.count_matrix)
print(mc.transition_matrix)   # row-normalized
print(mc.stationary_distribution())

# --- Temporal distribution p(t | hierarchy), pre-fitted on CAPTURE-24 ------
from pymaslow import vonMisesMixture as vmmm

print(vmmm.p_x)        # fitted prior p(hierarchy)
print(vmmm.best_k)     # BIC-selected components per hierarchy
print(vmmm.models["1"])  # fitted mixture for physiological needs (H1)

# the embedded resampled occurrence times (hours of day)
print({k: len(v) for k, v in vmmm.data.items()})

# plot the fitted joint model p(hierarchy, t) = p(hierarchy) * p(t | hierarchy)
fig, axes = vmmm.plot_vmmm_results(vmmm.data, vmmm.p_x, vmmm.models, vmmm.best_k)

# sample new occurrence times from the fitted model
cls, times = pymaslow.sample_joint_vmmm(vmmm.p_x, vmmm.models, n_samples=1000, seed=42)
h1_times = pymaslow.sample_vonmises_mixture(vmmm.models["1"], n_samples=500, seed=1,
                                            return_radians=False)

# fit your own data (hours of day per class); also returns a results table
p_x, models, best_k, table = pymaslow.fit_vmmm_dictionary(
    {"meals": np.array([7.5, 8.0, 12.2, 12.5, 18.7, 19.1] * 20)}, verbose=False
)
print(table[["class", "n", "K", "BIC", "peak_times"]])  # per-class fitting summary

# --- Circular KDE of time-of-day ---------------------------------------------
rng = np.random.default_rng(42)
theta = pymaslow.sec_to_rad((rng.normal(23 * 3600, 3600, 500)) % 86400)
kde = pymaslow.fit_circular_kde(theta)
print(kde.pdf(theta[:5]))

# --- Dirichlet model of the compositional need profile ----------------------
freq = rng.dirichlet([5, 3, 2, 4, 1], size=200)   # (N, 5) simplex rows
alphas = pymaslow.dirichlet.mle(freq)
print(alphas)

# --- Hidden Markov model over durations -------------------------------------
hmm = pymaslow.ExponentialHMM(n_states=5)
hierarchy = [[1, 1, 2, 3], [2, 2, 1]]
durations = [[1800., 900., 3600., 1200.], [600., 300., 2400.]]
hmm.fit_supervised(hierarchy, durations, verbose=False)
print(hmm.predict([1500., 700., 3000.]))

# --- Sample a full synthetic day by chaining the models ---------------------
from pymaslow import sampling

# hierarchy ~ p(h|t) (von Mises mixture posterior), activity ~ p(a|h) (HMM
# emission), duration ~ p(d|t) (positive circular KDE); time advances by the
# sampled duration. Uses the embedded pre-fitted models by default.
diary = sampling.sample_sequence(t0="02:00", max_days=1, random_state=42)
for rec in diary[:5]:
    print(rec["start_time"], rec["hierarchy_name"], rec["activity"])
```

## Methodological background

The package accompanies the manuscript *"Diurnal Rhythms of Human
Motivational Needs: A Computational Framework Based on Temporal Maslow
Hierarchy"*. Human behavior is represented as tuples *(t, A, M, L)* of time,
activity, (multi-label) Maslow hierarchy, and location. Because raw Human
Activity Recognition datasets do not annotate *why* an activity is performed,
need hierarchies are inferred with a reasoning LLM (see
`pymaslow.hierarchy.LLM_PROMPT_TEMPLATE`). The temporal structure of the
resulting hierarchy streams is then characterized with the Markov, von Mises,
circular-KDE, Dirichlet, and HMM models implemented here.

## Citation

If you use `pymaslow` in academic work, please cite the companion paper
(see `docs/pymaslow_manuscript/` in the source repository).

## License

MIT License. See `LICENSE`.
