"""
pymaslow.hmm
============

Hidden Markov models with continuous duration emissions for decoding the
latent Maslow need hierarchy underlying observed activity data.

Hidden states are the need-hierarchy levels (categorical ids); observations
are activity *durations* (continuous, positive). Two emission families are
provided:

- :class:`ExponentialHMM` -- exponential emissions, ``f(x | lambda) =
  lambda * exp(-lambda * x)``; the MLE of the rate is ``1 / mean(durations)``.
- :class:`LognormalHMM` -- lognormal emissions with per-state log-mean
  ``mu`` and log-standard-deviation ``sigma``.

Both are trained in *supervised* fashion (the hierarchy sequence is known
from the LLM annotation), handling states that never appear in the training
data -- unobserved states retain their initialized parameters. Decoding uses
the Viterbi algorithm.

Adapted from ``codes/opencode/hmm/hmm_exponential.py`` and
``codes/opencode/hmm/hmm_lognormal.py`` of the companion research repository.
"""

from __future__ import annotations

import itertools
import pickle
import warnings
from collections import defaultdict
from collections.abc import Sequence
from importlib import resources
from typing import cast

import numpy as np
from scipy.stats import expon, lognorm
from tqdm import tqdm

__all__ = [
    "CategoricalMaslowHMM",
    "ExponentialHMM",
    "LognormalHMM",
    "N_ACTIVITIES",
    "N_HIDDEN_STATES",
    "estimate_hmm_parameters",
    "hmm_states_definition",
    "load_etri_activity_definitions",
    "load_hmm_data",
    "metrics_classification",
    "train_test_split",
]


class _BaseDurationHMM:
    """Shared machinery for HMMs with continuous duration emissions."""

    def __init__(
        self,
        n_states: int,
        state_ids: Sequence | None = None,
        random_state: int = 42,
    ):
        if n_states is None or n_states <= 0:
            raise ValueError("n_states must be a positive integer")

        self.n_states = n_states
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        if state_ids is not None:
            if len(state_ids) != n_states:
                raise ValueError(
                    f"state_ids length ({len(state_ids)}) must match n_states ({n_states})"
                )
            self.state_ids_ = list(state_ids)
        else:
            self.state_ids_ = list(range(n_states))

        self.state_to_idx_ = {s: i for i, s in enumerate(self.state_ids_)}
        self.idx_to_state_ = dict(enumerate(self.state_ids_))

        self._initialize_params()
        self._observed_states_: set = set()

    # ------------------------------------------------------------------
    # Emission-specific hooks (implemented by subclasses)
    # ------------------------------------------------------------------

    def _init_emission_params(self, state_idx: int) -> dict:
        """Random initial emission parameters for one state."""
        raise NotImplementedError

    def _log_emission_prob(self, state_idx: int, obs: float) -> float:
        """Log probability of a duration observation under one state."""
        raise NotImplementedError

    def _mle_emission_params(self, obs_arr: np.ndarray) -> dict:
        """MLE emission parameters from the durations observed in one state."""
        raise NotImplementedError

    def expected_duration(self, state_id) -> float:
        """Expected (mean) duration of an activity serving the given state."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Model structure
    # ------------------------------------------------------------------

    def _initialize_params(self) -> None:
        """Initialize start/transition/emission parameters for all states."""
        n = self.n_states
        self.start_probs_ = np.ones(n) / n
        self.trans_probs_ = self.rng.rand(n, n)
        self.trans_probs_ /= self.trans_probs_.sum(axis=1, keepdims=True)
        self.emission_params_ = {i: self._init_emission_params(i) for i in range(n)}

    def _log_likelihood(self, state_seq, obs_seq) -> float:
        """Log-likelihood of labeled sequences under the current parameters."""
        total_ll = 0.0
        for states, obs in zip(state_seq, obs_seq, strict=True):
            for s in states:
                if s not in self.state_to_idx_:
                    warnings.warn(
                        f"State {s} not in model's state_ids. Skipping sequence.",
                        stacklevel=2,
                    )
                    return -np.inf

            state_idx = self.state_to_idx_[states[0]]
            total_ll += np.log(self.start_probs_[state_idx] + 1e-300)
            total_ll += self._log_emission_prob(state_idx, obs[0])

            for t in range(1, len(states)):
                prev_idx = self.state_to_idx_[states[t - 1]]
                curr_idx = self.state_to_idx_[states[t]]
                total_ll += np.log(self.trans_probs_[prev_idx, curr_idx] + 1e-300)
                total_ll += self._log_emission_prob(curr_idx, obs[t])

        return total_ll

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_supervised(
        self,
        hierarchy,
        durations,
        n_iter: int = 1,
        use_smoothing: bool = True,
        smoothing_weight: float = 0.01,
        verbose: bool = True,
    ) -> float:
        """Fit parameters from labeled sequences (maximum likelihood).

        Handles states that never appear in the data: unobserved states
        retain their initialized parameters.

        Parameters
        ----------
        hierarchy : list of list of state ids
            Observed hidden-state (need-hierarchy) sequences.
        durations : list of list of float
            Observed duration sequences, aligned with ``hierarchy``.
        n_iter : int
            Number of estimation passes (1 suffices for supervised MLE).
        use_smoothing : bool
            Add ``smoothing_weight`` pseudo-counts (Dirichlet prior).
        smoothing_weight : float
            Pseudo-count weight.
        verbose : bool
            Show a progress bar with the log-likelihood.

        Returns
        -------
        float
            Final log-likelihood of the training data.
        """
        if not hierarchy or not durations:
            raise ValueError("hierarchy and durations must be non-empty")
        if len(hierarchy) != len(durations):
            raise ValueError("hierarchy and durations must have same length")

        observed_states = set()
        for seq in hierarchy:
            observed_states.update(seq)
        self._observed_states_ = observed_states

        n_unobserved = self.n_states - len(observed_states)
        if verbose and n_unobserved > 0:
            unobserved = set(self.state_ids_) - observed_states
            print(
                f"Note: {n_unobserved} state(s) not observed in training data: "
                f"{unobserved}"
            )
            print("These states will retain their initialized parameters.")

        trans_counts = np.zeros((self.n_states, self.n_states))
        start_counts = np.zeros(self.n_states)
        emission_data = defaultdict(list)

        for states, obs in zip(hierarchy, durations, strict=True):
            if states is None:
                continue
            start_state = states[0]
            if start_state in self.state_to_idx_:
                start_counts[self.state_to_idx_[start_state]] += 1

            for t in range(len(states)):
                state = states[t]
                if state not in self.state_to_idx_:
                    continue
                state_idx = self.state_to_idx_[state]
                emission_data[state_idx].append(obs[t])

                if t > 0:
                    prev_state = states[t - 1]
                    if prev_state in self.state_to_idx_:
                        prev_idx = self.state_to_idx_[prev_state]
                        trans_counts[prev_idx, state_idx] += 1

        progress = tqdm(range(n_iter), disable=not verbose)
        new_ll = -np.inf
        for _iteration in progress:
            # Start probabilities (with optional smoothing)
            total_starts = start_counts.sum()
            if total_starts > 0:
                if use_smoothing:
                    smoothed = start_counts + smoothing_weight
                    new_start_probs = smoothed / smoothed.sum()
                else:
                    new_start_probs = start_counts / total_starts
            else:
                new_start_probs = np.ones(self.n_states) / self.n_states

            # Transition probabilities (with optional smoothing)
            new_trans_probs = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                row_sum = trans_counts[i].sum()
                if row_sum > 0:
                    if use_smoothing:
                        smoothed_row = trans_counts[i] + smoothing_weight
                        new_trans_probs[i] = smoothed_row / smoothed_row.sum()
                    else:
                        new_trans_probs[i] = trans_counts[i] / row_sum
                else:
                    new_trans_probs[i] = np.ones(self.n_states) / self.n_states

            # Emission parameters: MLE for observed states, keep the rest
            new_emission_params = {
                i: self.emission_params_[i].copy() for i in range(self.n_states)
            }
            for state_idx, obs_list in emission_data.items():
                if len(obs_list) > 0:
                    obs_arr = np.array([o for o in obs_list if o > 0], dtype=float)
                    if len(obs_arr) > 0:
                        new_emission_params[state_idx] = self._mle_emission_params(
                            obs_arr
                        )

            self.start_probs_ = new_start_probs
            self.trans_probs_ = new_trans_probs
            self.emission_params_ = new_emission_params

            new_ll = self._log_likelihood(hierarchy, durations)
            if verbose:
                progress.set_description(f"LL: {new_ll:.2f}")

        return new_ll

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, durations) -> list:
        """Decode the most likely state sequence (Viterbi algorithm).

        Parameters
        ----------
        durations : sequence of float
            Observed durations.

        Returns
        -------
        list
            Predicted state ids, one per observation.
        """
        n_timesteps = len(durations)
        if n_timesteps == 0:
            return []

        viterbi = np.full((self.n_states, n_timesteps), -np.inf)
        backpointer = np.zeros((self.n_states, n_timesteps), dtype=int)

        for i in range(self.n_states):
            viterbi[i, 0] = np.log(
                self.start_probs_[i] + 1e-300
            ) + self._log_emission_prob(i, durations[0])

        for t in range(1, n_timesteps):
            for j in range(self.n_states):
                scores = viterbi[:, t - 1] + np.log(self.trans_probs_[:, j] + 1e-300)
                best_prev = np.argmax(scores)
                viterbi[j, t] = scores[best_prev] + self._log_emission_prob(
                    j, durations[t]
                )
                backpointer[j, t] = best_prev

        best_path = np.zeros(n_timesteps, dtype=int)
        best_path[-1] = np.argmax(viterbi[:, -1])
        for t in range(n_timesteps - 2, -1, -1):
            best_path[t] = backpointer[best_path[t + 1], t + 1]

        return [self.idx_to_state_[i] for i in best_path]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a copy of the model parameters as a dictionary."""
        return {
            "start_probs": self.start_probs_.copy(),
            "trans_probs": self.trans_probs_.copy(),
            "emission_params": self.emission_params_.copy(),
            "state_ids": self.state_ids_.copy(),
            "n_states": self.n_states,
        }

    def log_likelihood(self, hierarchy, durations) -> float:
        """Log-likelihood of labeled sequences under the current parameters."""
        return self._log_likelihood(hierarchy, durations)

    def get_observed_states(self) -> set:
        """States observed during the last call to :meth:`fit_supervised`."""
        return self._observed_states_.copy()

    def get_unobserved_states(self) -> set:
        """States never observed during the last call to :meth:`fit_supervised`."""
        return set(self.state_ids_) - self._observed_states_


class ExponentialHMM(_BaseDurationHMM):
    """Hidden Markov model with exponential duration emissions.

    Emission density ``f(x | lambda) = lambda * exp(-lambda * x)`` for
    ``x >= 0`` with rate ``lambda > 0`` (mean ``1 / lambda``). Stored as the
    scipy ``scale`` parameter ``beta = 1 / lambda`` (the mean duration).

    Parameters
    ----------
    n_states : int
        Number of hidden states (need-hierarchy levels).
    state_ids : sequence or None
        State ids; defaults to ``0..n_states - 1``.
    random_state : int
        Seed for parameter initialization.
    """

    def _init_emission_params(self, state_idx: int) -> dict:
        mean_duration = self.rng.uniform(0.5, 5.0)
        return {"rate": 1.0 / mean_duration, "scale": mean_duration}

    def _log_emission_prob(self, state_idx: int, obs: float) -> float:
        if state_idx not in self.emission_params_:
            return -np.inf
        scale = self.emission_params_[state_idx]["scale"]
        return cast(float, expon.logpdf(obs, loc=0, scale=scale))

    def _mle_emission_params(self, obs_arr: np.ndarray) -> dict:
        # MLE for the exponential distribution: beta_hat = sample mean
        mean_duration = max(np.mean(obs_arr), 0.1)
        return {"rate": 1.0 / mean_duration, "scale": mean_duration}

    def expected_duration(self, state_id) -> float:
        """Mean duration ``beta = 1 / lambda`` of the given state."""
        if state_id not in self.state_to_idx_:
            raise ValueError(f"Unknown state ID: {state_id}")
        state_idx = self.state_to_idx_[state_id]
        if state_idx not in self.emission_params_:
            return 0.0
        return self.emission_params_[state_idx]["scale"]


class LognormalHMM(_BaseDurationHMM):
    """Hidden Markov model with lognormal duration emissions.

    Emission density with per-state log-mean ``mu`` and log-standard
    deviation ``sigma``: ``log(x) ~ Normal(mu, sigma^2)``.

    Parameters
    ----------
    n_states : int
        Number of hidden states (need-hierarchy levels).
    state_ids : sequence or None
        State ids; defaults to ``0..n_states - 1``.
    random_state : int
        Seed for parameter initialization.
    """

    def _init_emission_params(self, state_idx: int) -> dict:
        return {"mu": self.rng.uniform(-1, 2), "sigma": self.rng.uniform(0.3, 1.5)}

    def _log_emission_prob(self, state_idx: int, obs: float) -> float:
        if state_idx not in self.emission_params_:
            return -np.inf
        mu = self.emission_params_[state_idx]["mu"]
        sigma = self.emission_params_[state_idx]["sigma"]
        return cast(float, lognorm.logpdf(obs, sigma, loc=0, scale=np.exp(mu)))

    def _mle_emission_params(self, obs_arr: np.ndarray) -> dict:
        # MLE for the lognormal distribution: mean/std of the log-durations
        log_obs = np.log(obs_arr)
        mu = np.mean(log_obs)
        sigma = max(np.std(log_obs), 0.1)
        return {"mu": mu, "sigma": sigma}

    def expected_duration(self, state_id) -> float:
        """Mean duration ``exp(mu + sigma^2 / 2)`` of the given state."""
        if state_id not in self.state_to_idx_:
            raise ValueError(f"Unknown state ID: {state_id}")
        state_idx = self.state_to_idx_[state_id]
        if state_idx not in self.emission_params_:
            return 0.0
        mu = self.emission_params_[state_idx]["mu"]
        sigma = self.emission_params_[state_idx]["sigma"]
        return np.exp(mu + sigma**2 / 2)


# =============================================================================
# Hidden state definitions (multi-label hierarchy states)
# =============================================================================

#: Number of hidden states: all non-empty subsets of the five need levels
#: (``2**5 - 1``).
N_HIDDEN_STATES = 31

#: Number of observable activities (Compendium of Physical Activities codes
#: 0..823 for CAPTURE-24).
N_ACTIVITIES = 824


def hmm_states_definition() -> tuple[dict[int, str], dict[str, int]]:
    """Mapping between hidden-state ids and multi-label hierarchy states.

    The hidden state of the Maslow HMM is the *set* of need levels active at
    a moment; with five levels there are ``2**5 - 1 = 31`` non-empty
    subsets, e.g. state id 5 is ``"1,2"`` (physiological + safety).

    Returns
    -------
    id_to_state : dict
        ``{0: '1', 1: '2', ..., 5: '1,2', ..., 30: '1,2,3,4,5'}``.
    state_to_id : dict
        The inverse mapping.
    """
    id_to_state: dict[int, str] = {}
    state_to_id: dict[str, int] = {}
    states: list[str] = []
    for n in range(1, 6):
        for tup in itertools.combinations(range(1, 6), n):
            states.append(",".join(map(str, tup)))
    for i, state in enumerate(states):
        id_to_state[i] = state
        state_to_id[state] = i
    return id_to_state, state_to_id


# =============================================================================
# Supervised parameter estimation (MLE with Laplace smoothing)
# =============================================================================


def estimate_hmm_parameters(
    hidden_state_sequences,
    observation_sequences,
    all_hidden_state_ids=None,
    all_observation_ids=None,
    alpha: float = 1e-10,
) -> tuple[dict, dict, dict]:
    """Estimate categorical-HMM parameters from labeled sequences (MLE).

    Parameters
    ----------
    hidden_state_sequences : list of list of int
        Observed hidden-state sequences.
    observation_sequences : list of list of int
        Observation (activity id) sequences, aligned with the states.
    all_hidden_state_ids : iterable of int or None
        Complete hidden-state vocabulary; defaults to ``range(31)``.
    all_observation_ids : iterable of int or None
        Complete observation vocabulary; defaults to ``range(824)``.
    alpha : float
        Laplace pseudo-count for unseen events.

    Returns
    -------
    initial_prob : dict
        ``{state_id: probability}``.
    transition_prob : dict
        ``{from_state: {to_state: probability}}``.
    emission_prob : dict
        ``{state_id: {obs_id: probability}}``.
    """
    if len(hidden_state_sequences) != len(observation_sequences):
        raise ValueError(
            "Number of hidden state sequences must equal number of observation sequences"
        )
    for i, (hs, obs) in enumerate(
        zip(hidden_state_sequences, observation_sequences, strict=True)
    ):
        if len(hs) != len(obs):
            raise ValueError(
                f"Sequence {i}: hidden state length {len(hs)} != observation length {len(obs)}"
            )

    if all_hidden_state_ids is None:
        all_hidden_state_ids = range(N_HIDDEN_STATES)
    if all_observation_ids is None:
        all_observation_ids = range(N_ACTIVITIES)
    states = sorted(all_hidden_state_ids)
    observations = sorted(all_observation_ids)

    initial_counts = defaultdict(lambda: alpha)
    transition_counts = defaultdict(lambda: defaultdict(lambda: alpha))
    emission_counts = defaultdict(lambda: defaultdict(lambda: alpha))

    # Seed all keys so unseen states/observations still appear in the output
    for s in states:
        initial_counts[s]
        for s2 in states:
            transition_counts[s][s2]
        for o in observations:
            emission_counts[s][o]

    for hs_seq, obs_seq in zip(
        hidden_state_sequences, observation_sequences, strict=True
    ):
        if len(hs_seq) == 0:
            continue
        initial_counts[hs_seq[0]] += 1
        for t, (state, obs) in enumerate(zip(hs_seq, obs_seq, strict=True)):
            emission_counts[state][obs] += 1
            if t < len(hs_seq) - 1:
                transition_counts[state][hs_seq[t + 1]] += 1

    total_initial = sum(initial_counts[s] for s in states)
    initial_prob = {s: initial_counts[s] / total_initial for s in states}

    transition_prob = {}
    for s in states:
        row_total = sum(transition_counts[s][s2] for s2 in states)
        transition_prob[s] = {
            s2: transition_counts[s][s2] / row_total for s2 in states
        }

    emission_prob = {}
    for s in states:
        row_total = sum(emission_counts[s][o] for o in observations)
        emission_prob[s] = {o: emission_counts[s][o] / row_total for o in observations}

    return initial_prob, transition_prob, emission_prob


# =============================================================================
# Train/test splitting with state conversion
# =============================================================================


def _state_to_id(state, state_to_id: dict) -> int:
    """Convert one raw state (multi-hot vector or label string) to its id."""
    if isinstance(state, str):
        key = ",".join(sorted(state.split(",")))
    else:
        arr = np.asarray(state).ravel()
        key = ",".join(str(i + 1) for i in range(len(arr)) if arr[i] != 0)
    return state_to_id[key]


def train_test_split(
    samples_dict,
    test_size_not_percent: int = 5,
    validation_size_not_percent: int | bool = False,
    random_state: int | None = None,
):
    """Split HMM samples into train/validation/test and convert states to ids.

    ``samples_dict`` maps sample ids to ``(state_sequences,
    activity_sequences, durations)`` triples. Raw states may be multi-hot
    vectors (CAPTURE-24, e.g. ``[1., 0., 0., 0., 0.]``) or (possibly
    unsorted) label strings (ETRI, e.g. ``"4,5,1"``); both are converted to
    hidden-state ids via :func:`hmm_states_definition`.

    Parameters
    ----------
    samples_dict : dict
        ``{sample_id: (state_sequences, activity_sequences, durations)}``.
    test_size_not_percent : int
        Number of *samples* (not a percentage) held out for testing.
    validation_size_not_percent : int or False
        Number of samples held out for validation (False = none).
    random_state : int or None
        Seed for the split. Note: the research prototype sampled with
        replacement and mixed positional/sample indices; this port samples
        without replacement and indexes by key, so splits are exact and
        reproducible.

    Returns
    -------
    train, validation, test
        Each ``(state_sequences, activity_sequences, durations, lengths)``
        with states as ids; ``validation`` is None when
        ``validation_size_not_percent`` is False.
    """
    _, state_to_id = hmm_states_definition()

    sample_keys = sorted(samples_dict.keys())
    n_samples = len(sample_keys)
    if test_size_not_percent >= n_samples:
        raise ValueError(
            "the number of the test size should be smaller than the samples size"
        )

    rng = np.random.default_rng(random_state)
    test_keys = set(rng.choice(sample_keys, size=test_size_not_percent, replace=False).tolist())
    remaining = [k for k in sample_keys if k not in test_keys]

    validation_keys: set = set()
    if not isinstance(validation_size_not_percent, bool):
        if validation_size_not_percent > len(remaining) / 2:
            raise ValueError("too much validation test data")
        validation_keys = set(
            rng.choice(remaining, size=validation_size_not_percent, replace=False).tolist()
        )
        remaining = [k for k in remaining if k not in validation_keys]

    def _collect(keys):
        state_seqs, activity_seqs, duration_seqs, lengths = [], [], [], []
        for k in keys:
            states_raw, activity_sequences, durations = samples_dict[k]
            state_ids = [_state_to_id(s, state_to_id) for s in states_raw]
            state_seqs.append(state_ids)
            activity_seqs.append(list(activity_sequences))
            duration_seqs.append(list(durations))
            lengths.append(len(durations))
        return state_seqs, activity_seqs, duration_seqs, lengths

    train = _collect(remaining)
    test = _collect(sorted(test_keys))
    validation = _collect(sorted(validation_keys)) if validation_keys else None
    return train, validation, test


# =============================================================================
# Categorical HMM (hmmlearn wrapper)
# =============================================================================


class CategoricalMaslowHMM:
    """Categorical-emission HMM over Maslow hidden states (hmmlearn wrapper).

    Hidden states are the 31 multi-label hierarchy states (see
    :func:`hmm_states_definition`); observations are activity ids
    (Compendium codes 0..823 for CAPTURE-24). Parameters are estimated by
    supervised MLE with Laplace smoothing via
    :func:`estimate_hmm_parameters` and injected into a
    :class:`hmmlearn.hmm.CategoricalHMM`, which performs Viterbi decoding.

    Parameters
    ----------
    n_components : int
        Number of hidden states (default 31).
    n_observations : int
        Observation vocabulary size (default 824).
    """

    def __init__(self, n_components: int = N_HIDDEN_STATES, n_observations: int = N_ACTIVITIES):
        if n_components <= 0 or n_observations <= 0:
            raise ValueError("n_components and n_observations must be positive")
        self.n_components = n_components
        self.n_observations = n_observations
        self._model = None

    def fit_supervised(self, state_sequences, observation_sequences) -> CategoricalMaslowHMM:
        """Estimate parameters from labeled sequences and build the model.

        Parameters
        ----------
        state_sequences : list of list of int
            Hidden-state id sequences (see :func:`train_test_split`).
        observation_sequences : list of list of int
            Activity id sequences, aligned with the states.

        Returns
        -------
        self
        """
        try:
            from hmmlearn.hmm import (  # pyright: ignore[reportMissingImports]
                CategoricalHMM,
            )
        except ImportError as exc:
            raise ImportError(
                "CategoricalMaslowHMM requires hmmlearn; "
                "install it with `pip install hmmlearn`."
            ) from exc

        initial_prob, transition_prob, emission_prob = estimate_hmm_parameters(
            state_sequences,
            observation_sequences,
            range(self.n_components),
            range(self.n_observations),
        )

        model = CategoricalHMM(n_components=self.n_components)
        model.startprob_ = np.array(
            [initial_prob[s] for s in range(self.n_components)]
        )
        model.transmat_ = np.array(
            [
                [transition_prob[s1][s2] for s2 in range(self.n_components)]
                for s1 in range(self.n_components)
            ]
        )
        model.emissionprob_ = np.array(
            [
                [emission_prob[s][o] for o in range(self.n_observations)]
                for s in range(self.n_components)
            ]
        )
        model.n_features = self.n_observations

        self._model = model
        self.initial_prob_ = initial_prob
        self.transition_prob_ = transition_prob
        self.emission_prob_ = emission_prob
        return self

    def predict(self, observations, lengths=None) -> np.ndarray:
        """Decode hidden states from observation sequences (Viterbi).

        Parameters
        ----------
        observations : 1D array-like or list of list of int
            Activity ids. Either a flat sequence (then pass ``lengths`` for
            sequence boundaries) or a list of sequences.
        lengths : list of int or None
            Sequence lengths when ``observations`` is flat.

        Returns
        -------
        ndarray of int
            Predicted hidden-state ids, one per observation.
        """
        if self._model is None:
            raise RuntimeError("Model is not fitted; call fit_supervised() first.")
        if lengths is None and len(observations) > 0 and isinstance(
            observations[0], (list, tuple, np.ndarray)
        ):
            lengths = [len(seq) for seq in observations]
            flat = [o for seq in observations for o in seq]
        else:
            flat = list(observations)
        x = np.asarray(flat, dtype=int).reshape(-1, 1)
        return self._model.predict(x, lengths)

    @property
    def model_(self):
        """The underlying :class:`hmmlearn.hmm.CategoricalHMM` (after fitting)."""
        return self._model


# =============================================================================
# Evaluation
# =============================================================================


def metrics_classification(y_true, y_pred, average: str = "weighted") -> dict:
    """Classification metrics for hidden-state decoding.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        Ground-truth and predicted state ids.
    average : {'weighted', 'macro'}
        Averaging mode for precision/recall/f1.

    Returns
    -------
    dict
        ``{'accuracy', 'precision', 'recall', 'f1'}``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    accuracy = np.mean(y_true == y_pred).item()
    labels = np.union1d(y_true, y_pred)

    precisions, recalls, f1s, supports = [], [], [], []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)
        precisions.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
        p, r = precisions[-1], recalls[-1]
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        supports.append(support)

    supports_arr = np.asarray(supports, dtype=float)
    if average == "macro":
        weights = np.ones_like(supports_arr) / len(supports_arr)
    else:  # weighted
        total = supports_arr.sum()
        weights = supports_arr / total if total > 0 else np.ones_like(supports_arr)

    return {
        "accuracy": accuracy,
        "precision": np.dot(weights, precisions).item(),
        "recall": np.dot(weights, recalls).item(),
        "f1": np.dot(weights, f1s).item(),
    }


# =============================================================================
# Embedded HMM training data (lazy)
# =============================================================================

_HMM_DATA_RESOURCES = {
    "capture24": "data/hmm_CAPTURE24.pickle",
    "etri": "data/hmm_ETRI.pickle",
}
_ETRI_TUPLES_RESOURCE = "data/etri_action_condition_place_tuples.pickle"


def _load_pickle_resource(resource_path: str):
    try:
        with resources.as_file(
            resources.files("pymaslow").joinpath(resource_path)
        ) as path, open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Failed to load embedded data ({resource_path}); "
            "the pymaslow installation appears corrupted."
        ) from exc


def load_hmm_data(dataset: str) -> dict:
    """Load an embedded HMM training set.

    Parameters
    ----------
    dataset : {'capture24', 'etri'}
        Which embedded dataset to load.

    Returns
    -------
    dict
        ``{sample_id: (state_sequences, activity_sequences, durations)}``.
        CAPTURE-24 (146 samples): states are 5-dim multi-hot vectors,
        activities are compendium ids. ETRI (22 samples): states are label
        strings like ``"1,2"``, activities are
        ``"action,condition,place"`` strings. Convert states to ids with
        :func:`train_test_split`.
    """
    key = dataset.lower()
    if key not in _HMM_DATA_RESOURCES:
        raise ValueError(
            f"Unknown dataset {dataset!r}; expected one of {tuple(_HMM_DATA_RESOURCES)}"
        )
    return _load_pickle_resource(_HMM_DATA_RESOURCES[key])


def load_etri_activity_definitions() -> set:
    """Load the ETRI activity vocabulary: a set of 184 ``(action, condition,
    place)`` tuples, e.g. ``('work', 'WITH_ONE', 'other_indoor')``."""
    return _load_pickle_resource(_ETRI_TUPLES_RESOURCE)


_LAZY_DATA = {
    "data_capture24": lambda: load_hmm_data("capture24"),
    "data_etri": lambda: load_hmm_data("etri"),
    "etri_activity_definitions": load_etri_activity_definitions,
}


def __getattr__(name: str):
    """Lazy access to embedded HMM datasets (loaded on first use)."""
    if name in _LAZY_DATA:
        value = _LAZY_DATA[name]()
        globals()[name] = value  # cache after first load
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
