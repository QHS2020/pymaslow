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

import warnings
from collections import defaultdict
from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy.stats import expon, lognorm
from tqdm import tqdm

__all__ = [
    "ExponentialHMM",
    "LognormalHMM",
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
