"""
pymaslow.vonMisesMixture
========================

Mixtures of von Mises distributions modeling the temporal occurrence
distribution ``p(t | hierarchy)`` of each Maslow need hierarchy.

Time of day is circular, so the occurrence times of activities serving a
given need level are mapped to angles ``theta in [0, 2*pi)`` and modeled as a
finite mixture of von Mises distributions -- the circular analogue of the
Gaussian. The number of components is selected per hierarchy by BIC (or
AIC); parameters are estimated with an EM algorithm.

Embedded assets
---------------
Because the raw CAPTURE-24 sequences are very large, the package ships the
KDE-resampled occurrence times instead (see the *resample data* section of
``notebooks/pymaslow.ipynb`` in the companion repository). On import, this
module loads:

- ``data`` -- dict mapping each hierarchy level (``"1"``..``"5"``) to the
  resampled occurrence times, in hours ``[0, 24)``;
- ``p_x`` -- the fitted hierarchy prior ``p(hierarchy)``;
- ``models`` -- dict mapping each hierarchy level to its fitted
  :class:`VonMisesMixture` (``p(t | hierarchy)``);
- ``best_k`` -- the BIC-selected number of components per hierarchy.

Adapted from ``codes/vonMises/utilities.py`` of the companion research
repository. Sampling is vectorized per component (the research prototype
sampled one observation at a time), and the per-class seeds are derived from
a deterministic CRC32 hash instead of the salted built-in ``hash()``, making
fits reproducible across processes.
"""

from __future__ import annotations

import json
import os
import zlib
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from scipy.stats import vonmises

from .hierarchy import HIERARCHY_NAMES  # pyright: ignore[reportMissingImports]
from .timeutils import (  # pyright: ignore[reportMissingImports]
    hours_to_rad,
    rad_to_hours,
)

__all__ = [
    "VonMisesMixture",
    "fit_vmmm_dictionary",
    "plot_vmmm_results",
    "sample_vonmises_mixture",
    "sample_vmmm_dictionary",
    "sample_joint_vmmm",
    "load_resampled_data",
    "load_fitted_models",
    "data",
    "p_x",
    "models",
    "best_k",
]

#: Resource paths of the embedded assets.
_RESAMPLED_RESOURCE = "data/resampleddata4vonMisesMixture.npz"
_FITTED_RESOURCE = "data/vonmises_mixture_fitted.json"

#: Time bounds (hours of day) of the embedded data and fitted models.
Y_LW, Y_UP = 0.0, 24.0


# =============================================================================
# Model class
# =============================================================================


class VonMisesMixture:
    """Mixture of von Mises distributions on the circle, fitted by EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on the combined parameter change.
    random_state : int or None
        Seed for the k-means-style initialization.

    Attributes
    ----------
    weights, mu, kappa : ndarray, shape (K,)
        Mixture weights, mean directions (radians), and concentrations,
        available after :meth:`fit` or :meth:`from_parameters`.
    """

    def __init__(self, n_components, max_iter=200, tol=1e-6, random_state=None):
        if n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @classmethod
    def from_parameters(
        cls, weights: npt.ArrayLike, mu: npt.ArrayLike, kappa: npt.ArrayLike
    ) -> VonMisesMixture:
        """Construct a mixture directly from known parameters (no fitting).

        Parameters
        ----------
        weights, mu, kappa : array_like, shape (K,)
            Mixture weights (must sum to 1), mean directions in radians,
            and concentrations (positive).

        Returns
        -------
        VonMisesMixture
        """
        weights = np.asarray(weights, dtype=float)
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        if not (len(weights) == len(mu) == len(kappa)):
            raise ValueError("weights, mu and kappa must have equal length")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if np.any(kappa <= 0):
            raise ValueError("kappa must be positive")
        model = cls(n_components=len(weights))
        model.weights = weights
        model.mu = mu
        model.kappa = kappa
        return model

    @staticmethod
    def _solve_kappa(r):
        """Approximate concentration from the mean resultant length (Banerjee)."""
        r = np.clip(r, 1e-10, 0.9999999)
        if r < 0.53:
            kappa = 2 * r + r**3 + (5 * r**5) / 6
        elif r < 0.85:
            kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            kappa = 1.0 / (r**3 - 4 * r**2 + 3 * r)
        return np.clip(kappa, 1e-3, 700)

    def fit(self, data_rad: npt.ArrayLike) -> VonMisesMixture:
        """Fit the mixture to angular data with the EM algorithm.

        Parameters
        ----------
        data_rad : array_like
            Observed angles in radians ``[0, 2*pi)``.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(self.random_state)
        x = np.asarray(data_rad, dtype=float).flatten()
        n = len(x)
        k = self.n_components

        if k == 1:
            cos_mean, sin_mean = np.cos(x).mean(), np.sin(x).mean()
            self.mu = np.array([np.arctan2(sin_mean, cos_mean)])
            self.kappa = np.array([self._solve_kappa(np.hypot(sin_mean, cos_mean))])
            self.weights = np.array([1.0])
            return self

        # Circular k-means-style initialization from sorted data quantiles
        sorted_x = np.sort(x)
        idx = np.linspace(0, n - 1, k, endpoint=False).astype(int)
        self.mu = (sorted_x[idx] + rng.normal(0, 0.1, k)) % (2 * np.pi)
        self.kappa = np.ones(k) * 5.0
        self.weights = np.ones(k) / k

        for _iteration in range(self.max_iter):
            # E-step: posterior responsibilities
            log_resp = np.zeros((n, k))
            for j in range(k):
                log_resp[:, j] = np.log(self.weights[j] + 1e-15) + vonmises.logpdf(
                    x, self.kappa[j], loc=self.mu[j]
                )
            resp = np.exp(log_resp - logsumexp(log_resp, axis=1, keepdims=True))

            # M-step: weights, circular means, concentrations
            nk = resp.sum(axis=0) + 1e-15
            new_weights = nk / n
            new_mu = np.zeros(k)
            new_kappa = np.zeros(k)
            for j in range(k):
                sin_sum = np.sum(resp[:, j] * np.sin(x))
                cos_sum = np.sum(resp[:, j] * np.cos(x))
                cos_bar, sin_bar = cos_sum / nk[j], sin_sum / nk[j]
                new_mu[j] = np.arctan2(sin_bar, cos_bar)
                new_kappa[j] = self._solve_kappa(np.hypot(sin_bar, cos_bar))

            mu_diff = np.abs(np.sin((new_mu - self.mu) / 2.0))
            diff = (
                np.mean(np.abs(new_weights - self.weights))
                + np.mean(mu_diff) * 2
                + np.mean(np.abs(new_kappa - self.kappa)) / 10.0
            )

            self.weights, self.mu, self.kappa = new_weights, new_mu, new_kappa
            if diff < self.tol:
                break

        return self

    def pdf(self, x_rad: npt.ArrayLike) -> np.ndarray:
        """Mixture density at angles ``x_rad`` (radians)."""
        x = np.asarray(x_rad, dtype=float)
        result = np.zeros_like(x, dtype=float)
        for j in range(self.n_components):
            result += self.weights[j] * vonmises.pdf(x, self.kappa[j], loc=self.mu[j])
        return result

    def component_pdf(self, x_rad: npt.ArrayLike, idx: int) -> np.ndarray:
        """Density of a single mixture component (unweighted)."""
        x = np.asarray(x_rad, dtype=float)
        return vonmises.pdf(x, self.kappa[idx], loc=self.mu[idx])

    def weighted_component_pdf(self, x_rad: npt.ArrayLike, idx: int) -> np.ndarray:
        """Density of a single mixture component (weighted by its prior)."""
        return self.weights[idx] * self.component_pdf(x_rad, idx)

    def log_likelihood(self, x_rad: npt.ArrayLike) -> float:
        """Total log-likelihood of angular data under the mixture."""
        return np.sum(np.log(self.pdf(x_rad) + 1e-15))

    def sample(
        self,
        n_samples: int = 1,
        seed: int | np.random.Generator | None = None,
        return_radians: bool = True,
        y_lw: float = Y_LW,
        y_up: float = Y_UP,
    ) -> np.ndarray:
        """Draw samples from the mixture; see :func:`sample_vonmises_mixture`."""
        return sample_vonmises_mixture(
            self, n_samples, seed, return_radians, y_lw, y_up
        )

    def peak_times(self, y_lw: float = Y_LW, y_up: float = Y_UP) -> np.ndarray:
        """Peak (modal) time of each component in time units (hours by default)."""
        return rad_to_hours(np.mod(self.mu, 2.0 * np.pi), y_lw, y_up)

    def __repr__(self) -> str:
        if not hasattr(self, "weights"):
            return f"VonMisesMixture(n_components={self.n_components}, unfitted)"
        lines = [f"VonMisesMixture(n_components={self.n_components})"]
        for j in range(self.n_components):
            peak = rad_to_hours(np.mod(self.mu[j], 2.0 * np.pi))[()]
            lines.append(
                f"  Comp {j + 1}: weight={self.weights[j]:.3f}, "
                f"kappa={self.kappa[j]:.2f}, peak={peak:05.2f}h"
            )
        return "\n".join(lines)


# =============================================================================
# Dictionary-based fitting with adaptive component selection
# =============================================================================


def fit_vmmm_dictionary(
    data_dict,
    y_lw: float = Y_LW,
    y_up: float = Y_UP,
    k_max: int = 8,
    criterion: str = "bic",
    random_state: int = 42,
    large_but_finite_kappa: float = 500.0,
    eps: float = 1e-7,
    verbose: bool = True,
):
    """Fit one von Mises mixture per class with BIC/AIC component selection.

    For each class (e.g. each Maslow hierarchy level), mixtures with
    ``K = 1..min(k_max, n//3 + 1)`` components are fitted by EM and the best
    ``K`` is selected by the chosen information criterion.

    Parameters
    ----------
    data_dict : dict
        Mapping of class labels to 1D arrays of occurrence times in
        ``[y_lw, y_up]`` time units (hours by default).
    y_lw, y_up : float
        Time-domain bounds; the domain is wrapped onto the circle.
    k_max : int
        Maximum number of mixture components to try per class.
    criterion : {'bic', 'aic'}
        Model-selection criterion.
    random_state : int
        Base seed; per-class seeds are derived deterministically via a CRC32
        hash of the class label (reproducible across processes, unlike the
        salted built-in ``hash()``).
    large_but_finite_kappa : float
        Concentration used for single-observation classes (forced K=1).
    eps : float
        Inward clip margin keeping observations strictly inside the domain.
    verbose : bool
        Print the selected K and criterion value per class.

    Returns
    -------
    p_x : dict
        Class prior probabilities ``p(x)`` (empirical class frequencies).
    models : dict
        Mapping of class labels to fitted :class:`VonMisesMixture` objects,
        i.e. ``p(t | x)``.
    best_k : dict
        Selected number of components per class.
    """
    p_x: dict = {}
    models: dict = {}
    best_k: dict = {}
    total_n = sum(len(v) for v in data_dict.values())

    for c, times in data_dict.items():
        y = np.asarray(times, dtype=float)
        n_c = len(y)
        if n_c == 0:
            continue

        p_x[c] = n_c / total_n
        y_clip = np.clip(y, y_lw + eps, y_up - eps)
        x_rad = hours_to_rad(y_clip, y_lw, y_up)

        if n_c == 1:
            # A single observation: force a sharp K=1 component
            vmm = VonMisesMixture(n_components=1, random_state=random_state)
            vmm.mu = np.array([x_rad[0]])
            vmm.kappa = np.array([large_but_finite_kappa])
            vmm.weights = np.array([1.0])
            models[c] = vmm
            best_k[c] = 1
            if verbose:
                print(f"Class '{c}': n=1 -> forced K=1 (kappa={vmm.kappa[0]:.1f})")
            continue

        cls_seed = random_state + zlib.crc32(str(c).encode()) % 1000
        best_score = np.inf
        best_model = None
        best_k_c = 1
        k_candidates = range(1, min(k_max, n_c // 3 + 1) + 1)

        for k in k_candidates:
            try:
                vmm = VonMisesMixture(
                    n_components=k, max_iter=150, random_state=cls_seed + k
                )
                vmm.fit(x_rad)
                log_l = vmm.log_likelihood(x_rad)
                k_eff = 3 * k - 1
                score = (
                    -2 * log_l + k_eff * np.log(n_c)
                    if criterion == "bic"
                    else -2 * log_l + 2 * k_eff
                )
                if score < best_score:
                    best_score = score
                    best_model = vmm
                    best_k_c = k
            except (ValueError, FloatingPointError):
                continue

        models[c] = best_model
        best_k[c] = best_k_c
        if verbose:
            print(
                f"Class '{c}': n={n_c}, selected K={best_k_c} "
                f"({criterion.upper()}={best_score:.1f})"
            )

    return p_x, models, best_k


# =============================================================================
# Plotting
# =============================================================================


def _default_class_label(c) -> str:
    """Display label for a class: Maslow level name for digit labels, else str."""
    s = str(c)
    if s.isdigit():
        try:
            level = int(s)
        except ValueError:
            return s
        if level in HIERARCHY_NAMES:
            return f"H{level} {HIERARCHY_NAMES[level]}"
    return s


def plot_vmmm_results(
    data_dict,
    p_x,
    models,
    best_k,
    y_lw: float = Y_LW,
    y_up: float = Y_UP,
    figsize=(14, 12),
    save_path: str | None = None,
    class_labels: dict | None = None,
):
    """Plot a fitted joint von Mises mixture model in three panels.

    The figure visualizes the joint model
    ``p(hierarchy, t) = p(hierarchy) * p(t | hierarchy)``:

    1. **Raw data** -- per-class histograms and rug plots of the observed
       occurrence times, showing the empirical temporal pattern of each
       class (e.g. need hierarchy);
    2. **Fitted conditional densities** ``p(t | x_j)`` -- the fitted mixture
       of each class (solid) and its individual components (dashed),
       showing *when during the day* each class concentrates;
    3. **Joint distribution** ``p(x, y) = p(x) * p(y | x)`` -- each class's
       contribution weighted by its prior, plus the marginal ``p(y)``
       (dashed black), showing the composition of the day by class.

    Parameters
    ----------
    data_dict : dict
        Mapping of class labels to observed occurrence times.
    p_x : dict
        Class priors, as returned by :func:`fit_vmmm_dictionary`.
    models : dict
        Fitted per-class :class:`VonMisesMixture` models.
    best_k : dict
        Selected component counts per class.
    y_lw, y_up : float
        Time-domain bounds.
    figsize : tuple
        Figure size in inches.
    save_path : str or None
        If given, save the figure to this path.
    class_labels : dict or None
        Optional display labels per class, e.g.
        ``{"1": "H1 Physiological", ...}``; defaults to
        :data:`pymaslow.HIERARCHY_NAMES` for the standard levels.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and the three axes.
    """
    time_grid = np.linspace(y_lw, y_up, 800)
    rad_grid = hours_to_rad(time_grid, y_lw, y_up)

    classes = list(models.keys())
    n_classes = len(classes)
    if class_labels is None:
        class_labels = {c: _default_class_label(c) for c in classes}

    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(n_classes)]

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1.3, 1.3]}
    )

    # --- Panel 1: raw data (rug + histogram) ---
    ax0 = axes[0]
    rng = np.random.default_rng(0)
    for idx, c in enumerate(classes):
        y_c = np.asarray(data_dict[c], dtype=float)
        ax0.hist(
            y_c,
            bins=np.linspace(y_lw, y_up, 50),
            alpha=0.35,
            color=colors[idx],
            label=f"{class_labels.get(c, c)} (n={len(y_c)})",
        )
        jitter = rng.uniform(-0.15, 0.15, size=len(y_c))
        ax0.scatter(y_c, idx + 1 + jitter, c=[colors[idx]], s=12, alpha=0.5, zorder=3)

    ax0.set_xlim(y_lw, y_up)
    ax0.set_ylim(0.3, n_classes + 0.7)
    ax0.set_xlabel("Time of day (hours)", fontsize=12)
    ax0.set_ylabel("Class", fontsize=12)
    ax0.set_title(
        "Observed occurrence times per class (rug + histogram)",
        fontsize=14,
        fontweight="bold",
    )
    ax0.set_yticks(range(1, n_classes + 1))
    ax0.set_yticklabels([class_labels.get(c, c) for c in classes])
    ax0.legend(loc="upper right", fontsize=9, ncols=2)
    for bound in (y_lw, y_up):
        ax0.axvline(bound, color="gray", ls="--", alpha=0.3)

    # --- Panel 2: fitted conditional densities p(t | class) ---
    ax1 = axes[1]
    for idx, c in enumerate(classes):
        pdf_vals = models[c].pdf(rad_grid)
        for k in range(models[c].n_components):
            ax1.plot(
                time_grid,
                models[c].weighted_component_pdf(rad_grid, k),
                color=colors[idx],
                lw=1.0,
                alpha=0.4,
                ls="--",
            )
        ax1.plot(
            time_grid,
            pdf_vals,
            color=colors[idx],
            lw=2.5,
            label=f"{class_labels.get(c, c)} (K={best_k[c]})",
        )
        ax1.fill_between(time_grid, 0, pdf_vals, color=colors[idx], alpha=0.12)

    ax1.set_xlim(y_lw, y_up)
    ax1.set_xlabel("Time of day (hours)", fontsize=12)
    ax1.set_ylabel("p(t | class)", fontsize=12)
    ax1.set_title(
        "Fitted von Mises mixture densities p(t | class) — periodic over [0, 24h]",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9, ncols=2)
    for bound in (y_lw, y_up):
        ax1.axvline(bound, color="gray", ls="--", alpha=0.3)

    # --- Panel 3: joint p(class, t) and marginal p(t) ---
    ax2 = axes[2]
    joint_curves = []
    for idx, c in enumerate(classes):
        joint = p_x[c] * models[c].pdf(rad_grid)
        joint_curves.append(joint)
        ax2.plot(
            time_grid,
            joint,
            color=colors[idx],
            lw=2,
            label=f"p({class_labels.get(c, c)}, t)",
        )

    total_y = np.sum(joint_curves, axis=0)
    ax2.plot(time_grid, total_y, "k--", lw=2.5, label="Marginal p(t) = Σ p(class, t)")
    ax2.fill_between(time_grid, 0, total_y, color="black", alpha=0.08)

    ax2.set_xlim(y_lw, y_up)
    ax2.set_xlabel("Time of day (hours)", fontsize=12)
    ax2.set_ylabel("p(class, t)", fontsize=12)
    ax2.set_title(
        "Joint distribution p(class, t) = p(class) · p(t | class)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=9, ncols=2)
    for bound in (y_lw, y_up):
        ax2.axvline(bound, color="gray", ls="--", alpha=0.3)

    fig.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            try:
                os.makedirs(dirname, exist_ok=True)
            except OSError as exc:
                raise OSError(
                    f"Cannot create directory for save_path {save_path!r}"
                ) from exc
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, axes


# =============================================================================
# Sampling
# =============================================================================


def sample_vonmises_mixture(
    model,
    n_samples: int = 1,
    seed: int | np.random.Generator | None = None,
    return_radians: bool = True,
    y_lw: float = Y_LW,
    y_up: float = Y_UP,
) -> np.ndarray:
    """Sample from a fitted von Mises mixture model.

    Component assignments are drawn from the mixture weights, then each
    component's samples are drawn vectorized from
    :class:`scipy.stats.vonmises` and wrapped onto ``[0, 2*pi)``.

    Parameters
    ----------
    model : VonMisesMixture
        Fitted mixture with ``weights``, ``mu`` and ``kappa`` attributes.
    n_samples : int
        Number of samples to draw.
    seed : int or np.random.Generator, optional
        Random seed or generator for reproducibility.
    return_radians : bool
        If True, return angles in ``[0, 2*pi)``; if False, map back to time
        units ``[y_lw, y_up)``.
    y_lw, y_up : float
        Time-domain bounds (only used when ``return_radians=False``).

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    k = len(model.weights)

    components = rng.choice(k, size=n_samples, p=model.weights)
    samples = np.empty(n_samples, dtype=float)
    for j in range(k):
        mask = components == j
        n_j = mask.sum()
        if n_j == 0:
            continue
        theta = vonmises.rvs(
            model.kappa[j], loc=model.mu[j], size=n_j, random_state=rng
        )
        samples[mask] = np.mod(theta, 2.0 * np.pi)

    if not return_radians:
        samples = rad_to_hours(samples, y_lw, y_up)
    return samples


def sample_vmmm_dictionary(
    models: dict,
    n_samples: int | dict,
    seed: int | np.random.Generator | None = None,
    return_radians: bool = False,
    y_lw: float = Y_LW,
    y_up: float = Y_UP,
) -> dict:
    """Sample from a dictionary of fitted mixtures (batch over classes).

    Parameters
    ----------
    models : dict
        Mapping of class labels to fitted :class:`VonMisesMixture` objects.
    n_samples : int or dict
        If int, draw this many samples per class; if dict, per-class counts.
    seed : int or np.random.Generator, optional
        Base seed; each class gets a derived independent seed.
    return_radians : bool
        If True, return angles in ``[0, 2*pi)``; else time units.
    y_lw, y_up : float
        Time-domain bounds.

    Returns
    -------
    dict
        Mapping of class labels to sample arrays.
    """
    rng = np.random.default_rng(seed)
    base_seed = rng.integers(0, 2**31)

    samples_dict = {}
    for idx, (cls, model) in enumerate(models.items()):
        n = n_samples[cls] if isinstance(n_samples, dict) else n_samples
        cls_seed = base_seed + idx * 10007  # large prime jump
        samples_dict[cls] = sample_vonmises_mixture(
            model=model,
            n_samples=n,
            seed=cls_seed,
            return_radians=return_radians,
            y_lw=y_lw,
            y_up=y_up,
        )
    return samples_dict


def sample_joint_vmmm(
    p_x: dict,
    models: dict,
    n_samples: int = 1,
    seed: int | np.random.Generator | None = None,
    return_radians: bool = False,
    y_lw: float = Y_LW,
    y_up: float = Y_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample jointly from ``p(x, t) = p(x) * p(t | x)``.

    Classes are drawn from the prior ``p(x)``, then times are drawn from the
    corresponding conditional mixture -- vectorized per class.

    Parameters
    ----------
    p_x : dict
        Marginal class probabilities (must sum to 1).
    models : dict
        Conditional mixtures ``p(t | x=c)`` per class.
    n_samples : int
        Total number of joint samples.
    seed : int or np.random.Generator, optional
        Random seed.
    return_radians : bool
        If True, times are in ``[0, 2*pi)``; otherwise in ``[y_lw, y_up)``.
    y_lw, y_up : float
        Time-domain bounds.

    Returns
    -------
    classes : ndarray, shape (n_samples,)
        Sampled class labels.
    times : ndarray, shape (n_samples,)
        Sampled times conditioned on the drawn classes.
    """
    rng = np.random.default_rng(seed)
    classes_arr = np.array(list(p_x.keys()))
    probs = np.array([p_x[c] for c in classes_arr], dtype=float)
    probs = probs / probs.sum()

    sampled_classes = rng.choice(classes_arr, size=n_samples, p=probs)
    times = np.empty(n_samples, dtype=float)
    for cls in classes_arr:
        mask = sampled_classes == cls
        n_c = mask.sum()
        if n_c == 0:
            continue
        times[mask] = sample_vonmises_mixture(
            model=models[cls],
            n_samples=n_c,
            seed=rng.integers(0, 2**31),
            return_radians=return_radians,
            y_lw=y_lw,
            y_up=y_up,
        )
    return sampled_classes, times


# =============================================================================
# Embedded assets: resampled data and fitted parameters
# =============================================================================


def load_resampled_data() -> dict[str, np.ndarray]:
    """Load the embedded resampled CAPTURE-24 hierarchy occurrence times.

    The raw CAPTURE-24 activity sequences are far too large to ship; this is
    the KDE-resampled proxy produced in the *resample data* section of
    ``notebooks/pymaslow.ipynb``.

    Returns
    -------
    dict
        Mapping of hierarchy level (``"1"``..``"5"``) to occurrence times in
        hours ``[0, 24)`` (about 20,000 samples in total).
    """
    with resources.as_file(
        resources.files("pymaslow").joinpath(_RESAMPLED_RESOURCE)
    ) as path:
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}


def load_fitted_models() -> tuple[dict, dict[str, VonMisesMixture], dict]:
    """Load the embedded von Mises mixture parameters fitted on the resampled data.

    Returns
    -------
    p_x : dict
        Fitted hierarchy prior ``p(hierarchy)``.
    models : dict
        Mapping of hierarchy level to fitted :class:`VonMisesMixture`,
        i.e. ``p(t | hierarchy)``.
    best_k : dict
        BIC-selected number of components per hierarchy.
    """
    try:
        with (
            resources.as_file(
                resources.files("pymaslow").joinpath(_FITTED_RESOURCE)
            ) as path,
            open(path, encoding="utf-8") as fh,
        ):
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to load embedded fitted parameters ({_FITTED_RESOURCE}); "
            "the pymaslow installation appears corrupted."
        ) from exc

    fitted_models = {
        c: VonMisesMixture.from_parameters(
            params["weights"], params["mu"], params["kappa"]
        )
        for c, params in payload["models"].items()
    }
    return payload["p_x"], fitted_models, payload["best_k"]


#: Embedded resampled CAPTURE-24 occurrence times per hierarchy (hours).
data: dict[str, np.ndarray] = load_resampled_data()

#: Embedded fitted model: prior, conditional mixtures, and selected K.
p_x, models, best_k = load_fitted_models()
