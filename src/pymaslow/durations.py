"""
pymaslow.durations
==================

Joint modeling of activity *durations* and their time of day.

Every activity serving a need hierarchy lasts some duration; hence at any
instant of the day (say 09:00), each hierarchy is associated with a
distribution of durations. This module models the joint distribution
``p(d, t)`` of duration ``d > 0`` and time of day ``t in [0, 24)`` with a
**positive circular KDE** (:class:`PositiveCircularKDE`): a log-normal
(Gaussian-in-log-space) kernel for the positive duration axis and a von
Mises kernel for the circular time axis. The conditional
``p(d | t = t_query)`` can then be sampled, answering e.g. "how long do
physiological activities typically last at 08:00 vs 18:00?".

Embedded data
-------------
The module ships the duration table computed from raw CAPTURE-24 sequences
in the *Duration* section of ``notebooks/pymaslow.ipynb``
(``datas/t_mhn_activity_dAct_dMhn``), exposed at import as :data:`data`
with keys:

- ``moment`` -- activity start time, seconds since midnight;
- ``mhns`` -- multi-label Maslow hierarchy annotation, e.g. ``"1,3"``;
- ``activity`` -- Compendium of Physical Activities code;
- ``duration`` -- activity duration in seconds.

Adapted from ``codes/vonMises/utilities.py`` (``PositiveCircularKDE`` and
``plot_moments_duration_grid``) of the companion research repository.
"""

from __future__ import annotations

import os
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde, norm, vonmises

__all__ = [
    "PositiveCircularKDE",
    "fit",
    "plot",
    "sample_conditional",
    "load_data",
    "data",
]

#: Resource path of the embedded duration table.
_DATA_RESOURCE = "data/t_mhn_activity_dAct_dMhn.npz"


# =============================================================================
# Positive circular KDE
# =============================================================================


class PositiveCircularKDE:
    """Joint KDE for ``(d, t)`` with ``d > 0`` and ``t`` circular in ``[0, 24)``.

    Uses a log-transform for ``d`` (Gaussian kernel in log-space, with the
    Jacobian factor ``1 / d``) and a von Mises kernel for ``t``.

    Parameters
    ----------
    d_data : array_like
        Positive observations (e.g. durations in seconds).
    t_data : array_like
        Time-of-day observations in hours ``[0, 24)``.
    bw_d : float or None
        Bandwidth of the Gaussian kernel in log-duration space;
        defaults to Silverman's rule on ``log(d)``.
    bw_t : float or None
        Concentration ``kappa`` of the von Mises time kernel;
        defaults to a rule-of-thumb from the mean resultant length.

    Attributes
    ----------
    d, t : ndarray
        The (validated) observations.
    N : int
        Number of observations.
    log_d : ndarray
        ``log(d)``.
    bw_d, bw_t : float
        The bandwidths in use.
    """

    def __init__(
        self,
        d_data: npt.ArrayLike,
        t_data: npt.ArrayLike,
        bw_d: float | None = None,
        bw_t: float | None = None,
    ):
        self.d = np.asarray(d_data, dtype=float).ravel()
        self.t = np.asarray(t_data, dtype=float).ravel()
        if self.d.shape[0] != self.t.shape[0]:
            raise ValueError(
                f"d_data and t_data must have same length, got "
                f"{self.d.shape[0]} and {self.t.shape[0]}"
            )
        self.N = self.d.shape[0]
        if self.N == 0:
            raise ValueError("d_data and t_data must be non-empty")
        if np.any(self.d <= 0):
            raise ValueError("All d must be > 0")
        if np.any((self.t < 0) | (self.t >= 24)):
            raise ValueError("All t must be in [0, 24)")

        self.log_d = np.log(self.d)
        self.bw_d = bw_d if bw_d is not None else self._silverman_log(self.log_d)
        self.bw_t = bw_t if bw_t is not None else self._vm_concentration(self.t)

    # ---------- Bandwidth selection ----------

    def _silverman_log(self, x: np.ndarray) -> float:
        """Silverman's rule of thumb on ``log(d)``."""
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        return 0.9 * sigma * self.N ** (-0.2)

    def _vm_concentration(self, t: np.ndarray) -> float:
        """Rule-of-thumb von Mises concentration from the mean resultant length."""
        theta = 2 * np.pi * t / 24.0
        cos_mean, sin_mean = np.mean(np.cos(theta)), np.mean(np.sin(theta))
        r = np.hypot(cos_mean, sin_mean)
        if r < 0.85:
            kappa = r * (2 - r**2) / (1 - r**2)
        else:
            kappa = 1.0 / (2 * (1 - r))
        return max(kappa, 0.5)  # avoid degenerate near-uniform kernels

    # ---------- Kernel helpers ----------

    def _kt(self, t_eval: np.ndarray, t_center: np.ndarray) -> np.ndarray:
        """Von Mises time kernel, shape ``(n_eval, N)``."""
        diff = np.mod(t_eval[:, None] - t_center[None, :] + 12, 24) - 12
        theta_diff = 2 * np.pi * diff / 24.0
        return vonmises.pdf(theta_diff, self.bw_t)

    def _kd(self, d_eval: np.ndarray, d_center: np.ndarray) -> np.ndarray:
        """Log-normal duration kernel (with Jacobian), shape ``(n_eval, N)``."""
        log_eval = np.log(d_eval[:, None])
        log_cen = np.log(d_center[None, :])
        return norm.pdf(log_eval, loc=log_cen, scale=self.bw_d) / d_eval[:, None]

    # ---------- Density evaluation ----------

    def pdf(self, d: npt.ArrayLike, t: npt.ArrayLike) -> np.ndarray:
        """Joint density ``p(d, t)``.

        ``d`` and ``t`` may be scalars or 1D arrays of matching length.
        """
        d = np.atleast_1d(np.asarray(d, dtype=float))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if d.shape != t.shape:
            raise ValueError("d and t must have the same shape")
        kt = self._kt(t, self.t)
        kd = self._kd(d, self.d)
        return np.mean(kt * kd, axis=1)

    def marginal_t(self, t: npt.ArrayLike) -> np.ndarray:
        """Marginal ``p(t)`` (each duration kernel integrates to 1)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.mean(self._kt(t, self.t), axis=1)

    # ---------- Conditional sampling ----------

    def sample_conditional(
        self, t_query: npt.ArrayLike, n_samples: int = 1, random_state=None
    ) -> np.ndarray:
        """Sample durations from the conditional ``p(d | t = t_query)``.

        Observations are reweighted by their von Mises time-kernel weights
        at ``t_query``; a data point is then drawn from those weights and a
        duration is sampled from its log-normal kernel.

        Parameters
        ----------
        t_query : float or array_like
            Query time(s) in hours ``[0, 24)``.
        n_samples : int
            Number of durations to draw per query time.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        ndarray
            Shape ``(len(t_query), n_samples)``, or ``(n_samples,)`` for a
            scalar ``t_query``.
        """
        rng = np.random.default_rng(random_state)
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        out = np.empty((t_query.size, n_samples))

        for idx, tq in enumerate(t_query):
            diff = np.mod(self.t - tq + 12, 24) - 12
            theta_diff = 2 * np.pi * diff / 24.0
            w = np.exp(self.bw_t * np.cos(theta_diff))
            w /= w.sum()

            chosen = rng.choice(self.N, size=n_samples, p=w)
            log_samp = rng.normal(self.log_d[chosen], self.bw_d)
            out[idx] = np.exp(log_samp)

        return out if out.shape[0] > 1 else out[0]


# =============================================================================
# Embedded data
# =============================================================================


def load_data() -> dict[str, np.ndarray]:
    """Load the embedded CAPTURE-24 duration table.

    Returns
    -------
    dict
        Keys ``moment`` (seconds since midnight), ``mhns`` (multi-label
        hierarchy annotations), ``activity`` (compendium codes) and
        ``duration`` (seconds); each a 1D array of the same length.
    """
    with resources.as_file(
        resources.files("pymaslow").joinpath(_DATA_RESOURCE)
    ) as path:
        npz = np.load(path)
        return {k: npz[k] for k in npz.files}


#: Embedded CAPTURE-24 duration table (see :func:`load_data`).
data: dict[str, np.ndarray] = load_data()


# =============================================================================
# Fitting
# =============================================================================


def fit(
    d_data: npt.ArrayLike | None = None,
    t_data: npt.ArrayLike | None = None,
    bw_d: float | None = None,
    bw_t: float | None = None,
    log_duration: bool = True,
) -> PositiveCircularKDE:
    """Fit a :class:`PositiveCircularKDE` to duration/time-of-day data.

    When called without arguments, reproduces the notebook workflow on the
    embedded CAPTURE-24 table: durations are log-transformed, entries with
    non-positive log-duration (durations ``<= 1`` second) are dropped, and
    moments are converted from seconds to hours. Note that the returned
    model's positive variable is then the *log-duration*:
    :meth:`~PositiveCircularKDE.sample_conditional` returns samples on the
    log-duration scale, and ``np.exp(samples)`` converts them back to
    seconds.

    Parameters
    ----------
    d_data : array_like or None
        Duration observations (must be positive). If None, the embedded
        ``data['duration']`` is used.
    t_data : array_like or None
        Time-of-day observations in hours ``[0, 24)``. If None, the embedded
        ``data['moment'] / 3600`` is used.
    bw_d, bw_t : float or None
        Optional bandwidths (see :class:`PositiveCircularKDE`).
    log_duration : bool
        If True (default), use ``log(d)`` as the positive-axis variable and
        drop non-positive values, as in the notebook. Note the fitted model
        then models the *log-duration* scale internally in the same way as
        raw durations (it log-transforms internally as well).

    Returns
    -------
    PositiveCircularKDE
        The fitted model.
    """
    if d_data is None:
        d_arr = np.asarray(data["duration"], dtype=float)
        t_arr = np.asarray(data["moment"], dtype=float) / 3600.0
    else:
        d_arr = np.asarray(d_data, dtype=float).ravel()
        if t_data is None:
            raise ValueError("t_data must be provided together with d_data")
        t_arr = np.asarray(t_data, dtype=float).ravel()

    if log_duration:
        # Notebook workflow: model the log-duration values (kept > 0) as the
        # positive variable; sample_conditional then returns samples on the
        # log-duration scale -- np.exp() converts them back to seconds.
        log_d = np.log(d_arr)
        mask = log_d > 0
        d_fit = log_d[mask]
        t_fit = t_arr[mask]
    else:
        d_fit, t_fit = d_arr, t_arr

    return PositiveCircularKDE(d_fit, t_fit, bw_d=bw_d, bw_t=bw_t)


def sample_conditional(
    t_query: npt.ArrayLike,
    n_samples: int = 1,
    model: PositiveCircularKDE | None = None,
    random_state=None,
) -> np.ndarray:
    """Sample durations from ``p(d | t = t_query)``.

    Module-level convenience wrapper around
    :meth:`PositiveCircularKDE.sample_conditional`; when ``model`` is None a
    default model fitted on the embedded data (see :func:`fit`) is used.

    Parameters
    ----------
    t_query : float or array_like
        Query time(s) in hours ``[0, 24)``.
    n_samples : int
        Number of durations per query time.
    model : PositiveCircularKDE or None
        A fitted model; fitted on the embedded data if omitted.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    ndarray
        Shape ``(len(t_query), n_samples)``, or ``(n_samples,)`` for a
        scalar ``t_query``.
    """
    if model is None:
        model = fit()
    return model.sample_conditional(t_query, n_samples=n_samples, random_state=random_state)


# =============================================================================
# Plotting
# =============================================================================


def plot(
    moments: npt.ArrayLike | None = None,
    duration: npt.ArrayLike | None = None,
    log_duration: bool = False,
    figsize: tuple[float, float] = (10, 8),
    bins: int = 50,
    scatter_alpha: float = 0.3,
    heatmap_bins: int = 50,
    duration_label: str | None = None,
    moments_label: str = "Moments (hours)",
    title: str | None = None,
    save_path: str | None = None,
):
    """2x2 grid analyzing the relationship between moments and durations.

    Layout (port of ``plot_moments_duration_grid`` from the research code):

    - **top-left** -- marginal distribution of duration ``p(duration)``
      (histogram + KDE);
    - **top-right** -- smoothed heatmap of ``(moment, duration)`` overlaid
      with the scatter points;
    - **bottom-left** -- scatter plot with the Pearson correlation
      coefficient;
    - **bottom-right** -- marginal distribution of moments ``p(moment)``.

    Called without arguments, plots the embedded CAPTURE-24 table (moments
    converted to hours).

    Parameters
    ----------
    moments : array_like or None
        Time-of-day values in hours; defaults to the embedded moments.
    duration : array_like or None
        Duration values; defaults to the embedded durations (seconds).
    log_duration : bool
        If True, plot ``log(duration)`` instead of raw duration.
    figsize : tuple
        Figure size in inches.
    bins : int
        Histogram bins for the marginal distributions.
    scatter_alpha : float
        Scatter point transparency.
    heatmap_bins : int
        Bins per axis for the 2D heatmap.
    duration_label : str or None
        Duration axis label (auto-set when ``log_duration`` is used).
    moments_label : str
        Moments axis label.
    title : str or None
        Overall figure title.
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and the 2x2 axes array.
    """
    if moments is None:
        moments_arr = np.asarray(data["moment"], dtype=float) / 3600.0
    else:
        moments_arr = np.asarray(moments, dtype=float).flatten()
    if duration is None:
        duration_arr = np.asarray(data["duration"], dtype=float)
    else:
        duration_arr = np.asarray(duration, dtype=float).flatten()

    if len(moments_arr) != len(duration_arr):
        raise ValueError(
            f"moments and duration must have same length, got "
            f"{len(moments_arr)} and {len(duration_arr)}"
        )

    if log_duration:
        duration_arr = np.log(duration_arr)
        if duration_label is None:
            duration_label = "log(Duration)"
    if duration_label is None:
        duration_label = "Duration (seconds)"

    fig, axes = plt.subplots(
        2, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.2], "height_ratios": [1.2, 1]}
    )

    # (0, 0): marginal distribution of duration (horizontal, KDE overlay)
    ax_duration_dist = axes[0, 0]
    ax_duration_dist.hist(
        duration_arr,
        bins=bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        orientation="horizontal",
    )
    try:
        duration_kde = gaussian_kde(duration_arr)
        duration_range = np.linspace(duration_arr.min(), duration_arr.max(), 200)
        ax_duration_dist.plot(
            duration_kde(duration_range), duration_range, "r-", lw=2, label="KDE"
        )
        ax_duration_dist.legend()
    except (ValueError, np.linalg.LinAlgError):
        pass
    ax_duration_dist.set_xlabel("Density")
    ax_duration_dist.set_ylabel(duration_label)
    ax_duration_dist.set_title(f"Distribution of {duration_label}")
    ax_duration_dist.grid(True, alpha=0.3)

    # (0, 1): heatmap + scatter overlay
    ax_heat_scatter = axes[0, 1]
    heatmap, xedges, yedges = np.histogram2d(moments_arr, duration_arr, bins=heatmap_bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_heat_scatter.imshow(
        heatmap.T, origin="lower", extent=extent, aspect="auto", cmap="YlOrRd", alpha=0.7
    )
    ax_heat_scatter.scatter(
        moments_arr, duration_arr, alpha=scatter_alpha, s=10, c="blue", edgecolors="none"
    )
    fig.colorbar(im, ax=ax_heat_scatter, label="Count")
    ax_heat_scatter.set_xlabel(moments_label)
    ax_heat_scatter.set_ylabel(duration_label)
    ax_heat_scatter.set_title(f"{moments_label} vs {duration_label} (Heatmap + Scatter)")

    # (1, 0): scatter with correlation coefficient
    ax_scatter = axes[1, 0]
    ax_scatter.scatter(
        moments_arr, duration_arr, alpha=scatter_alpha, s=15, c="darkblue", edgecolors="none"
    )
    ax_scatter.set_xlabel(moments_label)
    ax_scatter.set_ylabel(duration_label)
    ax_scatter.set_title(f"Scatter: {moments_label} vs {duration_label}")
    ax_scatter.grid(True, alpha=0.3)
    corr = np.corrcoef(moments_arr, duration_arr)[0, 1]
    ax_scatter.text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=ax_scatter.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    # (1, 1): marginal distribution of moments
    ax_moments_dist = axes[1, 1]
    ax_moments_dist.hist(
        moments_arr, bins=bins, density=True, alpha=0.6, color="forestgreen", edgecolor="black"
    )
    try:
        moments_kde = gaussian_kde(moments_arr)
        moments_range = np.linspace(moments_arr.min(), moments_arr.max(), 200)
        ax_moments_dist.plot(moments_range, moments_kde(moments_range), "r-", lw=2, label="KDE")
        ax_moments_dist.legend()
    except (ValueError, np.linalg.LinAlgError):
        pass
    ax_moments_dist.set_xlabel(moments_label)
    ax_moments_dist.set_ylabel("Density")
    ax_moments_dist.set_title(f"Distribution of {moments_label}")
    ax_moments_dist.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

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
