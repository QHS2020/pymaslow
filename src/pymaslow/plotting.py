"""
pymaslow.plotting
=================

Publication-ready visualizations for circular time-of-day models: full
24-hour clock (polar) plots, AM/PM semicircular radar plots, linear
time-density plots, and BIC/AIC model-selection curves, plus the
:class:`CircularTimeAnalyzer` batch driver for dictionary-structured data
(e.g. one time series per Maslow hierarchy level).

Adapted from ``codes/vonMises/circular_time_analysis.py`` of the companion
research repository.
"""

from __future__ import annotations

import warnings
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from scipy.integrate import trapezoid

from .circularkde import (  # pyright: ignore[reportMissingImports]
    CircularKDE,
    fit_circular_kde,
)
from .timeutils import (  # pyright: ignore[reportMissingImports]
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    format_time,
    rad_to_sec,
    sec_to_rad,
)
from .vonMisesMixture import VonMisesMixture  # pyright: ignore[reportMissingImports]

__all__ = [
    "CircularTimeAnalyzer",
    "plot_am_pm_radar",
    "plot_bic_curve",
    "plot_circular_summary",
    "plot_linear_time",
]


def _get_component_colors(n: int) -> np.ndarray:
    """Distinct colors for mixture components."""
    return colormaps["tab10"](np.linspace(0, 1, max(n, 10)))[:n]


def _fit_vmmm_bic(
    theta_data: np.ndarray,
    max_components: int = 6,
    n_restarts: int = 5,
    random_state: int | None = None,
) -> tuple[VonMisesMixture, list[dict]]:
    """Fit a :class:`VonMisesMixture` by EM with BIC component selection.

    For each ``K = 1..max_components``, EM is run from ``n_restarts``
    differently-seeded initializations and the best log-likelihood is kept;
    the model with the lowest BIC across all ``K`` is returned together with
    per-K result records (``n_comp``, ``model``, ``logL``, ``aic``, ``bic``).
    """
    x = np.asarray(theta_data, dtype=float)
    n = len(x)
    base_seed = 42 if random_state is None else random_state

    all_results: list[dict] = []
    for k in range(1, max_components + 1):
        best_model = None
        best_log_l = -np.inf
        for restart in range(n_restarts):
            model = VonMisesMixture(
                n_components=k, random_state=base_seed + 1000 * restart + k
            )
            model.fit(x)
            log_l = model.log_likelihood(x)
            if log_l > best_log_l:
                best_log_l = log_l
                best_model = model

        k_free = 3 * k - 1
        aic = -2.0 * best_log_l + 2.0 * k_free
        bic = -2.0 * best_log_l + k_free * np.log(n)
        all_results.append(
            {
                "n_comp": k,
                "model": best_model,
                "logL": best_log_l,
                "aic": aic,
                "bic": bic,
            }
        )

    best_idx = np.argmin([r["bic"] for r in all_results]).item()
    return all_results[best_idx]["model"], all_results


def plot_circular_summary(
    theta_data: np.ndarray,
    model: VonMisesMixture | None = None,
    kde: CircularKDE | None = None,
    title: str = "",
    figsize: tuple[float, float] = (10, 10),
    show_components: bool = True,
) -> Figure:
    """24-hour clock (polar) plot of data, mixture fit, and KDE.

    Parameters
    ----------
    theta_data : array_like
        Observed angles in radians.
    model : VonMisesMixture or None
        Fitted mixture to overlay.
    kde : CircularKDE or None
        Fitted circular KDE to overlay.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    show_components : bool
        Whether to draw the individual mixture components.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    ax = cast(PolarAxes, fig.add_subplot(111, projection="polar"))

    n_bins = 48
    bins = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    counts, _ = np.histogram(theta_data, bins=bins)
    width = np.diff(bins)[0]
    density = counts / (len(theta_data) * width + 1e-12)

    ax.bar(
        bins[:-1] + width / 2.0,
        density,
        width=width,
        alpha=0.35,
        color="lightgray",
        edgecolor="gray",
        label="Observed data",
    )

    theta_grid = np.linspace(0, 2.0 * np.pi, 1000)

    if model is not None:
        ax.plot(
            theta_grid,
            model.pdf(theta_grid),
            "r-",
            lw=2.5,
            label="Mixture fit",
            zorder=5,
        )
        if show_components:
            colors = _get_component_colors(model.n_components)
            for i in range(model.n_components):
                ax.plot(
                    theta_grid,
                    model.weighted_component_pdf(theta_grid, i),
                    "--",
                    color=colors[i],
                    alpha=0.7,
                    lw=1.5,
                    label=f"Comp {i + 1} ({format_time(rad_to_sec(model.mu[i]))})",
                )

    if kde is not None:
        pdf_kde = kde.pdf(theta_grid)
        pdf_kde_norm = pdf_kde / (trapezoid(pdf_kde, theta_grid) + 1e-12)
        ax.plot(
            theta_grid,
            pdf_kde_norm,
            "g-",
            lw=2,
            alpha=0.7,
            label="Circular KDE",
            zorder=4,
        )

    ax.set_theta_zero_location("N")  # 00:00 at the top
    ax.set_theta_direction(-1)  # clockwise
    ax.set_xticks(np.linspace(0, 2.0 * np.pi, 24, endpoint=False))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=8)
    ax.set_title(title, pad=25, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0.08, 1.08), fontsize=8)

    fig.tight_layout()
    return fig


def plot_am_pm_radar(
    theta_data: np.ndarray,
    model: VonMisesMixture | None = None,
    kde: CircularKDE | None = None,
    title: str = "",
    figsize: tuple[float, float] = (14, 7),
) -> Figure:
    """AM (00:00-12:00) and PM (12:00-24:00) semicircular radar plots.

    Fitted densities are shown conditional on each half-day, enabling direct
    shape comparison between the AM and PM periods.

    Parameters
    ----------
    theta_data : array_like
        Observed angles in radians.
    model : VonMisesMixture or None
        Fitted mixture to overlay.
    kde : CircularKDE or None
        Fitted circular KDE to overlay.
    title : str
        Overall figure title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_am_raw, ax_pm_raw) = plt.subplots(
        1, 2, figsize=figsize, subplot_kw={"projection": "polar"}
    )
    ax_am = cast(PolarAxes, ax_am_raw)
    ax_pm = cast(PolarAxes, ax_pm_raw)

    for ax in (ax_am, ax_pm):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.grid(True, alpha=0.3)

    t_data = np.asarray(rad_to_sec(theta_data), dtype=float)

    # AM period: 00:00-12:00 mapped to [0, pi]
    am_mask = t_data < 12 * SECONDS_PER_HOUR
    am_theta = np.pi * t_data[am_mask] / (12.0 * SECONDS_PER_HOUR)
    if len(am_theta) > 0:
        am_bins = np.linspace(0, np.pi, 13)
        am_counts, _ = np.histogram(am_theta, bins=am_bins)
        am_width = np.diff(am_bins)[0]
        am_density = am_counts / (len(am_theta) * am_width + 1e-12)
        ax_am.bar(
            am_bins[:-1] + am_width / 2.0,
            am_density,
            width=am_width,
            alpha=0.6,
            color="gold",
            edgecolor="darkorange",
            linewidth=1.2,
            label="Observed",
        )

    # PM period: 12:00-24:00 mapped to [0, pi] for display
    pm_mask = t_data >= 12 * SECONDS_PER_HOUR
    pm_theta = (
        np.pi * (t_data[pm_mask] - 12.0 * SECONDS_PER_HOUR) / (12.0 * SECONDS_PER_HOUR)
    )
    if len(pm_theta) > 0:
        pm_bins = np.linspace(0, np.pi, 13)
        pm_counts, _ = np.histogram(pm_theta, bins=pm_bins)
        pm_width = np.diff(pm_bins)[0]
        pm_density = pm_counts / (len(pm_theta) * pm_width + 1e-12)
        ax_pm.bar(
            pm_bins[:-1] + pm_width / 2.0,
            pm_density,
            width=pm_width,
            alpha=0.6,
            color="steelblue",
            edgecolor="navy",
            linewidth=1.2,
            label="Observed",
        )

    # Overlay fitted densities, conditional on AM/PM
    theta_am_fine = np.linspace(0, np.pi, 500)
    theta_pm_orig = np.linspace(np.pi, 2.0 * np.pi, 500)
    theta_pm_plot = theta_pm_orig - np.pi

    if model is not None:
        pdf_am = model.pdf(theta_am_fine)
        prob_am = trapezoid(pdf_am, theta_am_fine)
        if prob_am > 1e-12:
            cond_am = pdf_am / prob_am
            ax_am.plot(theta_am_fine, cond_am, "r-", lw=2.5, label="Mixture", zorder=5)
            ax_am.fill_between(theta_am_fine, 0, cond_am, alpha=0.12, color="red")

        pdf_pm = model.pdf(theta_pm_orig)
        prob_pm = trapezoid(pdf_pm, theta_pm_orig)
        if prob_pm > 1e-12:
            cond_pm = pdf_pm / prob_pm
            ax_pm.plot(theta_pm_plot, cond_pm, "r-", lw=2.5, label="Mixture", zorder=5)
            ax_pm.fill_between(theta_pm_plot, 0, cond_pm, alpha=0.12, color="red")

    if kde is not None:
        pdf_kde_am = kde.pdf(theta_am_fine)
        norm_am = trapezoid(pdf_kde_am, theta_am_fine)
        if norm_am > 1e-12:
            ax_am.plot(
                theta_am_fine, pdf_kde_am / norm_am, "g-", lw=2, alpha=0.7, label="KDE"
            )

        pdf_kde_pm = kde.pdf(theta_pm_orig)
        norm_pm = trapezoid(pdf_kde_pm, theta_pm_orig)
        if norm_pm > 1e-12:
            ax_pm.plot(
                theta_pm_plot, pdf_kde_pm / norm_pm, "g-", lw=2, alpha=0.7, label="KDE"
            )

    ax_am.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax_am.set_xticklabels(["00:00", "03:00", "06:00", "09:00", "12:00"])
    ax_am.set_title(
        "AM Period\n(00:00 - 12:00)", pad=20, fontsize=13, fontweight="bold"
    )

    ax_pm.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax_pm.set_xticklabels(["12:00", "15:00", "18:00", "21:00", "24:00"])
    ax_pm.set_title(
        "PM Period\n(12:00 - 24:00)", pad=20, fontsize=13, fontweight="bold"
    )

    # Equal radial limits for direct comparison
    max_r = max(ax.get_ylim()[1] for ax in (ax_am, ax_pm))
    for ax in (ax_am, ax_pm):
        ax.set_ylim(0, max_r * 1.15)
        ax.set_yticklabels([])
        ax.legend(loc="lower left", bbox_to_anchor=(0.02, -0.08), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    return fig


def plot_linear_time(
    t_data: np.ndarray,
    model: VonMisesMixture | None = None,
    kde: CircularKDE | None = None,
    title: str = "",
    figsize: tuple[float, float] = (12, 5),
    n_bins: int = 48,
) -> Figure:
    """Density on a linear 24-hour time axis.

    Parameters
    ----------
    t_data : array_like
        Times in seconds ``[0, 86400]``.
    model : VonMisesMixture or None
        Fitted mixture to overlay.
    kde : CircularKDE or None
        Fitted circular KDE to overlay.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        t_data,
        bins=n_bins,
        density=True,
        alpha=0.35,
        color="lightgray",
        edgecolor="gray",
        label="Observed data",
    )

    t_grid = np.linspace(0, SECONDS_PER_DAY, 1000)
    theta_grid = np.asarray(sec_to_rad(t_grid), dtype=float)
    jacobian = 2.0 * np.pi / SECONDS_PER_DAY

    if model is not None:
        ax.plot(
            t_grid,
            model.pdf(theta_grid) * jacobian,
            "r-",
            lw=2.5,
            label="Mixture fit",
        )

    if kde is not None:
        ax.plot(
            t_grid,
            kde.pdf(theta_grid) * jacobian,
            "g-",
            lw=2,
            alpha=0.7,
            label="Circular KDE",
        )

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(0, SECONDS_PER_DAY + 1, SECONDS_PER_HOUR))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(25)], rotation=45, ha="right")
    ax.set_xlim(0, SECONDS_PER_DAY)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_bic_curve(
    mixture_results: list[dict],
    title: str = "Model Selection",
    figsize: tuple[float, float] = (8, 5),
) -> Figure:
    """BIC and AIC curves across candidate component counts.

    Parameters
    ----------
    mixture_results : list of dict
        The ``all_results`` output of the BIC selection loop in
        :meth:`CircularTimeAnalyzer.analyze`.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ns = [r["n_comp"] for r in mixture_results]
    bics = [r["bic"] for r in mixture_results]
    aics = [r["aic"] for r in mixture_results]

    best_idx = np.argmin(bics).item()

    ax.plot(ns, bics, "bo-", lw=2, markersize=8, label="BIC")
    ax.plot(ns, aics, "gs--", lw=2, markersize=8, label="AIC")
    ax.axvline(
        ns[best_idx],
        color="b",
        linestyle=":",
        alpha=0.5,
        label=f"Best (N={ns[best_idx]})",
    )

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Information Criterion", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


class CircularTimeAnalyzer:
    """Batch analyzer for dictionary-structured circular time-of-day data.

    Each entry of the input dictionary is a 1D array of timestamps in
    seconds ``[0, 86400)`` -- typically the occurrence times of one Maslow
    hierarchy level.

    Parameters
    ----------
    data_dict : dict
        Mapping of labels (e.g. ``"H1 Physiological"``) to 1D arrays of
        timestamps in seconds.

    Example
    -------
    >>> import numpy as np
    >>> from pymaslow.plotting import CircularTimeAnalyzer
    >>> data = {"H1": np.array([7 * 3600, 12 * 3600, 18 * 3600])}
    >>> analyzer = CircularTimeAnalyzer(data)
    >>> analyzer.analyze_all(max_components=3, n_restarts=2, random_state=42)
    >>> analyzer.summary()
    """

    def __init__(self, data_dict: dict[str, np.ndarray]):
        self.data_dict: dict[str, np.ndarray] = {}
        for k, v in data_dict.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim != 1:
                raise ValueError(
                    f"Data for key '{k}' must be 1D, got shape {arr.shape}"
                )
            if len(arr) == 0:
                warnings.warn(f"Key '{k}' has empty data array.", stacklevel=2)
            self.data_dict[k] = arr

        self.results: dict[str, dict] = {}

    def analyze(
        self,
        key: str,
        max_components: int = 6,
        n_restarts: int = 5,
        bandwidth: str | float | None = None,
        random_state: int | None = None,
    ) -> dict:
        """Fit the von Mises mixture and circular KDE for one key.

        Parameters
        ----------
        key : str
            Key of ``data_dict`` to analyze.
        max_components : int
            Maximum mixture components for BIC selection.
        n_restarts : int
            Random restarts per component count.
        bandwidth : str, float, or None
            KDE bandwidth (``None`` = Scott's rule).
        random_state : int or None
            Reproducibility seed.

        Returns
        -------
        dict
            Fitted ``mixture``, ``kde``, ``mixture_results``, and the raw
            ``time``/``theta`` data.
        """
        if key not in self.data_dict:
            raise KeyError(f"Key '{key}' not found in data_dict.")

        t_data = self.data_dict[key] % SECONDS_PER_DAY
        theta_data = np.asarray(sec_to_rad(t_data), dtype=float)

        mixture, mixture_results = _fit_vmmm_bic(
            theta_data,
            max_components=max_components,
            n_restarts=n_restarts,
            random_state=random_state,
        )
        kde = fit_circular_kde(theta_data, bandwidth=bandwidth)

        result = {
            "key": key,
            "time": t_data,
            "theta": theta_data,
            "mixture": mixture,
            "kde": kde,
            "mixture_results": mixture_results,
        }
        self.results[key] = result
        return result

    def analyze_all(
        self,
        max_components: int = 6,
        n_restarts: int = 5,
        bandwidth: str | float | None = None,
        random_state: int | None = None,
    ) -> dict[str, dict]:
        """Analyze every key in the data dictionary."""
        for key in self.data_dict:
            print(f"Analyzing '{key}'...")
            self.analyze(
                key,
                max_components=max_components,
                n_restarts=n_restarts,
                bandwidth=bandwidth,
                random_state=random_state,
            )
        return self.results

    def plot(
        self,
        key: str,
        plot_type: str = "all",
        save_path: str | None = None,
        **kwargs,
    ) -> list[tuple[str, Figure]]:
        """Generate plots for one analyzed key.

        Parameters
        ----------
        key : str
            Key to plot (must have been analyzed first).
        plot_type : {'all', 'circular', 'radar', 'linear', 'bic'}
            Which plot(s) to generate.
        save_path : str or None
            If given, figures are saved as ``{save_path}_{key}_{type}.png``.
        **kwargs
            Forwarded to the individual plotting functions.

        Returns
        -------
        list of (name, Figure)
        """
        if key not in self.results:
            raise KeyError(
                f"Key '{key}' not analyzed yet. Call analyze() or analyze_all() first."
            )

        res = self.results[key]
        figs: list[tuple[str, Figure]] = []

        if plot_type in ("circular", "all"):
            figs.append(
                (
                    "circular",
                    plot_circular_summary(
                        res["theta"],
                        model=res["mixture"],
                        kde=res["kde"],
                        title=f"{key} — Circular Distribution",
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k in ("figsize", "show_components")
                        },
                    ),
                )
            )

        if plot_type in ("radar", "all"):
            figs.append(
                (
                    "radar",
                    plot_am_pm_radar(
                        res["theta"],
                        model=res["mixture"],
                        kde=res["kde"],
                        title=f"{key} — AM/PM Radar",
                        **{k: v for k, v in kwargs.items() if k in ("figsize",)},
                    ),
                )
            )

        if plot_type in ("linear", "all"):
            figs.append(
                (
                    "linear",
                    plot_linear_time(
                        res["time"],
                        model=res["mixture"],
                        kde=res["kde"],
                        title=f"{key} — Linear Time Density",
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k in ("figsize", "n_bins")
                        },
                    ),
                )
            )

        if plot_type in ("bic", "all"):
            figs.append(
                (
                    "bic",
                    plot_bic_curve(
                        res["mixture_results"],
                        title=f"{key} — Model Selection (BIC/AIC)",
                    ),
                )
            )

        if save_path is not None:
            for name, fig in figs:
                fname = f"{save_path}_{key}_{name}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"Saved: {fname}")
                plt.close(fig)

        return figs

    def plot_all(
        self,
        plot_type: str = "all",
        save_path: str | None = None,
        **kwargs,
    ) -> dict[str, list[tuple[str, Figure]]]:
        """Generate plots for all analyzed keys."""
        return {
            key: self.plot(key, plot_type=plot_type, save_path=save_path, **kwargs)
            for key in self.results
        }

    def summary(self, key: str | None = None) -> None:
        """Print a text summary of the analysis results.

        Parameters
        ----------
        key : str or None
            If None, summarize all analyzed keys.
        """
        keys = [key] if key is not None else list(self.results.keys())

        for k in keys:
            if k not in self.results:
                print(f"\n[WARNING] Key '{k}' not found or not analyzed.\n")
                continue

            res = self.results[k]
            print(f"\n{'=' * 60}")
            print(f"  Key: {k}")
            print(f"{'=' * 60}")
            print(f"  Sample size: {len(res['time'])}")

            print("\n  Best Mixture Model (selected by BIC):")
            print(f"  {res['mixture']}")

            print("\n  Model Selection Table:")
            print(f"  {'N':>3} {'k':>3} {'logL':>12} {'AIC':>12} {'BIC':>12}")
            print(f"  {'-' * 50}")
            for r in res["mixture_results"]:
                marker = (
                    "  * BEST" if r["n_comp"] == res["mixture"].n_components else ""
                )
                print(
                    f"  {r['n_comp']:>3} {3 * r['n_comp'] - 1:>3} "
                    f"{r['logL']:>12.1f} {r['aic']:>12.1f} {r['bic']:>12.1f}{marker}"
                )

            print(f"\n  KDE: {res['kde']}")
            print(f"{'=' * 60}\n")
