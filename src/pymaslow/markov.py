"""
pymaslow.markov
===============

First-order Markov chains over Maslow need hierarchies.

Given sequences of (possibly multi-label) hierarchy observations -- e.g. the
per-moment MHN labels of a CAPTURE-24 participant -- the transition-count
matrix is built by counting, for every pair of consecutive time steps, all
transitions from each active source level to each active destination level.
For example, a step from ``"1,3"`` to ``"3,4"`` contributes one count to each
of ``1->3``, ``1->4``, ``3->3`` and ``3->4``.

Adapted from the *Markov chain* section of ``notebooks/pymaslow.ipynb`` and
``maslownet.visualize_markov_chain`` of the companion research repository.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from .hierarchy import HIERARCHY_NAMES, N_HIERARCHIES, parse_mhn

__all__ = [
    "MarkovChain",
    "build_transition_counts",
]

#: Type of one observed step in a hierarchy sequence: a multi-label string
#: such as ``"1,3"``, a single int, or a sequence of ints.
StepLike = str | int | Sequence[int]


def build_transition_counts(
    sequences: Iterable[Sequence[StepLike]],
    n_states: int = N_HIERARCHIES,
) -> np.ndarray:
    """Count hierarchy-to-hierarchy transitions from observed sequences.

    Parameters
    ----------
    sequences : iterable of sequences
        Each inner sequence is an ordered series of steps; each step is a
        multi-label annotation (``"1,3"``), an int, or a sequence of ints.
    n_states : int
        Number of hierarchy levels (default 5).

    Returns
    -------
    (n_states, n_states) ndarray of float
        ``counts[i, j]`` is the number of observed transitions from level
        ``i + 1`` to level ``j + 1``.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for seq in sequences:
        steps = [parse_mhn(step) for step in seq]
        for prev, curr in pairwise(steps):
            for src in prev:
                for des in curr:
                    counts[src - 1, des - 1] += 1.0
    return counts


class MarkovChain:
    """First-order Markov chain over the Maslow need hierarchies.

    Parameters
    ----------
    count_matrix : array_like, shape (n_states, n_states)
        Transition counts; ``count_matrix[i, j]`` counts transitions from
        state ``i + 1`` to state ``j + 1``.
    state_labels : list of str or None
        Display labels for the states; defaults to the Maslow level names
        for ``n_states == 5``, otherwise ``["1", "2", ...]``.

    Attributes
    ----------
    count_matrix : pandas.DataFrame
        Transition counts with labeled rows/columns.
    transition_matrix : pandas.DataFrame
        Row-normalized transition probabilities.
    """

    def __init__(
        self,
        count_matrix: np.ndarray | pd.DataFrame,
        state_labels: list[str] | None = None,
    ):
        counts = np.asarray(count_matrix, dtype=float)
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
            raise ValueError("count_matrix must be a square 2D array")
        if np.any(counts < 0):
            raise ValueError("count_matrix must be non-negative")

        self.n_states = counts.shape[0]
        if state_labels is None:
            if self.n_states == N_HIERARCHIES:
                state_labels = [
                    f"H{i} {HIERARCHY_NAMES[i]}" for i in range(1, N_HIERARCHIES + 1)
                ]
            else:
                state_labels = [str(i + 1) for i in range(self.n_states)]
        if len(state_labels) != self.n_states:
            raise ValueError("state_labels length must match count_matrix shape")
        self.state_labels = list(state_labels)
        labels_index = pd.Index(self.state_labels)

        self.count_matrix = pd.DataFrame(
            counts, index=labels_index, columns=labels_index
        )

        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(
            counts,
            row_sums,
            out=np.full_like(counts, 1.0 / self.n_states),
            where=row_sums > 0,
        )
        self.transition_matrix = pd.DataFrame(
            probs, index=labels_index, columns=labels_index
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[Sequence[StepLike]],
        n_states: int = N_HIERARCHIES,
        state_labels: list[str] | None = None,
    ) -> MarkovChain:
        """Build a Markov chain from observed hierarchy sequences.

        Parameters
        ----------
        sequences : iterable of sequences
            Ordered hierarchy observations; see :func:`build_transition_counts`.
        n_states : int
            Number of hierarchy levels (default 5).
        state_labels : list of str or None
            Display labels for the states.

        Returns
        -------
        MarkovChain
        """
        counts = build_transition_counts(sequences, n_states=n_states)
        return cls(counts, state_labels=state_labels)

    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution of the chain (left eigenvector for eigenvalue 1).

        Returns
        -------
        (n_states,) ndarray
            Stationary probabilities in state order. Falls back to the
            uniform distribution if the chain is not irreducible.
        """
        p = self.transition_matrix.to_numpy(dtype=float)
        eigenvalues, eigenvectors = np.linalg.eig(p.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        total = stationary.sum()
        if not np.isfinite(stationary).all() or np.all(stationary <= 0) or total == 0:
            return np.full(self.n_states, 1.0 / self.n_states)
        stationary = np.abs(stationary) / np.abs(stationary).sum()
        return stationary

    def plot(
        self,
        save_path: str | None = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8, 6),
        min_threshold: float = 0.01,
    ):
        """Visualize the transition matrix as a weighted directed graph.

        Node placement uses a spring layout; edge width and color encode the
        transition probability; self-loops are drawn as circles. Requires the
        optional ``networkx`` dependency (``pip install "pymaslow[plot]"``).

        Parameters
        ----------
        save_path : str or None
            If given, save the figure to this path.
        dpi : int
            Resolution used when saving.
        figsize : tuple
            Figure size in inches.
        min_threshold : float
            Minimum transition probability to display (filters weak edges).

        Returns
        -------
        (fig, ax)
            Matplotlib figure and axes.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "MarkovChain.plot() requires networkx; "
                'install it with `pip install "pymaslow[plot]"`.'
            ) from exc

        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.patches import Circle, FancyArrowPatch

        p = self.transition_matrix

        graph = nx.DiGraph()
        states = p.index.tolist()
        graph.add_nodes_from(states)

        max_val = p.max().max()
        min_val = p[p > 0].min().min() if (p > 0).any().any() else 0.0

        edges, weights = [], []
        for i in states:
            for j in states:
                if p.loc[i, j] > min_threshold:
                    edges.append((i, j))
                    weights.append(p.loc[i, j])

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

        nx.draw_networkx_nodes(
            graph, pos, node_color="lightblue", node_size=2000, alpha=0.9, ax=ax
        )
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax)

        norm = Normalize(vmin=min_val, vmax=max_val)
        cmap = colormaps["Reds"]

        for (u, v), weight in zip(edges, weights, strict=True):
            arrow_width = 1 + 5 * (weight / max_val)
            color = cmap(norm(weight))

            if u == v:
                # Self-loop
                x, y = pos[u]
                circle = Circle(
                    (x, y + 0.15),
                    0.08,
                    fill=False,
                    edgecolor=color,
                    linewidth=arrow_width,
                )
                ax.add_patch(circle)
                ax.annotate(
                    "",
                    xy=(x + 0.08, y + 0.15),
                    xytext=(x + 0.07, y + 0.18),
                    arrowprops={
                        "arrowstyle": "->",
                        "lw": arrow_width,
                        "color": color,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
                ax.text(
                    x + 0.12,
                    y + 0.15,
                    f"{weight:.2f}",
                    fontsize=9,
                    ha="left",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                    },
                )
            else:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                arrow = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    arrowstyle="-|>",
                    mutation_scale=20 + 10 * (weight / max_val),
                    linewidth=arrow_width,
                    color=color,
                    connectionstyle="arc3,rad=0.1",
                    alpha=0.7,
                    shrinkA=25,
                    shrinkB=25,
                )
                ax.add_patch(arrow)

                # Probability label at the edge midpoint, offset perpendicularly
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                offset_x = -dy / length * 0.05
                offset_y = dx / length * 0.05
                ax.text(
                    mid_x + offset_x,
                    mid_y + offset_y,
                    f"{weight:.2f}",
                    fontsize=9,
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                    },
                )

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition Probability", rotation=270, labelpad=20, fontsize=12)

        ax.axis("off")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return fig, ax

    def __repr__(self) -> str:
        return f"MarkovChain(n_states={self.n_states})"
