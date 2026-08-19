"""Tests for the pymaslow.markov module."""

import numpy as np
import pytest

from pymaslow import MarkovChain, build_transition_counts


def test_build_transition_counts_single_labels():
    seqs = [[1, 2, 3], [2, 1]]
    counts = build_transition_counts(seqs, n_states=5)
    assert counts.shape == (5, 5)
    assert counts[0, 1] == 1  # 1 -> 2
    assert counts[1, 2] == 1  # 2 -> 3
    assert counts[1, 0] == 1  # 2 -> 1


def test_build_transition_counts_multi_label_cartesian():
    # "1,3" -> "3,4" contributes to 1->3, 1->4, 3->3, 3->4
    counts = build_transition_counts([["1,3", "3,4"]], n_states=5)
    assert counts[0, 2] == 1
    assert counts[0, 3] == 1
    assert counts[2, 2] == 1
    assert counts[2, 3] == 1
    assert counts.sum() == 4


def test_transition_matrix_row_normalized():
    seqs = [[1, 2, 1, 1, 2], [3, 3, 2]]
    mc = MarkovChain.from_sequences(seqs)
    probs = mc.transition_matrix.to_numpy()
    rows_with_data = probs.sum(axis=1) > 0
    np.testing.assert_allclose(probs[rows_with_data].sum(axis=1), 1.0)


def test_count_matrix_labels():
    mc = MarkovChain.from_sequences([[1, 2]])
    assert "H1 Physiological Needs" in mc.count_matrix.index
    assert mc.count_matrix.shape == (5, 5)


def test_stationary_distribution():
    seqs = [[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]]
    mc = MarkovChain.from_sequences(seqs)
    stat = mc.stationary_distribution()
    assert stat.shape == (5,)
    np.testing.assert_allclose(stat.sum(), 1.0)
    # A pure cycle 1->2->3->4->5->1 is uniform
    np.testing.assert_allclose(stat, 0.2, atol=1e-6)


def test_invalid_count_matrix():
    with pytest.raises(ValueError):
        MarkovChain(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        MarkovChain(-np.ones((3, 3)))
