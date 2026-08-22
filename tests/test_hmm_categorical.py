"""Unit tests for the categorical HMM approach in pymaslow.hmm."""

import numpy as np
import pytest

from pymaslow import hmm

hmmlearn = pytest.importorskip("hmmlearn", reason="hmmlearn not installed")


# ---------------------------------------------------------------------
# Hidden state definitions
# ---------------------------------------------------------------------


def test_hmm_states_definition():
    id_to_state, state_to_id = hmm.hmm_states_definition()
    assert len(id_to_state) == 31 == len(state_to_id)
    assert id_to_state[0] == "1"
    assert id_to_state[4] == "5"
    assert id_to_state[5] == "1,2"
    assert id_to_state[30] == "1,2,3,4,5"
    assert state_to_id["1,4,5"] == 20
    # inverse mapping is consistent
    for i, s in id_to_state.items():
        assert state_to_id[s] == i


def test_n_hidden_states_constant():
    assert hmm.N_HIDDEN_STATES == 31
    assert hmm.N_ACTIVITIES == 824


# ---------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------


def test_estimate_hmm_parameters_shapes_and_normalization():
    states = [[0, 1, 1], [0, 0, 1], [1, 2, 2]]
    obs = [[10, 11, 11], [10, 10, 12], [11, 12, 12]]
    initial, transition, emission = hmm.estimate_hmm_parameters(
        states, obs, range(3), range(13)
    )
    assert set(initial) == {0, 1, 2}
    np.testing.assert_allclose(sum(initial.values()), 1.0)
    for s in range(3):
        np.testing.assert_allclose(sum(transition[s].values()), 1.0)
        np.testing.assert_allclose(sum(emission[s].values()), 1.0)
    # state 0 -> 1 and 0 -> 0 transitions observed, never 0 -> 2
    assert transition[0][2] < transition[0][0]
    assert transition[0][2] < transition[0][1]


def test_estimate_hmm_parameters_length_mismatch():
    with pytest.raises(ValueError):
        hmm.estimate_hmm_parameters([[0, 1]], [[0]])


# ---------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------


def test_train_test_split_reproducible_and_converts_states():
    data = hmm.load_hmm_data("capture24")
    train1, val1, test1 = hmm.train_test_split(
        data, test_size_not_percent=20, validation_size_not_percent=10, random_state=42
    )
    train2, _, _ = hmm.train_test_split(
        data, test_size_not_percent=20, validation_size_not_percent=10, random_state=42
    )
    assert val1 is not None
    assert len(train1[0]) == 116 and len(val1[0]) == 10 and len(test1[0]) == 20
    assert train1[0] == train2[0]  # reproducible
    assert test1[3] == [len(d) for d in test1[2]]  # lengths match durations
    # states converted to ids in [0, 31)
    all_ids = {s for seq in train1[0] for s in seq}
    assert min(all_ids) >= 0 and max(all_ids) < 31


def test_train_test_split_etri_string_states():
    data = hmm.load_hmm_data("etri")
    train, _, test = hmm.train_test_split(data, test_size_not_percent=4, random_state=0)
    all_ids = {s for seq in train[0] + test[0] for s in seq}
    assert min(all_ids) >= 0 and max(all_ids) < 31


# ---------------------------------------------------------------------
# CategoricalMaslowHMM
# ---------------------------------------------------------------------


def test_categorical_hmm_fit_predict():
    data = hmm.load_hmm_data("capture24")
    train, _, test = hmm.train_test_split(
        data, test_size_not_percent=20, random_state=42
    )
    model = hmm.CategoricalMaslowHMM()
    model.fit_supervised(train[0], train[1])

    assert model.model_ is not None
    assert model.model_.startprob_.shape == (31,)
    assert model.model_.transmat_.shape == (31, 31)
    assert model.model_.emissionprob_.shape == (31, 824)

    pred = model.predict(test[1])
    gt = [s for seq in test[0] for s in seq]
    assert len(pred) == len(gt)

    metrics = hmm.metrics_classification(gt, pred)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    # deterministic activity->state mapping makes decoding near-perfect
    assert metrics["accuracy"] > 0.9


def test_categorical_hmm_predict_flat_with_lengths():
    data = hmm.load_hmm_data("capture24")
    train, _, test = hmm.train_test_split(data, test_size_not_percent=5, random_state=1)
    model = hmm.CategoricalMaslowHMM().fit_supervised(train[0], train[1])
    flat = [o for seq in test[1] for o in seq]
    pred = model.predict(flat, lengths=test[3])
    assert len(pred) == len(flat)


def test_categorical_hmm_unfitted_raises():
    model = hmm.CategoricalMaslowHMM()
    with pytest.raises(RuntimeError):
        model.predict([1, 2, 3])


# ---------------------------------------------------------------------
# metrics_classification
# ---------------------------------------------------------------------


def test_metrics_classification_perfect_and_partial():
    assert hmm.metrics_classification([0, 1, 2], [0, 1, 2])["accuracy"] == 1.0
    m = hmm.metrics_classification([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    with pytest.raises(ValueError):
        hmm.metrics_classification([0, 1], [0])


# ---------------------------------------------------------------------
# Embedded data
# ---------------------------------------------------------------------


def test_embedded_hmm_data():
    cap = hmm.load_hmm_data("capture24")
    assert len(cap) == 146
    states, activities, durations = cap[min(cap.keys())]
    assert len(states) == len(activities) == len(durations)

    etri = hmm.load_hmm_data("etri")
    assert len(etri) == 22

    tuples = hmm.load_etri_activity_definitions()
    assert len(tuples) == 184

    with pytest.raises(ValueError):
        hmm.load_hmm_data("unknown")


def test_lazy_data_attributes():
    assert len(hmm.data_capture24) == 146
    assert len(hmm.data_etri) == 22
    assert len(hmm.etri_activity_definitions) == 184
    with pytest.raises(AttributeError):
        _ = hmm.nonexistent_attribute
