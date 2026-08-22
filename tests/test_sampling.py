"""Unit tests for pymaslow.sampling."""

import numpy as np
import pytest

import pymaslow
from pymaslow import sampling
from pymaslow import vonMisesMixture as vmmm

hmmlearn = pytest.importorskip("hmmlearn", reason="hmmlearn not installed")


# ---------------------------------------------------------------------
# sample_hierarchy_given_time
# ---------------------------------------------------------------------


def test_sample_hierarchy_returns_valid_level():
    h = sampling.sample_hierarchy_given_time(8.0, random_state=42)
    assert h in {"1", "2", "3", "4", "5"}


def test_sample_hierarchy_time_dependence():
    # the posterior p(h|t) should put more mass on H1 (physiological)
    # at meal times than H4 (esteem) does
    counts_8am = np.zeros(6)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        h = sampling.sample_hierarchy_given_time(
            8.0, (vmmm.p_x, vmmm.models), random_state=rng
        )
        counts_8am[int(h)] += 1
    # H1 dominates early morning in the fitted model
    assert counts_8am[1] > counts_8am[4]
    assert counts_8am[1] > counts_8am[5]


def test_model_vonmisesmixture_accepted_forms():
    # tuple (p_x, models)
    h1 = sampling.sample_hierarchy_given_time(
        8.0, (vmmm.p_x, vmmm.models), random_state=1
    )
    # tuple (p_x, models, best_k) as returned by fit_vmmm_dictionary
    h2 = sampling.sample_hierarchy_given_time(
        8.0, (vmmm.p_x, vmmm.models, vmmm.best_k), random_state=1
    )
    # dict form
    h3 = sampling.sample_hierarchy_given_time(
        8.0, {"p_x": vmmm.p_x, "models": vmmm.models}, random_state=1
    )
    # the module itself (has p_x/models attributes)
    h4 = sampling.sample_hierarchy_given_time(8.0, vmmm, random_state=1)
    assert h1 == h2 == h3 == h4
    with pytest.raises(ValueError):
        sampling.sample_hierarchy_given_time(8.0, {"wrong": "keys"})
    with pytest.raises(ValueError):
        sampling.sample_hierarchy_given_time(8.0, 12345)


def test_sample_hierarchy_reproducible():
    h1 = sampling.sample_hierarchy_given_time(12.0, random_state=7)
    h2 = sampling.sample_hierarchy_given_time(12.0, random_state=7)
    assert h1 == h2


def test_sample_hierarchy_invalid_time():
    with pytest.raises(ValueError):
        sampling.sample_hierarchy_given_time("not-a-time")  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------
# sample_activity_given_hierarchy
# ---------------------------------------------------------------------


def test_sample_activity_returns_valid_id():
    a = sampling.sample_activity_given_hierarchy("1", random_state=1)
    assert isinstance(a, int)
    assert 0 <= a < 824


def test_sample_activity_accepts_int_and_str():
    a1 = sampling.sample_activity_given_hierarchy(2, random_state=3)
    a2 = sampling.sample_activity_given_hierarchy("2", random_state=3)
    assert a1 == a2


def test_sample_activity_invalid_hierarchy():
    with pytest.raises(ValueError):
        sampling.sample_activity_given_hierarchy("9", random_state=0)


def _raw_hmmlearn_model(random_state=42):
    """A raw hmmlearn CategoricalHMM fitted manually, as in the notebook."""
    from hmmlearn.hmm import (  # pyright: ignore[reportMissingImports]
        CategoricalHMM,
    )

    from pymaslow import hmm

    train, _, _ = hmm.train_test_split(
        hmm.data_capture24, test_size_not_percent=20, random_state=random_state
    )
    init_p, trans_p, emit_p = hmm.estimate_hmm_parameters(
        train[0], train[1], range(31), range(824)
    )
    m = CategoricalHMM(n_components=31)
    m.startprob_ = np.array([init_p[s] for s in range(31)])
    m.transmat_ = np.array([[trans_p[a][b] for b in range(31)] for a in range(31)])
    m.emissionprob_ = np.array([[emit_p[s][o] for o in range(824)] for s in range(31)])
    m.n_features = 824
    return m


def test_sample_activity_accepts_raw_hmmlearn_model():
    # regression: raw hmmlearn models expose 'emissionprob_' (not 'emission_prob_')
    raw = _raw_hmmlearn_model()
    a = sampling.sample_activity_given_hierarchy("1", raw, random_state=5)
    assert isinstance(a, int) and 0 <= a < 824


def test_sample_sequence_accepts_raw_hmmlearn_model():
    raw = _raw_hmmlearn_model()
    diary = sampling.sample_sequence(
        t0="07:30", n_activities=3, model_categoricalhmm=raw, random_state=0
    )
    assert len(diary) == 3
    assert diary[0]["start_time"] == "07:30"


def test_sample_activity_unfitted_model_raises_runtimeerror():
    from pymaslow import hmm

    unfitted = hmm.CategoricalMaslowHMM()
    with pytest.raises(RuntimeError):
        sampling.sample_activity_given_hierarchy("1", unfitted, random_state=0)


def test_wrapper_exposes_hmmlearn_convention_attributes():
    from pymaslow import hmm

    train, _, _ = hmm.train_test_split(
        hmm.data_capture24, test_size_not_percent=20, random_state=42
    )
    model = hmm.CategoricalMaslowHMM().fit_supervised(train[0], train[1])
    assert model.startprob_ is not None and model.startprob_.shape == (31,)
    assert model.transmat_ is not None and model.transmat_.shape == (31, 31)
    assert model.emissionprob_ is not None and model.emissionprob_.shape == (31, 824)
    # unfitted wrapper exposes None instead of raising AttributeError
    unfitted = hmm.CategoricalMaslowHMM()
    assert unfitted.emissionprob_ is None
    assert unfitted.emission_prob_ is None


# ---------------------------------------------------------------------
# sample_sequence
# ---------------------------------------------------------------------


def test_sample_sequence_structure():
    diary = sampling.sample_sequence(t0="02:00", max_days=1, random_state=42)
    assert len(diary) > 3
    for rec in diary:
        assert set(rec) == {
            "day",
            "start",
            "end",
            "start_time",
            "end_time",
            "hierarchy",
            "hierarchy_name",
            "activity_id",
            "duration_seconds",
            "activity",
        }
        assert 0 <= rec["start"] < 24
        assert rec["duration_seconds"] >= 1.0
        assert rec["hierarchy"] in {"1", "2", "3", "4", "5"}
        assert 0 <= rec["activity_id"] < 824
    # times are non-decreasing within the day and the chain is consistent
    prev_end = diary[0]["start"]
    for rec in diary:
        assert rec["start"] == pytest.approx(prev_end % 24, abs=1e-6)
        prev_end = rec["start"] + rec["duration_seconds"] / 3600.0


def test_sample_sequence_reproducible():
    d1 = sampling.sample_sequence(t0=2.0, max_days=1, random_state=11)
    d2 = sampling.sample_sequence(t0=2.0, max_days=1, random_state=11)
    assert d1 == d2


def test_sample_sequence_t0_formats():
    d_str = sampling.sample_sequence(t0="06:30", max_days=1, random_state=5)
    assert d_str[0]["start_time"] == "06:30"
    d_float = sampling.sample_sequence(t0=6.5, max_days=1, random_state=5)
    assert d_float[0]["start_time"] == "06:30"


def test_sample_sequence_invalid_t0():
    with pytest.raises(ValueError):
        sampling.sample_sequence(t0="25:00")
    with pytest.raises(ValueError):
        sampling.sample_sequence(t0=-1.0)
    with pytest.raises(ValueError):
        sampling.sample_sequence(t0="bad")


def test_sample_sequence_n_activities_cap():
    diary = sampling.sample_sequence(
        t0=8.0, n_activities=5, max_days=10, random_state=2
    )
    assert len(diary) == 5


def test_sample_sequence_multi_day():
    diary = sampling.sample_sequence(t0=22.0, max_days=2, random_state=9)
    assert max(r["day"] for r in diary) >= 1  # crossed midnight


def test_sample_sequence_activity_names():
    diary = sampling.sample_sequence(t0=23.5, n_activities=3, random_state=13)
    for rec in diary:
        assert isinstance(rec["activity"], str) and len(rec["activity"]) > 0
    # the first activity after 23:30 is very likely sleep-related
    # (H1 dominates at night); just check the name resolves from compendium
    df = pymaslow.load_compendium()
    assert diary[0]["activity"] in set(df["activity"])


def test_sample_sequence_explicit_models():
    from pymaslow import durations, hmm

    duration_model = durations.fit()
    cat = hmm.CategoricalMaslowHMM().fit_supervised(
        *hmm.train_test_split(
            hmm.data_capture24, test_size_not_percent=20, random_state=42
        )[0][:2]
    )
    diary = sampling.sample_sequence(
        t0="07:00",
        n_activities=4,
        model_duration=duration_model,
        model_categoricalhmm=cat,
        model_vonmisesmixture=(vmmm.p_x, vmmm.models),
        random_state=3,
    )
    assert len(diary) == 4
    assert diary[0]["start_time"] == "07:00"


def test_top_level_exposure():
    assert pymaslow.sampling is sampling
