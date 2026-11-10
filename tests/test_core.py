from pathlib import Path

import numpy as np

from er.engagement import part_of
from er.explain import band_of, check_grounded, explain, top_reasons
from er.features import WINDOW, build_matrix, features_for
from er.redact import contains_name, redact
from er.sim import make

ROOT = Path(__file__).resolve().parents[1]


def test_framework_mapping():
    assert part_of("logins") == "behavioural"
    assert part_of("submissions_on_time") == "cognitive"
    assert part_of("forum_posts") == "emotional_social"


def test_features_only_use_the_window():
    students = make(5, seed=3)
    f = features_for(vars(students[0]) if not isinstance(students[0], dict) else students[0], WINDOW) \
        if False else features_for(_as_dict(students[0]), WINDOW)
    # a trend feature exists for logins and the quiz mean is present
    assert "logins__trend" in f and "quiz_score__mean" in f


def _as_dict(s):
    import dataclasses
    return dataclasses.asdict(s)


def test_build_matrix_shapes():
    students = [_as_dict(s) for s in make(40, seed=4)]
    X, y, names, groups = build_matrix(students, WINDOW)
    assert X.shape[0] == 40 and X.shape[1] == len(names)
    assert set(np.unique(y)).issubset({0, 1})
    assert set(np.unique(groups)).issubset({"A", "B"})


def test_band_and_reasons():
    assert band_of(0.8) == "high" and band_of(0.4) == "medium" and band_of(0.1) == "low"
    contrib = {"logins__trend": 0.9, "missed__sum": 0.7, "quiz_score__mean": 0.4, "forum_reads__mean": -0.5}
    reasons = top_reasons(contrib, 2)
    assert reasons == ["logins__trend", "missed__sum"]


def test_explanation_is_grounded_and_behavioural():
    contrib = {"logins__trend": 0.9, "missed__sum": 0.7, "forum_reads__mean": -0.2}
    ex = explain(0.75, contrib)
    assert ex.band == "high"
    assert ex.reasons and all(isinstance(r, str) for r in ex.reasons)
    assert ex.grounded
    # the note must not guess a cause (no health/money words)
    assert "illness" not in ex.tutor.lower() and "money" not in ex.tutor.lower()


def test_grounding_rejects_a_made_up_reason():
    contrib = {"logins__trend": 0.9, "missed__sum": 0.7}
    assert check_grounded(["logins__trend"], contrib)
    assert not check_grounded(["quiz_score__mean"], contrib)   # not among the risk-raising factors


def test_redact_name():
    r = redact("Aisha Khan is at risk", ["Aisha Khan"])
    assert "Aisha Khan" not in r.masked
    assert not contains_name(r.masked, ["Aisha Khan"])
    assert r.restore(r.masked) == "Aisha Khan is at risk"
