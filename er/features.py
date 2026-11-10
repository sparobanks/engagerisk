"""Turn the first few weeks of a student's activity into model features.

For each signal we compute three things over the observation window:
- its average (the overall level),
- its trend (is it rising or falling, the slope over the weeks), and
- its most recent value (where it is now).

A falling trend is often the earliest warning, before the average looks bad,
which is why the trend matters. We also count work handed in, handed in late,
and missed, and take the average quiz score so far.

Every feature keeps the name of the signal it came from, so a model
explanation can always be traced back to a part of engagement (see
engagement.py).
"""
from __future__ import annotations

import numpy as np

from .engagement import DIRECTION
from .sim import ASSIGNMENT_WEEKS

WINDOW = 6   # weeks of data used to make the prediction (the early-warning window)
TREND_SIGNALS = ["logins", "active_minutes", "days_active", "resource_views",
                 "forum_posts", "forum_reads", "help_requests"]


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def features_for(student: dict, window: int = WINDOW) -> dict[str, float]:
    weeks = [w for w in student["weeks"] if w["week"] <= window]
    f: dict[str, float] = {}
    for sig in TREND_SIGNALS:
        series = [float(w[sig]) for w in weeks]
        f[f"{sig}__mean"] = float(np.mean(series)) if series else 0.0
        f[f"{sig}__trend"] = _trend(series)
        f[f"{sig}__last"] = series[-1] if series else 0.0
    # assignments due within the window
    due = [w for w in weeks if w["week"] in ASSIGNMENT_WEEKS]
    f["submissions_on_time__sum"] = float(sum(w["submissions_on_time"] for w in due))
    f["late_submissions__sum"] = float(sum(w["late_submissions"] for w in due))
    f["missed__sum"] = float(sum(1 for w in due if not w["submissions_on_time"] and not w["late_submissions"]))
    quizzes = [w["quiz_score"] for w in due if w["quiz_score"] is not None]
    f["quiz_score__mean"] = float(np.mean(quizzes)) if quizzes else 0.0
    return f


def feature_names(window: int = WINDOW) -> list[str]:
    sample = features_for({"weeks": [dict(week=w, logins=0, active_minutes=0, days_active=0,
                                          resource_views=0, forum_posts=0, forum_reads=0,
                                          help_requests=0, submissions_on_time=0, late_submissions=0,
                                          quiz_score=None) for w in range(1, window + 1)]}, window)
    return list(sample)


def build_matrix(students: list[dict], window: int = WINDOW):
    names = feature_names(window)
    X = np.array([[features_for(s, window)[n] for n in names] for s in students], dtype=float)
    y = np.array([1 if s["disengaged"] else 0 for s in students], dtype=int)
    groups = np.array([s["group"] for s in students])
    return X, y, names, groups
