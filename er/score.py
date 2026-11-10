"""Score a cohort: risk and a plain-English reason for each student.

    rows = score_cohort(students, model, names)

Each row is ready for the dashboard: who the student is, their risk and band,
the main reasons, the two notes, and their weekly logins so the tutor can see
the trend. Students are returned highest risk first.
"""
from __future__ import annotations

import numpy as np

from .explain import explain
from .features import WINDOW, features_for, feature_names


def score_cohort(students: list[dict], model, names: list[str], writer=None, window: int = WINDOW):
    rows = []
    for s in students:
        x = np.array([features_for(s, window)[n] for n in names], dtype=float)
        risk = float(model.predict_proba(x.reshape(1, -1))[0])
        # never show absolute certainty about one person: cap the shown risk
        risk = min(0.98, max(0.02, risk))
        contrib, _ = model.contributions(x)
        ex = explain(risk, contrib, writer=writer, student_name=s["name"])
        logins = [w["logins"] for w in s["weeks"] if w["week"] <= window]
        rows.append({
            "id": s["id"], "name": s["name"], "group": s["group"],
            "risk": round(risk, 3), "band": ex.band,
            "reasons": ex.reasons, "tutor": ex.tutor, "support": ex.support,
            "grounded": ex.grounded, "written_by": ex.source,
            "weekly_logins": logins,
        })
    rows.sort(key=lambda r: -r["risk"])
    return rows
