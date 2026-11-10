import dataclasses
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from er.features import WINDOW, build_matrix
from er.model import LogisticModel, RiskModel
from er.sim import make

ROOT = Path(__file__).resolve().parents[1]


def _cohort(n, seed):
    return [dataclasses.asdict(s) for s in make(n, seed)]


def test_model_predicts_disengagement_well():
    tr, te = _cohort(800, 1), _cohort(500, 2)
    Xtr, ytr, names, _ = build_matrix(tr, WINDOW)
    Xte, yte, _, _ = build_matrix(te, WINDOW)
    for M in (RiskModel, LogisticModel):
        m = M.train(Xtr, ytr, names, seed=0)
        auc = roc_auc_score(yte, m.predict_proba(Xte))
        assert auc > 0.8


def test_probabilities_are_in_range():
    tr = _cohort(400, 1)
    X, y, names, _ = build_matrix(tr, WINDOW)
    m = RiskModel.train(X, y, names, seed=0)
    p = m.predict_proba(X)
    assert p.min() >= 0 and p.max() <= 1


def test_contributions_cover_all_features():
    tr = _cohort(300, 1)
    X, y, names, _ = build_matrix(tr, WINDOW)
    m = LogisticModel.train(X, y, names, seed=0)
    contrib, base = m.contributions(X[0])
    assert set(contrib) == set(names)


def test_api_requires_auth_and_lists_cohort():
    client = TestClient(__import__("er.api", fromlist=["app"]).app)
    assert client.get("/cohort").status_code == 401
    r = client.post("/login", json={"username": "tutor", "password": "engagerisk"})
    assert r.status_code == 200
    tok = r.json()["token"]
    c = client.get("/cohort", headers={"Authorization": f"Bearer {tok}"})
    assert c.status_code == 200
    body = c.json()
    assert "students" in body and "counts" in body


def test_api_rejects_bad_login():
    client = TestClient(__import__("er.api", fromlist=["app"]).app)
    assert client.post("/login", json={"username": "tutor", "password": "x"}).status_code == 401
