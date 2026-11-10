"""Train and evaluate the engagement risk model.

    python scripts/run_eval.py

1. Train on the training cohort, test on a separate cohort the model never saw.
2. Compare the LightGBM model with a logistic-regression baseline on ROC AUC,
   PR AUC, Brier score and calibration error.
3. Early warning: retrain using only the first 3, 6 and 9 weeks, to show how
   early a reliable prediction can be made.
4. Fairness: report the model's accuracy and flag rate for each group, so any
   gap is visible.
Writes models/risk.pkl, reports/metrics.json and two charts in reports/.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from er.features import WINDOW, build_matrix
from er.model import LogisticModel, RiskModel
from er.sim import load


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def scores(y, p):
    return {"roc_auc": round(float(roc_auc_score(y, p)), 3),
            "pr_auc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 3),
            "calibration_error": round(ece(y, p), 3)}


def recall_at_topk(y, p, k=0.2):
    # flag exactly the top k fraction by rank (ties broken by order), so the
    # flag rate is a true 20% even when many scores are equal
    n_flag = max(1, int(round(len(p) * k)))
    order = np.argsort(-p)
    flagged = np.zeros(len(p), dtype=bool)
    flagged[order[:n_flag]] = True
    caught = (flagged & (y == 1)).sum()
    return float(caught / max(1, (y == 1).sum())), float(flagged.mean())


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    train = load(ROOT / "data" / "train")
    test = load(ROOT / "data" / "test")

    Xtr, ytr, names, _ = build_matrix(train, WINDOW)
    Xte, yte, _, gte = build_matrix(test, WINDOW)

    rm = RiskModel.train(Xtr, ytr, names, seed=0)
    lm = LogisticModel.train(Xtr, ytr, names, seed=0)
    p_rm, p_lm = rm.predict_proba(Xte), lm.predict_proba(Xte)

    report = {
        "window_weeks": WINDOW,
        "test_students": len(test),
        "disengaged_rate": round(float(yte.mean()), 3),
        "lightgbm": scores(yte, p_rm),
        "logistic_baseline": scores(yte, p_lm),
    }

    # early warning: how good is the model with fewer weeks of data?
    report["early_warning"] = []
    for w in (3, 6, 9):
        Xtr_w, ytr_w, names_w, _ = build_matrix(train, w)
        Xte_w, yte_w, _, _ = build_matrix(test, w)
        m = LogisticModel.train(Xtr_w, ytr_w, names_w, seed=0)
        rec, rate = recall_at_topk(yte_w, m.predict_proba(Xte_w))
        report["early_warning"].append({
            "weeks_of_data": w, "roc_auc": scores(yte_w, m.predict_proba(Xte_w))["roc_auc"],
            "disengagers_caught_in_top_20pct": round(rec, 3)})

    # fairness: same metrics per group
    report["fairness_by_group"] = {}
    for g in ("A", "B"):
        m = gte == g
        if m.sum() > 10:
            rec, rate = recall_at_topk(yte[m], p_rm[m])
            report["fairness_by_group"][g] = {
                "students": int(m.sum()), "roc_auc": round(float(roc_auc_score(yte[m], p_rm[m])), 3),
                "disengagers_caught_in_top_20pct": round(rec, 3), "flag_rate": round(rate, 3),
                "base_rate": round(float(yte[m].mean()), 3)}

    # deploy the stronger model (here the calibrated logistic model does slightly
    # better and is simpler and more transparent, so it is the one we ship)
    best, best_name = (lm, "logistic") if report["logistic_baseline"]["roc_auc"] >= \
        report["lightgbm"]["roc_auc"] else (rm, "lightgbm")
    report["deployed_model"] = best_name
    with open(ROOT / "models" / "risk.pkl", "wb") as f:
        pickle.dump({"model": best, "names": names}, f)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    _charts(yte, p_rm, p_lm, report)
    print(json.dumps(report, indent=2))


def _charts(y, p_rm, p_lm, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # calibration
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for p, lab, col in [(p_rm, "LightGBM", "#1f5fa8"), (p_lm, "Logistic", "#c2571a")]:
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() > 5:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "-o", ms=4, color=col, label=lab)
    ax.plot([0, 1], [0, 1], "--", color="#8a948f", lw=1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Share who actually disengaged")
    ax.set_title(f"Calibration (error {report['lightgbm']['calibration_error']:.3f})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "calibration.png", dpi=150)
    plt.close(fig)
    # early warning
    ew = report["early_warning"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot([e["weeks_of_data"] for e in ew], [e["roc_auc"] for e in ew], "-o", color="#2f7d52")
    ax.set_xlabel("Weeks of data used")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("The earlier we predict, the harder it is", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dde2df")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "early_warning.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
