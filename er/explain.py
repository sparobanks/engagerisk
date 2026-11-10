"""Turn a risk score into a plain-English reason, grounded in the data.

For each flagged student the model gives a contribution for every feature: how
much it pushed the risk up or down. This picks the few features that pushed the
risk up the most, turns them into plain words using the engagement framework,
and writes two short notes:

- for the tutor: what the signs are and what to look at,
- for the student support team: a gentler summary for a wellbeing conversation.

Two safeguards:
- the reasons only ever use the student's real top factors, so the explanation
  cannot drift from the data. check_grounded() confirms the named factors are
  really among the top ones.
- the reason describes behaviour (less activity, missed work), never a guess
  about the cause, because the cause needs a human conversation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engagement import DIRECTION, SIGNAL_LABEL, part_of


def _signal_of(feature: str) -> str:
    return feature.split("__")[0]


def _phrase(feature: str) -> str:
    sig = _signal_of(feature)
    label = SIGNAL_LABEL.get(sig, sig)
    if feature.endswith("__trend"):
        return f"a falling trend in {label}"
    if feature.endswith("__last"):
        return f"low recent {label}"
    if feature == "missed__sum":
        return "missed assignments"
    if feature == "late_submissions__sum":
        return "work handed in late"
    if feature == "submissions_on_time__sum":
        return "fewer assignments handed in on time"
    if feature == "quiz_score__mean":
        return "lower quiz scores"
    return f"lower {label}"


def top_reasons(contrib: dict[str, float], n: int = 3) -> list[str]:
    # features that pushed risk UP the most (positive contribution)
    up = sorted([(f, v) for f, v in contrib.items() if v > 0], key=lambda kv: -kv[1])
    return [f for f, _ in up[:n]]


@dataclass
class Explanation:
    risk: float
    band: str
    reasons: list[str]
    tutor: str
    support: str
    grounded: bool
    source: str


def band_of(risk: float) -> str:
    return "high" if risk >= 0.6 else "medium" if risk >= 0.3 else "low"


def tutor_note(risk: float, reason_feats: list[str]) -> str:
    phrases = [_phrase(f) for f in reason_feats]
    if not phrases:
        return "No clear warning signs in the data so far."
    joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return (f"Estimated {risk:.0%} chance of disengaging. The main signs are {joined}. "
            f"It may be worth checking in with them.")


def support_note(risk: float, reason_feats: list[str]) -> str:
    parts = {part_of(_signal_of(f)) for f in reason_feats}
    if "cognitive" in parts and ("behavioural" in parts or "emotional_social" in parts):
        focus = "both their coursework and how connected they feel to the course"
    elif "cognitive" in parts:
        focus = "keeping up with the coursework"
    elif "emotional_social" in parts:
        focus = "feeling connected to the course"
    else:
        focus = "staying active on the course"
    return (f"This student may be starting to struggle with {focus}. A friendly check-in could help. "
            f"The flag is based on their activity only, so the reason is for them to tell us.")


def check_grounded(reason_feats: list[str], contrib: dict[str, float]) -> bool:
    """Every named reason must be one of the student's real risk-raising factors."""
    allowed = set(top_reasons(contrib, n=5))
    return all(f in allowed for f in reason_feats)


def explain(risk: float, contrib: dict[str, float], writer=None, student_name: str | None = None) -> Explanation:
    feats = top_reasons(contrib, n=3)
    tutor, support, source = tutor_note(risk, feats), support_note(risk, feats), "template"
    if writer is not None and band_of(risk) != "low":
        try:
            cand = writer.write(risk, feats, student_name=student_name)
            if check_grounded(feats, contrib):   # model text uses the same facts
                tutor, source = cand, "llm"
        except Exception:
            pass
    return Explanation(risk, band_of(risk), [_phrase(f) for f in feats],
                       tutor, support, check_grounded(feats, contrib), source)
