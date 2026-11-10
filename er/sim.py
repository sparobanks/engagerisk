"""Simulate a student cohort over a term, with hidden engagement and outcomes.

Each student has a hidden weekly engagement level that we never see directly.
What we see are the signals it produces: logins, time on the platform,
resources opened, work handed in, quiz scores, forum activity and help
requests. Some students stay engaged; others slide, and by the end of term
they have disengaged (their activity has fallen away and they miss the final
pieces of work).

The task the model learns: using only the first few weeks, predict whether a
student will have disengaged by the end of term. Because we built the data, we
know the true outcome for every student, so the prediction can be scored.

A group attribute (two groups) is included so the model can be checked for
fairness: one group starts from a slightly lower baseline, which is the kind of
difference a real system must not amplify.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

WEEKS = 12
ASSIGNMENT_WEEKS = [3, 6, 9, 12]
FIRST = ["Aisha", "Ben", "Chloe", "David", "Emeka", "Fatima", "Grace", "Harry", "Ivy", "Jacob",
         "Kofi", "Lily", "Mo", "Nadia", "Oliver", "Priya", "Ruth", "Sam", "Tolu", "Uma",
         "Violet", "Will", "Yusuf", "Zara", "Amara", "Chidi", "Ella", "Femi", "Isla", "Noah"]
LAST = ["Khan", "Smith", "Okafor", "Jones", "Patel", "Brown", "Adeyemi", "Taylor", "Wilson", "Evans"]


@dataclass
class Student:
    id: int
    name: str
    group: str
    weeks: list[dict]          # one record per week of raw signals
    disengaged: bool           # the outcome at end of term (what we predict)
    _engagement: list[float] = field(default_factory=list)  # hidden, for reference only


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _student(i: int, rng: random.Random) -> Student:
    group = rng.choice(["A", "B"])
    base = rng.gauss(0.70 if group == "A" else 0.64, 0.14)   # group B slightly lower on average
    slope = rng.gauss(0.0, 0.015)
    # about a quarter of students are on a declining path
    if rng.random() < 0.25:
        slope = rng.gauss(-0.03, 0.012)
    shock_week = rng.randint(3, 9) if rng.random() < 0.20 else None
    weeks, eng = [], []
    missed = 0
    for w in range(1, WEEKS + 1):
        e = base + slope * w + rng.gauss(0, 0.06)
        if shock_week and w >= shock_week:
            e -= 0.20   # a mid-term drop (illness, money worries, etc.)
        e = _clip(e)
        eng.append(e)
        logins = max(0, int(rng.gauss(7 * e + 0.5, 1.2)))
        active = max(0, round(rng.gauss(240 * e, 40), 1))
        days = max(0, min(7, int(rng.gauss(7 * e, 1.0))))
        views = max(0, int(rng.gauss(18 * e, 4)))
        forum_posts = max(0, int(rng.gauss(2.5 * e - 0.3, 0.9)))
        forum_reads = max(0, int(rng.gauss(14 * e, 4)))
        help_req = max(0, int(rng.gauss(1.4 * (1 - e), 0.6)))
        on_time = late = 0
        quiz = None
        if w in ASSIGNMENT_WEEKS:
            p_submit = _clip(0.25 + 0.7 * e)
            r = rng.random()
            if r < p_submit:
                on_time = 1
            elif r < p_submit + 0.12:
                late = 1
            else:
                missed += 1
            quiz = round(_clip(rng.gauss(e, 0.12)) * 100, 1)
        weeks.append(dict(week=w, logins=logins, active_minutes=active, days_active=days,
                          resource_views=views, forum_posts=forum_posts, forum_reads=forum_reads,
                          help_requests=help_req, submissions_on_time=on_time, late_submissions=late,
                          quiz_score=quiz))
    end_engagement = sum(eng[-3:]) / 3
    disengaged = end_engagement < 0.33 or missed >= 3
    return Student(i, f"{rng.choice(FIRST)} {rng.choice(LAST)}", group, weeks, disengaged, eng)


def make(n: int, seed: int) -> list[Student]:
    rng = random.Random(seed)
    return [_student(i, rng) for i in range(n)]


def write(students: list[Student], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in students:
        d = asdict(s)
        d.pop("_engagement", None)   # never ship the hidden truth
        rows.append(d)
    (out_dir / "students.json").write_text(json.dumps(rows, indent=2))


def load(out_dir: Path) -> list[dict]:
    return json.loads((out_dir / "students.json").read_text())
