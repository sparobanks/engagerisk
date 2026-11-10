"""A framework that turns student engagement research into measurable signals.

Student engagement is usually described in three parts (Fredricks and others):

- behavioural engagement: do they turn up and take part (logins, activity)
- cognitive engagement: do they do the academic work (resources, submissions)
- emotional/social engagement: do they connect with the course and peers
  (forum activity, help-seeking)

The job is to turn each of these into something we can actually measure from a
learning platform. This file lists, for each part of engagement, the raw
signals that stand in for it, and the human-readable name for each signal. The
feature builder (features.py) uses this so every number the model sees can be
traced back to a part of engagement, which keeps the model explainable and
grounded in the research rather than being a black box.
"""
from __future__ import annotations

# part of engagement -> raw signals that measure it
FRAMEWORK: dict[str, list[str]] = {
    "behavioural": ["logins", "active_minutes", "days_active"],
    "cognitive": ["resource_views", "submissions_on_time", "quiz_score", "late_submissions"],
    "emotional_social": ["forum_posts", "forum_reads", "help_requests"],
}

# a plain-English label for every raw signal
SIGNAL_LABEL: dict[str, str] = {
    "logins": "how often they log in",
    "active_minutes": "time spent on the platform",
    "days_active": "days active in the week",
    "resource_views": "course materials opened",
    "submissions_on_time": "work handed in on time",
    "late_submissions": "work handed in late",
    "quiz_score": "quiz scores",
    "forum_posts": "posts in the discussion forum",
    "forum_reads": "discussion posts read",
    "help_requests": "times they asked for help",
}

# which way is good: +1 means more is better, -1 means more is worse
DIRECTION: dict[str, int] = {
    "logins": 1, "active_minutes": 1, "days_active": 1, "resource_views": 1,
    "submissions_on_time": 1, "quiz_score": 1, "forum_posts": 1, "forum_reads": 1,
    "help_requests": 1, "late_submissions": -1,
}

SIGNALS = list(SIGNAL_LABEL)


def part_of(signal: str) -> str:
    for part, sigs in FRAMEWORK.items():
        base = signal.split("__")[0]
        if base in sigs:
            return part
    return "other"
