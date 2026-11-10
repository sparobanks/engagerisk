"""Create the student cohorts.

    python scripts/make_data.py

Writes data/train and data/test (different cohorts) and a small data/demo set
used by the dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from er.sim import make, write


def main():
    for name, seed, n in [("train", 1, 1600), ("test", 2, 800), ("demo", 7, 60)]:
        students = make(n, seed)
        write(students, ROOT / "data" / name)
        rate = sum(s.disengaged for s in students) / len(students)
        print(f"{name}: {n} students, {rate:.0%} disengaged")


if __name__ == "__main__":
    main()
