"""Remove a student's personal data before any explanation is sent to a model.

A student's name is personal data. If a language model is used to write the
reason for a flag, it should not receive the name. This replaces names with
tokens and keeps a private map so the real name can be put back only in what
staff see.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Redaction:
    masked: str
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        out = text
        for token, real in self.mapping.items():
            out = out.replace(token, real)
        return out


def redact(text: str, names: list[str] | None = None) -> Redaction:
    mapping: dict[str, str] = {}
    out = text
    for i, real in enumerate(sorted([n for n in (names or []) if n], key=len, reverse=True), 1):
        tok = f"STUDENT_{i}"
        out = re.sub(rf"\b{re.escape(real)}\b", tok, out)
        mapping[tok] = real
    return Redaction(out, mapping)


def contains_name(text: str, names: list[str] | None = None) -> bool:
    for n in names or []:
        if n and re.search(rf"\b{re.escape(n)}\b", text):
            return True
    return False
