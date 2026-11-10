"""Simple, correct authentication for the dashboard API.

Passwords are stored only as a salted PBKDF2-HMAC-SHA256 hash, checked in
constant time. A successful login returns an HMAC-SHA256 signed token carrying
the user and an expiry, checked on every request, so no session store is
needed. The signing key comes from ER_SECRET (a fixed demo key is used if it
is not set; set it in real use).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

SECRET = os.environ.get("ER_SECRET", "dev-only-secret-change-me").encode()
PBKDF2_ROUNDS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, rounds, salt_b64, dk_b64 = stored.split("$")
        salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(rounds))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk, expected)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(user: str, ttl_seconds: int = 3600) -> str:
    payload = _b64(json.dumps({"user": user, "exp": int(time.time()) + ttl_seconds},
                              separators=(",", ":")).encode())
    sig = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def read_token(token: str) -> str | None:
    try:
        payload, sig = token.split(".")
    except ValueError:
        return None
    good = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, good):
        return None
    try:
        body = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if body.get("exp", 0) < int(time.time()):
        return None
    return body.get("user")


def demo_users() -> dict[str, str]:
    return {"tutor": hash_password("engagerisk")}
