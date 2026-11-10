"""The tutor dashboard API.

  POST /login            -> token
  GET  /cohort           -> every student with risk, band and reasons (needs token)
  GET  /student/{id}     -> one student in full (needs token)
  GET  /healthz          -> liveness, no auth
  GET  /                 -> dashboard page

At start-up it loads the demo cohort and the trained model (models/risk.pkl).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import auth
from .features import feature_names
from .score import score_cohort
from .sim import load

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="EngageRisk", version="0.1.0")
bearer = HTTPBearer(auto_error=False)
USERS = auth.demo_users()


def _load():
    students = load(Path(os.environ.get("ER_DATA", ROOT / "data" / "demo")))
    path = ROOT / "models" / "risk.pkl"
    if not path.exists():
        return students, None, [], "none"
    with open(path, "rb") as f:
        blob = pickle.load(f)
    rows = score_cohort(students, blob["model"], blob["names"])
    return students, {r["id"]: r for r in rows}, rows, blob["model"].name


STUDENTS, BY_ID, ROWS, MODEL_NAME = _load()


class Login(BaseModel):
    username: str
    password: str


def require_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    if creds is None or not auth.read_token(creds.credentials):
        raise HTTPException(status_code=401, detail="missing or invalid token")
    return auth.read_token(creds.credentials)


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": MODEL_NAME, "students": len(STUDENTS)}


@app.post("/login")
def login(body: Login):
    stored = USERS.get(body.username)
    if not stored or not auth.verify_password(body.password, stored):
        raise HTTPException(status_code=401, detail="wrong username or password")
    return {"token": auth.make_token(body.username), "user": body.username}


@app.get("/cohort")
def cohort(user: str = Depends(require_user)):
    brief = [{k: r[k] for k in ("id", "name", "group", "risk", "band", "reasons")} for r in ROWS]
    counts = {b: sum(1 for r in ROWS if r["band"] == b) for b in ("high", "medium", "low")}
    return {"model": MODEL_NAME, "counts": counts, "students": brief}


@app.get("/student/{sid}")
def student(sid: int, user: str = Depends(require_user)):
    r = BY_ID.get(sid)
    if r is None:
        raise HTTPException(status_code=404, detail="no such student")
    return r


@app.get("/")
def index():
    page = ROOT / "web" / "index.html"
    return FileResponse(page) if page.exists() else JSONResponse({"detail": "no page"}, status_code=404)
