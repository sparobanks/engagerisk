"""Produce the dashboard screenshot for the README.

    python scripts/make_demo.py
"""
from __future__ import annotations
import os, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import uvicorn
PORT = 8096

def serve():
    os.environ.setdefault("ER_DATA", str(ROOT / "data" / "demo"))
    from er.api import app
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

def main():
    threading.Thread(target=serve, daemon=True).start()
    for _ in range(40):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=1); break
        except Exception:
            time.sleep(0.5)
    from playwright.sync_api import sync_playwright
    out = ROOT / "docs" / "img"; out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 940, "height": 1100})
        pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        pg.click("button")
        pg.wait_for_selector(".stu", timeout=8000)
        for a in pg.query_selector_all(".stu"):
            if "high" in (a.get_attribute("class") or ""):
                a.click(); break
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(out / "dashboard.png"), full_page=True)
        b.close()
    print("wrote", out / "dashboard.png")

if __name__ == "__main__":
    main()
