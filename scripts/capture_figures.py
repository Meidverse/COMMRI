"""Capture report figures from the running app (http://127.0.0.1:5000) using Playwright + Microsoft Edge.
Usage: python scripts/capture_figures.py <email> <password>
Saves PNGs into report_figures/."""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_figures")
BASE = "http://127.0.0.1:5000"
SAMPLE = r"C:\all projects\zz_projects\medAI\mri\dataset\classification\test\glioma\glioma_1008.jpg"
os.makedirs(OUT, exist_ok=True)
email, pw = sys.argv[1], sys.argv[2]

with sync_playwright() as p:
    br = p.chromium.launch(channel="msedge")
    ctx = br.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    pg = ctx.new_page()

    def shot(name, full=True, wait=1800):
        pg.wait_for_timeout(wait)
        pg.screenshot(path=os.path.join(OUT, name), full_page=full)
        print("saved", name)

    pg.goto(BASE + "/register"); shot("fig_4_1_registration.png", full=False, wait=600)
    pg.goto(BASE + "/login"); shot("fig_4_2_login.png", full=False, wait=600)
    pg.fill("input[name=identifier]", email); pg.fill("input[name=password]", pw); pg.click("button")
    pg.wait_for_url(BASE + "/")
    shot("fig_4_3_dashboard.png", full=False, wait=1600)

    pg.goto(BASE + "/analyze?patient=1&modality=mri")
    pg.set_input_files("#file", SAMPLE)
    shot("fig_4_4_image_upload.png", full=False, wait=900)

    pg.goto(BASE + "/analysis/3/processing"); shot("fig_4_5_image_processing.png")
    pg.goto(BASE + "/analysis/3"); shot("fig_4_6_analysis_result.png", wait=2800)
    pg.goto(BASE + "/report/1"); pg.wait_for_timeout(1800)
    pg.locator(".paper").screenshot(path=os.path.join(OUT, "fig_4_7_result_report.png")); print("saved fig_4_7_result_report.png")
    pg.set_viewport_size({"width": 1680, "height": 1000})
    pg.goto(BASE + "/history"); shot("fig_4_8_analysis_history.png", full=False)
    br.close()
