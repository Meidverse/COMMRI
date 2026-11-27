# NeuroScan AI — Medical Image Analysis System

Flask + PyTorch web app: login/registration with roles, patient management, brain MRI tumour
classification (4 classes), bone X-ray fracture detection (2 classes), preprocessing pipeline view,
Grad-CAM explanations, printable reports and analysis history (SQLite).

## Run
Double-click `app\run.bat`, or:
```
python app\app.py      # then open http://127.0.0.1:5000
```

## Database
- Local: SQLite file `app/instance/neuroscan.db` (created automatically).
- Hosted: PostgreSQL — set `DATABASE_URL` and the app uses it automatically.
- Tables: `users`, `patients`, `analyses`, `images` (scan images are stored in the database, so no disk is needed).

## Hosting on Render
1. Push the repo to GitHub (already done).
2. Render Dashboard → **New → Blueprint** → choose this repository. `render.yaml` creates:
   - `neuroscan-db` — PostgreSQL 16 (Singapore)
   - `neuroscan-ai` — web service (gunicorn, CPU-only PyTorch)
3. Enter `ADMIN_EMAIL` and `ADMIN_PASSWORD` when asked — that account is created on first start.
4. Open the service URL, sign in as admin, add doctors under Doctor Management; patients self-register.

Optional: load the demo data into the hosted DB from your PC (uses the local datasets):
```
$env:DATABASE_URL="<External Database URL from Render>"; python scripts\seed_demo.py
```
Other hosts (Railway, Heroku-style): `Procfile` + `requirements-render.txt`, set `DATABASE_URL`, `SECRET_KEY`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`.

## Demo accounts (created by `python scripts\seed_demo.py`, which RESETS the database)
| Role | Sign in with | Password |
|---|---|---|
| Administrator | admin@neuroscan.local / NS-AD-0001 | Admin@2026 |
| Doctor | dr.mehta@neuroscan.local / NS-DR-0001 | Doctor@2026 |
| Doctor | dr.kapoor@neuroscan.local / NS-DR-0002 | Doctor@2026 |
| Patient (portal) | sunita.verma@neuroscan.local / NS-PA-0001 | Patient@2026 |

Demo patients are fictional. Every demo analysis is real model output (images uploaded through the app).

## Models
- MRI: `kaggle_tumor_2dcnn_best.pth` — TumorCNN, 93.95 % test accuracy (training-time held-out split).
- X-ray: `xray_fracture_mnv3.pth` — MobileNetV3-Large fine-tuned by `scripts\train_xray.py`.
  - 97.7 % accuracy on 383 held-out images **from the same Kaggle dataset** (after removing duplicate files).
  - **On an independent dataset (FracAtlas, 4,083 X-rays) it found only 50 % of fractures and cleared 78 % of
    normal X-rays** (hand 39 % / 78 %, leg 65 % / 88 %, hip and shoulder normals almost always flagged).
    The Kaggle number does not carry over to real-world X-rays.
  - The app therefore accepts only hand / wrist / forearm / elbow X-rays and answers "Inconclusive" below 85 %
    confidence. Treat X-ray output as a research prototype, not a diagnosis.
  - v2 (`scripts\train_xray_v2.py`, Kaggle + FracAtlas, body-part head) is used when `XRAY_MODEL=v2` (set on Render).
    FracAtlas test: AUC 0.84; score < 0.2 = no fracture (73 % of fractures caught), 0.2-0.6 = Inconclusive,
    >= 0.6 = fracture (96 % of normals cleared). Hip/shoulder films are refused; the body-part head cannot tell
    forearm from leg, so knees are still not reliable. Details: `app/model_meta/xray_v2_metrics.json`.

## Scripts
- `scripts\train_xray.py` — train the X-ray model
- `scripts\train_xray_v3.py` — v3 X-ray training run for a rented GPU (steps in `scripts\V3_VASTAI.md`; v2 stays live until v3 beats it)
- `scripts\eval_mri.py` — write MRI metrics
- `scripts\seed_demo.py` — reset DB with demo data
- `scripts\capture_figures.py` — regenerate report screenshots into `report_figures\`
- `scripts\insert_figures.py` — insert figures into a copy of the report .docx
