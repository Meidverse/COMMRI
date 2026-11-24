# NeuroScan AI — Medical Image Analysis System

Flask + PyTorch web app: login/registration with roles, patient management, brain MRI tumour
classification (4 classes), bone X-ray fracture detection (2 classes), preprocessing pipeline view,
Grad-CAM explanations, printable reports and analysis history (SQLite).

## Run
Double-click `app\run.bat`, or:
```
python app\app.py      # then open http://127.0.0.1:5000
```

## Demo accounts (created by `python scripts\seed_demo.py`, which RESETS the database)
| Role | Sign in with | Password |
|---|---|---|
| Administrator | admin@neuroscan.local / NS-AD-0001 | Admin@2026 |
| Doctor | dr.mehta@neuroscan.local / NS-DR-0001 | Doctor@2026 |
| Doctor | dr.kapoor@neuroscan.local / NS-DR-0002 | Doctor@2026 |
| Technician | tech.rao@neuroscan.local / NS-TC-0001 | Tech@2026 |

Demo patients are fictional. Every demo analysis is real model output (images uploaded through the app).

## Models
- MRI: `kaggle_tumor_2dcnn_best.pth` — TumorCNN, 93.95 % test accuracy (training-time held-out split).
- X-ray: `xray_fracture_mnv3.pth` — MobileNetV3-Large fine-tuned by `scripts\train_xray.py`;
  97.7 % accuracy on 383 unseen test images after removing duplicate files from the Kaggle split.

## Scripts
- `scripts\train_xray.py` — train the X-ray model
- `scripts\eval_mri.py` — write MRI metrics
- `scripts\seed_demo.py` — reset DB with demo data
- `scripts\capture_figures.py` — regenerate report screenshots into `report_figures\`
- `scripts\insert_figures.py` — insert figures into a copy of the report .docx
