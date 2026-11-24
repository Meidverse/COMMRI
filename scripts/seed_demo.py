"""
Reset the NeuroScan AI database and fill it with demo staff, fictional patients and REAL analyses
(every analysis is produced by uploading an image through the app, so results are genuine model output).

Usage: python scripts/seed_demo.py
Demo sign-in accounts are listed in app/README.md.
"""
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
os.chdir(os.path.join(ROOT, "app"))

import db as dbm  # noqa: E402

if os.path.exists(dbm.INSTANCE):
    for p in (dbm.DB_PATH, dbm.UPLOADS):
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)

from werkzeug.security import generate_password_hash  # noqa: E402

import app as webapp  # noqa: E402  (creates schema)
import ml  # noqa: E402

MRI_DIR = r"C:\all projects\zz_projects\medAI\mri\dataset\classification\test"
XRAY_DIR = r"C:\all projects\zz_projects\medAI\xray\dataset\test"

STAFF = [
    ("NS-AD-0001", "Vishal Kumar Upadhyay", "admin@neuroscan.local", "Admin@2026", "admin", "IT & Systems", ""),
    ("NS-DR-0001", "Dr. Ananya Mehta", "dr.mehta@neuroscan.local", "Doctor@2026", "doctor", "Radiology", "MCI-58213"),
    ("NS-DR-0002", "Dr. Rohan Kapoor", "dr.kapoor@neuroscan.local", "Doctor@2026", "doctor", "Orthopaedics", "MCI-60477"),
    ("NS-PA-0001", "Sunita Verma", "sunita.verma@neuroscan.local", "Patient@2026", "patient", "", ""),
]
PATIENTS = [
    ("Rajesh Kumar", 54, "Male", "B+", "Dr. Ananya Mehta", "Persistent headaches for 3 months, blurred vision."),
    ("Sunita Verma", 47, "Female", "O+", "Dr. Ananya Mehta", "Seizure episode, referred for brain MRI."),
    ("Arjun Singh", 29, "Male", "A+", "Dr. Rohan Kapoor", "Fall from bike, wrist pain and swelling."),
    ("Meera Iyer", 38, "Female", "AB+", "Dr. Ananya Mehta", "Hormonal imbalance, irregular cycles; pituitary evaluation."),
    ("Kabir Malhotra", 63, "Male", "O-", "Dr. Ananya Mehta", "Memory complaints, routine screening MRI."),
    ("Neha Gupta", 22, "Female", "B-", "Dr. Rohan Kapoor", "Sports injury, ankle pain on weight bearing."),
    ("Farhan Ali", 41, "Male", "A-", "Dr. Rohan Kapoor", "Elbow pain after lifting injury."),
]
# (patient index, modality, image path glob, uploader index)
JOBS = [
    (1, "mri", os.path.join(MRI_DIR, "meningioma", "meningioma_1002.jpg"), 3),
    (0, "mri", os.path.join(MRI_DIR, "glioma", "glioma_1008.jpg"), 1),
    (3, "mri", os.path.join(MRI_DIR, "pituitary", "pituitary_1031.jpg"), 3),
    (4, "mri", os.path.join(MRI_DIR, "notumor", "notumor_10.jpg"), 1),
    (0, "mri", os.path.join(MRI_DIR, "glioma", "glioma_1035.jpg"), 1),
    (3, "mri", os.path.join(MRI_DIR, "pituitary", "pituitary_1021.jpg"), 3),
    (4, "mri", os.path.join(MRI_DIR, "notumor", "notumor_1017.jpg"), 3),
    (1, "mri", os.path.join(MRI_DIR, "meningioma", "meningioma_1037.jpg"), 1),
]
XRAY_JOBS = [(2, "fractured", 0, "Hand / Wrist"), (5, "not fractured", 1, "Ankle / Foot"), (6, "fractured", 4, "Elbow"),
             (2, "not fractured", 7, "Hand / Wrist"), (5, "fractured", 9, "Ankle / Foot")]
REMARKS = {
    1: "AI finding noted. Recommend contrast-enhanced MRI and neurosurgery consultation.",
    2: "Correlates with presenting symptoms. Plan follow-up imaging in 6 weeks.",
    4: "No abnormality seen. Routine clinical follow-up advised.",
}

client = webapp.app.test_client()
con = dbm.connect()
for i, (name, age, sex, bg, ref, hist) in enumerate(PATIENTS, 1):
    con.execute("INSERT INTO patients (patient_code, full_name, age, sex, phone, blood_group, referred_by, history, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"PT-{i:04d}", name, age, sex, f"+91 98{i:03d} 4{i:02d}17", bg, ref, hist, None, dbm.now()))
for code, name, email, pw, role, dept, lic in STAFF:
    con.execute("INSERT INTO users (user_code, full_name, email, password_hash, role, patient_id, department, license_no, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (code, name, email, generate_password_hash(pw), role, 2 if role == "patient" else None, dept, lic, dbm.now()))
con.commit()

xray_jobs = []
if ml.available("xray"):
    for pi, cls, k, part in XRAY_JOBS:
        files = sorted(glob.glob(os.path.join(XRAY_DIR, cls, "*")))
        xray_jobs.append((pi, "xray", files[k * 7 % len(files)], 2, part))
jobs = [(p, m, f, 3 if p == 1 else 1, None) for p, m, f, u in JOBS]  # patient account uploads own scans
order = []
for i in range(max(len(jobs), len(xray_jobs))):
    if i < len(jobs): order.append(jobs[i])
    if i < len(xray_jobs): order.append(xray_jobs[i])

def login(idx):
    client.get("/logout")
    r = client.post("/login", data={"identifier": STAFF[idx][2], "password": STAFF[idx][3]})
    assert r.status_code == 302, "login failed"

for n, (pi, mod, path, uidx, part) in enumerate(order, 1):
    login(uidx)
    cands = [path] + sorted(glob.glob(os.path.join(os.path.dirname(path), "*")))
    for path in cands:  # skip images the input validation rejects
        with open(path, "rb") as fh:
            r = client.post("/analyze", data={"patient": str(pi + 1), "modality": mod, "body_part": part or "",
                                              "image": (fh, os.path.basename(path))}, content_type="multipart/form-data")
        if r.status_code != 400:
            break
        print("  validation rejected", os.path.basename(path))
    assert r.status_code == 302, f"analysis failed for {path}: {r.status_code}"
    aid = int(r.headers["Location"].rsplit("/", 1)[1])
    print(f"AN #{aid}: {mod} {os.path.basename(path)}")
    if aid in REMARKS:
        login(1 if mod == "mri" else 2)
        client.post(f"/analysis/{aid}/remarks", data={"remarks": REMARKS[aid]})

for row in dbm.connect().execute("SELECT analysis_code, modality, label, round(confidence*100,1) FROM analyses"):
    print(tuple(row))
print("Seed complete.")
