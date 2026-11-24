"""Load demo staff, fictional patients and REAL analyses (sample scans run through the app) into an
empty database. Enabled on hosting with AUTO_DEMO=1 so the live site always has demo content,
even on hosts whose temporary database resets on restart. Demo logins are listed in app/README.md."""
import os

from werkzeug.security import generate_password_hash

import db as dbm

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
STAFF = [  # code, name, email, password, role, department, patient_id
    ("NS-AD-0001", "Administrator", "admin@neuroscan.local", "Admin@2026", "admin", "IT & Systems", None),
    ("NS-DR-0001", "Dr. Ananya Mehta", "dr.mehta@neuroscan.local", "Doctor@2026", "doctor", "Radiology", None),
    ("NS-DR-0002", "Dr. Rohan Kapoor", "dr.kapoor@neuroscan.local", "Doctor@2026", "doctor", "Orthopaedics", None),
    ("NS-PA-0001", "Sunita Verma", "sunita.verma@neuroscan.local", "Patient@2026", "patient", "", 2),
]
PATIENTS = [
    ("Rajesh Kumar", 54, "Male", "B+", "Dr. Ananya Mehta", "Persistent headaches for 3 months, blurred vision."),
    ("Sunita Verma", 47, "Female", "O+", "Dr. Ananya Mehta", "Seizure episode, referred for brain MRI."),
    ("Arjun Singh", 29, "Male", "A+", "Dr. Rohan Kapoor", "Fall from bike, wrist pain and swelling."),
    ("Meera Iyer", 38, "Female", "AB+", "Dr. Ananya Mehta", "Hormonal imbalance; pituitary evaluation."),
    ("Kabir Malhotra", 63, "Male", "O-", "Dr. Ananya Mehta", "Memory complaints, routine screening MRI."),
    ("Neha Gupta", 22, "Female", "B-", "Dr. Rohan Kapoor", "Sports injury, ankle pain on weight bearing."),
    ("Farhan Ali", 41, "Male", "A-", "Dr. Rohan Kapoor", "Elbow pain after lifting injury."),
]
# patient no., modality, sample file, uploader (index in STAFF), body part, doctor remark
JOBS = [
    (2, "mri", "meningioma_1.jpg", 3, None, "AI finding noted. Recommend contrast-enhanced MRI and neurosurgery consultation."),
    (3, "xray", "xray_fracture_1.png", 2, "Hand / Wrist", "Fracture confirmed on review. Immobilise and refer to orthopaedics."),
    (1, "mri", "glioma_3.jpg", 1, None, None),
    (6, "xray", "xray_normal_1.png", 2, "Ankle / Foot", "No fracture. Soft-tissue injury; rest and follow-up."),
    (4, "mri", "pituitary_2.jpg", 1, None, None),
    (7, "xray", "xray_fracture_2.png", 2, "Elbow", None),
    (5, "mri", "notumor_2.jpg", 1, None, "No abnormality seen. Routine clinical follow-up advised."),
    (3, "xray", "xray_normal_2.png", 2, "Hand / Wrist", None),
    (1, "mri", "glioma_2.jpg", 1, None, None),
    (4, "mri", "pituitary_3.jpg", 1, None, None),
    (2, "mri", "meningioma_3.jpg", 3, None, None),
]


def run(app):
    con = dbm.connect()
    try:
        if con.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            return  # database already has data
        for i, (name, age, sex, bg, ref, hist) in enumerate(PATIENTS, 1):
            con.execute("""INSERT INTO patients (patient_code, full_name, age, sex, phone, blood_group, referred_by, history, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?)""", (f"PT-{i:04d}", name, age, sex, f"+91 98{i:03d} 4{i:02d}17", bg, ref, hist, dbm.now()))
        for code, name, email, pw, role, dept, pid in STAFF:
            con.execute("""INSERT INTO users (user_code, full_name, email, password_hash, role, patient_id, department, created_at)
                           VALUES (?,?,?,?,?,?,?,?)""", (code, name, email, generate_password_hash(pw), role, pid, dept, dbm.now()))
        con.commit()
    finally:
        con.close()

    c = app.test_client()
    for pno, mod, fn, uidx, part, remark in JOBS:
        c.get("/logout")
        c.post("/login", data={"identifier": STAFF[uidx][2], "password": STAFF[uidx][3]})
        with open(os.path.join(SAMPLES, fn), "rb") as fh:
            r = c.post("/analyze", data={"patient": str(pno), "modality": mod, "body_part": part or "", "image": (fh, fn)},
                       content_type="multipart/form-data")
        if r.status_code == 302 and remark:
            aid = r.headers["Location"].rstrip("/").rsplit("/", 1)[1]
            c.get("/logout")
            c.post("/login", data={"identifier": STAFF[1 if mod == "mri" else 2][2], "password": "Doctor@2026"})
            c.post(f"/analysis/{aid}/remarks", data={"remarks": remark})
    c.get("/logout")
    print("Demo data loaded.", flush=True)
