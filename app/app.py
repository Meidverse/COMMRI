"""
NeuroScan AI — Medical Image Analysis System.

Flask web application: user authentication with roles, patient management,
MRI brain tumour classification and X-ray fracture detection (PyTorch),
image preprocessing pipeline, Grad-CAM explanation, reports and analysis history.

Run:  python app\\app.py   then open http://127.0.0.1:5000
"""
import json
import os
import secrets
from datetime import datetime
from functools import wraps

from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   send_from_directory, session, url_for)
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash

import db as dbm
import ml

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROLES = {"admin": "Administrator", "doctor": "Doctor", "patient": "Patient"}
ROLE_PREFIX = {"admin": "NS-AD-", "doctor": "NS-DR-", "patient": "NS-PA-"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
dbm.init_db()
_key_file = os.path.join(dbm.INSTANCE, "secret.key")
if not os.path.exists(_key_file):
    with open(_key_file, "w") as fh:
        fh.write(secrets.token_hex(32))
app.secret_key = open(_key_file).read().strip()
app.teardown_appcontext(dbm.close_db)


# ------------------------------------------------------------------ helpers
@app.before_request
def load_user():
    g.user = None
    uid = session.get("uid")
    if uid:
        g.user = dbm.get_db().execute("SELECT * FROM users WHERE id=? AND active=1", (uid,)).fetchone()


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped


def roles_required(*roles):
    def deco(view):
        @wraps(view)
        @login_required
        def wrapped(*a, **kw):
            if g.user["role"] not in roles:
                abort(403)
            return view(*a, **kw)
        return wrapped
    return deco


_icons = app.jinja_env.get_template("_icons.html").module
app.jinja_env.globals.update(icon=_icons.icon, logo=_icons.logo)


@app.context_processor
def inject():
    return {"ROLES": ROLES, "MODALITIES": ml.MODALITIES, "nav": request.endpoint, "now_hour": datetime.now().hour}


@app.template_filter("pct")
def pct(v, digits=1):
    return f"{v * 100:.{digits}f}%"


@app.template_filter("initials")
def initials(name):
    parts = [p for p in (name or "").replace("Dr.", "").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def is_patient():
    return g.user is not None and g.user["role"] == "patient"


def can_see(patient_id):
    return not is_patient() or g.user["patient_id"] == patient_id


def scope(alias="a"):
    """SQL filter limiting analyses to the signed-in patient (no filter for staff)."""
    return (f" AND {alias}.patient_id=?", [g.user["patient_id"]]) if is_patient() else ("", [])


def get_analysis(aid):
    row = dbm.get_db().execute(
        """SELECT a.*, p.full_name AS patient_name, p.patient_code, p.age, p.sex, p.referred_by,
                  p.phone AS patient_phone, p.blood_group, p.history,
                  u.full_name AS user_name, u.user_code, u.role AS user_role,
                  r.full_name AS reviewer_name, r.user_code AS reviewer_code
           FROM analyses a JOIN patients p ON p.id=a.patient_id JOIN users u ON u.id=a.user_id
           LEFT JOIN users r ON r.id=a.reviewed_by WHERE a.id=?""", (aid,)).fetchone()
    if row is None:
        abort(404)
    if not can_see(row["patient_id"]):
        abort(403)
    a = dict(row)
    a["probabilities"] = json.loads(a["probabilities"])
    a["details"] = json.loads(a["details"])
    return a


# ------------------------------------------------------------------ auth
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ident = request.form.get("identifier", "").strip()
        pw = request.form.get("password", "")
        db = dbm.get_db()
        u = db.execute("SELECT * FROM users WHERE (lower(email)=lower(?) OR upper(user_code)=upper(?))", (ident, ident)).fetchone()
        if u and u["active"] and check_password_hash(u["password_hash"], pw):
            session.clear()
            session["uid"] = u["id"]
            db.execute("UPDATE users SET last_login=? WHERE id=?", (dbm.now(), u["id"]))
            db.commit()
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Invalid user ID / email or password.", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    form = request.form
    if request.method == "POST":
        role = form.get("role")
        errors = []
        if role not in ("doctor", "patient"):
            errors.append("Choose a valid role.")
        if len(form.get("full_name", "").strip()) < 3:
            errors.append("Enter your full name.")
        if "@" not in form.get("email", ""):
            errors.append("Enter a valid email address.")
        if len(form.get("password", "")) < 8:
            errors.append("Password must be at least 8 characters.")
        if form.get("password") != form.get("confirm"):
            errors.append("Passwords do not match.")
        db = dbm.get_db()
        if not errors and db.execute("SELECT 1 FROM users WHERE lower(email)=lower(?)", (form["email"].strip(),)).fetchone():
            errors.append("An account with this email already exists.")
        if errors:
            for e in errors:
                flash(e, "error")
        else:
            pid = None
            if role == "patient":  # patient portal account is linked to a patient record
                pcode = dbm.next_code(db, "patients", "patient_code", "PT-")
                age = form.get("department", "").strip()  # register form: "Department / Age"
                pid = db.execute("""INSERT INTO patients (patient_code, full_name, age, sex, phone, created_at)
                                    VALUES (?,?,?,?,?,?)""",
                                 (pcode, form["full_name"].strip(), int(age) if age.isdigit() else None, form.get("sex"),
                                  form.get("phone", "").strip(), dbm.now())).lastrowid
            code = dbm.next_code(db, "users", "user_code", ROLE_PREFIX[role])
            db.execute("""INSERT INTO users (user_code, full_name, email, password_hash, role, patient_id, department, license_no, phone, created_at)
                          VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (code, form["full_name"].strip(), form["email"].strip(), generate_password_hash(form["password"]),
                        role, pid, form.get("department", "").strip(), form.get("license_no", "").strip(),
                        form.get("phone", "").strip(), dbm.now()))
            db.commit()
            flash(f"Account created. Your unique user ID is {code} — use it or your email to sign in.", "success")
            return redirect(url_for("login"))
    return render_template("register.html", form=form)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------ dashboard
@app.get("/")
@login_required
def dashboard():
    db = dbm.get_db()
    f, prm = scope()
    q = lambda s: db.execute(s.replace("WHERE 1", "WHERE 1" + f), prm).fetchone()[0]
    stats = {
        "analyses": q("SELECT COUNT(*) FROM analyses a WHERE 1"),
        "patients": db.execute("SELECT COUNT(*) FROM patients").fetchone()[0],
        "abnormal": q("SELECT COUNT(*) FROM analyses a WHERE 1 AND abnormal=1"),
        "pending": q("SELECT COUNT(*) FROM analyses a WHERE 1 AND reviewed_by IS NULL"),
        "today": q("SELECT COUNT(*) FROM analyses a WHERE 1 AND date(created_at)=date('now','localtime')"),
        "avg_conf": q("SELECT COALESCE(AVG(confidence),0) FROM analyses a WHERE 1"),
    }
    dist = db.execute(f"SELECT modality, label, abnormal, COUNT(*) n FROM analyses a WHERE 1{f} GROUP BY modality, label ORDER BY n DESC", prm).fetchall()
    recent = db.execute(f"""SELECT a.*, p.full_name patient_name, p.patient_code FROM analyses a
                           JOIN patients p ON p.id=a.patient_id WHERE 1{f} ORDER BY a.id DESC LIMIT 6""", prm).fetchall()
    return render_template("dashboard.html", stats=stats, dist=dist, recent=recent)


# ------------------------------------------------------------------ patients
@app.route("/patients", methods=["GET", "POST"])
@roles_required("doctor", "admin")
def patients():
    db = dbm.get_db()
    if request.method == "POST":
        f = request.form
        if len(f.get("full_name", "").strip()) < 2:
            flash("Patient name is required.", "error")
        else:
            code = dbm.next_code(db, "patients", "patient_code", "PT-")
            age = f.get("age", "").strip()
            db.execute("""INSERT INTO patients (patient_code, full_name, age, sex, phone, blood_group, referred_by, history, created_by, created_at)
                          VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (code, f["full_name"].strip(), int(age) if age.isdigit() else None, f.get("sex"), f.get("phone", "").strip(),
                        f.get("blood_group"), f.get("referred_by", "").strip(), f.get("history", "").strip(), g.user["id"], dbm.now()))
            db.commit()
            flash(f"Patient {code} registered.", "success")
            if request.form.get("then") == "analyze":
                pid = db.execute("SELECT id FROM patients WHERE patient_code=?", (code,)).fetchone()[0]
                return redirect(url_for("analyze", patient=pid))
        return redirect(url_for("patients"))
    s = request.args.get("q", "").strip()
    rows = db.execute("""SELECT p.*, COUNT(a.id) n, MAX(a.created_at) last_at,
                                SUM(a.abnormal) n_abn FROM patients p LEFT JOIN analyses a ON a.patient_id=p.id
                         WHERE p.full_name LIKE ? OR p.patient_code LIKE ?
                         GROUP BY p.id ORDER BY p.id DESC""", (f"%{s}%", f"%{s}%")).fetchall()
    return render_template("patients.html", patients=rows, q=s)


@app.get("/patients/<int:pid>")
@login_required
def patient_detail(pid):
    db = dbm.get_db()
    if not can_see(pid):
        abort(403)
    p = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone() or abort(404)
    rows = db.execute("SELECT a.*, u.full_name user_name FROM analyses a JOIN users u ON u.id=a.user_id WHERE patient_id=? ORDER BY a.id DESC", (pid,)).fetchall()
    return render_template("patient_detail.html", p=p, rows=rows)


# ------------------------------------------------------------------ analysis
@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    db = dbm.get_db()
    f_, prm = scope("patients")
    pats = db.execute(f"SELECT id, patient_code, full_name, age, sex FROM patients WHERE 1{f_.replace('patients.patient_id', 'patients.id')} ORDER BY id DESC", prm).fetchall()
    ctx = {"patients": pats, "sel_patient": g.user["patient_id"] if is_patient() else request.values.get("patient", type=int),
           "sel_modality": request.values.get("modality", "mri"), "available": {k: ml.available(k) for k in ml.MODALITIES}}
    if request.method == "GET":
        return render_template("analyze.html", **ctx)

    modality = request.form.get("modality")
    pid = g.user["patient_id"] if is_patient() else request.form.get("patient", type=int)
    f = request.files.get("image")
    err = None
    if modality not in ml.MODALITIES or not ml.available(modality):
        err = "Select an available imaging modality."
    elif not pid or not db.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone():
        err = "Select the patient this scan belongs to."
    elif f is None or not f.filename:
        err = "Choose an image file to upload."
    elif os.path.splitext(f.filename)[1].lower() not in ALLOWED_EXT:
        err = "Unsupported file type. Upload JPG, PNG, BMP or TIFF."
    if err is None:
        try:
            img = Image.open(f.stream)
            img.load()
        except (UnidentifiedImageError, OSError):
            err = "The file could not be read as an image."
    if err is None:
        ok, checks = ml.validate(img, modality)
        if not ok:
            failed = next(c for c in checks if not c[1])
            name = ml.MODALITIES[modality]["name"]
            if failed[0].startswith("Recognised"):
                err = f"This image does not look like a {name} scan, so it was not analysed. Upload a {name} image or choose the correct modality."
            elif failed[0].startswith("Greyscale"):
                err = f"This looks like a colour photograph, not a medical scan ({failed[2]}). Upload a {name} image."
            else:
                err = f"Input validation failed: {failed[0]} ({failed[2]})."
    if err:
        flash(err, "error")
        return render_template("analyze.html", **ctx), 400

    code = dbm.next_code(db, "analyses", "analysis_code", "AN-2026-")
    out_dir = os.path.join(dbm.UPLOADS, code)
    res = ml.run_analysis(modality, img, out_dir)
    details = {k: res[k] for k in ("description", "image_w", "image_h", "histogram", "stats")}
    details["checks"] = checks
    cur = db.execute(
        """INSERT INTO analyses (analysis_code, patient_id, user_id, modality, body_part, original_filename, prediction, label,
                                 abnormal, confidence, probabilities, details, inference_ms, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (code, pid, g.user["id"], modality, "Brain" if modality == "mri" else request.form.get("body_part") or "Bone",
         os.path.basename(f.filename), res["prediction"], res["label"], int(res["abnormal"]), res["confidence"],
         json.dumps(res["probabilities"]), json.dumps(details), res["inference_ms"], dbm.now()))
    db.commit()
    return redirect(url_for("result", aid=cur.lastrowid))


@app.get("/analysis/<int:aid>")
@login_required
def result(aid):
    return render_template("result.html", a=get_analysis(aid), m=ml.MODALITIES)


@app.get("/analysis/<int:aid>/processing")
@login_required
def processing(aid):
    return render_template("processing.html", a=get_analysis(aid), size=ml.INPUT_SIZE)


@app.post("/analysis/<int:aid>/remarks")
@roles_required("doctor", "admin")
def remarks(aid):
    db = dbm.get_db()
    db.execute("UPDATE analyses SET remarks=?, reviewed_by=?, reviewed_at=? WHERE id=?",
               (request.form.get("remarks", "").strip(), g.user["id"], dbm.now(), aid))
    db.commit()
    flash("Review saved and attached to the report.", "success")
    return redirect(url_for("result", aid=aid))


@app.get("/report/<int:aid>")
@login_required
def report(aid):
    return render_template("report.html", a=get_analysis(aid))


@app.get("/media/<code>/<name>")
@login_required
def media(code, name):
    if name not in {"original.png", "resized.png", "normalized.png", "enhanced.png"}:
        abort(404)
    row = dbm.get_db().execute("SELECT patient_id FROM analyses WHERE analysis_code=?", (code,)).fetchone()
    if row is None or not can_see(row[0]):
        abort(404)
    return send_from_directory(os.path.join(dbm.UPLOADS, os.path.basename(code)), name)


@app.get("/history")
@login_required
def history():
    db = dbm.get_db()
    s, mod, res = request.args.get("q", "").strip(), request.args.get("modality", ""), request.args.get("result", "")
    sql = """SELECT a.*, p.full_name patient_name, p.patient_code, u.full_name user_name FROM analyses a
             JOIN patients p ON p.id=a.patient_id JOIN users u ON u.id=a.user_id
             WHERE (p.full_name LIKE ? OR p.patient_code LIKE ? OR a.analysis_code LIKE ?)"""
    params = [f"%{s}%"] * 3
    f_, prm = scope()
    sql += f_; params += prm
    if mod in ml.MODALITIES:
        sql += " AND a.modality=?"; params.append(mod)
    if res == "abnormal":
        sql += " AND a.abnormal=1"
    elif res == "normal":
        sql += " AND a.abnormal=0"
    elif res == "pending":
        sql += " AND a.reviewed_by IS NULL"
    rows = db.execute(sql + " ORDER BY a.id DESC", params).fetchall()
    total = db.execute(f"SELECT COUNT(*) FROM analyses a WHERE 1{f_}", prm).fetchone()[0]
    return render_template("history.html", rows=rows, q=s, modality=mod, res=res, total=total)


# ------------------------------------------------------------------ staff (doctor management)
@app.route("/staff", methods=["GET", "POST"])
@roles_required("admin")
def staff():
    db = dbm.get_db()
    if request.method == "POST":
        f = request.form
        if f.get("action") == "toggle":
            uid = int(f["uid"])
            if uid != g.user["id"]:
                db.execute("UPDATE users SET active = 1 - active WHERE id=?", (uid,))
                db.commit()
            return redirect(url_for("staff"))
        role = f.get("role")
        if role not in ("doctor", "admin") or "@" not in f.get("email", "") or len(f.get("password", "")) < 8 or len(f.get("full_name", "").strip()) < 3:
            flash("Fill name, valid email, role and a password of 8+ characters.", "error")
        elif db.execute("SELECT 1 FROM users WHERE lower(email)=lower(?)", (f["email"].strip(),)).fetchone():
            flash("Email already registered.", "error")
        else:
            code = dbm.next_code(db, "users", "user_code", ROLE_PREFIX[role])
            db.execute("""INSERT INTO users (user_code, full_name, email, password_hash, role, department, license_no, phone, created_at)
                          VALUES (?,?,?,?,?,?,?,?,?)""",
                       (code, f["full_name"].strip(), f["email"].strip(), generate_password_hash(f["password"]), role,
                        f.get("department", "").strip(), f.get("license_no", "").strip(), f.get("phone", "").strip(), dbm.now()))
            db.commit()
            flash(f"{ROLES[role]} account {code} created.", "success")
        return redirect(url_for("staff"))
    rows = db.execute("""SELECT u.*, COUNT(a.id) n FROM users u LEFT JOIN analyses a ON a.user_id=u.id
                         GROUP BY u.id ORDER BY CASE u.role WHEN 'admin' THEN 0 WHEN 'doctor' THEN 1 ELSE 2 END, u.id""").fetchall()
    return render_template("staff.html", rows=rows)


@app.errorhandler(403)
def forbidden(_e):
    return render_template("error.html", code=403, msg="Your role does not have access to this page."), 403


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, msg="That page or record does not exist."), 404


@app.errorhandler(413)
def too_big(_e):
    flash("File too large (max 20 MB).", "error")
    return redirect(url_for("analyze"))


if __name__ == "__main__":
    print("NeuroScan AI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
