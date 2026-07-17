"""
Persistence layer for the ECG app — doctors, patients, ECG records and the
dashboard aggregates.  Plain SQLite (one file, ``ecg_app.db``) so the data is
persistent across restarts with zero external services.

Design
------
* Each doctor owns their own patients (``patients.doctor_id``).  The dashboard,
  however, aggregates across *all* records in the system ("todo el programa").
* An ECG record is de-duplicated per patient by the SHA-256 of the uploaded
  file, so re-analysing the same file for the same patient returns the stored
  result instead of creating a duplicate.
* Passwords are hashed with Werkzeug (never stored in plain text).  Sessions are
  opaque bearer tokens kept in the ``sessions`` table.
"""

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import date, datetime

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecg_app.db")

# Age buckets used by the dashboard.
_AGE_GROUPS = [
    ("0-17",  0,  17),
    ("18-39", 18, 39),
    ("40-59", 40, 59),
    ("60-79", 60, 79),
    ("80+",   80, 200),
]


# ──────────────────────────────────────────────────────────────────────────────
# Connection & schema
# ──────────────────────────────────────────────────────────────────────────────

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if missing and seed a demo doctor (admin / admin)."""
    with _conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name          TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                doctor_id  INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patients (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id  INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                name       TEXT NOT NULL,
                dob        TEXT,               -- ISO YYYY-MM-DD
                gender     TEXT,               -- 'F' | 'M' | 'Other'
                notes      TEXT,               -- extra relevant info
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS records (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id   INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                doctor_id    INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                filename     TEXT,
                file_hash    TEXT NOT NULL,
                prediction   TEXT,
                confidence   REAL,
                class_probs  TEXT,             -- JSON {code: prob}
                result_json  TEXT,             -- full /predict response, for re-render
                doctor_notes TEXT,
                verdict      TEXT,             -- 'correct' | 'incorrect' | NULL (unreviewed)
                true_label   TEXT,             -- actual class when marked incorrect
                created_at   TEXT NOT NULL,
                UNIQUE(patient_id, file_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_patients_doctor ON patients(doctor_id);
            CREATE INDEX IF NOT EXISTS idx_records_patient ON records(patient_id);
            """
        )

        # ── Lightweight migrations for databases created before these columns ──
        rec_cols = {r["name"] for r in conn.execute("PRAGMA table_info(records)")}
        for col in ("verdict", "true_label"):
            if col not in rec_cols:
                conn.execute(f"ALTER TABLE records ADD COLUMN {col} TEXT")

        pat_cols = {r["name"] for r in conn.execute("PRAGMA table_info(patients)")}
        for col in ("cedula", "first_name", "last_name"):
            if col not in pat_cols:
                conn.execute(f"ALTER TABLE patients ADD COLUMN {col} TEXT")

        doc_cols = {r["name"] for r in conn.execute("PRAGMA table_info(doctors)")}
        if "avatar_color" not in doc_cols:
            conn.execute("ALTER TABLE doctors ADD COLUMN avatar_color TEXT")

    # Seed a demo account so the app is usable immediately.
    if get_doctor_by_username("admin") is None:
        create_doctor("admin", "admin", "Doctor Demo")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now():
    # Local time so that stored dates line up with the doctor's calendar and the
    # dashboard date-range filters (which use the client's local dates).
    return datetime.now().isoformat(timespec="seconds")


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def compute_age(dob: str):
    """Age in whole years from an ISO YYYY-MM-DD string, or None if unparseable."""
    if not dob:
        return None
    try:
        b = date.fromisoformat(dob[:10])
    except (ValueError, TypeError):
        return None
    today = date.today()
    years = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
    return years if 0 <= years <= 200 else None


def _age_group(age):
    if age is None:
        return "Unknown"
    for label, lo, hi in _AGE_GROUPS:
        if lo <= age <= hi:
            return label
    return "Unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Doctors & auth
# ──────────────────────────────────────────────────────────────────────────────

def get_doctor_by_username(username: str):
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM doctors WHERE username = ?", (username,)
        ).fetchone()


def create_doctor(username: str, password: str, name: str):
    """Create a doctor. Returns the new row, or raises ValueError if taken."""
    username = (username or "").strip().lower()
    name = (name or "").strip() or username
    if not username or not password:
        raise ValueError("Username and password are required.")
    if get_doctor_by_username(username) is not None:
        raise ValueError("That username is already taken.")
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO doctors (username, password_hash, name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), name, _now()),
        )
        doctor_id = cur.lastrowid
        return conn.execute(
            "SELECT * FROM doctors WHERE id = ?", (doctor_id,)
        ).fetchone()


def verify_login(username: str, password: str):
    doc = get_doctor_by_username((username or "").strip().lower())
    if doc and check_password_hash(doc["password_hash"], password or ""):
        return doc
    return None


def create_session(doctor_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, doctor_id, created_at) VALUES (?, ?, ?)",
            (token, doctor_id, _now()),
        )
    return token


def get_doctor_by_token(token: str):
    if not token:
        return None
    with _conn() as conn:
        return conn.execute(
            "SELECT d.* FROM doctors d JOIN sessions s ON s.doctor_id = d.id "
            "WHERE s.token = ?",
            (token,),
        ).fetchone()


def delete_session(token: str):
    with _conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def doctor_public(doc) -> dict:
    keys = doc.keys()
    return {
        "id": doc["id"],
        "username": doc["username"],
        "name": doc["name"],
        "avatar_color": doc["avatar_color"] if "avatar_color" in keys else None,
    }


def update_doctor(doctor_id, name=None, avatar_color=None, password=None):
    """Update the logged-in doctor's own profile. Only provided fields change."""
    with _conn() as conn:
        doc = conn.execute(
            "SELECT * FROM doctors WHERE id = ?", (doctor_id,)
        ).fetchone()
        if doc is None:
            return None
        new_name = (name or "").strip() or doc["name"]
        new_color = avatar_color if avatar_color is not None else doc["avatar_color"]
        if password:
            conn.execute(
                "UPDATE doctors SET name = ?, avatar_color = ?, password_hash = ? "
                "WHERE id = ?",
                (new_name, new_color, generate_password_hash(password), doctor_id),
            )
        else:
            conn.execute(
                "UPDATE doctors SET name = ?, avatar_color = ? WHERE id = ?",
                (new_name, new_color, doctor_id),
            )
        doc = conn.execute(
            "SELECT * FROM doctors WHERE id = ?", (doctor_id,)
        ).fetchone()
    return doctor_public(doc)


# ──────────────────────────────────────────────────────────────────────────────
# Patients
# ──────────────────────────────────────────────────────────────────────────────

def _full_name(first, last, fallback=None):
    full = f"{(first or '').strip()} {(last or '').strip()}".strip()
    return full or (fallback or "").strip()


def patient_public(row, record_count=None) -> dict:
    keys = row.keys()
    first = row["first_name"] if "first_name" in keys else None
    last = row["last_name"] if "last_name" in keys else None
    out = {
        "id": row["id"],
        "cedula": row["cedula"] if "cedula" in keys else None,
        "first_name": first,
        "last_name": last,
        "name": _full_name(first, last, row["name"]),
        "dob": row["dob"],
        "age": compute_age(row["dob"]),
        "gender": row["gender"],
        "notes": row["notes"],
        "created_at": row["created_at"],
    }
    if record_count is not None:
        out["record_count"] = record_count
    return out


def _patient_fields(d: dict):
    """Normalise incoming patient fields; returns (cedula, first, last, name, dob, gender, notes)."""
    first = (d.get("first_name") or "").strip()
    last = (d.get("last_name") or "").strip()
    name = _full_name(first, last, d.get("name"))
    if not name:
        raise ValueError("El nombre del paciente es obligatorio.")
    return (
        (d.get("cedula") or "").strip() or None,
        first or None,
        last or None,
        name,
        d.get("dob") or None,
        d.get("gender") or None,
        (d.get("notes") or "").strip() or None,
    )


def list_patients(doctor_id: int) -> list:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT p.*, COUNT(r.id) AS n
            FROM patients p
            LEFT JOIN records r ON r.patient_id = p.id
            WHERE p.doctor_id = ?
            GROUP BY p.id
            ORDER BY p.name COLLATE NOCASE
            """,
            (doctor_id,),
        ).fetchall()
    return [patient_public(r, record_count=r["n"]) for r in rows]


def create_patient(doctor_id, fields: dict) -> dict:
    cedula, first, last, name, dob, gender, notes = _patient_fields(fields)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients "
            "(doctor_id, cedula, first_name, last_name, name, dob, gender, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doctor_id, cedula, first, last, name, dob, gender, notes, _now()),
        )
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return patient_public(row, record_count=0)


def get_patient(patient_id, doctor_id):
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM patients WHERE id = ? AND doctor_id = ?",
            (patient_id, doctor_id),
        ).fetchone()


def update_patient(patient_id, doctor_id, fields: dict):
    if get_patient(patient_id, doctor_id) is None:
        return None
    cedula, first, last, name, dob, gender, notes = _patient_fields(fields)
    with _conn() as conn:
        conn.execute(
            "UPDATE patients SET cedula = ?, first_name = ?, last_name = ?, "
            "name = ?, dob = ?, gender = ?, notes = ? "
            "WHERE id = ? AND doctor_id = ?",
            (cedula, first, last, name, dob, gender, notes, patient_id, doctor_id),
        )
        row = conn.execute(
            "SELECT * FROM patients WHERE id = ?", (patient_id,)
        ).fetchone()
    return patient_public(row)


def delete_patient(patient_id, doctor_id) -> bool:
    if get_patient(patient_id, doctor_id) is None:
        return False
    with _conn() as conn:
        conn.execute(
            "DELETE FROM patients WHERE id = ? AND doctor_id = ?",
            (patient_id, doctor_id),
        )
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Records
# ──────────────────────────────────────────────────────────────────────────────

def record_public(row, include_full=False) -> dict:
    keys = row.keys()
    out = {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "filename": row["filename"],
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "class_probs": json.loads(row["class_probs"] or "{}"),
        "doctor_notes": row["doctor_notes"],
        "verdict": row["verdict"] if "verdict" in keys else None,
        "true_label": row["true_label"] if "true_label" in keys else None,
        "created_at": row["created_at"],
    }
    if include_full:
        out["result"] = json.loads(row["result_json"] or "{}")
    return out


def find_record_by_hash(patient_id, fhash):
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM records WHERE patient_id = ? AND file_hash = ?",
            (patient_id, fhash),
        ).fetchone()


def create_record(patient_id, doctor_id, filename, fhash, result: dict) -> dict:
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO records
                (patient_id, doctor_id, filename, file_hash, prediction,
                 confidence, class_probs, result_json, doctor_notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id, doctor_id, filename, fhash,
                result.get("prediction"),
                result.get("confidence"),
                json.dumps(result.get("class_probs", {})),
                json.dumps(result),
                None,
                _now(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return record_public(row, include_full=True)


def list_records(patient_id) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM records WHERE patient_id = ? ORDER BY created_at DESC, id DESC",
            (patient_id,),
        ).fetchall()
    return [record_public(r) for r in rows]


def get_record(record_id, doctor_id):
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM records WHERE id = ? AND doctor_id = ?",
            (record_id, doctor_id),
        ).fetchone()
    return record_public(row, include_full=True) if row else None


def update_record_notes(record_id, doctor_id, notes):
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE records SET doctor_notes = ? WHERE id = ? AND doctor_id = ?",
            (notes or None, record_id, doctor_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    return record_public(row, include_full=True)


def set_record_verdict(record_id, doctor_id, verdict, true_label=None):
    """
    Record the doctor's assessment of a prediction.

    verdict: 'correct' | 'incorrect' | None (clears the review).
    true_label: the actual class code, used when verdict == 'incorrect'.
    """
    if verdict not in ("correct", "incorrect", None):
        raise ValueError("verdict must be 'correct', 'incorrect' or null.")

    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM records WHERE id = ? AND doctor_id = ?",
            (record_id, doctor_id),
        ).fetchone()
        if row is None:
            return None

        if verdict == "correct":
            true_label = row["prediction"]
        elif verdict is None:
            true_label = None

        conn.execute(
            "UPDATE records SET verdict = ?, true_label = ? WHERE id = ?",
            (verdict, true_label, record_id),
        )
        row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()
    return record_public(row, include_full=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard — global aggregates across every record in the system
# ──────────────────────────────────────────────────────────────────────────────

def dashboard_stats(from_date=None, to_date=None, gender=None) -> dict:
    """
    Aggregate stats for the whole program: prediction counts overall, by gender
    and by age group, plus totals.  Returns raw counts keyed by class code; the
    Flask route enriches them with class colours/names.

    Optional filters (all inclusive):
      from_date / to_date : 'YYYY-MM-DD' bounds on the ECG date.
      gender              : 'F' | 'M' | 'Other' to restrict to one gender.
    """
    where = ["r.prediction IS NOT NULL"]
    params = []
    if from_date:
        where.append("date(r.created_at) >= date(?)")
        params.append(from_date)
    if to_date:
        where.append("date(r.created_at) <= date(?)")
        params.append(to_date)
    if gender:
        where.append("p.gender = ?")
        params.append(gender)
    where_sql = " AND ".join(where)

    with _conn() as conn:
        rows = conn.execute(
            f"""
            SELECT r.prediction AS pred, r.confidence AS conf,
                   r.verdict AS verdict, r.true_label AS true_label,
                   r.created_at AS created_at,
                   r.patient_id AS patient_id,
                   p.gender AS gender, p.dob AS dob
            FROM records r
            JOIN patients p ON p.id = r.patient_id
            WHERE {where_sql}
            """,
            params,
        ).fetchall()
        n_doctors = conn.execute("SELECT COUNT(*) c FROM doctors").fetchone()["c"]

    # Totals reflect the current filter.
    totals = {
        "doctors": n_doctors,
        "patients": len({r["patient_id"] for r in rows}),
        "records": len(rows),
    }

    by_class = {}
    conf_sum = {}
    by_gender = {}      # gender -> {code: count}
    by_age = {}         # group  -> {code: count}
    by_day = {}         # 'YYYY-MM-DD'    -> {code: count}
    by_hour = {}        # 'YYYY-MM-DDTHH' -> {code: count}

    # Accuracy from the doctor's verdicts.
    acc_by_class = {}   # predicted code -> [reviewed, correct]
    confusion = {}      # 'PRED→ACTUAL' -> count (incorrect only)
    reviewed = correct = 0

    def _bump(bucket, key, code):
        d = bucket.setdefault(key, {})
        d[code] = d.get(code, 0) + 1

    for r in rows:
        code = r["pred"]
        by_class[code] = by_class.get(code, 0) + 1
        conf_sum[code] = conf_sum.get(code, 0.0) + (r["conf"] or 0.0)

        gender = (r["gender"] or "Unknown").strip() or "Unknown"
        _bump(by_gender, gender, code)

        grp = _age_group(compute_age(r["dob"]))
        _bump(by_age, grp, code)

        created = r["created_at"] or ""
        if len(created) >= 10:
            _bump(by_day, created[:10], code)        # YYYY-MM-DD
        if len(created) >= 13:
            _bump(by_hour, created[:13], code)        # YYYY-MM-DDTHH

        verdict = r["verdict"]
        if verdict in ("correct", "incorrect"):
            reviewed += 1
            stat = acc_by_class.setdefault(code, [0, 0])
            stat[0] += 1
            if verdict == "correct":
                correct += 1
                stat[1] += 1
            else:
                actual = r["true_label"] or "?"
                key = f"{code}→{actual}"
                confusion[key] = confusion.get(key, 0) + 1

    avg_conf = {c: (conf_sum[c] / by_class[c]) for c in by_class}

    # Preserve a sensible, fixed age-group order (+ Unknown last if present).
    age_order = [g[0] for g in _AGE_GROUPS]
    if "Unknown" in by_age:
        age_order = age_order + ["Unknown"]

    accuracy = {
        "overall": {
            "reviewed": reviewed,
            "correct": correct,
            "unreviewed": totals["records"] - reviewed,
            "accuracy": (correct / reviewed) if reviewed else None,
        },
        "by_class": {
            c: {"reviewed": v[0], "correct": v[1],
                "accuracy": (v[1] / v[0]) if v[0] else None}
            for c, v in acc_by_class.items()
        },
        "confusion": confusion,
    }

    def _series(bucket):
        return [
            {"date": k, "counts": bucket[k], "total": sum(bucket[k].values())}
            for k in sorted(bucket)
        ]

    timelines = {"day": _series(by_day), "hour": _series(by_hour)}

    return {
        "totals": totals,
        "by_class": by_class,
        "avg_confidence": avg_conf,
        "accuracy": accuracy,
        "timeline": timelines["day"],   # back-compat
        "timelines": timelines,
        "by_gender": [
            {"gender": g, "counts": by_gender[g], "total": sum(by_gender[g].values())}
            for g in sorted(by_gender)
        ],
        "by_age": [
            {"group": g, "counts": by_age[g], "total": sum(by_age[g].values())}
            for g in age_order if g in by_age
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Risk & alerts — prioritise the doctor's patients by the model's findings
# ──────────────────────────────────────────────────────────────────────────────

# Rhythms considered "normal"; everything else counts as abnormal for triage.
NORMAL_CLASSES = {"SR", "NORM"}

# Clinical severity weight per class (higher = more urgent). Tuned for the 4
# classes the model handles; unknown abnormal classes default to 2.
_RISK_WEIGHT = {"AFIB": 3.0, "STACH": 2.0, "SBRAD": 2.0, "SR": 0.0, "NORM": 0.0}


def _risk_level(score):
    if score >= 2.4:
        return "alto"
    if score >= 1.2:
        return "medio"
    if score > 0:
        return "bajo"
    return "normal"


def risk_overview(doctor_id: int) -> dict:
    """
    Triage view for one doctor: patients ranked by risk (based on their latest
    ECG), recent abnormal findings, and abnormal ECGs still awaiting the doctor's
    confirmation (pending follow-up).
    """
    with _conn() as conn:
        patients = conn.execute(
            "SELECT * FROM patients WHERE doctor_id = ?", (doctor_id,)
        ).fetchall()

        recs = conn.execute(
            """
            SELECT r.*, p.name AS patient_name, p.dob AS dob, p.gender AS gender
            FROM records r
            JOIN patients p ON p.id = r.patient_id
            WHERE p.doctor_id = ?
            ORDER BY r.created_at DESC, r.id DESC
            """,
            (doctor_id,),
        ).fetchall()

    # Group records by patient (already newest-first).
    by_patient = {}
    for r in recs:
        by_patient.setdefault(r["patient_id"], []).append(r)

    def _abnormal(code):
        return code is not None and code not in NORMAL_CLASSES

    prioritized = []
    for p in patients:
        rlist = by_patient.get(p["id"], [])
        total = len(rlist)
        abnormal_count = sum(1 for r in rlist if _abnormal(r["prediction"]))
        latest = rlist[0] if rlist else None

        if latest is not None:
            code = latest["prediction"]
            conf = latest["confidence"] or 0.0
            weight = _RISK_WEIGHT.get(code, 2.0 if _abnormal(code) else 0.0)
            score = weight * (0.5 + 0.5 * conf)   # scale by confidence
            pending = latest["verdict"] is None and _abnormal(code)
        else:
            code, conf, score, pending = None, 0.0, 0.0, False

        prioritized.append({
            "patient_id": p["id"],
            "name": p["name"],
            "age": compute_age(p["dob"]),
            "gender": p["gender"],
            "total_ecgs": total,
            "abnormal_count": abnormal_count,
            "latest_prediction": code,
            "latest_confidence": conf,
            "latest_date": latest["created_at"] if latest else None,
            "pending_review": pending,
            "risk_score": round(score, 3),
            "risk_level": _risk_level(score),
        })

    prioritized.sort(key=lambda x: x["risk_score"], reverse=True)

    new_abnormal = [
        {
            "record_id": r["id"],
            "patient_id": r["patient_id"],
            "name": r["patient_name"],
            "prediction": r["prediction"],
            "confidence": r["confidence"] or 0.0,
            "created_at": r["created_at"],
            "verdict": r["verdict"],
        }
        for r in recs if _abnormal(r["prediction"])
    ][:12]

    pending_followup = [a for a in new_abnormal if a["verdict"] is None]

    counts = {
        "alto":   sum(1 for x in prioritized if x["risk_level"] == "alto"),
        "medio":  sum(1 for x in prioritized if x["risk_level"] == "medio"),
        "bajo":   sum(1 for x in prioritized if x["risk_level"] == "bajo"),
        "normal": sum(1 for x in prioritized if x["risk_level"] == "normal"),
    }

    return {
        "counts": counts,
        "prioritized": prioritized,
        "new_abnormal": new_abnormal,
        "pending_followup": pending_followup,
    }
