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
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(records)")}
        for col in ("verdict", "true_label"):
            if col not in existing:
                conn.execute(f"ALTER TABLE records ADD COLUMN {col} TEXT")

    # Seed a demo account so the app is usable immediately.
    if get_doctor_by_username("admin") is None:
        create_doctor("admin", "admin", "Doctor Demo")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now():
    return datetime.utcnow().isoformat(timespec="seconds")


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
    return {"id": doc["id"], "username": doc["username"], "name": doc["name"]}


# ──────────────────────────────────────────────────────────────────────────────
# Patients
# ──────────────────────────────────────────────────────────────────────────────

def patient_public(row, record_count=None) -> dict:
    age = compute_age(row["dob"])
    out = {
        "id": row["id"],
        "name": row["name"],
        "dob": row["dob"],
        "age": age,
        "gender": row["gender"],
        "notes": row["notes"],
        "created_at": row["created_at"],
    }
    if record_count is not None:
        out["record_count"] = record_count
    return out


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


def create_patient(doctor_id, name, dob, gender, notes) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Patient name is required.")
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients (doctor_id, name, dob, gender, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doctor_id, name, dob or None, gender or None, notes or None, _now()),
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


def update_patient(patient_id, doctor_id, name, dob, gender, notes):
    if get_patient(patient_id, doctor_id) is None:
        return None
    name = (name or "").strip()
    if not name:
        raise ValueError("Patient name is required.")
    with _conn() as conn:
        conn.execute(
            "UPDATE patients SET name = ?, dob = ?, gender = ?, notes = ? "
            "WHERE id = ? AND doctor_id = ?",
            (name, dob or None, gender or None, notes or None, patient_id, doctor_id),
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

def dashboard_stats() -> dict:
    """
    Aggregate stats for the whole program: prediction counts overall, by gender
    and by age group, plus totals.  Returns raw counts keyed by class code; the
    Flask route enriches them with class colours/names.
    """
    with _conn() as conn:
        totals = {
            "doctors":  conn.execute("SELECT COUNT(*) c FROM doctors").fetchone()["c"],
            "patients": conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"],
            "records":  conn.execute("SELECT COUNT(*) c FROM records").fetchone()["c"],
        }

        rows = conn.execute(
            """
            SELECT r.prediction AS pred, r.confidence AS conf,
                   r.verdict AS verdict, r.true_label AS true_label,
                   r.created_at AS created_at,
                   p.gender AS gender, p.dob AS dob
            FROM records r
            JOIN patients p ON p.id = r.patient_id
            WHERE r.prediction IS NOT NULL
            """
        ).fetchall()

    by_class = {}
    conf_sum = {}
    by_gender = {}      # gender -> {code: count}
    by_age = {}         # group  -> {code: count}
    by_date = {}        # 'YYYY-MM-DD' -> {code: count}

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

        day = (r["created_at"] or "")[:10]
        if day:
            _bump(by_date, day, code)

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

    timeline = [
        {"date": d, "counts": by_date[d], "total": sum(by_date[d].values())}
        for d in sorted(by_date)
    ]

    return {
        "totals": totals,
        "by_class": by_class,
        "avg_confidence": avg_conf,
        "accuracy": accuracy,
        "timeline": timeline,
        "by_gender": [
            {"gender": g, "counts": by_gender[g], "total": sum(by_gender[g].values())}
            for g in sorted(by_gender)
        ],
        "by_age": [
            {"group": g, "counts": by_age[g], "total": sum(by_age[g].values())}
            for g in age_order if g in by_age
        ],
    }
